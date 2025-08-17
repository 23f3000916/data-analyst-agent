import io
import os
import re
import math
import json
from typing import Dict, Any, List, Tuple, Optional
import base64
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import UploadFile
import matplotlib
matplotlib.use("Agg") # headless
import matplotlib.pyplot as plt
from PIL import Image
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize FastAPI app
app = FastAPI(title="Data Analyst Agent API")

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
if GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
else:
    llm = None

IMAGE_SIZE_LIMIT = 100_000 # 100 kB cap per guideline

# ---------------------------
# Multipart reading
# ---------------------------
async def read_form(request: Request) -> Tuple[str, Dict[str, UploadFile]]:
    form = await request.form()
    questions_text = ""
    files: Dict[str, UploadFile] = {}

    if "questions.txt" in form and isinstance(form["questions.txt"], UploadFile):
        questions_text = (await form["questions.txt"].read()).decode("utf-8", errors="replace")

    for key, value in form.items():
        if isinstance(value, UploadFile):
            files[key] = value

    return questions_text.strip(), files

def read_csv_upload(upload: UploadFile) -> Optional[pd.DataFrame]:
    try:
        data = upload.file.read()
        upload.file.seek(0)
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return None

def load_first_csv(files: Dict[str, UploadFile]) -> Optional[pd.DataFrame]:
    for k, f in files.items():
        if k.lower() == "data.csv":
            return read_csv_upload(f)
    for _, f in files.items():
        name = (f.filename or "").lower()
        if name.endswith(".csv"):
            return read_csv_upload(f)
    return None

# ---------------------------
# Web Scraping
# ---------------------------
def scrape_wikipedia_table(url: str) -> Optional[pd.DataFrame]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table", {"class": "wikitable"})
        if table:
            return pd.read_html(str(table))[0]
    except Exception as e:
        print(f"Error scraping Wikipedia table: {e}")
    return None

# ---------------------------
# Data Analysis
# ---------------------------
def analyze_movie_data(df: pd.DataFrame) -> Dict[str, Any]:
    # Clean and preprocess data
    df.columns = df.columns.str.replace(r"\[.*\]", "", regex=True)
    df["Worldwide gross"] = df["Worldwide gross"].replace(r'[\$,]', '', regex=True).astype(float)
    df["Year"] = df["Year"].astype(int)

    # Perform analysis
    movies_before_2000 = df[df["Year"] < 2000]
    two_billion_movies_before_2000 = movies_before_2000[movies_before_2000["Worldwide gross"] > 2_000_000_000]
    earliest_film_over_1_5_billion = df[df["Worldwide gross"] > 1_500_000_000].sort_values("Year").iloc[0]

    # Correlation and scatterplot
    df["Rank"] = df.index + 1
    correlation = df["Rank"].corr(df["Peak"])
    scatterplot = create_scatterplot(df["Rank"], df["Peak"], "Rank", "Peak")

    return {
        "two_billion_movies_before_2000": len(two_billion_movies_before_2000),
        "earliest_film_over_1_5_billion": earliest_film_over_1_5_billion["Title"],
        "correlation": correlation,
        "scatterplot": scatterplot,
    }

# ---------------------------
# Plotting
# ---------------------------
def base64_encode(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def encode_png_ensure_limit(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    png = buf.getvalue()

    if len(png) <= IMAGE_SIZE_LIMIT:
        return "data:image/png;base64," + base64_encode(png)

    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size
    for scale in (0.85, 0.7, 0.55, 0.45, 0.35, 0.25):
        resized = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        resized.save(out, format="PNG", optimize=True)
        data = out.getvalue()
        if len(data) <= IMAGE_SIZE_LIMIT:
            return "data:image/png;base64," + base64_encode(data)
    return "data:image/png;base64," + base64_encode(data)

def create_scatterplot(x, y, xlabel, ylabel) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y)
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, color='red', linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. {xlabel}")
    return encode_png_ensure_limit(fig)

# ---------------------------
# API Endpoints
# ---------------------------
@app.post("/api/")
async def api(request: Request):
    questions_text, files = await read_form(request)

    if not questions_text:
        return JSONResponse(content={"error": "questions.txt is missing."}, status_code=400)

    # Handle movie data analysis task
    if "highest grossing films" in questions_text.lower():
        url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        df = scrape_wikipedia_table(url)
        if df is not None:
            results = analyze_movie_data(df)
            return JSONResponse(content=[
                results["two_billion_movies_before_2000"],
                results["earliest_film_over_1_5_billion"],
                results["correlation"],
                results["scatterplot"],
            ])
        else:
            return JSONResponse(content={"error": "Failed to scrape or process data."}, status_code=500)

    # Generic LLM-based analysis
    if llm:
        df = load_first_csv(files)
        if df is not None:
            prompt = PromptTemplate(
                input_variables=["data", "questions"],
                template="Analyze the following data:\n{data}\n\nAnswer these questions:\n{questions}"
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(data=df.to_string(), questions=questions_text)
            return JSONResponse(content=json.loads(response))

    return JSONResponse(content={"error": "Unsupported request or LLM not configured."}, status_code=400)

@app.get("/health")
def health():
    return PlainTextResponse("ok")

@app.get("/ls")
def ls():
    try:
        files = os.listdir(".")
        return JSONResponse({"cwd": os.getcwd(), "files": files})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)