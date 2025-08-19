import os
import tempfile
import pandas as pd
import fitz  # PyMuPDF
import easyocr
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from dotenv import load_dotenv
from agent_core import handle_task

load_dotenv()

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OCR Reader
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_pdf(file_path):
    """Extract text from PDF pages using OCR."""
    pdf_document = fitz.open(file_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        result = reader.readtext(np.array(img))
        page_text = " ".join([res[1] for res in result])
        text += page_text + "\n"
    return text.strip()

@app.post("/api/")
async def analyze_file(files: list[UploadFile] = File(...)):
    """
    Accept multiple files:
    - Any .txt file = question
    - CSV = data
    - PNG/JPG = OCR image data
    - PDF = OCR PDF data
    """
    question = ""
    csv_info, image_info, pdf_text = "", "", ""

    for file in files:
        content = await file.read()
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)

        # Detect question file dynamically (any .txt)
        if file.filename.lower().endswith(".txt") and not question:
            try:
                question = content.decode("utf-8").strip()
            except UnicodeDecodeError:
                question = content.decode("latin-1").strip()
            continue

        if file.filename.endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
                csv_info += f"\nCSV Preview:\n{df.head().to_csv(index=False)}"
            except Exception as e:
                csv_info += f"\nError reading CSV: {str(e)}"

        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(file_path)
                img_array = np.array(img)
                result = reader.readtext(img_array)
                image_info += " ".join([res[1] for res in result])
            except Exception as e:
                image_info += f"\nError reading image: {str(e)}"

        elif file.filename.lower().endswith(".pdf"):
            try:
                pdf_text += extract_text_from_pdf(file_path)
            except Exception as e:
                pdf_text += f"\nError reading PDF: {str(e)}"

    if not question:
        return JSONResponse(content={"error": "No question text file provided"}, status_code=400)

    # Build full prompt
    full_prompt = (
        f"Question: {question}\n"
        f"{csv_info}\n"
        f"{image_info}\n"
        f"{pdf_text}\n"
        "Return ONLY a valid JSON array or object as the final answer. "
        "Plots must be PNG, base64 encoded, and under 100,000 bytes."
    )

    try:
        result = await handle_task(full_prompt)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Data Analyst API is running"}
