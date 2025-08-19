import os
import networkx as nx
import re
import json
import base64
import tempfile
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import re
import json
import base64
import tempfile
import subprocess
import logging
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from dotenv import load_dotenv
import requests

# Optional image conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# LangChain / LLM imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# -------------------- Robust Gemini LLM with fallback --------------------
from collections import defaultdict
import time

# Config
GEMINI_KEYS = [os.getenv(f"gemini_api_{i}") for i in range(1, 11)]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

MODEL_HIERARCHY = [
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-pro",
]

MAX_RETRIES_PER_KEY = 2
TIMEOUT = 30
QUOTA_KEYWORDS = ["quota", "exceeded", "rate limit", "403", "too many requests"]

if not GEMINI_KEYS:
    # Fallback to a single GOOGLE_API_KEY if the numbered keys are not found
    single_key = os.getenv("GOOGLE_API_KEY")
    if single_key:
        GEMINI_KEYS.append(single_key)
    else:
        raise RuntimeError("No Gemini API keys found. Please set GOOGLE_API_KEY or gemini_api_1, gemini_api_2, etc. in your environment.")

# -------------------- LLM wrapper --------------------
class LLMWithFallback:
    def __init__(self, keys=None, models=None, temperature=0):
        self.keys = keys or GEMINI_KEYS
        self.models = models or MODEL_HIERARCHY
        self.temperature = temperature
        self.key_model_attempts = defaultdict(list)
        self.current_llm = None

    def _get_llm_instance(self):
        last_error = None
        for model in self.models:
            for key in self.keys:
                try:
                    llm_instance = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=self.temperature,
                        google_api_key=key,
                        request_timeout=TIMEOUT,
                    )
                    # A light check to see if the model is accessible
                    llm_instance.invoke("ping")
                    self.current_llm = llm_instance
                    logger.info(f"Successfully connected to LLM with model {model}")
                    return llm_instance
                except Exception as e:
                    last_error = e
                    msg = str(e).lower()
                    self.key_model_attempts[(key[:5], model)].append(str(e))
                    logger.warning(f"Failed to connect with model {model} using key {key[:5]}... Error: {e}")
                    time.sleep(0.5)
        raise RuntimeError(f"All models/keys failed. Last error: {last_error}")

    def bind_tools(self, tools):
        if not self.current_llm:
            self._get_llm_instance()
        return self.current_llm.bind_tools(tools)

    def invoke(self, prompt):
        if not self.current_llm:
            self._get_llm_instance()
        return self.current_llm.invoke(prompt)

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 240))

# -----------------------------
# FastAPI Frontend & Health
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as app.py</p>", status_code=404)

_FAVICON_FALLBACK_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII=")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve favicon.ico or a fallback transparent PNG."""
    if os.path.exists("favicon.ico"):
        return FileResponse("favicon.ico", media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    """Health/info endpoint."""
    return JSONResponse({
        "ok": True,
        "message": "Server is running. Use POST /api with 'questions_file' and optional 'data_file'.",
    })

def parse_keys_and_types(raw_questions: str):
    """
    Parses the key/type section from the questions file.
    Returns:
        type_map: dict mapping short_key -> casting function
    """
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    matches = re.findall(pattern, raw_questions)
    type_map_def = {"number": float, "string": str, "integer": int, "int": int, "float": float}
    type_map = {key: type_map_def.get(t.lower(), str) for key, t in matches}
    return type_map

# -----------------------------
# Tools
# -----------------------------
@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame (supports HTML tables, CSV, Excel, Parquet, JSON, and plain text).
    Always returns {"status": "success", "data": [...], "columns": [...]} if fetch works.
    """
    logger.info(f"Scraping URL: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Referer": "https://www.google.com/",
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        df = None

        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(resp.content))
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(io.BytesIO(resp.content))
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(resp.content))
        elif "application/json" in ctype or url.lower().endswith(".json"):
            df = pd.json_normalize(resp.json())
        elif "text/html" in ctype:
            tables = pd.read_html(io.StringIO(resp.text), flavor="bs4")
            if tables:
                df = tables[0]
            else: # Fallback to plain text from HTML
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                df = pd.DataFrame({"text": [soup.get_text(separator="\n", strip=True)]})
        else: # Fallback for any other text-based format
            df = pd.DataFrame({"text": [resp.text]})

        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()
        return {"status": "success", "data": df.to_dict(orient="records"), "columns": df.columns.tolist()}
    except Exception as e:
        logger.error(f"Scrape tool failed for URL {url}: {e}")
        return {"status": "error", "message": str(e)}

# -----------------------------
# Utilities for executing code safely
# -----------------------------
def clean_llm_output(output: str) -> Dict:
    """Extract JSON object from LLM output robustly."""
    try:
        s = re.search(r'\{.*\}', output, re.DOTALL).group(0)
        return json.loads(s)
    except (AttributeError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse JSON from LLM output. Error: {e}. Raw output: {output}")
        return {"error": "No valid JSON object found in LLM output", "raw": output}

def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Executes Python code in a sandboxed environment.
    - Provides pandas, numpy, matplotlib.
    - Injects a DataFrame 'df' if a pickle path is provided.
    - Defines a 'plot_to_base64()' helper to generate size-constrained images.
    - Expects the user code to populate a 'results' dictionary.
    - Returns the parsed JSON output from the script.
    """
    helper = r'''
import base64, io
from matplotlib import pyplot as plt
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def plot_to_base64(max_bytes=100000):
    """Saves current plot to a base64 string, trying to keep it under max_bytes."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=90)
    img_bytes = buf.getvalue()

    if len(img_bytes) <= max_bytes:
        plt.clf() # Clear figure for next plot
        plt.close() # Close figure
        return base64.b64encode(img_bytes).decode('ascii')

    if PIL_AVAILABLE:
        try:
            buf.seek(0)
            img = Image.open(buf)
            # Try converting to WEBP which is often smaller
            for quality in [85, 75, 65]:
                out_buf = io.BytesIO()
                img.save(out_buf, format='WEBP', quality=quality)
                webp_bytes = out_buf.getvalue()
                if len(webp_bytes) <= max_bytes:
                    plt.clf()
                    plt.close()
                    return base64.b64encode(webp_bytes).decode('ascii')
        except Exception:
            pass # Fallback to downscaling PNG

    # Fallback: iteratively reduce DPI
    for dpi in [70, 50, 30]:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            plt.clf()
            plt.close()
            return base64.b64encode(b).decode('ascii')

    # If still too large, return the smallest PNG we generated
    plt.clf()
    plt.close()
    return base64.b64encode(b).decode('ascii')
'''
    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "import networkx as nx"
    ]
    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')")
    else:
        preamble.append("df = None") # Ensure df exists

    script_code = "\n".join(preamble) + f"\n{helper}\nresults = {{}}\n{code}\n"
    script_code += "print(json.dumps({'status':'success','result':results}, default=str), flush=True)\n"

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp_path = tmp.name
            tmp.write(script_code)

        completed = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=timeout)

        if completed.returncode != 0:
            error_msg = completed.stderr.strip() or completed.stdout.strip()
            logger.error(f"Code execution failed. Stderr: {completed.stderr}. Stdout: {completed.stdout}")
            return {"status": "error", "message": error_msg}

        out = completed.stdout.strip()
        return json.loads(out)
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Could not parse JSON output: {e}", "raw": out}
    finally:
        for path in [tmp_path, injected_pickle]:
            if path and os.path.exists(path):
                os.unlink(path)

# -----------------------------
# LLM agent setup
# -----------------------------
llm = LLMWithFallback(temperature=0)

# **CRITICAL PROMPT UPDATE**
# This new prompt solves the key mapping and styling issues.
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an autonomous data analyst agent. Your goal is to generate a Python script to answer a user's questions.

You will receive:
- A set of **rules**.
- The user's **questions**, which include short, backticked keys like `edge_count`.
- An optional **dataset preview**.

You MUST follow these rules:
1.  Your final output must be a single, valid JSON object and nothing else.
2.  The JSON object must contain two keys: "questions" and "code".
3.  The "questions" value must be a list of the original full question strings.
4.  The "code" value must be a string of Python code.
5.  The Python code MUST create a dictionary named `results`.
6.  For each question, the code must compute the answer and store it in the `results` dictionary.
7.  **CRITICAL**: The keys of the `results` dictionary MUST be the short, backticked keys (e.g., `edge_count`, `bar_chart`) provided in the user's questions. DO NOT use the full question string as the key.
8.  **PLOTTING**: If a question asks for a plot with specific styling (e.g., "draw a bar chart with blue bars"), you MUST include the necessary parameters (e.g., `color='blue'`) in your plotting code.
9.  All plots MUST be converted to a base64 string using the provided `plot_to_base64()` helper function.
10. Your code will be executed in a sandbox with pandas, numpy, matplotlib, and networkx available. A pandas DataFrame named `df` will be available if a dataset was uploaded."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, [scrape_url_to_dataframe], prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[scrape_url_to_dataframe],
    verbose=True,
    max_iterations=4,
    early_stopping_method="generate",
    handle_parsing_errors=True
)

# -----------------------------
# Runner: orchestrates agent -> execute
# -----------------------------
def run_agent_safely(llm_input: str, pickle_path: str = None) -> Dict:
    """
    Runs the LLM agent to get Python code, executes it, and returns the results.
    - If pickle_path is None, the agent is expected to use the scrape_url_to_dataframe tool.
    - If pickle_path is provided, the DataFrame is injected directly into the execution environment.
    """
    try:
        response = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
        raw_out = response.get("output", "")
        if not raw_out:
            return {"error": f"Agent returned no output. Full response: {response}"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        code = parsed.get("code")
        if not code:
            return {"error": f"Invalid agent response format, missing 'code' key: {parsed}"}

        # If no data was uploaded, the agent's code should contain a scrape call.
        # We let the generated code handle the scraping itself now.
        # The scrape_url_to_dataframe tool is available to the agent if needed.

        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message', 'Unknown error')}", "raw": exec_result.get("raw")}

        return exec_result.get("result", {})

    except Exception as e:
        logger.exception("run_agent_safely failed")
        return {"error": str(e)}

# -----------------------------
# Main API Endpoint
# -----------------------------
@app.post("/api")
async def analyze_data(request: Request):
    try:
        form = await request.form()
        questions_file = form.get("questions_file")
        data_file = form.get("data_file")

        if not questions_file or not hasattr(questions_file, "read"):
            raise HTTPException(400, "Missing or invalid questions file (.txt)")

        raw_questions = (await questions_file.read()).decode("utf-8")
        type_map = parse_keys_and_types(raw_questions)

        pickle_path = None
        df_preview = ""
        dataset_uploaded = False

        if data_file and hasattr(data_file, "filename") and data_file.filename:
            dataset_uploaded = True
            filename = data_file.filename.lower()
            content = await data_file.read()
            df = None

            if filename.endswith(".csv"): df = pd.read_csv(io.BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")): df = pd.read_excel(io.BytesIO(content))
            elif filename.endswith(".parquet"): df = pd.read_parquet(io.BytesIO(content))
            elif filename.endswith(".json"): df = pd.read_json(io.BytesIO(content))
            else: raise HTTPException(400, f"Unsupported data file type: {filename}")

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                pickle_path = temp_pkl.name
                df.to_pickle(pickle_path)

            df_preview = (
                f"\n\nDataset Preview:\n"
                f"The uploaded dataset has {len(df)} rows and columns: {', '.join(df.columns.astype(str))}.\n"
                f"First 5 rows:\n{df.head(5).to_markdown(index=False)}\n"
            )

        rules = (
            "You have access to a pandas DataFrame called `df` containing the uploaded data. DO NOT call `scrape_url_to_dataframe`."
            if dataset_uploaded else
            "No dataset was uploaded. You MUST use the `scrape_url_to_dataframe(url)` tool to fetch data if required by the questions."
        )

        llm_input = f"Rules:\n{rules}\n\nQuestions:\n{raw_questions}\n{df_preview}"

        result = run_agent_safely(llm_input, pickle_path)

        if "error" in result:
            raise HTTPException(500, detail=result)

        # **SIMPLIFIED POST-PROCESSING**
        # Apply type casting directly to the results from the agent.
        final_result = {}
        for key, value in result.items():
            caster = type_map.get(key)
            if caster:
                try:
                    # Don't cast base64 strings
                    if isinstance(value, str) and len(value) > 200 and ("=" in value or "+" in value):
                         final_result[key] = value
                    else:
                        final_result[key] = caster(value)
                except (ValueError, TypeError):
                    final_result[key] = value # Keep original value if casting fails
            else:
                final_result[key] = value

        return JSONResponse(content=final_result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)