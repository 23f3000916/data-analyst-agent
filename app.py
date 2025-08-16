import os
import re
import json
import base64
import tempfile
import sys
import subprocess
import logging
import concurrent.futures
from io import BytesIO, StringIO
from typing import Dict, Any, List

# --- Third-party Imports ---
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response

# --- LangChain / LLM Imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --- Optional Image Conversion ---
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# --- Initial Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 150))

# -----------------------------
# Agent Tools
# -----------------------------

@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return its content as a pandas DataFrame.
    This tool supports HTML tables, CSV, Excel, Parquet, and JSON data sources.
    It returns a dictionary with status, data, and columns.
    """
    logger.info(f"Scraping URL: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        df = None

        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))
        elif "application/json" in ctype or url.lower().endswith(".json"):
            df = pd.json_normalize(resp.json())
        elif "text/html" in ctype:
            try:
                tables = pd.read_html(StringIO(resp.text), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError: # No tables found
                pass
            if df is None: # Fallback to text
                soup = BeautifulSoup(resp.text, "html.parser")
                df = pd.DataFrame({"text": [soup.get_text(separator="\n", strip=True)]})
        else: # Fallback for any other content type
            df = pd.DataFrame({"text": [resp.text]})

        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()
        return {"status": "success", "data": df.to_dict(orient="records"), "columns": df.columns.tolist()}

    except Exception as e:
        logger.error(f"Scraping failed for URL {url}: {e}")
        return {"status": "error", "message": str(e)}

# -----------------------------
# Code Execution Utilities
# -----------------------------

def clean_llm_output(output: str) -> Dict:
    """
    Extracts a JSON object from the LLM's raw output string.
    """
    if not output:
        return {"error": "Empty LLM output"}
    
    # Remove markdown fences
    match = re.search(r"```(?:json)?\s*({.*})\s*```", output, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback for outputs without fences
        first_brace = output.find('{')
        last_brace = output.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = output[first_brace:last_brace+1]
        else:
            return {"error": "No JSON object found in LLM output", "raw": output}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parsing failed: {e}", "raw": json_str}


def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Executes Python code in a sandboxed environment via a temporary file.
    Injects a DataFrame from a pickle file if provided.
    Includes a helper function to generate base64 plots.
    """
    helper_code = r'''
def plot_to_base64(max_bytes=100000):
    """Saves the current matplotlib plot to a base64 string, ensuring it's under a size limit."""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    img_bytes = buf.getvalue()

    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')

    # Try reducing DPI if image is too large
    for dpi in [80, 60, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')

    # Last resort: return smallest version even if oversized
    return base64.b64encode(b).decode('ascii')
'''
    
    preamble = [
        "import json, sys, base64",
        "from io import BytesIO",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "import networkx as nx" # Added networkx for network analysis test case
    ]

    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')")
        preamble.append("data = df.to_dict(orient='records')")
    else:
        preamble.append("data = {}") # Ensure 'data' exists

    # Ensure the generated code doesn't have accidental prints
    cleaned_code = code.strip()

    full_code = "\n".join(preamble) + "\n" + helper_code + "\nresults = {}\n" + cleaned_code
    full_code += "\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n"

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp_path = tmp.name
            tmp.write(full_code)
        
        completed = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout
        )

        if completed.returncode != 0:
            error_message = completed.stderr.strip() or completed.stdout.strip()
            return {"status": "error", "message": f"Script execution failed: {error_message}"}
        
        return json.loads(completed.stdout.strip())

    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Could not parse script output: {e}", "raw": completed.stdout.strip()}
    finally:
        # Cleanup temporary files
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if injected_pickle and os.path.exists(injected_pickle):
            os.unlink(injected_pickle)

# -----------------------------
# LLM Agent Setup
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash-latest"),
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an autonomous data analyst agent. Your goal is to write Python code to answer questions.

You will receive rules, questions, and an optional dataset preview.

You must follow these rules strictly:
1.  **CRITICAL**: Your generated code must NOT include any `print()` statements. The execution environment handles the final output. Your only task is to populate the `results` dictionary.
2.  Return ONLY a valid JSON object with two keys: "questions" (a list of original question strings) and "code" (a string of Python code).
3.  Your Python code must populate a dictionary named `results` where each key is a question string and the value is the computed answer.
4.  Your code will run in a sandbox with pandas, numpy, and matplotlib.
5.  A helper function `plot_to_base64()` is available for creating base64-encoded images under 100KB. ALWAYS use it for plots."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm=llm, tools=[scrape_url_to_dataframe], prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[scrape_url_to_dataframe],
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True,
)

# -----------------------------
# Main Analysis Pipeline
# -----------------------------
def run_analysis_pipeline(llm_input: str, pickle_path: str = None) -> Dict:
    """
    Orchestrates the entire process: LLM invocation, code generation, and safe execution.
    """
    try:
        response = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
        raw_out = response.get("output", "")
        if not raw_out:
            return {"error": "Agent returned no output."}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        code = parsed.get("code")
        questions = parsed.get("questions")

        # Log the generated code for debugging purposes
        logger.info(f"--- AGENT GENERATED CODE ---\n{code}\n--------------------------")

        if not code or not questions:
            return {"error": "Invalid agent response: 'code' or 'questions' key missing.", "raw": parsed}

        # If no data was uploaded, check if the agent's code needs to scrape data
        if pickle_path is None:
            urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
            if urls:
                url = urls[0]
                tool_resp = scrape_url_to_dataframe(url)
                if tool_resp.get("status") != "success":
                    return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
                
                df = pd.DataFrame(tool_resp["data"])
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                    pickle_path = temp_pkl.name
                    df.to_pickle(pickle_path)

        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message')}", "raw": exec_result.get("raw")}

        # Map results back to the original questions
        results_dict = exec_result.get("result", {})
        return {q: results_dict.get(q, "Answer not found") for q in questions}

    except Exception as e:
        logger.exception("Analysis pipeline failed")
        return {"error": str(e)}

def parse_keys_and_types(raw_questions: str) -> (List[str], Dict):
    """Parses keys and their expected types from the questions file."""
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    matches = re.findall(pattern, raw_questions)
    type_map_def = {"number": float, "string": str, "integer": int, "int": int, "float": float}
    type_map = {key: type_map_def.get(t.lower(), str) for key, t in matches}
    keys_list = [k for k, _ in matches]
    return keys_list, type_map

# -----------------------------
# FastAPI Endpoints
# -----------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the main HTML interface for user interaction."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


@app.post("/")
async def analyze_data(request: Request):
    """
    Main endpoint for data analysis. Accepts multipart/form-data with a
    questions file and an optional data file.
    """
    try:
        form = await request.form()
        
        # Correctly get the questions file from the form field named "questions.txt"
        questions_file = form.get("questions.txt")
        if not questions_file or not hasattr(questions_file, "read"):
            raise HTTPException(400, "Missing 'questions.txt' in form data.")

        # Find the optional data file by iterating through the form items
        data_file = None
        for key in form:
            if key != 'questions.txt':
                item = form[key]
                if hasattr(item, 'filename') and item.filename:
                    data_file = item
                    break # Found the data file, stop looking

        raw_questions = (await questions_file.read()).decode("utf-8")
        keys_list, type_map = parse_keys_and_types(raw_questions)

        pickle_path = None
        df_preview = ""
        
        if data_file:
            filename = data_file.filename.lower()
            content = await data_file.read()
            
            df = None
            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
            # ... add other file types as needed
            else:
                raise HTTPException(400, f"Unsupported data file type: {filename}")

            if df is not None:
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                    pickle_path = temp_pkl.name
                    df.to_pickle(pickle_path)
                
                df_preview = (
                    f"\n\nDataset Preview:\n"
                    f"- Rows: {len(df)}, Columns: {len(df.columns)}\n"
                    f"- Columns: {', '.join(df.columns.astype(str))}\n"
                    f"- Head:\n{df.head(3).to_markdown(index=False)}\n"
                )
            
            llm_rules = (
                "Rules:\n1. Use the provided dataset (`df`). Do NOT scrape external data.\n"
                "2. Generate Python code to answer the questions.\n"
                "3. For plots, use the `plot_to_base64()` helper function."
            )
        else:
            llm_rules = (
                "Rules:\n1. If data is needed, use the `scrape_url_to_dataframe(url)` tool.\n"
                "2. Generate Python code to answer the questions.\n"
                "3. For plots, use the `plot_to_base64()` helper function."
            )

        llm_input = f"{llm_rules}\n\nQuestions:\n{raw_questions}\n{df_preview}"

        # Run analysis in a thread to avoid blocking the event loop
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_analysis_pipeline, llm_input, pickle_path)
            try:
                result = future.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timed out")

        if "error" in result:
            raise HTTPException(500, detail=result.get("error", "Unknown error"))

        # Post-process results to match expected keys and types
        if keys_list and type_map:
            mapped_result = {}
            result_keys = list(result.keys())
            for i, key in enumerate(keys_list):
                if i < len(result_keys):
                    original_question = result_keys[i]
                    val = result[original_question]
                    caster = type_map.get(key, str)
                    try:
                        # Handle base64 strings before casting
                        if isinstance(val, str) and val.startswith("data:image/"):
                            val = val.split(',', 1)[1]
                        mapped_result[key] = caster(val)
                    except (ValueError, TypeError):
                        mapped_result[key] = val # Keep original if cast fails
            return JSONResponse(content=mapped_result)
        
        return JSONResponse(content=result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data endpoint failed")
        raise HTTPException(500, detail=f"An unexpected error occurred: {e}")

# --- Health and Info Endpoints ---
@app.get("/api")
async def api_info():
    return JSONResponse({
        "ok": True,
        "message": "This is the info endpoint. Use POST / to submit data for analysis.",
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)