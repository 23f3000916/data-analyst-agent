# main.py

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Any
import base64
import io

from .processor import process_question

app = FastAPI()

class IncomingFile:
    def __init__(self, filename, file_obj):
        self.filename = filename
        self.file = file_obj

def parse_files(files_data) -> list:
    """
    Convert uploaded files in request to the format process_question expects.
    Handles both base64 files from JSON and standard UploadFile.
    """
    result = []
    if files_data:
        for f in files_data:
            # JSON base64 files
            if isinstance(f, dict) and "filename" in f and "content" in f:
                try:
                    decoded_content = base64.b64decode(f["content"])
                    file_obj = io.BytesIO(decoded_content)
                    result.append(IncomingFile(f["filename"], file_obj))
                except Exception as e:
                    print(f"Error decoding base64 file {f.get('filename','unknown')}: {e}")
            # FastAPI UploadFile
            elif hasattr(f, "filename") and hasattr(f, "file"):
                file_obj = f.file
                result.append(IncomingFile(f.filename, file_obj))
            # Already a file-like object as fallback
            elif hasattr(f, "filename") and hasattr(f, "file"):
                result.append(f)
    return result

@app.post("/api/")
async def analyze(request: Request):
    """
    POST endpoint that accepts a question about uploaded files (JSON or multipart).
    """
    files = []
    question = None

    try:
        content_type = request.headers.get("content-type", "")

        # Handle application/json uploads (API/test/automation)
        if "application/json" in content_type:
            data = await request.json()
            question = data.get("question")
            # Try test frameworks
            if not question and isinstance(data.get("vars"), dict):
                question = data["vars"].get("question")
            file_objs = data.get("files", [])
            files = parse_files(file_objs)

        # Handle multipart/form-data uploads
        elif "multipart/form-data" in content_type:
            form = await request.form()
            question = form.get("question")
            form_files = form.getlist("files")
            files = parse_files(form_files)

        # Fallback: query param or empty-body requests
        if not question:
            question = request.query_params.get("question")

        if not question or not question.strip():
            return JSONResponse({"error": "Missing required field: question"}, status_code=400)

        result = process_question(question, files)
        if isinstance(result, dict):
            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"error": "Invalid JSON response from processor"}, status_code=500)

    except Exception as e:
        print(f"API Error: {str(e)}")
        return JSONResponse(
            content={"error": f"An unexpected error occurred: {str(e)}"},
            status_code=500
        )

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
