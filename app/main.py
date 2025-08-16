from fastapi import FastAPI, File, UploadFile, Form
from typing import List, Optional
from .processor import process_request

app = FastAPI()

@app.post("/api/")
async def process_api(
    files: List[UploadFile] = File(...),
    qtext: Optional[str] = Form(None)
):
    """
    API endpoint to process uploaded files and/or direct text questions.
    - files: CSV or TXT files (questions and/or data)
    - qtext: Optional direct question text
    """
    try:
        answers = process_request(files=files, qtext=qtext)
        return answers
    except Exception as e:
        return {"error": str(e)}
