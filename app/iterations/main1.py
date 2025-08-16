from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
from typing import List
from .processor import process_request
from pathlib import Path

app = FastAPI(title="Data Analyst Agent")

@app.post("/api/")
async def analyze(request: Request, files: List[UploadFile] = File(...)):
    """
    Accepts a multipart form with at least one file `questions.txt` and optional other files.
    Returns JSON payload(s) as required by the input questions.
    """
    # save uploaded files to a temp dir
    tmpdir = Path(tempfile.mkdtemp(prefix="data-agent-"))
    saved_files = {}
    for upload in files:
        dest = tmpdir / upload.filename
        content = await upload.read()
        dest.write_bytes(content)
        saved_files[upload.filename] = str(dest)

    if 'questions.txt' not in saved_files:
        return JSONResponse(status_code=400, content={"error": "questions.txt is required and must be uploaded as the filename questions.txt"})

    # Read in the questions
    qtext = open(saved_files['questions.txt'], 'r', encoding='utf-8').read()

    try:
        result = process_request(qtext=qtext, files=saved_files, workdir=str(tmpdir))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Return the result as JSON. This assumes the result is JSON-serializable.
    return JSONResponse(content=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    uvicorn.run('app.main:app', host='0.0.0.0', port=port, reload=False)
