from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pipeline import run_pipeline
from utils import setup_logger
from typing import List, Dict, Any
import traceback
import os
from datetime import datetime
import json


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify domains like ["https://app.example.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



   
# Default GET endpoint
@app.get("/")
async def read_root():
    return {"message": "Hello, world! v1.2"}

def save_to_tempfile(upload: UploadFile) -> str:
    # Create tmp folder in app root
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Prepare filename
    name, ext = os.path.splitext(upload.filename)
    if not ext:
        ext = ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{name}_{timestamp}{ext}"
    abs_path = os.path.join(tmp_dir, safe_name)

    # Rewind before reading
    try:
        upload.file.seek(0)
    except Exception:
        print("Failed to reset pointer before read:", traceback.format_exc())

    # Save the file
    with open(abs_path, "wb") as f:
        f.write(upload.file.read())

    # Reset again for possible reuse
    try:
        upload.file.seek(0)
    except Exception:
        pass

    # Return relative path
    return os.path.relpath(abs_path, os.getcwd())

def process_attachments(files: List[UploadFile]) -> List[Dict[str, Any]]:
    attachments = []
    for f in files:
        # Read full content
        contents = f.file.read()

        # Always save to temp file
        tmp_path = save_to_tempfile(f)

        attachments.append({
            "filename": f.filename,
            "content_bytes": contents,
            "content_type": f.content_type,
            "tmp_path": tmp_path,
        })

        # Reset file pointer in case something else needs to read UploadFile later
        try:
            f.file.seek(0)
        except Exception:
            pass

    return attachments



# POST endpoint to handle uploaded data analysis task
@app.post("/api/")
async def analyze_task(request: Request):
    log, log_path = setup_logger()
    try:
        start_time = datetime.now()

        form = await request.form()

        # Extract the main question file (must be present)
        if "questions.txt" not in form:
            return JSONResponse(status_code=400, content={"error": "questions.txt is required"})

        qfile = form["questions.txt"]
        if not hasattr(qfile, "filename"):
            return JSONResponse(status_code=400, content={"error": "questions.txt must be a file"})

        # Read task description
        contents = await qfile.read()
        try:
            task_description = contents.decode("utf-8").strip()
        except UnicodeDecodeError:
            task_description = contents.decode("latin-1").strip()

        log("\nReceived Task:\n" + (task_description[:1000] + ("..." if len(task_description) > 1000 else "")))

        # Gather all other files as attachments
        attachments_files = [
            v for k, v in form.multi_items()
            if hasattr(v, "filename") and (v.filename != "question.txt" and v.filename != "questions.txt")
        ]
        attachments = process_attachments(attachments_files)

        # Run pipeline
        answer = run_pipeline(task_description, log, attachments=attachments)
        end_time = datetime.now()
        log("total time taken to process (mins): " + str((end_time - start_time).total_seconds() / 60))
        if not isinstance(answer, str):
            answer = json.dumps(answer)

        return JSONResponse(content=json.loads(answer))


    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"API: error occurred: {str(e)}"}
        )