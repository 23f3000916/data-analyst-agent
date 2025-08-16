from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from .processor import process_question  # relative import for Railway
import uvicorn

app = FastAPI()

@app.post("/api/")
async def analyze(question: str = Form(...), files: list[UploadFile] = []):
    try:
        result = process_question(question, files)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
