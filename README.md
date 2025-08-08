# Data Analyst Agent API

## Deploy to Render
1. Push this repo to GitHub.
2. Go to [Render](https://render.com/) → **New Web Service**.
3. Connect your GitHub repo.
4. Render will detect the Dockerfile and deploy.
5. Once live, access your API at:
   ```
   https://<your-service-name>.onrender.com/api/
   ```
6. Send POST requests with `questions.txt` + optional attachments.

## Local run (optional)
```bash
pip install -r requirements.txt
uvicorn app_main:app --reload --host 0.0.0.0 --port 8000
```
