# ⚡ TDS Project 2 - Data Analyst Agent — Fast, AI‑Assisted Analytics

A lightweight agent that turns raw data and plain‑English questions into concise insights and clean visuals. Bring a dataset, add a questions.txt, and get answers—fast.

---

## What It Does
- Parses your questions and runs data analysis automatically
- Produces short summaries and ready‑to‑use charts
- Works locally; no data is stored in the cloud

## Highlights
- **AI insights** (Google Generative AI)
- **Rich visuals** (Matplotlib/Seaborn)
- **Web source mode** (fetch data from URLs)
- **Multi‑format input** (CSV/Excel/JSON/Parquet/TXT)
- **Batch Q&A** (multiple questions in one go)
- **FastAPI backend** with a minimal UI

---

## Quick Start
```bash
git clone https://github.com/24ds1000034/TDS_P2_DataAnalystAgent.git
cd data-analyst-agent
pip install -r requirements.txt
```

Create `.env` (either single key or multi‑key fallback):
```bash
GEMINI_API_KEY=your_key
# or
gemini_api_1=your_key
gemini_api_2=your_key
LLM_TIMEOUT_SECONDS=240
```

Run locally:
```bash
python -m uvicorn app:app --reload
# open http://localhost:8000
```

---

## Usage
1. Create `questions.txt` (one request per line).  
2. (Optional) Prepare a dataset file.  
3. Upload via the home page or call the API:

```bash
curl -X POST http://127.0.0.1:8000/api   -F "questions=@questions.txt"   -F "dataset=@data.csv"
```

**Endpoints**
- `GET /` – Web app  
- `POST /api` – Submit questions (+ optional dataset)  
- `GET /summary` – Diagnostics

**Supported formats**
CSV (`.csv`), Excel (`.xlsx`, `.xls`), JSON (`.json`), Parquet (`.parquet`), Text (`.txt`).

---

## Deploy on Railway (3 steps)
1. Push this repo to GitHub.  
2. In Railway: **New Project → Deploy from GitHub**.  
3. Add variables (`gemini_api_1..10`, `LLM_TIMEOUT_SECONDS`, optional `PORT`).

_Check logs with `railway logs`._

---

## Config (essentials)
- `GEMINI_API_KEY` **or** `gemini_api_1..10` (at least one key required)  
- `LLM_TIMEOUT_SECONDS` (default 240)  
- `PORT` (Railway injects this automatically)

---

## Security & License
- Keys live in `.env` (do not commit).  
- CORS should be restricted in production.  
- Licensed under **MIT**.