# FastAPI AI Task Runner

This project provides a simple API that uses an AI model to perform data analysis and web scraping tasks. You provide a text prompt and optional data files, and the service processes them in a sandboxed environment to generate a result.

It's designed for easy deployment on Railway or for running locally.

---

## Features

- **AI-Powered Tasks**: Uses Google Gemini to interpret natural language instructions for data processing.
- **Sandboxed Execution**: Each task runs in a secure, isolated folder to ensure safety.
- **Simple API**: A single `POST` endpoint to submit jobs and receive results.
- **Dynamic Package Installation**: Automatically installs required Python libraries for each task.
- **Easy Deployment**: Ready to deploy on Railway with minimal configuration.
- **Local Development**: Fully configured for local testing and development.

---

## Project Structure

```
.
├── main.py             # FastAPI application and main logic
├── task_engine.py      # Handles sandboxed task execution
├── gemini.py           # Client for interacting with the Gemini API
├── requirements.txt    # Project dependencies
├── Procfile            # Deployment configuration for Railway
├── .env.example        # Example environment variables
└── uploads/            # Directory for temporary request data
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- A Google Gemini API Key

### 1. Local Setup

First, clone the repository and set up a virtual environment:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory and add your Gemini API key.

**.env**
```
GENAI_API_KEY="your_gemini_api_key_here"
```

The application loads this key from the environment; no keys are ever hardcoded.

### 3. Run the Server

Start the application using Uvicorn:

```bash
uvicorn main.py:app --host 0.0.0.0 --port 8000
```

You can check if the server is running by visiting `http://localhost:8000` in your browser.

---

## How It Works

1.  The API receives a request containing a `question.txt` prompt and optional data files.
2.  The prompt and file list are sent to the AI model (Gemini), which interprets the request and defines the necessary processing steps and Python libraries for the task.
3.  The task engine prepares a secure, isolated environment for the job. It installs the required libraries if they are not already present.
4.  The processing steps are executed within the sandbox. The script can read the provided files and write its output (`result.json` or `result.txt`) to its working directory.
5.  The final result file is read and its content is returned as the API response.

---

## Deploy to Railway

1.  Push the code to a GitHub repository.
2.  Create a new project on Railway and connect it to your repository.
3.  Add your `GENAI_API_KEY` in the **Variables** tab in your Railway project settings.
4.  Railway will automatically deploy the application using the provided `Procfile`.

---

## API Usage

Send a `POST` request to the `/api` endpoint with your instructions in a `question.txt` file and any additional data files.

-   **Endpoint**: `POST /api`
-   **Body**: `multipart/form-data`
-   **Required Field**: `question.txt` (A file containing the prompt for the AI).
-   **Optional Fields**: Attach any other files (`.csv`, `.json`, etc.) needed for the task.

### Example `curl` Request

```bash
curl -X POST "http://localhost:8000/api" \
  -H "Accept: application/json" \
  -F "question.txt=@/path/to/your/question.txt" \
  -F "data.csv=@/path/to/your/data.csv"
```

The API will return a JSON object with the results of the task.