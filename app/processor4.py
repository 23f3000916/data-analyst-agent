import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from typing import List, Optional
from fastapi import UploadFile
from dotenv import load_dotenv
import openai  # old API style

# Load environment variables for local testing
load_dotenv()

# Get OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

# Configure API key for old-style openai
openai.api_key = api_key

def process_request(files: List[UploadFile], qtext: Optional[str] = None):
    """
    Processes uploaded files and optional qtext.
    Returns a valid JSON array or object.
    """

    questions = None
    csv_dataframes = []
    other_files_content = {}

    # Read uploaded files
    for file in files:
        content = file.file.read()
        if file.filename.lower().endswith(".txt"):
            questions = content.decode("utf-8").strip()
        elif file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            csv_dataframes.append(df)
        else:
            try:
                other_files_content[file.filename] = content.decode("utf-8")
            except:
                pass

    # Append inline qtext if provided
    if qtext:
        if questions:
            questions += "\n" + qtext
        else:
            questions = qtext

    if not questions:
        raise ValueError("No questions provided")

    # Summarise CSV data for LLM
    data_summary = ""
    for i, df in enumerate(csv_dataframes):
        data_summary += f"\nCSV File {i+1} (first 5 rows):\n{df.head().to_csv(index=False)}"

    # Build LLM prompt
    prompt = f"""
You are a data analyst.
The user provided the following questions:

{questions}

Additional data:
{data_summary}

Please answer the questions strictly in valid JSON format.
If multiple questions are asked, return a JSON array of answers in the same order.
If plotting is requested, return the plot as base64-encoded string in place of the answer.
"""

    try:
        # Call OpenAI API using old-style ChatCompletion
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw_output = response["choices"][0]["message"]["content"].strip()

        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            # Fallback: wrap string in array
            return [raw_output]

    except Exception:
        # If all fails, return empty strings for each question
        q_count = len([q for q in questions.split("\n") if q.strip()])
        return ["" for _ in range(q_count)]
