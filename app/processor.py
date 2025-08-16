import pandas as pd
import io
import openai
import os
from typing import List, Optional
from fastapi import UploadFile

openai.api_key = os.getenv("OPENAI_API_KEY")

def process_request(files: List[UploadFile], qtext: Optional[str] = None):
    csv_dataframes = []
    questions_text = ""

    # If qtext is provided directly, include it
    if qtext:
        questions_text += qtext.strip() + "\n"

    # Process uploaded files
    for file in files:
        content = file.file.read()
        try:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
                csv_dataframes.append(df)
            else:
                questions_text += content.decode("utf-8") + "\n"
        except Exception:
            questions_text += content.decode("utf-8", errors="ignore") + "\n"

    # Merge all CSVs into a single CSV text
    csv_text = ""
    if csv_dataframes:
        merged_df = pd.concat(csv_dataframes, ignore_index=True)
        csv_text = merged_df.to_csv(index=False)

    # Build LLM prompt
    prompt = f"""
You are a data analyst AI. You are given:
Questions:
{questions_text}

Data:
{csv_text}

Answer the questions as a JSON array of strings. 
If a question cannot be answered from the data, still answer it to the best of your knowledge.
"""

    # Call OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        answer = response.choices[0].message["content"].strip()
        return answer
    except Exception as e:
        return {"error": str(e)}
