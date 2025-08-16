import os
import io
import json
import pandas as pd
import openai
from fastapi import UploadFile
from typing import List

def process_request(files: List[UploadFile]):
    # Prepare CSV and question text
    csv_texts = []
    questions_text = ""

    for file in files:
        content = file.file.read()
        file.file.seek(0)

        if file.filename.lower().endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content))
                csv_texts.append(df.to_csv(index=False))
            except Exception as e:
                csv_texts.append(content.decode("utf-8"))
        else:
            questions_text += content.decode("utf-8") + "\n"

    # Build prompt
    prompt_parts = []
    if csv_texts:
        for idx, csv_text in enumerate(csv_texts, start=1):
            prompt_parts.append(f"CSV Dataset {idx}:\n{csv_text}")
    prompt_parts.append(
        "Answer the following questions based on the above datasets (if relevant). "
        "If the dataset is not needed, answer using your general knowledge. "
        "Respond ONLY with a valid JSON array of strings, in the same order as the questions.\n"
        f"Questions:\n{questions_text}"
    )

    full_prompt = "\n\n".join(prompt_parts)

    # Call OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # you can change to another available model
        messages=[
            {"role": "system", "content": "You are a data analyst. Be concise and accurate. Always respond in JSON array format."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0
    )

    # Parse output
    answer_text = response.choices[0].message["content"].strip()
    try:
        answers = json.loads(answer_text)
    except json.JSONDecodeError:
        answers = [answer_text]  # fallback if not JSON

    return answers
