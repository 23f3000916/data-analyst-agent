# validator.py
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

async def validate_output(task, output):
    """
    Ensure the output is exactly valid JSON and meets spec requirements.
    """
    prompt = f"""
Task:
{task}

The following output was produced by the data agent:

{output}

Return ONLY a valid JSON array or JSON object answering the task.
If plots are included, ensure they are base64-encoded PNGs under 100,000 bytes.
Do not include explanations or any text outside the JSON.
"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=60
    )
    return response.choices[0].message.content.strip()
