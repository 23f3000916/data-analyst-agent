# agent_core.py
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from code_executor import run_generated_code
from validator import validate_output

load_dotenv()
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

async def handle_task(task: str):
    """
    Generate Python code from a task, execute it, and validate output.
    """
    # 1. Ask GPT to generate Python code
    prompt = f"""You are a Python data analyst.
Given the task below, write Python code that fetches, prepares, analyzes, and visualizes the data.
Only return valid Python code without explanations.

Task:
{task}
"""

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=100
    )

    code = response.choices[0].message.content.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    # 2. Execute generated code
    result = run_generated_code(code)

    # 3. Validate output for correct JSON format
    final_response = await validate_output(task, result)

    return final_response
