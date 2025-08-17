from llm_client import call_llm
from executor import execute_code
from utils import extract_python_code, format_metadata_list, fix_code_with_llm, summarize_attachments_for_llm, load_allowed_packages
import json

def scraping_required(task: str, attachment_info: str) -> bool:
    # Load system instructions
    with open("prompts/scraping_required.txt", "r") as f:
        instructions = f.read()

    # Include both task and available file info for LLM
    user_prompt = f"""
The data-analysis task is:
{task}

The user has also provided the following files:
{attachment_info}
"""

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = call_llm(messages)
    print("\nScraping Required:", response)
    return "yes" in response.lower()





def generate_metadata_extraction_code(task: str, attachment_info: str) -> str:
    # Load instructions from prompt file
    with open("prompts/extract_metadata.txt", "r") as f:
        instructions = f.read()

    allowed_packages = load_allowed_packages()
    instructions += (
        "\n\nIMPORTANT:\n"
        "You are restricted to using libraries only from the below list of allowed packages present in the environment:\n"
        + "\n".join(f"- {pkg}" for pkg in allowed_packages)
    )

    # Combine task description with attachment details
    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": (
                f"The data-analysis task is:\n{task}\n\n"
                f"Attached files (if any):\n{attachment_info}"
            )
        }
    ]
    return call_llm(messages)



def generate_solution_code(task: str, metadata_list: list, attachment_info: str) -> str:
    """
    Use the task + optional metadata list to prompt the LLM to generate final solving code
    """

    allowed_packages = load_allowed_packages()

    metadata_text = format_metadata_list(metadata_list) if metadata_list else "No metadata required."

    prompt = f"""
You are a data analysis expert. Generate Python code to solve the following data analysis task.

## Task:
{task}

## Metadata:
{metadata_text}

##Attachments:
{attachment_info}


(Metadata describes potential data sources and structures. Use only the relevant parts.)

---

## Instructions:

- You are restricted to using **only** the following Python libraries:
{chr(10).join(f"- {pkg}" for pkg in allowed_packages)}
- Do NOT use any libraries that are not listed above.
- The final code **must be executable immediately**, without requiring the user to call any function manually.
- The code must define and populate two variables by the end:
    - `result` - containing the final JSON output as specified by the task.
    - `error_list` - a list that collects all error messages or exceptions encountered during execution.
- When building `result`, ensure all NumPy and Pandas datatypes (e.g., np.int64, np.float64, pd.Timestamp) are converted to native Python types (`int`, `float`, `str`, etc.) before passing to `json.dumps`.
- If reading from attachments, use the file paths exactly as given in Attachments.
- If a function is defined, ensure it is also **called** within the same script.
- Each question or part of the solution must be inside a separate `try/except` block.
    - On exception, append a message to `error_list` and continue.
- Clean the data before use.
    - Specifically, strip or ignore `<sup>` tags (such as footnotes/references in Wikipedia).
- If matplotlib is needed, always begin with:
    ```python
    import matplotlib
    matplotlib.use('Agg')
    ```

---

## Output Format:

- Provide **only** a single clean Python code block — no markdown formatting or extra text.
- Do not include explanations or comments — just the code.
- Ensure the code is minimal, correct, and ready to `exec()`.

"""

    messages = [
        {"role": "system", "content": "You are a data analysis expert."},
        {"role": "user", "content": prompt}
    ]



    return call_llm(messages)



def run_pipeline(task: str, log, attachments):

    attachment_info = summarize_attachments_for_llm(attachments)
    log("\n--- Attachments ---\n"+ attachment_info)

    
    # Step 1: Generate and execute metadata code
    metadata_code = extract_python_code(generate_metadata_extraction_code(task, attachment_info), True)
    log("\n--- Metadata Code ---\n"+ metadata_code)
    try:
        meta_env = execute_code(metadata_code)
        metadata_list = meta_env.get("metadata_list", [])
    except Exception as e:
        log(f"\n❌ Error extracting metadata: {e}\n")
        metadata_list = []
        
    log("\n--- Extracted Metadata ---\n")
    log(metadata_list)

    
    # Step 2: Ask LLM to generate the final code using task + metadata (if any)
    final_code = extract_python_code(generate_solution_code(task, metadata_list, attachment_info), True)
    log("\n--- Initial Generated Code ---\n"+ final_code)
    MAX_RETRIES = 7
    result = json.dumps({}) # empty result json

    for attempt in range(1, MAX_RETRIES + 1):
        log(f"\n▶️ Attempt {attempt} at executing the code...\n")
        try:        
            final_env = execute_code(final_code)
            result = final_env.get("result", result)
            if not isinstance(result, str):
                result = json.dumps(result)
            error_list = final_env.get("error_list")
        except Exception as e:
            log(f"\n❌ Error executing code: {e}\n")
            error_list = [str(e)]

        if not error_list:
            log("\n✅ Final result:\n"+ result)
            return result

        log("\n❌ Errors found:\n"+ str(error_list))
        log("\n🔁 Fixing code based on above errors...\n")
        final_code = fix_code_with_llm(final_code, error_list)
        log("\n--- Fixed Code ---\n"+ final_code)

        if(attempt == MAX_RETRIES):
            log("\n❌ Max retries reached. returning last attempt result.\n")
            return result

