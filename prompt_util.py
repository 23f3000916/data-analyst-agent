import re 
import traceback

PROBLEM_METADATA_TEMPLATE = """
**INPUT ANALYSIS TASK**
Extract metadata from the provided input with absolute precision. Do not add, infer, or modify any information.

**INPUT CONTENT**
{questions_text}

{attachments_text}

**SECTION A: QUESTIONS EXTRACTION**
1. Identify EVERY analytical question/problem
2. PRESERVE EXACT wording from input
3. For ambiguous questions, formulate CLEAR descriptions maintaining original intent
4. Include ALL question-related text including calculation specifics

**SECTION B: DATA SOURCES & STRUCTURE**
1. Identify ALL data locations (DBs, APIs, files, URLs)
2. Specify EXACT access methods (SQL queries, API endpoints, file paths)
3. Extract STRUCTURAL DETAILS:
   - File paths/patterns (use wildcards where specified)
   - Data formats (CSV, Parquet, JSON, etc.)
   - Schema details (tables, columns, data types)
   - Sample data representations
   - Partitioning/organization schemes
4. Include credentials ONLY if explicitly provided in input
5. If database query present:
   - Explicitly state "Public dataset - no credentials required" if no credentials in input
   - Include ALL credentials if present in input

**SECTION C: OUTPUT REQUIREMENTS**
1. Extract EVERY formatting instruction
2. Capture ALL precision/unit requirements
3. Specify encoding requirements (base64, etc.)
4. Note ALL constraints (length limits, file formats)
5. Include output schemas or example outputs
6. If no schema provided, create one based EXCLUSIVELY on input

**RULES YOU MUST FOLLOW**
- COMBINE information from text and images WITHOUT ADDITION
- REFERENCE files by exact name when mentioned
- PRESERVE technical details verbatim
- ALL input details MUST appear in output
- DO NOT INFER missing information"""

PROBLEM_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "question string"
            }
        },
        "data_source_text": {
            "type": "string",
            "description": "data sources and structure"
        },
        "output_format_text": {
            "type": "string",
            "description": "output schema and requirements"
        }
    },
    "required": ["questions", "data_source_text", "output_format_text"],
    "additionalProperties": False
}

FILE_LOADING_SCRIPT_TEMPLATE = """
You will be given information about various data sources. Your task is to identify which of them can be loaded from a file path, and write Python code to load them into pandas DataFrames.

INPUT:
{data_source_text}

INSTRUCTIONS:

- For each attached file, determine the correct method to load it into one or more pandas DataFrames based on its file extension.
- Supported formats include:
  - CSV, TSV
  - Excel files (.xls, .xlsx) — may contain multiple sheets
  - JSON
  - Parquet
  - ZIP files — may contain multiple CSVs or other supported files
  - PDF — use `tabula` (via `py-tabula`) to extract tables

- If a file contains multiple tables (e.g., multi-sheet Excel, ZIP with multiple CSVs), extract **all** tables and include them as separate DataFrames.

- Use the file path pattern: `request_data/{request_id}/<filename>` when loading each file.

YOUR TASK:

Write a single Python script that:
- Imports all required libraries
- Loads all tables into a list named `files_dfs`
- Appends one DataFrame per table into `files_dfs`

✅ Example:
```python
import pandas as pd

files_dfs = [
    pd.read_csv("request_data/{request_id}/data1.csv"),
    pd.read_excel("request_data/{request_id}/data2.xlsx", sheet_name="Sheet1"),
    pd.read_excel("request_data/{request_id}/data2.xlsx", sheet_name="Sheet2")
]
```

IMPORTANT RULES:

- Do **not** use any try-except blocks in the script.
- Return **only the final Python script**, with imports and construction of `files_dfs`.
- No markdown, explanations, or comments — just raw executable code.

"""


FILE_LOADING_SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "script": {
            "type": "string",
            "description": "A single valid Python script that loads DataFrames into a list called files_dfs."
        }
    },
    "required": ["script"],
    "additionalProperties": False
}

FIX_FILE_LOADING_TEMPLATE = """
You must FIX the below script so that it loads the dataframes correctly from the directory into a list named files_dfs.

FAILED SCRIPT:
{script}

ERROR TRACEBACK:
{traceback_text}

TRIED FIXES:
{fix_history_text}

Only fix the error causing parts of the script. Do not touch anything else.

IMPORTANT POINT
{attempt_specific_text}
"""

FIX_FILE_LOADING_SCHEMA = {
    "type": "object",
    "properties": {
        "fixed_script": {
            "type": "string",
            "description": "A single valid Python script that loads DataFrames into a list called files_dfs."
        },
        "fix_description": {
            "type": "string",
            "description": "short description of applied fix in 10-15 words"
        }

    },
    "required": ["fixed_script", "fix_description"],
    "additionalProperties": False

}

WEBSCRAPE_URL_TEMPLATE = """
You will be given an INPUT that explains the sources of data for a data analyis problem.

YOUR TASK 
Identify the URL of the webpage that needs to scraped else return null.

KEY INSTRUCTIONS:
- Check if it mentions **web scraping**; if not, make the URL null.
- Otherwise return the url for the webpage that needs to be scraped.
- The URL MUST NOT be related to a database connection, direct file download, or API endpoint.
- It MUST point to a WEBPAGE that needs to be scraped.
- The URL must be a valid webpage URL.

INPUT:
{data_source_text}
"""

WEBSCRAPE_URL_SCHEMA = {
    "type": "object",
    "properties": {
        "URL": {
            "type": ["string", "null"]
        }
    },
    "required": ["URL"],
    "additionalProperties": False
}


QUESTION_SCRIPTS_TEMPLATE = """
You will be given an INPUT containing:
- Data sources
- A list of questions

Your task is to write a separate Python script for **each question**, implementing a `find_answer({find_answer_args})` function that returns the answer.

---

INPUT:
{data_source_text}

{questions_list_text}

{dfs_text}

INSTRUCTIONS:

- ⚠️ [VERY IMPORTANT] Use only `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `sklearn`.
- The output must include one script per question, in the **same order** as the questions appear.
- Do **not** use `try-except` blocks in any of the scripts.
- Each script must define a function named exactly `find_answer({find_answer_args})` that returns the answer.
- All necessary packages must be explicitly imported in each script.
- Your output must be **only the Python code**, no explanations, no comments, no markdown formatting.
"""

QUESTION_SCRIPTS_SCHEMA = {
    "type": "object",
    "required": ["scripts"],
    "properties": {
        "scripts": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "script containing the 'find_answer' function"
            }
        }
    }
}

FIX_QUESTION_SCRIPT_TEMPLATE = """
You will be given the following inputs:
1. Problem metadata containing data sources and questions.
2. A TARGET QUESTION from the questions list.
2. Script containing 'find_answer' function for the TARGET QUESTION.
3. A detailed error traceback.
4. Short description of failed fixes that did not work.

Here are the inputs:

PROBLEM METADATA:
{data_source_text}

{questions_list_text}

{dfs_text}

TARGET QUESTION
{question_string}

BROKEN SCRIPT:
```python
{script}
```

ERROR TRACEBACK:
```plaintext
{traceback_text}
```

FAILED FIXES: 
{fix_history_text}

KEY INSTRUCTIONS:
- Do not use any try-except blocks in the function definition
- THE 'find_answer' must be present and only return the answer to: '{question_string}'
- If the error is due to missing packages, solve the question using ONLY `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `sklearn` .
"""

FIX_QUESTION_SCRIPT_SCHEMA = {
    "type": "object",
    "required": ["fixed_script", "fix_description"],
    "properties": {
        "fixed_script": {
            "type": "string",
            "description": "the fixed function definition"
        },
        "fix_description": {
            "type": "string",
            "description": "compact description of the tried fix in 10-15 words"
        }
    }
}

OUTPUT_SCRIPT_TEMPLATE = """
You are given the following inputs related to a data analysis problem:
- A list of questions
- A list of corresponding calculated answers
- A description of the expected output format

Your task is to generate a Python script that defines the function:

    def create_output(answers: list) -> Any

This function should take the list of answers and return a structured output that conforms to the specified format.

HERE ARE THE INPUTS:

QUESTIONS LIST:
{questions_list_text}

CALCULATED ANSWERS LIST:
```python
[
{answers_list_text}
]
```

OUTPUT FORMAT INSTRUCTIONS:
{output_format_text}

YOUR TASK:

Implement the function `create_output(answers: list)` that transforms the raw answers into the required output structure.

YOU MUST FOLLOW THESE RULES:

- Carefully examine the CALCULATED ANSWERS LIST.

- Convert each answer to the exact data type specified in the OUTPUT FORMAT INSTRUCTIONS.

- For any question where the answer is `None`, generate a valid dummy answer:
  - For base64-encoded image strings, use: 'dummy_base64_string'
  - For other types, use a plausible placeholder that matches the expected format.
  
- Ensure the returned value is a valid native Python data structure (e.g., dict, list, etc.).
- Ensure that the base64 strings ARE NOT PREFIXED by `data:image/<filetype>;base64,`, remove them if they are.
- DO NOT use json.dumps() or any form of manual JSON serialization.
- The output must be JSON-serializable (i.e., contain only data types that can be converted to JSON).
- Only output the complete Python script containing necessary imports (if any) and the create_output function.
- Do not include any explanations, comments, or example usages—only output the code.
"""


OUTPUT_SCRIPT_SCHEMA = {
    "type": "object",
    "required": ["script"],
    "properties": {
        "script": {
            "type": "string",
            "description": "python script containing create_output(answers: list)"
        }, 
    }
}

FIX_OUTPUT_SCRIPT_TEMPLATE = """
You are given the following inputs related to a data analysis task:
- A list of questions
- A list of calculated answer snippets
- Output format instructions
- A Python script that defines 'create_output(answers)' but currently raises an error
- The traceback of the error
- A list of previously attempted (but unsuccessful) fixes

Your task is to return a corrected version of the script such that it works as intended — assembling the output in the specified format without errors.

HERE ARE THE INPUTS:

QUESTIONS LIST:
{questions_list_text}

CALCULATED ANSWERS LIST:
```python
[
{answers_list_text}
]
```

OUTPUT FORMAT INSTRUCTIONS:
{output_format_text}

BROKEN SCRIPT:
```python
{script}
```

TRACEBACK:
```plaintext
{traceback_text}
```

YOUR TASK:

- Carefully inspect the script and the traceback to identify and fix the issue.
- Ensure that the corrected script produces the expected structure, using the provided answers list and respecting the output format instructions.
- If any answer is `None`, generate a valid dummy placeholder:
  - For base64-encoded images, use: `'data:image/png;base64,iV...(truncated)'`
  - For other types, use appropriate dummy values.
- The function must return a native Python data structure (e.g., dict, list) — **do not use `json.dumps()`** or other serializers.
- The output must be valid and JSON-serializable.
- Return only the **fixed script** as a raw code snippet — no explanations, comments, or additional text.
"""


FIX_OUTPUT_SCRIPT_SCHEMA = {
    "type": "object",
    "required": ["fixed_script", "fix_description"],
    "properties": {
        "fixed_script": {
            "type": "string",
            "description": "python script containing create_output(answers: list)"
        }, 
        "fix_description": {
            "type": "string",
            "description": "description of the applied fix in 10-15 words"
        }
    }
}


DFS_TEXT_TEMPLATE = """
The required data has been loaded into a list of Pandas DataFrames named `dfs`.
Each DataFrame must be accessed using its index: `dfs[<table_index>]`.

DATA SNIPPETS:
{all_snippets_text}

INSTRUCTIONS FOR USING DATAFRAMES:

- Access data **only** via `dfs[<table_index>]`. Do not reference or recreate data directly from the snippets.
- ✅ Always select the **smallest possible subset** of DataFrames necessary to answer **all** the questions.
- 🚫 Do **not** join, merge, or concatenate DataFrames **unless** it is explicitly required to answer a question.
- 🧹 [VERY IMPORTANT] Clean and preprocess **numerical columns** using regular expressions if needed (e.g., remove symbols, convert types).

Adhere strictly to these rules to ensure clean, efficient, and accurate analysis.
"""

def create_questions_list_text(questions_list):
    text = ""
    for question in questions_list:
        text += f"-{question}\n"
    return text 


def create_metadata_text(metadata_dict):
    temp = """
PROBLEM METADATA: 

A. Data Source Information
{data_source_text}

B. Analytical Questions
{questions_list_text}

C. Output Instructions & Format
{output_format_text}
    """
    text = temp.format(
        data_source_text = metadata_dict["data_source_text"],
        questions_list_text = create_questions_list_text(metadata_dict["questions"]),
        output_format_text = metadata_dict["output_format_text"]
    )
    return text 

def create_attachments_text(p):
    text = ""
    if len(p.images) > 0:
        text += f"You have been given {len(p.images)} images with the problem." 
    if len(p.filenames) > 0:
        filepaths = [f"request_data/{p.request_id}/{fn}" for fn in p.filenames]
        text += f"\nAttached files can be read locally at: \n{'\n'.join(filepaths)}"
    return text


def create_traceback_text(e):
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    full_traceback = ''.join(tb_lines)
    sanitized_traceback = re.sub(r'File ".*?",', 'File "<redacted>",', full_traceback)
    return sanitized_traceback

def create_dfs_text(dfs) -> str:
    if not dfs:
        return ""
       
    all_snippets_text = ""
    for i, df in enumerate(dfs):
        # truncate cell values 
        df = df.map(lambda x: (str(x)[:30] + '...') if isinstance(x, str) and len(x) > 30 else x)

        # create markdown snippet 
        snippet_text = df.sample(min(3, df.shape[0])).to_markdown()
        all_snippets_text += f"\ndfs[{i}] snippet:\n{snippet_text}\n"

    dfs_text = DFS_TEXT_TEMPLATE.format(all_snippets_text=all_snippets_text)
    return dfs_text 

def create_answers_list_text(answers):
    fmtd_answers = []
    for ans in answers:
        ans = str(ans)
        ans = f"{ans[:20]}...(truncated)" if len(ans) > 20 else ans 
        fmtd_answers.append(ans)
    return ",\n".join(fmtd_answers)
