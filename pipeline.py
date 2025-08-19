import io 
import uuid
import builtins
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from fastapi import Request
from starlette.datastructures import UploadFile

from prompt_util import *
from llm_util import ask_llm



class Problem:
    def __init__(self) -> None:
        self.request_id = uuid.uuid4().hex[:8]
        self.request_data_path = Path(f"request_data/{self.request_id}")
        self.request_data_path.mkdir(parents=True, exist_ok=True)

        self.questions_text = ""              
        self.images = []              
        self.filenames = []                 

        self.metadata_dict = {}                  
        self.metadata_text = ""                 

        self.questions_list = []             
        self.questions_list_text = ""
        self.data_source_text = ""              
        self.output_format_text = ""              

        self.dfs = []              

        self.dfs_text = "" 

        self.answers = []




async def create_problem_instance(request: Request) -> Problem:
    uploads = [(n, v) for n, v in (await request.form()).items() if isinstance(v, UploadFile)]

    p = Problem() 
    found_questions = False

    for name, upload_file in uploads:
        if name == "questions.txt":
            p.questions_text = (await upload_file.read()).decode("utf-8")
            found_questions = True

        elif str(upload_file.content_type).startswith("image/"):
            p.images.append(Image.open(io.BytesIO(await upload_file.read())))
        else:
            file_content = await upload_file.read()
            file_path = Path(f"request_data/{p.request_id}/{name}")

            with open(file_path, "wb") as f:
                f.write(file_content)
            p.filenames.append(name)
    if not found_questions:
        raise ValueError("No 'questions.txt' file found in the request.")
    print(f"{'=' * 100}\n"
          f"Problem Instance Created:\n"
          f"questions_text: {p.questions_text[:30]}...\n"
          f"attached_files: {p.filenames if p.filenames else None}\n"
          f"images: {len(p.images)}")
    return p




async def generate_problem_metadata(p: Problem) -> dict:
    from prompt_util import PROBLEM_METADATA_TEMPLATE, PROBLEM_METADATA_SCHEMA
    attachments_text = create_attachments_text(p)
    prompt_text = PROBLEM_METADATA_TEMPLATE.format(
        questions_text = p.questions_text, 
        attachments_text = attachments_text
    )
    response_json = await ask_llm(contents = [prompt_text] + p.images, response_schema = PROBLEM_METADATA_SCHEMA)
    print(f"{'=' * 100}\n"
      f"Problem Metadata Generated:\n"
      f"{create_metadata_text(response_json)}")
    return response_json




async def load_files_as_dfs(p: Problem) -> list[pd.DataFrame]:
    from prompt_util import FILE_LOADING_SCRIPT_TEMPLATE, FILE_LOADING_SCRIPT_SCHEMA
    if len(p.filenames) == 0:
        return []

    prompt_text = FILE_LOADING_SCRIPT_TEMPLATE.format(
        data_source_text = p.data_source_text, 
        request_id = p.request_id,
    )
    response_json = await ask_llm(
        contents = [prompt_text], 
        response_schema = FILE_LOADING_SCRIPT_SCHEMA
    ) 
    script = response_json.get("script")
    if not script: 
        return []
    
    files_dfs = await run_file_loading_script(script)
    valid_dfs = [df for df in files_dfs if isinstance(df, pd.DataFrame) and not df.empty]
    print(f"{'=' * 100}\n"
          f"{len(p.filenames)} attached files loaded as {len(valid_dfs)} dataframes.")
    return valid_dfs




async def run_file_loading_script(script, max_tries = 4) -> list[pd.DataFrame]:
    from prompt_util import FIX_FILE_LOADING_TEMPLATE, FIX_FILE_LOADING_SCHEMA

    fix_history = []

    for attempt in range(max_tries):
        try:
            env = {"__builtins__": builtins}
            exec(script, env)
            files_dfs = env.get("files_dfs", [])
            if isinstance(files_dfs, list):
                return files_dfs
            else:
                print(
                    f"{'=' * 100}\n"
                    f"Final File Loading Script:\n"
                    f"{script}"
                )
                return []

        except Exception as e:
            if attempt < max_tries - 1:
                attempt_specific_text = "Do not use any try-except block in the script."
            else:
                attempt_specific_text = "Wrap each file load in try-except and skip failing ones."

            if fix_history:
                fix_history_text = "\n".join(desc for desc in fix_history)
            else:
                fix_history_text = "None"

            fix_prompt = FIX_FILE_LOADING_TEMPLATE.format(
                script = script,
                traceback_text = create_traceback_text(e),
                fix_history_text = fix_history_text,
                attempt_specific_text = attempt_specific_text
            )
            response = await ask_llm(contents=[fix_prompt], response_schema=FIX_FILE_LOADING_SCHEMA)

            script = response.get("fixed_script", "")

            fix_description = response.get("fix_description", "").strip()
            
            fix_history.append(f"Attempt {attempt + 1}: {fix_description}")
    return []




async def webscrape_tables_if_needed(p: Problem) -> list[pd.DataFrame]:
    from prompt_util import WEBSCRAPE_URL_TEMPLATE, WEBSCRAPE_URL_SCHEMA

    prompt_text = WEBSCRAPE_URL_TEMPLATE.format(data_source_text = p.data_source_text)

    response_json = await ask_llm(contents = [prompt_text], response_schema = WEBSCRAPE_URL_SCHEMA)
    URL = response_json.get("URL")
    if not URL: 
        return []
    try: 
        scraped_tables = pd.read_html(URL)

        valid_tables = [table for table in scraped_tables if not table.empty and 
            any(not str(col).startswith("Unnamed") and not isinstance(col, int) for col in table.columns)]  
        print(
            f"{'=' * 100}\n"
            f"{len(valid_tables)} tables from {URL} loaded as {len(valid_tables)} dataframes.\n"
        )
        return valid_tables
    
    except ValueError as e:
        return []




async def find_question_answers(p: Problem) -> list:
    from prompt_util import QUESTION_SCRIPTS_TEMPLATE, QUESTION_SCRIPTS_SCHEMA

    prompt_text = QUESTION_SCRIPTS_TEMPLATE.format(
        data_source_text = p.data_source_text,
        questions_list_text = p.questions_list_text, 
        dfs_text = p.dfs_text,
        find_answer_args = "dfs=None" if len(p.dfs) > 0 else ""
    )
    response_json = await ask_llm(
        contents=[prompt_text],
        response_schema=QUESTION_SCRIPTS_SCHEMA
    )
    scripts = response_json.get("scripts", [])

    answers = []

    for qno, script in enumerate(scripts):
        question_string = p.questions_list[qno]
        ans = await run_question_script(script, question_string, p)
        answers.append(json_compatible(ans))
    return answers



async def run_question_script(script: str, question_string: str, p: Problem, max_tries: int = 4):
    from prompt_util import FIX_QUESTION_SCRIPT_TEMPLATE, FIX_QUESTION_SCRIPT_SCHEMA

    fix_history = []
        
    for attempt in range(max_tries):
        try:
            env = {'__builtins__': builtins}

            exec(script, env)

            if "find_answer" not in env:
                raise NameError("The function 'find_answer' is not defined in the script.")

            if len(p.dfs) > 0:
                answer = env["find_answer"](p.dfs)
            else:
                answer = env["find_answer"]()
            print(
                f"{'=' * 100}\n"
                f"Final Script For '{question_string}':\n"
                f"{script}"
            )
            return answer
        
        except Exception as e:        
            if attempt < max_tries - 1:
                prompt_text = FIX_QUESTION_SCRIPT_TEMPLATE.format(
                    data_source_text = p.data_source_text,
                    questions_list_text = p.questions_list_text,
                    dfs_text = p.dfs_text,
                    script = script,
                    traceback_text = create_traceback_text(e),
                    fix_history_text = '\n-'.join(fix_history) if fix_history else 'None',
                    question_string = question_string
                )                
                response = await ask_llm(
                    contents = [prompt_text],
                    response_schema = FIX_QUESTION_SCRIPT_SCHEMA,
                )
                script = response.get("fixed_script", "")
                fix_description = response.get("fix_description", "").strip()
                
                fix_history.append(f"Attempt {attempt + 1}: {fix_description}")
            else:
                return None 
            


async def generate_output(p: Problem, max_tries=4):
    from prompt_util import OUTPUT_SCRIPT_TEMPLATE, OUTPUT_SCRIPT_SCHEMA

    fix_history = []
    
    prompt_text = OUTPUT_SCRIPT_TEMPLATE.format(
        questions_list_text = p.questions_list_text,
        answers_list_text = create_answers_list_text(p.answers),
        output_format_text = p.output_format_text
    )
    for attempt in range(max_tries):
        response = await ask_llm(contents=[prompt_text], response_schema=OUTPUT_SCRIPT_SCHEMA)
        script = response.get("script")
        try:
            env = {'__builtins__': builtins}
            exec(script, env)
            output = env["create_output"](p.answers) # type: ignore
            print(
                f"{'=' * 100}\n"
                f"Final Output Generation Script:\n"
                f"{script}"
            )
            return output
        except Exception as e:
            if attempt < max_tries - 1:
                prompt_text = FIX_OUTPUT_SCRIPT_TEMPLATE.format(
                    questions_list_text = p.questions_list_text,
                    answers_list_text = create_answers_list_text(p.answers),
                    output_format_text = p.output_format_text,
                    script = script,
                    traceback_text = create_traceback_text(e),
                    fix_history_text = '\n-'.join(fix_history) if fix_history else 'None',
                )                
                response = await ask_llm(
                    contents = [prompt_text],
                    response_schema = FIX_OUTPUT_SCRIPT_SCHEMA,
                )
                script = response.get("fixed_script")
                if not script:
                    raise Exception("fixed script not found")
                fix_description = response.get("fix_description", "").strip()
                
                fix_history.append(f"Attempt {attempt + 1}: {fix_description}")
            else:
                return str(e)


def json_compatible(obj):
    if obj is None:
        return None
    try:
        return obj.item()
    except:
        pass 

    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    if isinstance(obj, np.ndarray):
        return [json_compatible(x) for x in obj.tolist()]
    
    if isinstance(obj, (pd.Series, pd.Index)):
        return [json_compatible(x) for x in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        return obj.map(json_compatible).to_dict(orient='records')
    
    # Handle collections
    if isinstance(obj, dict):
        return {k: json_compatible(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [json_compatible(x) for x in obj]

    return str(obj)
