from dotenv import load_dotenv
from fastapi import FastAPI, Request

from pipeline import *
from prompt_util import *

import shutil

# initialize app
app = FastAPI()

@app.post("/analyze")
async def analyze(request: Request):

    try:
        p = await create_problem_instance(request)  
                
        p.metadata_dict = await generate_problem_metadata(p) 
        p.metadata_text = create_metadata_text(p.metadata_dict)

        p.questions_list = p.metadata_dict.get("questions", [])
        p.questions_list_text = create_questions_list_text(p.questions_list)

        if not p.questions_list:
            raise ValueError("No questions found.")
        
        p.data_source_text = p.metadata_dict.get("data_source_text", "")
        p.output_format_text = p.metadata_dict.get("output_format_text", "")

        files_dfs = await load_files_as_dfs(p)

        scraped_dfs = await webscrape_tables_if_needed(p)

        p.dfs = files_dfs + scraped_dfs
        p.dfs_text = create_dfs_text(p.dfs)

        # find the answers to the questions
        p.answers = await find_question_answers(p)

        output = await generate_output(p)
        try: shutil.rmtree(p.request_data_path)
        except: pass
        return output

    except Exception as e:
        try: shutil.rmtree(p.request_data_path)
        except: pass
        print(create_traceback_text(e))
        return {"error": "Analysis failed", "message": str(e)}
            
