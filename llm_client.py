import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in the .env file")

# Configure the Gemini client
genai.configure(api_key=GEMINI_API_KEY)
#model = genai.GenerativeModel("gemini-2.5-flash-lite") # fast but poor results (good for testing)
model = genai.GenerativeModel("gemini-2.5-flash") # very good results. but a bit slower
#model = genai.GenerativeModel("gemini-2.5-pro")

def call_llm(messages: list) -> str:
    """
    Call Gemini model with a conversation-style message list.
    messages: [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}
    ]
    """
    # Gemini does not have role awareness like OpenAI; we format the prompt ourselves
    prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])

    response = model.generate_content(prompt)
    return response.text.strip()
