# run.py
import os
import sys
import glob
import json
import requests

SEARCH_DIRS = [
    ".",  # current working dir
    "promptfoos/project-data-analyst-agent-sample-network",
    "promptfoos",  # just in case
]

ATTACH_PATTERNS = ["*.csv", "*.png", "*.jpg", "*.jpeg", "*.json", "*.pdf", "*.txt"]

def find_first_questions():
    for base in SEARCH_DIRS:
        # exact field name used by harness
        p1 = os.path.join(base, "questions.txt")
        if os.path.exists(p1):
            return p1
        # sometimes named 'question.txt'
        p2 = os.path.join(base, "question.txt")
        if os.path.exists(p2):
            return p2
    return None

def collect_attachments():
    found = []
    for base in SEARCH_DIRS:
        if not os.path.isdir(base):
            continue
        for pattern in ATTACH_PATTERNS:
            for path in glob.glob(os.path.join(base, pattern)):
                found.append(path)
    # de-dup, keep order
    seen = set()
    uniq = []
    for p in found:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error":"Usage: run.py <server_url>"}))
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")
    api_url = f"{base_url}/api/"

    qpath = find_first_questions()
    if not qpath:
        print(json.dumps({"error":"questions.txt not found"}))
        sys.exit(1)

    files = {}
    # The harness uses field key 'questions.txt'
    files["questions.txt"] = ("questions.txt", open(qpath, "rb"), "text/plain")

    # Attach other files if present
    for path in collect_attachments():
        name = os.path.basename(path)
        if name.lower() in ("questions.txt", "question.txt"):
            continue
        mime = "application/octet-stream"
        if name.lower().endswith(".csv"):
            mime = "text/csv"
        elif name.lower().endswith(".json"):
            mime = "application/json"
        elif name.lower().endswith(".png"):
            mime = "image/png"
        elif name.lower().endswith((".jpg", ".jpeg")):
            mime = "image/jpeg"
        elif name.lower().endswith(".pdf"):
            mime = "application/pdf"
        elif name.lower().endswith(".txt"):
            mime = "text/plain"
        # Use the filename as the multipart field key (works with our app)
        files[name] = (name, open(path, "rb"), mime)

    try:
        resp = requests.post(api_url, files=files, timeout=180)
        resp.raise_for_status()
        # Print raw JSON
        print(resp.text)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()