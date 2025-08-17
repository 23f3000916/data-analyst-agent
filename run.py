import sys
import requests
import json
import os

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python run.py <server_url>"}))
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")
    api_url = f"{base_url}/api"

    # Path to questions file (the evaluator mounts it automatically)
    qfile = "promptfoos/project-data-analyst-agent-sample-network/questions.txt"
    if not os.path.exists(qfile):
        print(json.dumps({"error": f"Questions file not found at {qfile}"}))
        sys.exit(1)

    with open(qfile, "rb") as f:
        files = {"questions_file": ("questions.txt", f, "text/plain")}
        try:
            resp = requests.post(api_url, files=files, timeout=180)
            resp.raise_for_status()
            # Print the API’s JSON response directly to stdout
            print(resp.text)
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(1)

if __name__ == "__main__":
    main()