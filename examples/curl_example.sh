#!/usr/bin/env bash
set -e

URL=${1:-http://localhost:8000/api/}

curl -X POST "$URL" \
  -F "questions.txt=@tests/sample_questions/sample_questions_1.txt" \
  -F "data.csv=@tests/sample_questions/sample_data.csv" \
  -H "Accept: application/json"
