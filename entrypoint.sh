#!/bin/sh
echo "Starting app on port ${PORT:-8000}..."
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}

