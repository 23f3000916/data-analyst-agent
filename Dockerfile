FROM python:3.10

WORKDIR /app

# Install system dependencies for OCR (EasyOCR + PyMuPDF need these)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face sets PORT automatically
ENV PORT=7860

CMD uvicorn app:app --host 0.0.0.0 --port $PORT
