FROM python:3.12-slim

WORKDIR /app

# System deps for PyMuPDF, sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps — CPU-only torch to keep image small
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir anthropic aiofiles

# Copy app
COPY backend/ ./
COPY frontend/ ./frontend/
COPY knowledge/ ./knowledge/

RUN mkdir -p /app/logs /app/chroma_db /app/.hf_cache /app/documents

EXPOSE 8010

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
