# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system deps + git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first → better Docker layer caching
COPY requirements.txt .

# Install Python deps (faster flags for Render free tier)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy code
COPY main.py .

EXPOSE ${PORT:-8080}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
