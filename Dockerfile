# Dockerfile - Optimized for fast + reliable build on Render
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies early (including full compiler stack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install setuptools early + force binary wheels where possible
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "setuptools<82.0.0" wheel \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy the app code
COPY main.py .

EXPOSE ${PORT:-8080}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
