FROM python:3.11-slim

WORKDIR /app

# System deps that prevent common build failures (pandas/numpy wheels usually avoid compiling,
# but this makes it robust if anything falls back to source).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip tooling to avoid metadata-generation failures
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway provides PORT
CMD ["bash", "-lc", "exec streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
