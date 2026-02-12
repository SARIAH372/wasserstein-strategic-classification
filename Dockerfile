 FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

COPY requirements.txt .

# Upgrade pip and install only wheels
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash", "-lc", "exec streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]

