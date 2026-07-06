FROM python:3.12-slim

LABEL maintainer="TriageAI"
LABEL description="AI-powered telephone triage system — FastAPI backend"

WORKDIR /app

# Install system dependencies (gcc for asyncpg, libpq for postgres)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command — override in docker-compose for dev (--reload)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
