# =============================================================================
# Dockerfile — Real-Time Fraud Detection API
# =============================================================================
# Build:  docker build -t fraud-api .
# Run:    docker run -p 8000:8000 fraud-api
# =============================================================================

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create model and static dirs
RUN mkdir -p models static

# Expose port
EXPOSE 8000

# Start API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
