# --- Stage 1: Builder ---
FROM python:3.11-slim as builder

# Avoid .pyc files and buffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Isolated virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Runner ---
FROM python:3.11-slim as runner

WORKDIR /app

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DATA_DIR=/app/data \
    NEW_RELIC_LOG=stderr \
    NEW_RELIC_LOG_LEVEL=info \
    NEW_RELIC_CAPTURE_PARAMS=false \
    NEW_RELIC_ATTRIBUTES_EXCLUDE=request.parameters.*,request.headers.authorization,request.headers.cookie

# Application source
COPY app/src/ ./src/
COPY app/main.py .

# Initial model artifact (required for first API boot)
COPY app/models/ ./models/

# Embedded dataset for Render Free
RUN mkdir -p /app/data
COPY app/data/PEDE_PASSOS_DATASET_FIAP.xlsx /app/data/PEDE_PASSOS_DATASET_FIAP.xlsx

# Permissions
RUN chown -R appuser:appuser /app

USER appuser

# Render injects PORT automatically
CMD sh -c "newrelic-admin run-program uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
