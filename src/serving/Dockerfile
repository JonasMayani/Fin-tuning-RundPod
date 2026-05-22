# ─────────────────────────────────────────────────────────────────────────────
# Multilingual African Health Assistant — Inference Server
# Phase 5.4 — Containerisation
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Python dependencies ────────────────────────────────────────────────────
COPY requirements.txt .

# Install CPU-only PyTorch for inference (lighter image)
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ─── Application code ────────────────────────────────────────────────────────
COPY src/             /app/src/
COPY src/training/config.yaml /app/src/training/config.yaml

# ─── Model artefacts (mounted at runtime OR baked in) ────────────────────────
# For production: mount model directory as a volume at /app/models/
# For embedded deployment: COPY models/ /app/models/
VOLUME ["/app/models"]

# ─── Non-root user ───────────────────────────────────────────────────────────
RUN adduser --disabled-password --gecos "" appuser
RUN chown -R appuser /app
USER appuser

# ─── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

# ─── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
