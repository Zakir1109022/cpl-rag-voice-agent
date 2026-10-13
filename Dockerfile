# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

# --- Build stage ---
FROM base AS build

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Playwright / Chromium runtime libraries
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    fonts-liberation fonts-noto-color-emoji \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN mkdir -p src

RUN uv sync

ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && \
    uv run --no-project python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Install Playwright + Chromium browser binary.
# The Python package is already in pyproject.toml; this command downloads
# the browser (~300MB) into the image so the runtime has no network
# dependency on first launch.
RUN uv run playwright install chromium

# Copy the application, including source PDFs under docs/.
COPY . .

# Build the local Qdrant index into the image. The runtime image will use
# this same packaged index; no external Qdrant service is required.
#
# LiveKit/Groq values are placeholders because ingestion only needs the
# Settings object to initialize. Qdrant is intentionally kept local.
COPY --chmod=0666 <<EOF /app/.ingest-marker
Build-time ingestion marker. Delete to force rebuild of the Qdrant index.
EOF

RUN rm -rf /app/data/qdrant && \
    LIVEKIT_URL="wss://placeholder.livekit.cloud" \
    LIVEKIT_API_KEY="placeholder" \
    LIVEKIT_API_SECRET="placeholder" \
    GROQ_API_KEY="placeholder" \
    QDRANT_URL="" \
    QDRANT_API_KEY="" \
    uv run python ingest.py && \
    test -d /app/data/qdrant && \
    uv run python -c "\
from qdrant_client import QdrantClient; \
q=QdrantClient(path='/app/data/qdrant'); \
info=q.get_collection('telecom_docs'); \
assert info.points_count > 0, 'Qdrant collection telecom_docs contains 0 points'; \
print(f'[OK] Verified telecom_docs with {info.points_count} points')"

# --- Production stage ---
FROM base

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

WORKDIR /app

COPY --from=build --chown=appuser:appuser /app /app

USER appuser

ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface/hub

# Keep the packaged local Qdrant index location explicit.
ENV QDRANT_PATH=/app/data/qdrant

CMD ["uv", "run", "python", "voice_agent/main.py", "start"]
