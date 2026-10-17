# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

# Chromium (Playwright) runtime libraries.
#
# These live in `base` rather than in the build stage on purpose: the
# production stage is `FROM base` and does NOT inherit the build stage's apt
# layers, so anything installed only in `build` is absent from the deployed
# image. Omitting them makes `chromium.launch()` fail with a missing shared
# library, and every browser_* tool returns an error the LLM reads aloud as
# "I'm unable to open a browser in this environment".
RUN apt-get update && apt-get install -y \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    libglib2.0-0 \
    fonts-liberation fonts-noto-color-emoji \
  && rm -rf /var/lib/apt/lists/*

# Download Chromium into /app so the `COPY --from=build /app /app` step below
# actually ships the browser binary. Unset, `playwright install chromium`
# writes to /root/.cache/ms-playwright in the build stage, which the
# production stage never copies — the worker then boots fine but every
# browser tool fails with "Executable doesn't exist".
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.cache/ms-playwright

# --- Build stage ---
FROM base AS build

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
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
#
# PLAYWRIGHT_BROWSERS_PATH (set in `base`) points this download at
# /app/.cache/ms-playwright, which the production stage copies from /app.
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

# Fail the build — not the first customer call — if the shipped image cannot
# launch Chromium as the runtime user. This is the guard against the two
# regression classes above: a browser binary that never made it into this
# stage, and Chromium runtime libraries missing from `base`.
RUN uv run --no-project python -c "\
from playwright.sync_api import sync_playwright; \
pw = sync_playwright().start(); \
browser = pw.chromium.launch(args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']); \
print(f'[OK] Chromium {browser.version} launches as the runtime user'); \
browser.close(); \
pw.stop()"

CMD ["uv", "run", "python", "voice_agent/main.py", "start"]
