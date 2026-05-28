# syntax=docker/dockerfile:1.7
# WATIIR V4 — worker AGPL de floutage vidéo.
#
# Build :
#   docker build -t watiir-video-redact:dev .
#
# Run :
#   docker run --rm --gpus all \
#     -e SUPABASE_URL=https://...supabase.co \
#     -e SUPABASE_SERVICE_ROLE_KEY=... \
#     watiir-video-redact:dev
#
# Recommandation prod : Scaleway L4 EU. Cf. docs/deploy-scaleway-l4.md.

# ── Stage 1 : build deps + download models ──────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Deps OS minimales pour opencv-python-headless (libGL/libgomp), ffmpeg pour
# certains codecs vidéo, ca-certificates pour HTTPS vers HF/Supabase.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache pip wheels avant le code (couche réutilisable si pyproject inchangé)
COPY pyproject.toml ./
COPY README.md LICENSE ./
RUN pip install --upgrade pip && pip install -e .

# Copier le code minimal pour faire tourner download.py
COPY models/ /build/models/

# Download + verify modèles ML (sha256 figés dans manifest.json).
# Le build échoue si checksums divergent ou si network HF est down.
RUN python -m models.download

# ── Stage 2 : runtime slim ──────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    WORK_DIR=/tmp/watiir-redact-work \
    MODELS_DIR=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 watiir

# Copier l'env Python construit + les modèles déjà téléchargés
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/watiir-redact-worker /usr/local/bin/watiir-redact-worker
COPY --from=builder /build/models /app/models

# Code application
COPY pipeline/ /app/pipeline/
COPY worker/ /app/worker/

WORKDIR /app
USER watiir

# Healthcheck minimal — Python check qu'on peut importer le worker.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import worker.main" || exit 1

ENTRYPOINT ["watiir-redact-worker"]
