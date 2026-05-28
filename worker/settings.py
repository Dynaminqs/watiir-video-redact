"""Configuration du worker — 100% lue depuis variables d'environnement.

Aucune valeur par défaut sensible (pas de fallback à un staging hardcodé).
Les credentials Supabase DOIVENT être passés par env vars (ou secret manager
au runtime — Scaleway Secret Manager, AWS Secrets Manager, etc.).

Cf. docs/deploy-scaleway-l4.md pour la procédure de provisioning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Buckets définis par la migration SQL côté WATIIR — voir
# `supabase/migrations/20260528030000_v4_vision_ai_backend_floutage.sql`.
BUCKET_UPLOADS = "vision-ai-uploads"
BUCKET_REDACTED = "vision-ai-redacted"

# Endpoints Edge Function (suffixés au SUPABASE_URL).
NOTIFY_ENDPOINT = "/functions/v1/notify-video-redacted"
DETECT_NOTIFY_ENDPOINT = "/functions/v1/notify-detection-complete"

# RPC PostgREST qui retournent la prochaine vidéo en attente (floutage / détection).
CLAIM_RPC = "claim_next_pending_redaction"
CLAIM_DETECTION_RPC = "claim_next_pending_detection"


@dataclass(frozen=True)
class WorkerSettings:
    """Snapshot immutable lu une fois au boot."""

    supabase_url: str
    supabase_service_role_key: str
    poll_interval_idle_s: int
    work_dir: Path
    face_model_path: Path
    plate_model_path: Path | None
    max_video_duration_s: int
    request_timeout_s: float

    @property
    def notify_url(self) -> str:
        return f"{self.supabase_url.rstrip('/')}{NOTIFY_ENDPOINT}"

    @property
    def detect_notify_url(self) -> str:
        return f"{self.supabase_url.rstrip('/')}{DETECT_NOTIFY_ENDPOINT}"


def load_settings() -> WorkerSettings:
    """Lit la config depuis l'env. Fail-fast si une variable critique manque."""

    def required(key: str) -> str:
        val = os.environ.get(key)
        if not val:
            raise RuntimeError(
                f"Variable d'environnement {key} requise et absente. "
                f"Cf. docs/deploy-scaleway-l4.md."
            )
        return val

    supabase_url = required("SUPABASE_URL").rstrip("/")
    service_role_key = required("SUPABASE_SERVICE_ROLE_KEY")

    poll_interval = int(os.environ.get("POLL_INTERVAL_IDLE_S", "30"))
    work_dir = Path(os.environ.get("WORK_DIR", "/tmp/watiir-redact-work"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # Les paths modèles sont relatifs au repo si en dev local, ou
    # /app/models/<file>.pt dans le container Docker (cf. Dockerfile).
    models_dir = Path(
        os.environ.get("MODELS_DIR", str(Path(__file__).resolve().parent.parent / "models"))
    )
    face_model_path = Path(os.environ.get("FACE_MODEL_PATH", str(models_dir / "model.pt")))
    plate_env = os.environ.get("PLATE_MODEL_PATH", str(models_dir / "best.pt"))
    plate_model_path: Path | None = (
        Path(plate_env) if plate_env and plate_env.lower() not in ("", "none") else None
    )

    # Limites de sécurité côté worker (en plus des CHECK SQL côté WATIIR).
    max_duration = int(os.environ.get("MAX_VIDEO_DURATION_S", "600"))  # 10 min hard cap
    request_timeout = float(os.environ.get("REQUEST_TIMEOUT_S", "30.0"))

    return WorkerSettings(
        supabase_url=supabase_url,
        supabase_service_role_key=service_role_key,
        poll_interval_idle_s=poll_interval,
        work_dir=work_dir,
        face_model_path=face_model_path,
        plate_model_path=plate_model_path,
        max_video_duration_s=max_duration,
        request_timeout_s=request_timeout,
    )
