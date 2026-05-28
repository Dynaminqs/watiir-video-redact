"""Wrapper Supabase pour le worker.

Encapsule les 3 opérations consommées par le polling loop :
  - claim_next_pending_redaction (RPC)
  - download_source / upload_redacted (Storage)
  - notify_redacted / notify_failed (Edge Function notify-video-redacted)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
from supabase import Client, create_client

from worker.settings import (
    BUCKET_REDACTED,
    BUCKET_UPLOADS,
    CLAIM_DETECTION_RPC,
    CLAIM_RPC,
    WorkerSettings,
)


@dataclass
class ClaimedVideo:
    """1 row réservée par claim_next_pending_redaction (status passé à 'redacting')."""

    id: str
    source_bucket_path: str
    gps_log_path: str | None
    file_size_bytes: int | None


@dataclass
class ClaimedDetection:
    """1 row réservée par claim_next_pending_detection (status passé à 'detecting')."""

    id: str
    redacted_bucket_path: str
    gps_log_path: str | None
    duration_s: int | None


class SupabaseWorkerClient:
    """Façade au-dessus de supabase-py + httpx (Edge Function notify)."""

    def __init__(self, settings: WorkerSettings):
        self._settings = settings
        self._supabase: Client = create_client(
            settings.supabase_url, settings.supabase_service_role_key
        )
        self._http = httpx.Client(timeout=settings.request_timeout_s)

    # ── Queue ────────────────────────────────────────────────────────────

    def claim_next_pending(self) -> ClaimedVideo | None:
        """Réserve la prochaine vidéo `uploaded` (la plus ancienne).

        Returns None si rien à traiter. Retourne ClaimedVideo (status flippé à
        'redacting' atomiquement côté PG via FOR UPDATE SKIP LOCKED) sinon.
        """
        result = self._supabase.rpc(CLAIM_RPC).execute()
        rows = result.data or []
        if not rows:
            return None
        row = rows[0]
        return ClaimedVideo(
            id=row["id"],
            source_bucket_path=row["source_bucket_path"],
            gps_log_path=row.get("gps_log_path"),
            file_size_bytes=row.get("file_size_bytes"),
        )

    def claim_next_pending_detection(self) -> ClaimedDetection | None:
        """Réserve la prochaine vidéo `redacted` à détecter (la plus ancienne).

        Returns None si rien à traiter. Status flippé à 'detecting' atomiquement
        côté PG (FOR UPDATE SKIP LOCKED)."""
        result = self._supabase.rpc(CLAIM_DETECTION_RPC).execute()
        rows = result.data or []
        if not rows:
            return None
        row = rows[0]
        return ClaimedDetection(
            id=row["id"],
            redacted_bucket_path=row["redacted_bucket_path"],
            gps_log_path=row.get("gps_log_path"),
            duration_s=row.get("duration_s"),
        )

    # ── Storage ──────────────────────────────────────────────────────────

    def download_source(self, source_bucket_path: str, dest: Path) -> Path:
        """Download bucket vision-ai-uploads/<path> → dest local. Retourne dest."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = self._supabase.storage.from_(BUCKET_UPLOADS).download(source_bucket_path)
        if not isinstance(data, (bytes, bytearray)):
            # supabase-py renvoie bytes ; couverture défensive
            raise RuntimeError(
                f"download_source: type inattendu {type(data).__name__} pour {source_bucket_path}"
            )
        dest.write_bytes(bytes(data))
        return dest

    def download_redacted(self, redacted_bucket_path: str, dest: Path) -> Path:
        """Download bucket vision-ai-redacted/<path> → dest local. Retourne dest."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = self._supabase.storage.from_(BUCKET_REDACTED).download(redacted_bucket_path)
        if not isinstance(data, (bytes, bytearray)):
            raise RuntimeError(
                f"download_redacted: type inattendu {type(data).__name__} pour {redacted_bucket_path}"
            )
        dest.write_bytes(bytes(data))
        return dest

    def download_gps_log(self, gps_log_path: str | None) -> bytes | None:
        """Download le log GPS depuis vision-ai-uploads. None si absent/illisible."""
        if not gps_log_path:
            return None
        data = self._supabase.storage.from_(BUCKET_UPLOADS).download(gps_log_path)
        return bytes(data) if isinstance(data, (bytes, bytearray)) else None

    def upload_redacted(self, local_path: Path, target_bucket_path: str) -> str:
        """Upload local → bucket vision-ai-redacted/<target>. Retourne target."""
        with local_path.open("rb") as f:
            self._supabase.storage.from_(BUCKET_REDACTED).upload(
                path=target_bucket_path,
                file=f,
                file_options={
                    "content-type": "video/mp4",
                    "upsert": "false",
                },
            )
        return target_bucket_path

    # ── Notify Edge Function ─────────────────────────────────────────────

    def notify_redacted(
        self,
        video_id: str,
        redacted_bucket_path: str,
        faces_count: int,
        plates_count: int,
    ) -> None:
        """POST /functions/v1/notify-video-redacted avec status=redacted."""
        self._post_notify(
            self._settings.notify_url,
            {
                "id": video_id,
                "status": "redacted",
                "redacted_bucket_path": redacted_bucket_path,
                "faces_count": faces_count,
                "plates_count": plates_count,
            },
            "notify-video-redacted",
        )

    def notify_failed(self, video_id: str, error_message: str) -> None:
        """POST /functions/v1/notify-video-redacted avec status=failed_redaction.

        Le `error_message` est conservé tel quel dans `video_imports.error_message`
        (visible par les admins via get-video-import-status + console G future).
        Garder court et snake_case (ex: `video_too_long_max_600s`,
        `ffmpeg_decode_error`, `model_inference_oom`)."""
        self._post_notify(
            self._settings.notify_url,
            {
                "id": video_id,
                "status": "failed_redaction",
                "error_message": error_message,
            },
            "notify-video-redacted",
        )

    # ── Notify détection (Edge Function notify-detection-complete) ───────

    def notify_detected(
        self,
        video_import_id: str,
        model_version: str,
        spots: list[dict],
    ) -> None:
        """POST /functions/v1/notify-detection-complete (succès).

        `spots` = liste de dicts {lng, lat, spot_kind?, spot_kind_confidence?,
        source_frame_ts_ms?, bbox?}."""
        self._post_notify(
            self._settings.detect_notify_url,
            {
                "video_import_id": video_import_id,
                "model_version": model_version,
                "spots": spots,
            },
            "notify-detection-complete",
        )

    def notify_detection_failed(self, video_import_id: str, error_message: str) -> None:
        """POST /functions/v1/notify-detection-complete (échec).

        `error_message` court snake_case (ex: `gps_required_for_georeferencing`,
        `detection_error`)."""
        self._post_notify(
            self._settings.detect_notify_url,
            {"video_import_id": video_import_id, "error": error_message},
            "notify-detection-complete",
        )

    def _post_notify(self, url: str, payload: dict, label: str = "notify-video-redacted") -> None:
        resp = self._http.post(
            url,
            headers={
                "Authorization": f"Bearer {self._settings.supabase_service_role_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"{label} HTTP {resp.status_code}: {resp.text[:500]}")

    def close(self) -> None:
        self._http.close()
