"""Polling loop principal du worker AGPL.

Démarrage :
    python -m worker.main
    # ou via entry-point pyproject : `watiir-redact-worker`

Stratégie polling (cf. design D2 de la sous-change E) :
  - Idle (aucune vidéo) → sleep POLL_INTERVAL_IDLE_S (30s par défaut)
  - Post-succès → boucle immédiate (la file peut contenir plus de travail)
  - Post-échec → boucle immédiate aussi (idem)
  - SIGINT/SIGTERM → graceful shutdown après la vidéo en cours

Concurrence : 1 worker = 1 vidéo à la fois (cf. design D8 — saturation VRAM
GPU L4). Scale horizontal en lançant N workers — `FOR UPDATE SKIP LOCKED`
côté PG empêche le double-traitement.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path

import cv2

from pipeline import load_models, process_video
from pipeline.detect import (
    GpsRequiredError,
    SpotDetector,
    StubDetector,
    Yolov8SpotDetector,
    detect_spots,
    parse_gps_log,
)
from pipeline.stats import VideoStats
from worker.settings import WorkerSettings, load_settings
from worker.supabase_client import (
    ClaimedDetection,
    ClaimedVideo,
    SupabaseWorkerClient,
)

logger = logging.getLogger("watiir.worker")


class GracefulShutdown:
    """Toggle bascule à True sur SIGINT/SIGTERM. Polling vérifie entre chaque
    itération pour finir proprement la vidéo en cours."""

    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, _frame):
        logger.info("Signal %s reçu — shutdown demandé.", signum)
        self.should_exit = True


def _build_redacted_bucket_path(source_bucket_path: str) -> str:
    """`<user>/<id>.<ext>` → `<user>/<id>_redacted.mp4` (output toujours mp4)."""
    p = Path(source_bucket_path)
    stem = p.stem  # nom sans extension
    parent = p.parent.as_posix() if str(p.parent) != "." else ""
    name = f"{stem}_redacted.mp4"
    return f"{parent}/{name}" if parent else name


def _probe_duration_s(video_path: Path) -> float:
    """Lit la durée via OpenCV. Retourne 0 si indéterminée."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0 and nframes > 0:
            return float(nframes) / float(fps)
        return 0.0
    finally:
        cap.release()


def process_one_video(
    claimed: ClaimedVideo,
    client: SupabaseWorkerClient,
    settings: WorkerSettings,
    models,
) -> VideoStats:
    """Pipeline complet pour 1 vidéo réservée par claim_next_pending_redaction.

    Étapes :
      1. Download bucket → fichier local
      2. Probe durée → reject si > MAX_VIDEO_DURATION_S
      3. Process via pipeline.process_video (blur + tracking)
      4. Upload résultat → bucket vision-ai-redacted
      5. notify-video-redacted (status=redacted, counts)
      6. Cleanup fichiers locaux

    Toute exception → caller notify_failed (orchestré par run_loop).
    """
    work_id = uuid.uuid4().hex[:8]
    local_in = settings.work_dir / f"{work_id}_in{Path(claimed.source_bucket_path).suffix}"
    local_out = settings.work_dir / f"{work_id}_out.mp4"

    try:
        logger.info("[%s] download %s", claimed.id, claimed.source_bucket_path)
        client.download_source(claimed.source_bucket_path, local_in)

        duration = _probe_duration_s(local_in)
        logger.info("[%s] duration ≈ %.1fs", claimed.id, duration)
        if duration > settings.max_video_duration_s:
            raise VideoTooLongError(
                f"video_too_long_max_{settings.max_video_duration_s}s (got {duration:.1f}s)"
            )

        logger.info("[%s] redact start", claimed.id)
        stats = process_video(
            video_path=str(local_in),
            models=models,
            output_path=str(local_out),
        )
        logger.info(
            "[%s] redact done — %d frames in %.1fs (RTF %.2fx), faces %d / plates %d",
            claimed.id,
            stats.n_frames,
            stats.process_s,
            stats.real_time_factor,
            stats.n_faces_total_tracks,
            stats.n_plates_total_tracks,
        )

        redacted_path = _build_redacted_bucket_path(claimed.source_bucket_path)
        logger.info("[%s] upload %s", claimed.id, redacted_path)
        client.upload_redacted(local_out, redacted_path)

        client.notify_redacted(
            video_id=claimed.id,
            redacted_bucket_path=redacted_path,
            faces_count=stats.n_faces_total_tracks,
            plates_count=stats.n_plates_total_tracks,
        )
        logger.info("[%s] notify-video-redacted OK", claimed.id)
        return stats
    finally:
        for p in (local_in, local_out):
            try:
                if p.exists():
                    p.unlink()
            except Exception as cleanup_err:
                logger.warning("[%s] cleanup %s failed: %s", claimed.id, p, cleanup_err)


class VideoTooLongError(RuntimeError):
    """Erreur applicative — sera convertie en notify_failed côté caller."""


def _load_detector() -> SpotDetector:
    """Sélectionne le détecteur de places (sous-change F).

    V4.0 : `StubDetector` par défaut (plomberie validée sans modèle). Le
    `Yolov8SpotDetector` est gaté sur un POC modèle (dataset V3.c floutées) —
    activable via SPOT_DETECTOR=yolov8 une fois fourni (cf. README)."""
    kind = os.environ.get("SPOT_DETECTOR", "stub").lower()
    if kind == "stub":
        return StubDetector()
    if kind == "yolov8":
        checkpoint = os.environ.get("SPOT_DETECTOR_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError(
                "SPOT_DETECTOR=yolov8 requiert SPOT_DETECTOR_CHECKPOINT "
                "(chemin vers le best.pt validé par le POC — cf. training/README.md)."
            )
        model_version = os.environ.get("SPOT_DETECTOR_MODEL_VERSION")
        return Yolov8SpotDetector(checkpoint, model_version=model_version)
    raise RuntimeError(f"SPOT_DETECTOR='{kind}' inconnu — valeurs supportées : 'stub', 'yolov8'.")


def process_one_detection(
    claimed: ClaimedDetection,
    client: SupabaseWorkerClient,
    settings: WorkerSettings,
    detector: SpotDetector,
) -> int:
    """Pipeline détection pour 1 vidéo floutée réservée par
    claim_next_pending_detection. Retourne le nombre de candidats publiés.

    Étapes :
      1. Download vidéo floutée → fichier local
      2. Download + parse log GPS
      3. detect_spots (détecteur pluggable + géoréférencement)
      4. notify-detection-complete (succès) → pending_moderation
      5. Cleanup fichier local

    GpsRequiredError est propagée au caller (→ notify_detection_failed)."""
    work_id = uuid.uuid4().hex[:8]
    local_in = settings.work_dir / f"{work_id}_redacted{Path(claimed.redacted_bucket_path).suffix}"

    try:
        logger.info("[%s] detect: download %s", claimed.id, claimed.redacted_bucket_path)
        client.download_redacted(claimed.redacted_bucket_path, local_in)

        gps_raw = client.download_gps_log(claimed.gps_log_path)
        gps_track = parse_gps_log(gps_raw)

        spots, model_version = detect_spots(local_in, gps_track, detector)
        logger.info("[%s] detect: %d candidats (model=%s)", claimed.id, len(spots), model_version)

        client.notify_detected(
            video_import_id=claimed.id,
            model_version=model_version,
            spots=[s.to_payload() for s in spots],
        )
        logger.info("[%s] notify-detection-complete OK", claimed.id)
        return len(spots)
    finally:
        try:
            if local_in.exists():
                local_in.unlink()
        except Exception as cleanup_err:
            logger.warning("[%s] cleanup %s failed: %s", claimed.id, local_in, cleanup_err)


def run_loop(settings: WorkerSettings) -> int:
    """Boucle polling. Retourne 0 sur shutdown propre."""
    logger.info(
        "worker démarré. supabase=%s  work_dir=%s", settings.supabase_url, settings.work_dir
    )
    logger.info("models: face=%s plate=%s", settings.face_model_path, settings.plate_model_path)

    models = load_models(settings.face_model_path, settings.plate_model_path)
    detector = _load_detector()
    logger.info("detector: %s (model=%s)", type(detector).__name__, detector.model_version)
    client = SupabaseWorkerClient(settings)
    shutdown = GracefulShutdown()

    try:
        while not shutdown.should_exit:
            # Priorité au floutage (étage amont), puis détection.
            try:
                claimed = client.claim_next_pending()
            except Exception as e:
                logger.exception("claim redaction RPC failed — sleep before retry: %s", e)
                _sleep_interruptibly(settings.poll_interval_idle_s, shutdown)
                continue

            if claimed is not None:
                _run_redaction(claimed, client, settings, models)
                continue  # boucle immédiate (file peut contenir plus)

            # Aucune vidéo à flouter → essaie l'étage détection.
            try:
                to_detect = client.claim_next_pending_detection()
            except Exception as e:
                logger.exception("claim detection RPC failed — sleep before retry: %s", e)
                _sleep_interruptibly(settings.poll_interval_idle_s, shutdown)
                continue

            if to_detect is not None:
                _run_detection(to_detect, client, settings, detector)
                continue

            # Les 2 files sont vides.
            logger.debug("rien à traiter — sleep %ds", settings.poll_interval_idle_s)
            _sleep_interruptibly(settings.poll_interval_idle_s, shutdown)
    finally:
        client.close()
        logger.info("worker arrêté.")
    return 0


def _run_redaction(
    claimed: ClaimedVideo,
    client: SupabaseWorkerClient,
    settings: WorkerSettings,
    models,
) -> None:
    """Traite 1 floutage avec gestion d'erreur → notify_failed."""
    try:
        process_one_video(claimed, client, settings, models)
    except VideoTooLongError as e:
        logger.warning("[%s] %s — notify failed", claimed.id, e)
        _safe_notify_failed(client, claimed.id, str(e))
    except Exception as e:
        logger.exception("[%s] pipeline error", claimed.id)
        _safe_notify_failed(client, claimed.id, f"pipeline_error: {type(e).__name__}")


def _run_detection(
    claimed: ClaimedDetection,
    client: SupabaseWorkerClient,
    settings: WorkerSettings,
    detector: SpotDetector,
) -> None:
    """Traite 1 détection avec gestion d'erreur → notify_detection_failed."""
    try:
        process_one_detection(claimed, client, settings, detector)
    except GpsRequiredError as e:
        logger.warning("[%s] détection sans GPS — notify failed", claimed.id)
        _safe_notify_detection_failed(client, claimed.id, str(e))
    except Exception as e:
        logger.exception("[%s] detection error", claimed.id)
        _safe_notify_detection_failed(client, claimed.id, f"detection_error: {type(e).__name__}")


def _safe_notify_failed(client: SupabaseWorkerClient, video_id: str, msg: str) -> None:
    """Best-effort notify_failed (log si la notif elle-même échoue, sans crash
    le worker). La row reste alors en 'redacting' et sera reset par le job
    pg_cron `vision_ai_reset_orphans` après 2h."""
    try:
        client.notify_failed(video_id, msg)
    except Exception as e:
        logger.exception("[%s] notify_failed failed: %s", video_id, e)


def _safe_notify_detection_failed(client: SupabaseWorkerClient, video_id: str, msg: str) -> None:
    """Best-effort notify_detection_failed. Si la notif échoue, la row reste en
    'detecting' et sera reset à 'redacted' par le job pg_cron après 2h."""
    try:
        client.notify_detection_failed(video_id, msg)
    except Exception as e:
        logger.exception("[%s] notify_detection_failed failed: %s", video_id, e)


def _sleep_interruptibly(seconds: int, shutdown: GracefulShutdown) -> None:
    """Sleep par ticks de 1s pour pouvoir sortir sur shutdown sans attendre."""
    for _ in range(max(1, seconds)):
        if shutdown.should_exit:
            return
        time.sleep(1)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        settings = load_settings()
    except RuntimeError as e:
        print(f"[fatal] {e}", file=sys.stderr)
        return 2
    return run_loop(settings)


if __name__ == "__main__":
    raise SystemExit(main())
