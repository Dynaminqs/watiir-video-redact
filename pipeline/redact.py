"""Pipeline floutage vidéo — port direct du POC validé 2026-05-27.

Stack : YOLOv8-face (HF arnabdhar) + YOLOv8-LP (HF yasirfaizahmed) + ByteTrack
(natif Ultralytics) + OpenCV GaussianBlur. Pas de fine-tune en V4.0.

Cf. README pour la vue d'ensemble + ARCHITECTURE V4 dans le repo WATIIR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.stats import VideoStats


@dataclass
class RedactionModels:
    """2 modèles chargés en mémoire (face + plate)."""

    face: YOLO
    plate: Optional[YOLO]


def load_models(face_model_path: Path, plate_model_path: Optional[Path]) -> RedactionModels:
    """Charge les modèles depuis disque. `plate_model_path=None` → mode face-only
    (POC a validé que c'est acceptable en dégradé, pas en V4.0 prod)."""
    face = YOLO(str(face_model_path))
    plate = YOLO(str(plate_model_path)) if plate_model_path is not None else None
    return RedactionModels(face=face, plate=plate)


def blur_boxes(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Applique un Gaussian blur sur chaque bbox (x1, y1, x2, y2) du frame.

    `boxes` est un array (N, 4) en float ou int. Le blur est appliqué sur une
    copie du frame (le frame d'origine n'est pas muté). Kernel size proportionnel
    à la largeur de la box pour un flou cohérent quelle que soit la résolution.

    Boxes hors bounds ou de surface 0 sont skippées silencieusement.
    """
    out = frame.copy()
    if len(boxes) == 0:
        return out
    h, w = out.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int) if box.dtype != np.int64 else box
        # Clamp aux bounds frame
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        roi = out[y1:y2, x1:x2]
        # Kernel impair, minimum 15 pour rester perceptible
        ksize = max(15, ((x2 - x1) // 3) | 1)
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        out[y1:y2, x1:x2] = blurred
    return out


def process_video(
    video_path: str,
    models: RedactionModels,
    output_path: str,
    conf_threshold: float = 0.25,
    tracker: str = "bytetrack.yaml",
    progress_every_n_frames: int = 30,
) -> VideoStats:
    """Lance le pipeline floutage temporal sur une vidéo entière.

    Retourne les statistiques agrégées (latence, n tracks, real-time factor) et
    écrit la vidéo floutée à `output_path` en MP4 H.264.

    Le tracker ByteTrack persiste son état entre frames (`persist=True`) :
    chaque visage / plaque identifié garde le même track_id tant qu'il reste
    visible, ce qui élimine le flicker du flou frame par frame (le pire des
    leaks RGPD).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"impossible d'ouvrir la vidéo {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = n_frames / fps if fps > 0 else 0.0

    stats = VideoStats(
        video_path=video_path,
        n_frames=n_frames,
        duration_s=duration_s,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"impossible d'ouvrir le writer pour {output_path}")

    face_track_ids: set[int] = set()
    plate_track_ids: set[int] = set()
    process_start = time.perf_counter()

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            inf_start = time.perf_counter()
            face_results = models.face.track(
                frame, persist=True, tracker=tracker, conf=conf_threshold, verbose=False
            )
            plate_results = (
                models.plate.track(
                    frame,
                    persist=True,
                    tracker=tracker,
                    conf=conf_threshold,
                    verbose=False,
                )
                if models.plate is not None
                else None
            )
            inf_ms = (time.perf_counter() - inf_start) * 1000
            stats.inference_ms.append(inf_ms)

            face_boxes = _extract_boxes(face_results)
            plate_boxes = _extract_boxes(plate_results)

            _collect_track_ids(face_results, face_track_ids)
            _collect_track_ids(plate_results, plate_track_ids)

            # Combine face + plate boxes pour 1 seule passe de blur
            if len(plate_boxes):
                all_boxes = np.vstack([face_boxes, plate_boxes])
            else:
                all_boxes = face_boxes
            blurred = blur_boxes(frame, all_boxes) if len(all_boxes) else frame
            writer.write(blurred)

            if frame_idx % progress_every_n_frames == 0:
                print(
                    f"  frame {frame_idx}/{n_frames}  "
                    f"inf {inf_ms:.1f}ms  "
                    f"faces tracked: {len(face_track_ids)}  "
                    f"plates: {len(plate_track_ids)}",
                    flush=True,
                )
    finally:
        cap.release()
        writer.release()

    stats.process_s = time.perf_counter() - process_start
    stats.n_faces_total_tracks = len(face_track_ids)
    stats.n_plates_total_tracks = len(plate_track_ids)
    return stats


def _extract_boxes(results) -> np.ndarray:
    """Renvoie un array (N, 4) xyxy depuis le résultat ultralytics, ou vide."""
    if not results or results[0].boxes is None:
        return np.empty((0, 4))
    xyxy = results[0].boxes.xyxy
    return xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)


def _collect_track_ids(results, sink: set[int]) -> None:
    """Ajoute les track_ids du résultat ultralytics dans `sink` (in-place)."""
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        return
    ids = results[0].boxes.id
    ids_np = ids.cpu().numpy() if hasattr(ids, "cpu") else np.asarray(ids)
    for tid in ids_np:
        sink.add(int(tid))
