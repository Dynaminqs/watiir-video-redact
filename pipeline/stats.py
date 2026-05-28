"""Dataclasses stats agrégées du pipeline.

Aucune dépendance ML : sert les tests pytest sans pull-in d'ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameStats:
    """Stats d'un frame traité (latence inference, nombre de boxes flouttées)."""

    frame_idx: int
    inference_ms: float
    n_face_boxes: int
    n_plate_boxes: int


@dataclass
class VideoStats:
    """Stats agrégées d'une vidéo entière traitée.

    Mesures consommées par le worker pour `notify-video-redacted` :
    - `n_faces_total_tracks` / `n_plates_total_tracks` → faces_count / plates_count
    - `process_s` / `duration_s` → SLO RTF (real-time factor) à surveiller
    """

    video_path: str
    n_frames: int
    duration_s: float
    process_s: float = 0.0
    inference_ms: list[float] = field(default_factory=list)
    n_faces_total_tracks: int = 0
    n_plates_total_tracks: int = 0

    @property
    def real_time_factor(self) -> float:
        """Rapport temps de traitement / durée vidéo. < 1.0 = plus rapide que
        temps réel. POC validé 2026-05-27 : 0.66× sur T4 free Colab."""
        return self.process_s / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def avg_inference_ms(self) -> float:
        return float(np.mean(self.inference_ms)) if self.inference_ms else 0.0

    @property
    def p95_inference_ms(self) -> float:
        return float(np.percentile(self.inference_ms, 95)) if self.inference_ms else 0.0
