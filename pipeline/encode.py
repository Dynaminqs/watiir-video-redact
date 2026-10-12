"""Écriture de la vidéo floutée en H.264 via ffmpeg.

`cv2.VideoWriter` + fourcc `mp4v` (MPEG-4 Part 2, sans contrôle de débit)
gonflait la sortie : 248 Mo source → >500 Mo floutés, rejetés par le bucket
`vision-ai-redacted` (413). On pipe désormais les frames BGR brutes vers ffmpeg
(libx264, CRF), présent dans l'image Docker.

Fallback : sans binaire ffmpeg (poste dev Windows), on retombe sur `mp4v` avec
un warning — fonctionnel mais volumineux, réservé aux petits clips de test.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# CRF 23 = défaut x264, qualité visuelle proche de la source dashcam/smartphone
# pour un débit très inférieur à mp4v. `veryfast` : le CPU ne doit pas devenir
# le goulot quand l'inférence tourne sur GPU.
DEFAULT_CRF = 23
DEFAULT_PRESET = "veryfast"


def resolve_ffmpeg() -> str | None:
    """Binaire ffmpeg : `FFMPEG_BIN` si défini, sinon celui du PATH."""
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")


def build_ffmpeg_command(
    ffmpeg: str,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
) -> list[str]:
    """Commande ffmpeg lisant des frames bgr24 sur stdin → MP4 H.264."""
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        # Pas de piste audio : OpenCV ne la lit pas et elle pourrait contenir
        # des voix (donnée personnelle non floutée).
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        # yuv420p exige des dimensions paires → pad d'1 px si besoin.
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        # moov atom en tête : lecture progressive dans la console admin.
        "-movflags",
        "+faststart",
        output_path,
    ]


class FfmpegWriter:
    """Interface minimale compatible `cv2.VideoWriter` (write / release)."""

    def __init__(self, output_path: str, fps: float, width: int, height: int, ffmpeg: str):
        self._shape = (height, width, 3)
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            build_ffmpeg_command(ffmpeg, output_path, fps, width, height),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
        )

    def isOpened(self) -> bool:  # miroir de cv2.VideoWriter
        return self._proc.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self._shape:
            raise ValueError(f"frame {frame.shape} ≠ attendu {self._shape}")
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f"ffmpeg s'est arrêté : {self._read_stderr()}") from exc

    def release(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        code = self._proc.wait()
        err = self._read_stderr()
        self._stderr.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg exit {code} : {err}")

    def _read_stderr(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace").strip()[-2000:]


def open_writer(output_path: str, fps: float, width: int, height: int):
    """Writer H.264 via ffmpeg, ou fallback `mp4v` si ffmpeg est introuvable."""
    ffmpeg = resolve_ffmpeg()
    if ffmpeg:
        return FfmpegWriter(output_path, fps, width, height, ffmpeg)
    logger.warning("ffmpeg introuvable — fallback OpenCV mp4v (sortie volumineuse)")
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
