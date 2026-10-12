"""Tests pipeline.encode — commande ffmpeg + aller-retour d'encodage réel."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipeline.encode import FfmpegWriter, build_ffmpeg_command, open_writer, resolve_ffmpeg


def test_build_ffmpeg_command_h264_crf_no_audio():
    cmd = build_ffmpeg_command("ffmpeg", "out.mp4", 29.97, 1920, 1080)
    assert cmd[0] == "ffmpeg"
    assert cmd[-1] == "out.mp4"
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-crf") + 1] == "23"
    assert cmd[cmd.index("-s") + 1] == "1920x1080"
    assert cmd[cmd.index("-pix_fmt") + 1] == "bgr24"
    assert "-an" in cmd
    assert "+faststart" in cmd


def test_open_writer_falls_back_to_opencv_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.encode.resolve_ffmpeg", lambda: None)
    writer = open_writer(str(tmp_path / "out.mp4"), 10.0, 64, 48)
    try:
        assert isinstance(writer, cv2.VideoWriter)
    finally:
        writer.release()


@pytest.mark.skipif(resolve_ffmpeg() is None, reason="ffmpeg absent")
def test_ffmpeg_writer_roundtrip_h264_odd_dimensions(tmp_path):
    """Dimensions impaires (pad) + relecture OpenCV : codec h264, frames intactes."""
    out = tmp_path / "out.mp4"
    width, height, n = 101, 75, 20
    writer = FfmpegWriter(str(out), 10.0, width, height, resolve_ffmpeg())
    for i in range(n):
        frame = np.full((height, width, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = fourcc.to_bytes(4, "little").decode(errors="replace").lower()
        assert codec in {"avc1", "h264"}
        frames = 0
        while cap.read()[0]:
            frames += 1
        assert frames == n
    finally:
        cap.release()


@pytest.mark.skipif(resolve_ffmpeg() is None, reason="ffmpeg absent")
def test_ffmpeg_writer_rejects_wrong_frame_shape(tmp_path):
    writer = FfmpegWriter(str(tmp_path / "out.mp4"), 10.0, 64, 48, resolve_ffmpeg())
    with pytest.raises(ValueError):
        writer.write(np.zeros((10, 10, 3), dtype=np.uint8))
    writer.release()
