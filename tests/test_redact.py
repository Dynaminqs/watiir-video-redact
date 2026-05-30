"""Tests unitaires pipeline.redact — couvre la logique de blur SANS YOLO.

Les tests qui chargeraient un vrai modèle YOLO sont marqués `e2e` et skip par
défaut (cf. conftest).
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.redact import blur_boxes

# ─── blur_boxes ─────────────────────────────────────────────────────────────


def test_blur_boxes_returns_copy_not_mutating_input():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 50, 50]], dtype=float)
    out = blur_boxes(frame, boxes)
    assert out is not frame
    # Frame original intact
    assert np.array_equal(frame, np.zeros((100, 100, 3), dtype=np.uint8))


def test_blur_boxes_empty_returns_unchanged_copy():
    frame = np.full((20, 20, 3), 128, dtype=np.uint8)
    out = blur_boxes(frame, np.empty((0, 4)))
    assert np.array_equal(out, frame)


def test_blur_boxes_modifies_pixels_inside_box():
    """Le blur appliqué à une zone non-uniforme doit changer les pixels."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Damier au milieu (sera modifié par le blur)
    frame[25:75:2, 25:75:2] = 255
    boxes = np.array([[20, 20, 80, 80]], dtype=float)
    out = blur_boxes(frame, boxes)
    assert not np.array_equal(out[20:80, 20:80], frame[20:80, 20:80])
    # En dehors de la box, intact
    assert np.array_equal(out[:20, :], frame[:20, :])
    assert np.array_equal(out[80:, :], frame[80:, :])


def test_blur_boxes_clamps_out_of_bounds_coordinates():
    frame = np.ones((50, 50, 3), dtype=np.uint8) * 128
    boxes = np.array([[-10, -10, 1000, 1000]], dtype=float)
    # Ne doit pas crasher, doit blur l'intégralité
    out = blur_boxes(frame, boxes)
    assert out.shape == frame.shape


def test_blur_boxes_skips_zero_area_boxes():
    frame = np.full((50, 50, 3), 200, dtype=np.uint8)
    # x2 <= x1 → skip silencieux
    boxes = np.array([[20, 20, 20, 30], [10, 10, 5, 5]], dtype=float)
    out = blur_boxes(frame, boxes)
    assert np.array_equal(out, frame)


def test_blur_boxes_multiple_boxes_independent():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:30:2, 10:30:2] = 255  # zone 1 non-uniforme
    frame[60:80:2, 60:80:2] = 255  # zone 2 non-uniforme
    boxes = np.array([[5, 5, 35, 35], [55, 55, 85, 85]], dtype=float)
    out = blur_boxes(frame, boxes)
    assert not np.array_equal(out[5:35, 5:35], frame[5:35, 5:35])
    assert not np.array_equal(out[55:85, 55:85], frame[55:85, 55:85])
    # Zone centrale 40-50 intacte
    assert np.array_equal(out[40:50, 40:50], frame[40:50, 40:50])


# ─── stats helpers ──────────────────────────────────────────────────────────


def test_video_stats_real_time_factor_zero_duration_returns_zero():
    from pipeline.stats import VideoStats

    stats = VideoStats(video_path="x", n_frames=0, duration_s=0.0, process_s=5.0)
    assert stats.real_time_factor == 0.0


def test_video_stats_real_time_factor_basic():
    from pipeline.stats import VideoStats

    stats = VideoStats(video_path="x", n_frames=100, duration_s=10.0, process_s=6.6)
    assert stats.real_time_factor == pytest.approx(0.66)


def test_video_stats_inference_avg_p95():
    from pipeline.stats import VideoStats

    stats = VideoStats(video_path="x", n_frames=3, duration_s=1.0)
    stats.inference_ms = [10.0, 20.0, 30.0]
    assert stats.avg_inference_ms == pytest.approx(20.0)
    assert stats.p95_inference_ms == pytest.approx(29.0, rel=0.02)


def test_video_stats_inference_empty_returns_zero():
    from pipeline.stats import VideoStats

    stats = VideoStats(video_path="x", n_frames=0, duration_s=0.0)
    assert stats.avg_inference_ms == 0.0
    assert stats.p95_inference_ms == 0.0
