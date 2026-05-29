"""Tests Yolov8SpotDetector — helpers purs + logique de collecte, SANS modèle.

On ne charge jamais ultralytics/torch ici : on teste le mapping de classes, la
conversion de bbox, le garde-fou checkpoint, et la dédup par track_id via un
faux objet `results` mimant l'API ultralytics.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.detect import (
    RawDetection,
    Yolov8SpotDetector,
    obb_corners_to_bbox_dict,
    spot_kind_from_class_id,
    xyxyn_to_bbox_dict,
)

# ─── spot_kind_from_class_id ────────────────────────────────────────────────


def test_class_mapping_matches_contract():
    assert spot_kind_from_class_id(0) == "livraison"
    assert spot_kind_from_class_id(1) == "pmr"
    assert spot_kind_from_class_id(2) == "recharge"
    assert spot_kind_from_class_id(3) == "standard"


def test_class_mapping_out_of_range_raises():
    with pytest.raises(ValueError):
        spot_kind_from_class_id(4)
    with pytest.raises(ValueError):
        spot_kind_from_class_id(-99)


# ─── xyxyn_to_bbox_dict ─────────────────────────────────────────────────────


def test_bbox_conversion_topleft_wh():
    box = xyxyn_to_bbox_dict(0.2, 0.3, 0.5, 0.7)
    assert box == {"x": 0.2, "y": 0.3, "w": pytest.approx(0.3), "h": pytest.approx(0.4)}


def test_bbox_conversion_sorts_and_clamps():
    # coins inversés + hors [0,1] → triés puis clampés
    box = xyxyn_to_bbox_dict(1.4, 0.9, -0.2, 0.1)
    assert box["x"] == 0.0
    assert box["y"] == pytest.approx(0.1)
    assert box["w"] == pytest.approx(1.0)
    assert box["h"] == pytest.approx(0.8)


# ─── garde-fou checkpoint ───────────────────────────────────────────────────


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Yolov8SpotDetector(tmp_path / "nope.pt")


def test_model_version_default_from_stem(tmp_path):
    ckpt = tmp_path / "spots_v1.pt"
    ckpt.write_bytes(b"fake")
    det = Yolov8SpotDetector(ckpt)
    assert det.model_version == "yolov8-spots_v1"
    det2 = Yolov8SpotDetector(ckpt, model_version="custom-v2")
    assert det2.model_version == "custom-v2"


# ─── obb_corners_to_bbox_dict ───────────────────────────────────────────────


def test_obb_corners_bounding_box():
    # OBB tournée → bbox englobant axis-aligné (min/max des coins).
    corners = [[0.2, 0.3], [0.6, 0.35], [0.55, 0.7], [0.15, 0.65]]
    box = obb_corners_to_bbox_dict(corners)
    assert box["x"] == pytest.approx(0.15)
    assert box["y"] == pytest.approx(0.30)
    assert box["w"] == pytest.approx(0.45)  # 0.60 - 0.15
    assert box["h"] == pytest.approx(0.40)  # 0.70 - 0.30


def test_obb_corners_clamps():
    corners = [[-0.1, 0.0], [1.2, 0.0], [1.2, 1.3], [-0.1, 1.3]]
    assert obb_corners_to_bbox_dict(corners) == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}


# ─── _collect : OBB + dédup par track_id ────────────────────────────────────


def _square(x0, y0, x1, y1):
    """4 coins (CCW) d'une boîte → format obb.xyxyxyxyn d'une détection."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class _FakeObb:
    def __init__(self, corners, cls, conf, ids):
        self.xyxyxyxyn = np.asarray(corners, dtype=float)
        self.cls = np.asarray(cls, dtype=float)
        self.conf = np.asarray(conf, dtype=float)
        self.id = None if ids is None else np.asarray(ids, dtype=float)

    def __len__(self):
        return len(self.xyxyxyxyn)


def _results(obb):
    return [type("R", (), {"obb": obb})()]


def test_collect_keeps_best_confidence_per_track():
    best: dict[int, RawDetection] = {}
    untracked: list[RawDetection] = []
    # track 7 vu 2x : conf 0.4 puis 0.9 → garde 0.9
    Yolov8SpotDetector._collect(
        _results(_FakeObb([_square(0, 0, 0.1, 0.1)], [3], [0.4], [7])), 1000, best, untracked
    )
    Yolov8SpotDetector._collect(
        _results(_FakeObb([_square(0, 0, 0.2, 0.2)], [3], [0.9], [7])), 2000, best, untracked
    )
    assert len(best) == 1
    assert best[7].confidence == pytest.approx(0.9)
    assert best[7].frame_ts_ms == 2000
    assert best[7].spot_kind == "standard"
    assert untracked == []


def test_collect_untracked_when_no_ids():
    best: dict[int, RawDetection] = {}
    untracked: list[RawDetection] = []
    Yolov8SpotDetector._collect(
        _results(_FakeObb([_square(0, 0, 0.1, 0.1), _square(0.5, 0.5, 0.6, 0.6)],
                          [1, 2], [0.7, 0.8], None)),
        500,
        best,
        untracked,
    )
    assert best == {}
    assert len(untracked) == 2
    assert {d.spot_kind for d in untracked} == {"pmr", "recharge"}


def test_collect_tolerates_empty():
    best: dict[int, RawDetection] = {}
    untracked: list[RawDetection] = []
    Yolov8SpotDetector._collect([], 0, best, untracked)
    Yolov8SpotDetector._collect(_results(_FakeObb([], [], [], None)), 0, best, untracked)
    assert best == {} and untracked == []
