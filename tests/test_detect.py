"""Tests unitaires pipeline.detect (sous-change F) — pas de réseau, pas de modèle.

Couvre : StubDetector déterministe, parsing GPS tolérant, interpolation GPS,
géoréférencement (succès + GpsRequiredError), propagation model_version.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.detect import (
    DetectedSpot,
    GpsFix,
    GpsRequiredError,
    GpsTrack,
    RawDetection,
    StubDetector,
    detect_spots,
    georeference,
    parse_gps_log,
)

# ─── StubDetector ───────────────────────────────────────────────────────────


def test_stub_detector_deterministic():
    det = StubDetector()
    a = det.detect(Path("whatever.mp4"))
    b = det.detect(Path("whatever.mp4"))
    assert a == b
    assert len(a) == 2
    assert a[0].spot_kind == "standard"
    assert a[1].spot_kind == "pmr"
    assert det.model_version == "stub-v0"


# ─── parse_gps_log ──────────────────────────────────────────────────────────


def test_parse_gps_log_json_valid():
    raw = '[{"t_ms":0,"lng":2.34,"lat":48.85},{"t_ms":2000,"lng":2.35,"lat":48.86,"heading":90}]'
    track = parse_gps_log(raw)
    assert len(track.fixes) == 2
    assert track.fixes[0].lng == 2.34
    assert track.fixes[1].heading == 90


def test_parse_gps_log_none_and_empty_give_empty_track():
    assert parse_gps_log(None).is_empty()
    assert parse_gps_log("").is_empty()
    assert parse_gps_log("   ").is_empty()


def test_parse_gps_log_invalid_json_gives_empty_track():
    assert parse_gps_log("not json").is_empty()
    assert parse_gps_log('{"not":"a list"}').is_empty()


def test_parse_gps_log_skips_malformed_fixes():
    raw = '[{"t_ms":0,"lng":2.3,"lat":48.8},{"bad":"row"},{"t_ms":1000,"lng":2.4}]'
    track = parse_gps_log(raw)
    assert len(track.fixes) == 1  # seul le premix valide est gardé


def test_parse_gps_log_bytes():
    track = parse_gps_log(b'[{"t_ms":0,"lng":2.3,"lat":48.8}]')
    assert len(track.fixes) == 1


# ─── GpsTrack.interpolate ───────────────────────────────────────────────────


def test_interpolate_midpoint():
    track = GpsTrack(fixes=[GpsFix(0, 0.0, 0.0), GpsFix(1000, 10.0, 20.0)])
    lng, lat, _ = track.interpolate(500)
    assert lng == pytest.approx(5.0)
    assert lat == pytest.approx(10.0)


def test_interpolate_clamps_before_first_and_after_last():
    track = GpsTrack(fixes=[GpsFix(1000, 2.0, 48.0), GpsFix(2000, 3.0, 49.0)])
    assert track.interpolate(0)[:2] == (2.0, 48.0)
    assert track.interpolate(9999)[:2] == (3.0, 49.0)


def test_interpolate_single_fix():
    track = GpsTrack(fixes=[GpsFix(500, 1.0, 2.0)])
    assert track.interpolate(123)[:2] == (1.0, 2.0)


def test_interpolate_empty_raises():
    with pytest.raises(GpsRequiredError):
        GpsTrack(fixes=[]).interpolate(0)


def test_gpstrack_sorts_fixes():
    track = GpsTrack(fixes=[GpsFix(2000, 3.0, 49.0), GpsFix(0, 2.0, 48.0)])
    assert [f.t_ms for f in track.fixes] == [0, 2000]


# ─── georeference / detect_spots ────────────────────────────────────────────


def test_georeference_positions_at_interpolated_gps():
    track = GpsTrack(fixes=[GpsFix(0, 2.0, 48.0), GpsFix(2000, 2.2, 48.2)])
    raw = [RawDetection(frame_ts_ms=1000, bbox={"x": 0.5}, spot_kind="pmr", confidence=0.9)]
    spots = georeference(raw, track)
    assert len(spots) == 1
    assert spots[0].lng == pytest.approx(2.1)
    assert spots[0].lat == pytest.approx(48.1)
    assert spots[0].spot_kind == "pmr"
    assert spots[0].spot_kind_confidence == 0.9
    assert spots[0].source_frame_ts_ms == 1000


def test_georeference_without_gps_raises():
    raw = [RawDetection(frame_ts_ms=1000, bbox={})]
    with pytest.raises(GpsRequiredError, match="gps_required_for_georeferencing"):
        georeference(raw, GpsTrack(fixes=[]))


def test_detect_spots_end_to_end_with_stub():
    track = GpsTrack(fixes=[GpsFix(0, 2.0, 48.0), GpsFix(5000, 2.5, 48.5)])
    spots, model_version = detect_spots(Path("x.mp4"), track, StubDetector())
    assert model_version == "stub-v0"
    assert len(spots) == 2
    # premier stub à 1000ms → interpolé entre les 2 fixes
    assert all(isinstance(s, DetectedSpot) for s in spots)
    assert spots[0].source_frame_ts_ms == 1000


def test_detected_spot_to_payload():
    s = DetectedSpot(
        lng=2.1,
        lat=48.1,
        spot_kind="livraison",
        spot_kind_confidence=0.6,
        source_frame_ts_ms=1200,
        bbox={"x": 0.4},
    )
    p = s.to_payload()
    assert p == {
        "lng": 2.1,
        "lat": 48.1,
        "spot_kind": "livraison",
        "spot_kind_confidence": 0.6,
        "source_frame_ts_ms": 1200,
        "bbox": {"x": 0.4},
    }
