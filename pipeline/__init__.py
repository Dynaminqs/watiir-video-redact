"""WATIIR V4 — pipeline AGPL de floutage visages + plaques (port du POC).

Exports publics consommés par le worker :

    from pipeline import blur_boxes, process_video, VideoStats
"""

from pipeline.detect import (
    DetectedSpot,
    GpsFix,
    GpsRequiredError,
    GpsTrack,
    RawDetection,
    SpotDetector,
    StubDetector,
    detect_spots,
    georeference,
    parse_gps_log,
)
from pipeline.redact import RedactionModels, blur_boxes, load_models, process_video
from pipeline.stats import FrameStats, VideoStats

__all__ = [
    "DetectedSpot",
    "FrameStats",
    "GpsFix",
    "GpsRequiredError",
    "GpsTrack",
    "RawDetection",
    "RedactionModels",
    "SpotDetector",
    "StubDetector",
    "VideoStats",
    "blur_boxes",
    "detect_spots",
    "georeference",
    "load_models",
    "parse_gps_log",
    "process_video",
]
