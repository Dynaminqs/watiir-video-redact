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
    Yolov8SpotDetector,
    detect_spots,
    georeference,
    obb_corners_to_bbox_dict,
    parse_gps_log,
    spot_kind_from_class_id,
    xyxyn_to_bbox_dict,
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
    "Yolov8SpotDetector",
    "blur_boxes",
    "detect_spots",
    "georeference",
    "load_models",
    "obb_corners_to_bbox_dict",
    "parse_gps_log",
    "process_video",
    "spot_kind_from_class_id",
    "xyxyn_to_bbox_dict",
]
