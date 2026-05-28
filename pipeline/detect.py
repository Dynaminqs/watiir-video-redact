"""WATIIR V4 — étage détection des places (sous-change F).

Détecte les places de stationnement sur la vidéo DÉJÀ floutée et les
géoréférence via le log GPS. Consommé par le worker après le claim
`claim_next_pending_detection`.

Le détecteur est PLUGGABLE (Protocol `SpotDetector`) :
  - `StubDetector` : sorties déterministes pour valider la plomberie E2E sans
    modèle ML (V4.0 par défaut).
  - `Yolov8SpotDetector` : gaté sur un POC modèle (dataset interne sur photos
    V3.c floutées). NON fourni ici — cf. README « POC détection de places ».

Géoréférencement (design D6) : approximation V4.0 assumée. Chaque détection est
positionnée à la position GPS interpolée du véhicule au timestamp de la frame.
La modération humaine (sous-change G) affine/rejette. Sans GPS exploitable →
`GpsRequiredError` (le worker notifie `failed_detection`).
"""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

# Aligné sur l'ENUM parking_spot_kind (migration A) + la validation Edge Function.
SPOT_KINDS = ("livraison", "pmr", "recharge", "standard")


class GpsRequiredError(RuntimeError):
    """Aucune source GPS exploitable → géoréférencement impossible.

    Le message est le code d'erreur snake_case consommé par
    notify-detection-complete (`error_message`)."""


@dataclass(frozen=True)
class GpsFix:
    """Un point GPS daté, relatif au début de la vidéo."""

    t_ms: int
    lng: float
    lat: float
    heading: float | None = None


@dataclass
class GpsTrack:
    """Trace GPS triée par timestamp croissant."""

    fixes: list[GpsFix] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.fixes = sorted(self.fixes, key=lambda f: f.t_ms)

    def is_empty(self) -> bool:
        return len(self.fixes) == 0

    def interpolate(self, t_ms: int) -> tuple[float, float, float | None]:
        """Position (lng, lat, heading) interpolée linéairement à `t_ms`.

        Clampée aux extrémités si `t_ms` est hors de la fenêtre couverte."""
        if self.is_empty():
            raise GpsRequiredError("gps_required_for_georeferencing")
        n = len(self.fixes)
        if n == 1 or t_ms <= self.fixes[0].t_ms:
            f = self.fixes[0]
            return (f.lng, f.lat, f.heading)
        if t_ms >= self.fixes[-1].t_ms:
            f = self.fixes[-1]
            return (f.lng, f.lat, f.heading)

        times = [f.t_ms for f in self.fixes]
        i = bisect_left(times, t_ms)
        # times[i-1] < t_ms <= times[i]
        a, b = self.fixes[i - 1], self.fixes[i]
        span = b.t_ms - a.t_ms
        ratio = 0.0 if span == 0 else (t_ms - a.t_ms) / span
        lng = a.lng + ratio * (b.lng - a.lng)
        lat = a.lat + ratio * (b.lat - a.lat)
        heading = a.heading if a.heading is not None else b.heading
        return (lng, lat, heading)


@dataclass(frozen=True)
class RawDetection:
    """Sortie brute du détecteur, en coordonnées frame (pas géoréférencée)."""

    frame_ts_ms: int
    bbox: dict
    spot_kind: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class DetectedSpot:
    """Place candidate géoréférencée, prête pour notify-detection-complete."""

    lng: float
    lat: float
    spot_kind: str | None
    spot_kind_confidence: float | None
    source_frame_ts_ms: int
    bbox: dict | None

    def to_payload(self) -> dict:
        """Sérialise au format attendu par l'Edge Function notify-detection-complete."""
        return {
            "lng": self.lng,
            "lat": self.lat,
            "spot_kind": self.spot_kind,
            "spot_kind_confidence": self.spot_kind_confidence,
            "source_frame_ts_ms": self.source_frame_ts_ms,
            "bbox": self.bbox,
        }


@runtime_checkable
class SpotDetector(Protocol):
    """Interface stable d'un détecteur de places. `model_version` est tracé
    jusqu'au staging pour audit."""

    model_version: str

    def detect(self, redacted_video: Path) -> list[RawDetection]: ...


class StubDetector:
    """Détecteur déterministe pour valider la plomberie sans modèle ML.

    Émet 2 détections fixes à des timestamps fixes — permet un smoke E2E
    reproductible (création de candidats staging) en attendant le modèle POC."""

    model_version = "stub-v0"

    def detect(self, redacted_video: Path) -> list[RawDetection]:
        return [
            RawDetection(
                frame_ts_ms=1000,
                bbox={"x": 0.40, "y": 0.50, "w": 0.10, "h": 0.20},
                spot_kind="standard",
                confidence=0.80,
            ),
            RawDetection(
                frame_ts_ms=3000,
                bbox={"x": 0.60, "y": 0.52, "w": 0.10, "h": 0.20},
                spot_kind="pmr",
                confidence=0.71,
            ),
        ]


def parse_gps_log(raw: bytes | str | None) -> GpsTrack:
    """Parseur GPS tolérant.

    V4.0 : format JSON `[{t_ms, lng, lat, heading?}, ...]`. GPX/NMEA sont une
    Open Question du design (le format dépend de la capture rail 1 / upload
    admin) — à brancher ici quand figé. Retourne une `GpsTrack` (peut être vide
    si `raw` est None/illisible)."""
    if raw is None:
        return GpsTrack(fixes=[])
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
    text = text.strip()
    if not text:
        return GpsTrack(fixes=[])
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return GpsTrack(fixes=[])
    if not isinstance(data, list):
        return GpsTrack(fixes=[])

    fixes: list[GpsFix] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            t_ms = int(item["t_ms"])
            lng = float(item["lng"])
            lat = float(item["lat"])
        except (KeyError, TypeError, ValueError):
            continue
        heading_raw = item.get("heading")
        heading = float(heading_raw) if isinstance(heading_raw, (int, float)) else None
        fixes.append(GpsFix(t_ms=t_ms, lng=lng, lat=lat, heading=heading))
    return GpsTrack(fixes=fixes)


def georeference(detections: list[RawDetection], gps_track: GpsTrack) -> list[DetectedSpot]:
    """Géoréférence chaque détection via interpolation GPS. Lève GpsRequiredError
    si la trace est vide."""
    if gps_track.is_empty():
        raise GpsRequiredError("gps_required_for_georeferencing")
    out: list[DetectedSpot] = []
    for d in detections:
        lng, lat, _heading = gps_track.interpolate(d.frame_ts_ms)
        out.append(
            DetectedSpot(
                lng=lng,
                lat=lat,
                spot_kind=d.spot_kind,
                spot_kind_confidence=d.confidence,
                source_frame_ts_ms=d.frame_ts_ms,
                bbox=d.bbox,
            )
        )
    return out


def detect_spots(
    redacted_video: Path,
    gps_track: GpsTrack,
    detector: SpotDetector,
) -> tuple[list[DetectedSpot], str]:
    """Orchestration : détecte puis géoréférence. Retourne (spots, model_version)."""
    raw = detector.detect(redacted_video)
    spots = georeference(raw, gps_track)
    return spots, detector.model_version
