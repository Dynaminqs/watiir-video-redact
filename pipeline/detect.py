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


def spot_kind_from_class_id(class_id: int) -> str:
    """Mappe un index de classe YOLO sur un `spot_kind`.

    L'ordre DOIT matcher `training/data.yaml` (names: 0..3) et `SPOT_KINDS`.
    Lève ValueError si l'index est hors plage (modèle incohérent avec le contrat).
    """
    try:
        return SPOT_KINDS[class_id]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"class_id={class_id!r} hors du contrat SPOT_KINDS={SPOT_KINDS}") from exc


def xyxyn_to_bbox_dict(x1: float, y1: float, x2: float, y2: float) -> dict:
    """Convertit une box YOLO normalisée (coins xyxy ∈ [0,1]) en dict bbox
    `{x, y, w, h}` où (x, y) est le coin haut-gauche normalisé (convention
    alignée sur `StubDetector`). Les valeurs sont clampées dans [0, 1]."""
    cx1, cx2 = sorted((float(x1), float(x2)))
    cy1, cy2 = sorted((float(y1), float(y2)))
    cx1 = min(max(cx1, 0.0), 1.0)
    cy1 = min(max(cy1, 0.0), 1.0)
    cx2 = min(max(cx2, 0.0), 1.0)
    cy2 = min(max(cy2, 0.0), 1.0)
    return {"x": cx1, "y": cy1, "w": cx2 - cx1, "h": cy2 - cy1}


def obb_corners_to_bbox_dict(corners) -> dict:
    """Boîte orientée (OBB) → bbox axis-aligné `{x, y, w, h}` normalisé et clampé.

    `corners` = les 4 sommets normalisés de l'OBB, séquence de longueur 4 de
    (x, y) ∈ [0,1] (format ultralytics `obb.xyxyxyxyn`). On expose la boîte
    ENGLOBANTE axis-alignée pour conserver le contrat `RawDetection.bbox`
    (le géoréférencement V4.0 est GPS — la forme orientée n'est pas requise en
    aval ; l'OBB sert uniquement à un fit serré à l'entraînement)."""
    xs = [min(max(float(p[0]), 0.0), 1.0) for p in corners]
    ys = [min(max(float(p[1]), 0.0), 1.0) for p in corners]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


class Yolov8SpotDetector:
    """Détecteur de places fine-tuné (POC modèle, cf. `training/`).

    Modèle YOLOv8-**OBB** (boîtes orientées) : les places en vue oblique sont des
    quadrilatères en biais, qu'un rectangle axis-aligné couvre mal. Le détecteur
    expose néanmoins un `RawDetection.bbox` axis-aligné (boîte englobante de
    l'OBB) pour ne rien changer au contrat aval — l'OBB n'améliore que le fit à
    l'entraînement/inférence.

    Gaté derrière `SPOT_DETECTOR=yolov8` + un checkpoint explicite : tant qu'aucun
    `best.pt` validé n'existe, le worker garde `StubDetector` par défaut.

    Stratégie : tracking ByteTrack sur la vidéo floutée (comme le floutage), une
    seule `RawDetection` émise par `track_id` (la frame de meilleure confiance).
    Cela évite N détections quasi-dupliquées de la même place sur des frames
    successives — la modération humaine G affine ensuite.

    `ultralytics` et `cv2` sont importés paresseusement (le chemin `StubDetector`
    n'a aucune dépendance ML)."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        *,
        model_version: str | None = None,
        conf_threshold: float = 0.35,
        tracker: str = "bytetrack.yaml",
        frame_stride: int = 5,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"checkpoint Yolov8SpotDetector introuvable : {self.checkpoint_path}"
            )
        # model_version tracé jusqu'au staging (audit) — défaut dérivé du fichier.
        self.model_version = model_version or f"yolov8-{self.checkpoint_path.stem}"
        self.conf_threshold = conf_threshold
        self.tracker = tracker
        self.frame_stride = max(1, int(frame_stride))
        self._model = None  # chargé paresseusement au 1er detect()

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy : évite torch pour le chemin stub

            self._model = YOLO(str(self.checkpoint_path))
        return self._model

    def detect(self, redacted_video: Path) -> list[RawDetection]:
        import cv2  # lazy

        model = self._ensure_model()
        cap = cv2.VideoCapture(str(redacted_video))
        if not cap.isOpened():
            raise RuntimeError(f"impossible d'ouvrir la vidéo {redacted_video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Garde la meilleure détection par track_id : {tid: RawDetection}.
        best_by_track: dict[int, RawDetection] = {}
        # Détections sans track_id (tracker non résolu) gardées telles quelles.
        untracked: list[RawDetection] = []
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % self.frame_stride != 0:
                    frame_idx += 1
                    continue
                ts_ms = int(frame_idx / fps * 1000) if fps > 0 else frame_idx
                results = model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker,
                    conf=self.conf_threshold,
                    verbose=False,
                )
                self._collect(results, ts_ms, best_by_track, untracked)
                frame_idx += 1
        finally:
            cap.release()

        return list(best_by_track.values()) + untracked

    @staticmethod
    def _collect(
        results,
        ts_ms: int,
        best_by_track: dict[int, RawDetection],
        untracked: list[RawDetection],
    ) -> None:
        """Extrait les OBB d'un résultat ultralytics → RawDetection (in-place).

        Modèle OBB : on lit `results[0].obb` (4 coins normalisés) et on dérive le
        bbox englobant axis-aligné. Pour les détections trackées, ne garde que la
        meilleure confiance par track_id. Tolérant aux attributs absents."""
        res = results[0] if results else None
        obb = getattr(res, "obb", None) if res is not None else None
        if obb is None or len(obb) == 0:
            return
        # Coins normalisés : shape (N, 4, 2).
        corners = obb.xyxyxyxyn
        corners = corners.cpu().numpy() if hasattr(corners, "cpu") else corners
        cls = obb.cls.cpu().numpy() if hasattr(obb.cls, "cpu") else obb.cls
        conf = obb.conf.cpu().numpy() if hasattr(obb.conf, "cpu") else obb.conf
        ids = None
        if getattr(obb, "id", None) is not None:
            ids = obb.id.cpu().numpy() if hasattr(obb.id, "cpu") else obb.id

        for i in range(len(corners)):
            det = RawDetection(
                frame_ts_ms=ts_ms,
                bbox=obb_corners_to_bbox_dict(corners[i]),
                spot_kind=spot_kind_from_class_id(int(cls[i])),
                confidence=float(conf[i]),
            )
            if ids is not None:
                tid = int(ids[i])
                prev = best_by_track.get(tid)
                if prev is None or (det.confidence or 0) > (prev.confidence or 0):
                    best_by_track[tid] = det
            else:
                untracked.append(det)


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
