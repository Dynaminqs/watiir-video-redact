"""WATIIR V4 — préparation du dataset POC détection de places.

Extrait des frames échantillonnées des vidéos rue, les **floute** (visages +
plaques) via le pipeline AGPL déjà livré, et écrit des images JPEG prêtes à
être annotées (Label Studio / CVAT / Roboflow).

Pourquoi flouter AVANT d'annoter :
  - Cohérence train/prod : le détecteur de places tournera en prod sur des
    vidéos DÉJÀ floutées (étage F après l'étage E). Il doit s'entraîner sur la
    même distribution visuelle.
  - RGPD : aucune frame non floutée ne quitte le pipeline. Les marquages au sol
    et panneaux (PMR / recharge / livraison) restent intacts → le TYPE reste
    annotable sur image floutée.

Usage :
    python -m training.prepare_dataset \
        --videos training/raw_videos \
        --out training/dataset/images/_to_annotate \
        --every-seconds 1.0

Les modèles de floutage (face + plaque) sont ceux pinned dans models/ :
    --face-model models/model.pt  --plate-model models/best.pt
(téléchargeables via `python -m models.download`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from pipeline.redact import blur_boxes, load_models

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _extract_boxes(results):
    """(N, 4) xyxy depuis un résultat ultralytics, ou array vide."""
    import numpy as np

    if not results or results[0].boxes is None:
        return np.empty((0, 4))
    xyxy = results[0].boxes.xyxy
    return xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)


def prepare_video(
    video_path: Path,
    out_dir: Path,
    models,
    every_seconds: float,
    conf_threshold: float,
    jpeg_quality: int,
) -> int:
    """Échantillonne + floute une vidéo. Retourne le nombre de frames écrites."""
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"impossible d'ouvrir la vidéo {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps * every_seconds))
    stem = video_path.stem

    written = 0
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                face_boxes = _extract_boxes(
                    models.face.predict(frame, conf=conf_threshold, verbose=False)
                )
                plate_boxes = (
                    _extract_boxes(models.plate.predict(frame, conf=conf_threshold, verbose=False))
                    if models.plate is not None
                    else np.empty((0, 4))
                )
                all_boxes = np.vstack([face_boxes, plate_boxes]) if len(plate_boxes) else face_boxes
                blurred = blur_boxes(frame, all_boxes) if len(all_boxes) else frame
                out_path = out_dir / f"{stem}_f{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), blurred, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                written += 1
            frame_idx += 1
    finally:
        cap.release()
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", required=True, type=Path, help="dossier de vidéos rue brutes")
    ap.add_argument("--out", required=True, type=Path, help="dossier de sortie des frames floutées")
    ap.add_argument("--face-model", type=Path, default=Path("models/model.pt"))
    ap.add_argument("--plate-model", type=Path, default=Path("models/best.pt"))
    ap.add_argument("--every-seconds", type=float, default=1.0, help="échantillonnage temporel")
    ap.add_argument("--conf", type=float, default=0.25, help="seuil de confiance floutage")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    if not args.videos.is_dir():
        raise SystemExit(f"--videos n'est pas un dossier : {args.videos}")
    args.out.mkdir(parents=True, exist_ok=True)

    plate_model = args.plate_model if args.plate_model and args.plate_model.exists() else None
    if plate_model is None:
        print("⚠️  modèle plaque absent → floutage face-only (dégradé, OK pour la prep POC)")
    models = load_models(args.face_model, plate_model)

    videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"aucune vidéo trouvée dans {args.videos} (exts: {sorted(VIDEO_EXTS)})")

    total = 0
    for v in videos:
        n = prepare_video(v, args.out, models, args.every_seconds, args.conf, args.jpeg_quality)
        print(f"  {v.name} → {n} frames floutées")
        total += n
    print(f"✅ {total} frames écrites dans {args.out}")
    print("Prochaine étape : annoter (4 classes : livraison/pmr/recharge/standard),")
    print("exporter au format YOLO, répartir en images/train|val + labels/train|val.")


if __name__ == "__main__":
    main()
