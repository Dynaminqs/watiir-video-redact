"""WATIIR V4 — fine-tune YOLOv8-nano OBB pour la détection de places (POC).

Modèle **OBB** (boîtes orientées) : les places en vue oblique sont des
quadrilatères en biais → l'OBB épouse leur orientation (un rectangle axis-aligné
couvre mal). Le détecteur d'inférence expose ensuite le bbox englobant
axis-aligné (contrat aval inchangé, cf. `pipeline/detect.py`).

Entraîne un YOLOv8n-OBB sur le dataset annoté (4 classes TYPE) et imprime les
métriques mAP pour la décision go/no-go. Pensé pour Colab GPU ou une machine
locale avec GPU ; tourne aussi sur CPU (lent).

Usage :
    python -m training.train --data training/data.yaml --epochs 80 --imgsz 640

Sortie : runs/detect/<name>/weights/best.pt (+ best.onnx si --export).
Pour livrer le détecteur :
  1. Copier best.pt et noter son SHA-256.
  2. L'ajouter à models/manifest.json (nouvelle entrée spot_detector + bump version).
  3. Lancer le worker avec SPOT_DETECTOR=yolov8 SPOT_DETECTOR_CHECKPOINT=<path>.

Critères go/no-go POC (indicatifs, à ajuster avec le dashboard d'éval J) :
  - mAP50 global ≥ 0.50 sur un petit batch → signal exploitable, on industrialise.
  - Surveiller le rappel par classe : PMR/recharge/livraison seront sous-
    représentées (classes déséquilibrées, risque accepté). Un mAP50 global flatté
    par la classe `standard` masque un mauvais rappel sur les types rares.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Doit matcher data.yaml / SPOT_KINDS.
EXPECTED_CLASSES = ("livraison", "pmr", "recharge", "standard")
GO_NOGO_MAP50 = 0.50


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("training/data.yaml"))
    ap.add_argument("--base-model", default="yolov8n-obb.pt", help="poids de départ (nano OBB)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="watiir-spots-poc")
    ap.add_argument("--device", default=None, help="'0' (GPU), 'cpu', ou None (auto)")
    ap.add_argument("--export", action="store_true", help="exporter aussi en ONNX")
    args = ap.parse_args()

    import yaml
    from ultralytics import YOLO  # lazy : lourd (torch)

    # Ultralytics résout un `path` RELATIF de data.yaml via son datasets_dir
    # global (défaut C:\...\datasets), pas via le cwd ni le dossier du yaml. Pour
    # rester portable (data.yaml versionné garde `path: ./dataset`), on génère à
    # la volée un yaml résolu avec un `path` ABSOLU, utilisé tel quel.
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    base = args.data.resolve().parent
    cfg["path"] = str((base / str(cfg.get("path", "."))).resolve())
    resolved = base / "_data.resolved.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    data_arg = str(resolved)

    model = YOLO(args.base_model)
    results = model.train(
        data=data_arg,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        patience=20,  # early stopping
        seed=42,  # reproductibilité POC
    )

    # Résumé métriques + verdict go/no-go.
    try:
        metrics = model.val(data=data_arg, imgsz=args.imgsz, device=args.device)
        map50 = float(metrics.box.map50)
        map5095 = float(metrics.box.map)
        print("\n-------- Resultats POC --------")
        print(f"mAP50      : {map50:.3f}")
        print(f"mAP50-95   : {map5095:.3f}")
        # Rappel par classe si disponible.
        names = getattr(metrics, "names", None) or {}
        per_class = getattr(metrics.box, "maps", None)
        if per_class is not None:
            for idx, ap_c in enumerate(per_class):
                print(f"  {names.get(idx, idx):>10} : mAP50-95 {float(ap_c):.3f}")
        verdict = "GO" if map50 >= GO_NOGO_MAP50 else "NO-GO"
        print(f"\nGo/no-go (seuil mAP50 >= {GO_NOGO_MAP50}) : {verdict}")
    except Exception as exc:  # le resume ne doit jamais casser le run
        print(f"[warn] resume metriques indisponible : {exc}")

    best = Path(results.save_dir) / "weights" / "best.pt" if hasattr(results, "save_dir") else None
    if best:
        print(f"\nbest.pt : {best}")
    if args.export and best and best.exists():
        YOLO(str(best)).export(format="onnx", imgsz=args.imgsz)


if __name__ == "__main__":
    main()
