"""WATIIR V4 — split train/val du dataset annoté (export CVAT YOLO).

Prend les images + labels YOLO exportés depuis CVAT et les répartit en
`training/dataset/images/{train,val}` + `training/dataset/labels/{train,val}`,
structure attendue par `training/data.yaml` et `train.py`.

Comportement (POC) :
  - Seules les images AYANT un fichier label `.txt` correspondant sont incluses
    (= images réellement annotées). Les images non annotées sont ignorées :
    en POC l'absence de label ne signifie pas « aucune place » mais « pas encore
    fait » — les inclure comme background introduirait des faux négatifs.
  - Un label `.txt` présent mais VIDE est gardé (= image background vérifiée
    sans place), utile pour réduire les faux positifs.
  - Split reproductible (tri par nom + seed fixe).

Usage (depuis la racine du repo) :
    python -m training.split_dataset \
        --images training/cvat_export/images \
        --labels training/cvat_export/labels \
        --val-ratio 0.2

CVAT : exporter au format « Ultralytics YOLO Oriented Bounding Boxes 1.0 » (OBB).
Labels OBB = `class x1 y1 x2 y2 x3 y3 x4 y4` ; l'histogramme par classe ci-dessous
lit le 1er token (class id), donc fonctionne aussi bien en OBB qu'en détection.
Selon le format, images et labels peuvent être dans des sous-dossiers
(ex. obj_train_data/) — pointe --images / --labels dessus. L'ordre des classes
dans CVAT DOIT rester livraison/pmr/recharge/standard (cf. cvat_labels.json) pour
matcher data.yaml (0..3).
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ("livraison", "pmr", "recharge", "standard")


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, type=Path, help="dossier des images exportées")
    ap.add_argument("--labels", required=True, type=Path, help="dossier des labels YOLO (.txt)")
    ap.add_argument("--out", type=Path, default=Path("training/dataset"))
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="déplacer au lieu de copier")
    args = ap.parse_args()

    if not args.images.is_dir() or not args.labels.is_dir():
        raise SystemExit("--images et --labels doivent être des dossiers existants")

    # Paires (image, label) pour chaque label .txt ayant une image correspondante.
    pairs: list[tuple[Path, Path]] = []
    orphan_labels = 0
    for lbl in sorted(args.labels.glob("*.txt")):
        if lbl.stem in {"classes", "obj"}:  # méta CVAT, pas un label d'image
            continue
        img = find_image(args.images, lbl.stem)
        if img is None:
            orphan_labels += 1
            continue
        pairs.append((img, lbl))

    if not pairs:
        raise SystemExit("aucune paire image+label trouvée — vérifie --images / --labels")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = max(1, round(len(pairs) * args.val_ratio)) if len(pairs) > 4 else 0
    val_set = set(range(n_val))

    counts = {"train": 0, "val": 0}
    class_hist = {c: 0 for c in CLASS_NAMES}
    empty_labels = 0

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    op = shutil.move if args.move else shutil.copy2
    for i, (img, lbl) in enumerate(pairs):
        split = "val" if i in val_set else "train"
        op(str(img), str(args.out / "images" / split / img.name))
        op(str(lbl), str(args.out / "labels" / split / lbl.name))
        counts[split] += 1
        # Histogramme classes (lecture du label).
        lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()] if lbl.exists() else []
        if not lines:
            empty_labels += 1
        for ln in lines:
            try:
                cid = int(ln.split()[0])
                if 0 <= cid < len(CLASS_NAMES):
                    class_hist[CLASS_NAMES[cid]] += 1
            except (ValueError, IndexError):
                pass

    print(f"[OK] split : {counts['train']} train / {counts['val']} val "
          f"(val-ratio={args.val_ratio}, seed={args.seed})")
    if orphan_labels:
        print(f"[warn] {orphan_labels} label(s) sans image correspondante - ignores")
    if empty_labels:
        print(f"[info] {empty_labels} image(s) background (label vide) incluses")
    print("Instances par classe :")
    for c in CLASS_NAMES:
        print(f"  {c:>10} : {class_hist[c]}")
    rares = [c for c in ("livraison", "pmr", "recharge") if class_hist[c] < 20]
    if rares:
        print(f"[warn] classes rares sous-representees (<20 instances) : {', '.join(rares)} "
              "- surveiller le rappel par classe au train (cf. dashboard eval J).")
    print("\nPrêt pour : python -m training.train --data training/data.yaml")


if __name__ == "__main__":
    main()
