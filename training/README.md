# POC — modèle de détection de places (TYPE, boîtes orientées OBB)

Socle du fine-tune qui remplacera le `StubDetector` (`stub-v0`) par un vrai
`Yolov8SpotDetector`. **Hors OpenSpec** (POC, comme le POC floutage qui a précédé
la spec E) : il informe l'intégration, on industrialise seulement après un signal
mAP positif.

## Scope V1 (verrouillé 2026-05-29)

- **DÉTECTION + TYPE en une passe** — 4 classes : `livraison`, `pmr`,
  `recharge`, `standard` (= `pipeline.detect.SPOT_KINDS`, = ENUM
  `parking_spot_kind` côté WATIIR). Pas d'état libre/occupé (reporté V4.1).
- **Risque accepté** : classes déséquilibrées (peu de PMR/recharge/livraison vs
  standard). Surveiller le **rappel par classe**, pas seulement le mAP50 global
  (flatté par `standard`). Suivi via le dashboard d'éval J (`/edge-ai-eval`).

## Données — annoter *from scratch*

Pas de dataset public exploitable : PKLot / CNRPark sont lot/aérien et ne
couvrent pas le **type FR** (marquages au sol spécifiques). On constitue le
corpus nous-mêmes :

1. **Filmer** quelques rues (mix de types, idéalement plusieurs heures/quartiers).
2. **Flouter AVANT d'annoter** via le pipeline livré (cohérence train/prod +
   RGPD) :
   ```bash
   python -m models.download                       # modèles floutage pinned
   python -m training.prepare_dataset \
       --videos training/raw_videos \
       --out   training/dataset/images/_to_annotate \
       --every-seconds 1.0
   ```
   Les marquages/panneaux restent intacts (seuls visages + plaques sont floutés)
   → le type reste annotable.
3. **Annoter** sous **CVAT** en **boîtes orientées (OBB)** — les places en vue
   oblique sont en biais (cf. `ANNOTATION_GUIDE.md` + `cvat_labels.json`). Export
   **« Ultralytics YOLO Oriented Bounding Boxes 1.0 »**, puis `split_dataset.py`
   répartit en :
   ```
   training/dataset/
     images/train/*.jpg   images/val/*.jpg
     labels/train/*.txt   labels/val/*.txt
   ```

## Fine-tune + go/no-go

```bash
python -m training.train --data training/data.yaml --epochs 80 --imgsz 640
```

- **Seuil go/no-go POC** : mAP50 ≥ 0.50 sur un petit batch → on industrialise
  l'annotation. Sinon on revoit le sourcing / l'équilibrage avant d'investir.
- Approche bornée : ne PAS sur-annoter avant d'avoir ce premier signal.

## Livraison du détecteur (après go)

1. Copier `best.pt`, noter son **SHA-256**.
2. Ajouter une entrée `spot_detector` à `models/manifest.json` + bump version.
3. Activer en prod :
   ```bash
   SPOT_DETECTOR=yolov8 \
   SPOT_DETECTOR_CHECKPOINT=models/spot_detector.pt \
   SPOT_DETECTOR_MODEL_VERSION=yolov8-spots-v1 \
   python -m worker.main
   ```
   Tant qu'aucun `best.pt` validé n'existe, **`stub` reste le défaut** (le worker
   refuse `yolov8` sans `SPOT_DETECTOR_CHECKPOINT`).

## Contrat d'intégration (stable — ne pas casser)

`Yolov8SpotDetector` (dans `pipeline/detect.py`) implémente le Protocol
`SpotDetector` : `model_version: str` + `detect(redacted_video) -> list[RawDetection]`.
Mapping classe → type : `class_id → SPOT_KINDS[class_id]` — **l'ordre des classes
dans `data.yaml` DOIT rester identique à `SPOT_KINDS`**. Le géoréférencement GPS,
la modération G et l'éval J sont en aval, inchangés.

## Licence

Toute amélioration du modèle est publiée **AGPL-3.0** (pas de fork privé) : le
détecteur est consommé dans ce repo AGPL. Le dataset brut/annoté n'est pas commité
(cf. `.gitignore`) ; seuls le code et `data.yaml` le sont.
