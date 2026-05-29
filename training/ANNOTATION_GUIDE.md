# Guide d'annotation — places de stationnement (CVAT)

Objectif : annoter les **places de stationnement** avec leur **TYPE** sur les
frames floutées de `training/dataset/images/_to_annotate/` (506 frames).

## Les 4 classes (ordre verrouillé)

L'ordre **DOIT** rester celui-ci (= `pipeline.detect.SPOT_KINDS`, indices 0→3) :

| id | classe | comment l'identifier |
|----|--------|----------------------|
| 0 | `livraison` | marquage/panneau **livraison** (zone jaune, « LIVRAISON ») |
| 1 | `pmr` | pictogramme **fauteuil roulant** bleu au sol / panneau PMR |
| 2 | `recharge` | **borne de recharge** + marquage vert / picto véhicule électrique |
| 3 | `standard` | place matérialisée **sans** marquage spécial (le cas courant) |

## Forme : boîte ORIENTÉE (OBB)

On annote en **boîtes orientées** (rectangles tournants), pas en rectangles
droits : les places en vue oblique sont en biais. Dans CVAT, trace un rectangle
puis **fais-le pivoter** (poignée de rotation en haut de la boîte) pour aligner
ses côtés sur les lignes de la place. Export attendu : **« Ultralytics YOLO
Oriented Bounding Boxes 1.0 »**.

## Quoi tracer

- **Une boîte (orientée) par PLACE** (l'emplacement matérialisé), **pas** par voiture.
  - Place **vide** → boîte sur le rectangle au sol.
  - Place **occupée** → boîte sur l'emprise de la place (les lignes restent le
    repère ; englober la voiture qui la remplit est OK si les lignes sont
    masquées).
- **Type décidé par le marquage/panneau visible** :
  - picto fauteuil bleu → `pmr` · borne + vert → `recharge` · jaune/livraison →
    `livraison` · sinon → `standard`.
- **Marquage non visible / ambigu** (place cachée, hors-champ) → `standard` par
  défaut, ou **ne pas annoter** si vraiment indéterminable. Ne jamais inventer un
  type rare sans preuve visuelle.
- **Ignorer** : la chaussée, les voies de circulation, les véhicules en
  mouvement, le mobilier urbain.

Astuce : privilégie la **qualité** sur la quantité pour ce 1er POC — mieux vaut
150 frames bien annotées (avec quelques pmr/recharge/livraison) que 506 bâclées.
Pense à chercher les **types rares** (la vidéo `131842` contient une recharge).

## Workflow CVAT

1. **Créer un projet** CVAT → onglet **Raw** des labels → coller le contenu de
   `training/cvat_labels.json` (garde l'ordre !).
2. **Créer une tâche** dans ce projet → uploader les images de
   `training/dataset/images/_to_annotate/`.
3. Annoter : outil **rectangle** → choisir le label → tracer la boîte → la
   **faire pivoter** (poignée de rotation) pour épouser la place en biais.
   Raccourcis utiles : `N` nouvelle boîte, `D`/`F` frame préc./suiv., `Ctrl+S`
   sauvegarder souvent.
4. **Exporter** la tâche : format **« Ultralytics YOLO Oriented Bounding Boxes
   1.0 »** (indices selon l'ordre des labels → d'où l'ordre verrouillé). Chaque
   label `.txt` = `class x1 y1 x2 y2 x3 y3 x4 y4` (4 sommets normalisés).

## Après l'export → split train/val

Décompresse l'export, repère le dossier des images et celui des `.txt`, puis :

```bash
cd C:\Claude_Code\watiir-video-redact
.venv\Scripts\python -m training.split_dataset \
    --images <dossier_images_export> \
    --labels <dossier_labels_export> \
    --val-ratio 0.2
```

Ça remplit `training/dataset/images/{train,val}` + `labels/{train,val}` et
affiche l'**histogramme par classe** (surveille les classes rares).

## Entraînement (go/no-go)

```bash
.venv\Scripts\python -m training.train --data training/data.yaml --epochs 80 --imgsz 640
```

Seuil POC indicatif : **mAP50 ≥ 0.50** → on industrialise l'annotation. Surveille
surtout le **rappel par classe** (pmr/recharge/livraison seront rares).

## RGPD

Les frames sont **déjà floutées** (visages + plaques). Ne ré-importe jamais de
source non floutée dans CVAT. Le dataset (`dataset/`, `cvat_export/`) est
gitignoré — seuls le code, `data.yaml`, ce guide et `cvat_labels.json` sont versionnés.
