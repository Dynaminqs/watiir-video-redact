# watiir-video-redact

**License : AGPL-3.0.** Pipeline Python de floutage de visages et de plaques
d'immatriculation pour le projet [WATIIR](https://watiir.com), composante V4
"Edge AI rail 2" (cartographie communautaire de stationnement par vidéo).

> ⚠ **AGPL est viral via réseau.** Toute modification ou intégration de ce
> pipeline DOIT être publiée sous licence compatible AGPL-3.0 si elle est
> exposée comme service réseau. Voir [LICENSE](./LICENSE) et la
> [FAQ AGPL](https://www.gnu.org/licenses/agpl-3.0.faq.html).

## Pourquoi public + AGPL ?

WATIIR collecte de la vidéo voie publique uploadée par des "cartographes"
désignés. Le pipeline de floutage est le garde-fou RGPD critique : il efface
visages + plaques avant toute modération humaine et avant tout usage des
images dérivées.

- **Transparence audit CNIL** : le DPO peut pointer ce repo public et le
  process exact qui floute les images.
- **Coût licence Ultralytics évité** : YOLOv8 est AGPL, donc tout code qui
  l'utilise en réseau l'est aussi. Publier WATIIR-video-redact sous AGPL est
  cohérent.
- **Frontière API stricte** : le repo principal WATIIR (privé) appelle ce
  pipeline via Supabase RPC + Storage uniquement — pas d'import statique de
  code. Aucune contamination AGPL du reste du codebase.

## Architecture

```
                            Supabase EU (project staging tdsdlptxsvlhscjoyuyc)
                            ┌──────────────────────────────────────────────┐
mobile/admin web            │                                              │
   │ POST /functions/v1/    │   Edge Function submit-video-import         │
   │ submit-video-import    │      │                                       │
   │ (Bearer user JWT)      │      │  INSERT video_imports (status=uploaded)
   ▼                        │      │  upload bucket vision-ai-uploads       │
                            │      ▼                                       │
                            │   table video_imports                        │
                            │      │                                       │
                            │      │  RPC claim_next_pending_redaction()   │
                            │      │  (FOR UPDATE SKIP LOCKED)             │
                            │      ▼                                       │
   THIS WORKER  ◀───────────┤  status=redacting                            │
   poll 30s                 │      │                                       │
      │                     │      │  download bucket vision-ai-uploads    │
      ▼                     │      │                                       │
   pipeline.redact          │      │                                       │
   (YOLOv8 face + LP        │      │                                       │
    + ByteTrack + OpenCV)   │      │                                       │
      │                     │      │  upload bucket vision-ai-redacted     │
      ▼                     │      │                                       │
   POST /functions/v1/      │   Edge Function notify-video-redacted        │
   notify-video-redacted    │   (Bearer SUPABASE_SERVICE_ROLE_KEY)         │
   { status: 'redacted',    │      │                                       │
     redacted_bucket_path,  │      │  UPDATE video_imports                 │
     faces_count,           │      │  status=redacted, redacted_at,        │
     plates_count }         │      │  redacted_bucket_path, counts         │
                            │      ▼                                       │
                            │  audit_log "video-import-redacted"           │
                            │                                              │
                            └──────────────────────────────────────────────┘
```

**État final** : la vidéo floutée reste 30j max dans `vision-ai-redacted`
(consommée par la modération humaine "G", future sous-change). La vidéo brute
est purgée >1h post-redacted par le job pg_cron `vision_ai_purge`.

## Stack ML (figée V4.0)

POC validé 2026-05-27 (cf. WATIIR repo `poc/edge-ai-floutage/results_2026-05-27.md`)
sur GPU T4 free Colab — **off-the-shelf suffit**, pas de fine-tune en V4.0 :

| Composant | Modèle | License | Source |
|---|---|---|---|
| Détection visages | YOLOv8-face | AGPL-3.0 | `arnabdhar/YOLOv8-Face-Detection` (HF) |
| Détection plaques | YOLOv8-LP | AGPL-3.0 | `yasirfaizahmed/license-plate-object-detection` (HF) |
| Tracker temporel | ByteTrack | MIT | Inclus dans Ultralytics 8.x |
| Floutage | OpenCV GaussianBlur | Apache 2.0 | `opencv-python-headless` |

Performance mesurée : **RTF 0.66×** sur T4 free Colab (2.3× la marge sur le
temps réel). Inférence mean 18.6ms / p95 27.3ms. Rappel visuel **100 %** sur
faces et plaques FR observées.

Les checkpoints sont pinned par commit hash + SHA-256 dans
[`models/manifest.json`](./models/manifest.json). Le `Dockerfile` télécharge
les modèles AU BUILD via [`models/download.py`](./models/download.py) — pas
de fetch HF à l'exécution (container auto-suffisant).

## Étage détection de places (sous-change F)

En plus du floutage, le worker exécute un **étage détection** : il poll
`claim_next_pending_detection()` (vidéos en `status=redacted`), détecte les
places de stationnement sur la vidéo **floutée** + le log GPS, et publie des
places candidates via `notify-detection-complete` → `status=pending_moderation`.
Une modération humaine (sous-change G, côté WATIIR) valide avant publication.

Flux : `redacted → (claim) detecting → pending_moderation | failed_detection`.

Le détecteur est **pluggable** (`pipeline/detect.py`, Protocol `SpotDetector`) :

| Détecteur | `SPOT_DETECTOR` | Usage |
|---|---|---|
| `StubDetector` | `stub` (défaut) | Sorties déterministes — valide la plomberie E2E sans modèle |
| `Yolov8SpotDetector` | `yolov8` | **NON fourni en V4.0** — gaté sur le POC ci-dessous |

Périmètre IA V4.0 : **TYPE de place uniquement** (`livraison` / `pmr` /
`recharge` / `standard`) avec score de confiance. L'état libre/occupé n'est
**pas** inféré (reporté V4.1). Géoréférencement : position GPS interpolée du
véhicule à la frame de détection (approximation assumée, affinée par la
modération G). Sans GPS exploitable → `failed_detection`
(`gps_required_for_georeferencing`).

### POC détection de places (prérequis du modèle réel)

Contrairement au floutage (modèles off-the-shelf validés), il n'existe **pas**
de modèle de détection de places de rue prêt à l'emploi. Le `Yolov8SpotDetector`
nécessite un **POC + dataset interne** avant d'être livré :

1. Constituer un dataset annoté à partir des photos V3.c **déjà floutées**
   (RLS-protégées côté WATIIR — jamais de photos non floutées).
2. Fine-tuner un YOLOv8-nano (TYPE de place) ; mesurer rappel / faux positifs.
3. Pinner le checkpoint dans `models/manifest.json` (commit hash + SHA-256).
4. Implémenter `Yolov8SpotDetector` dans `pipeline/detect.py` derrière le flag
   `SPOT_DETECTOR=yolov8`.

En attendant, la plomberie tourne avec `StubDetector` ; la modération humaine G
borne le risque qualité.

## Quickstart dev local

```bash
# Python 3.11 requis
python -m venv .venv
. .venv/bin/activate  # ou .venv\Scripts\activate sur Windows
pip install -e .[dev]

# Télécharger les modèles + vérifier SHA-256
python -m models.download

# Run tests (mocks ML — pas de GPU requis)
pytest

# Lancer le worker sur une vidéo de test (env vars requises)
export SUPABASE_URL=https://tdsdlptxsvlhscjoyuyc.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=...  # depuis Supabase → Settings → API
python -m worker.main
```

## Production : déploiement Scaleway L4

Recommandation V4.0 : 1 instance Scaleway L4 EU (~0.7 €/h, ~50 €/mois en
on-demand selon volumes). Procédure détaillée dans
[`docs/deploy-scaleway-l4.md`](./docs/deploy-scaleway-l4.md).

Image Docker : `ghcr.io/dynaminqs/watiir-video-redact:latest` (CI public).

## Tests

Les tests pytest **mockent** les modèles YOLO pour rester < 2 min en CI :

```bash
pytest                          # tous les tests (rapide)
pytest tests/test_redact.py     # uniquement pipeline (blur math)
pytest tests/test_worker.py     # uniquement worker (mock RPC + Storage)
ruff check .                    # lint
ruff format --check .           # format
```

Pour tester avec les vrais modèles (intégration, GPU requis) — pas inclus en
CI car coûteux :

```bash
WATIIR_REDACT_E2E=1 pytest tests/test_redact_e2e.py
```

## Contribution

Lire `CONTRIBUTING.md` (TBD) avant de soumettre un PR. Toute contribution est
acceptée sous AGPL-3.0.

## Sécurité

Aucun secret n'est commit dans ce repo. Les credentials Supabase passent par
variables d'environnement (`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`). En
prod, utiliser un secret manager (Scaleway Secret Manager, Vault, etc.).

Bug security ? Ouvrir une issue privée via GitHub Security Advisories
plutôt que public.

## License

Tout le code est sous licence **GNU AGPL v3.0** — voir [LICENSE](./LICENSE).
