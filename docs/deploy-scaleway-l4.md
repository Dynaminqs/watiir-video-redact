# Déploiement Scaleway L4 (EU) — guide opérationnel

Procédure recommandée pour faire tourner `watiir-video-redact` en production
contre Supabase EU. Volumes V4.0 attendus faibles → déploiement **on-demand**
plutôt que permanent (estimation ~50 €/mois).

## Pourquoi Scaleway L4 EU

| Critère | Scaleway L4 24Go | OVHcloud L4 24Go | AWS g6.xlarge (L4) |
|---|---|---|---|
| Localisation EU | ✅ Paris/Amsterdam | ✅ Strasbourg/Roubaix | ✅ Paris/Frankfurt |
| Prix horaire | ~0.70 € | ~0.80 € | ~0.85 $ |
| Minimum facturé | Heure pleine | Heure pleine | Minute |
| API CLI | `scw` (Go) | `ovh-cli` | `aws-cli` |
| RGPD souveraineté | ✅ Société française | ✅ Société française | ⚠ Sous-traitance possible (Schrems II) |

**Choix par défaut : Scaleway** pour la cohérence "souveraineté française"
exigée par le PRD (cf. ARCHITECTURE.md → section RGPD). OVHcloud reste un
fallback équivalent.

## Prérequis

1. Compte Scaleway avec quota GPU activé (demande à
   [console.scaleway.com](https://console.scaleway.com) → quotas).
2. Image Docker `ghcr.io/dynaminqs/watiir-video-redact:latest` (poussée par
   le CI public à chaque merge sur main).
3. Service_role key Supabase staging ou prod (Dashboard → Settings → API).
4. `scw` CLI installée + configurée (`scw init`).

## Variables d'environnement

Le worker lit **toute** sa config depuis l'env (`worker/settings.py`, fail-fast si une variable requise manque).

| Variable | Requis | Défaut | Rôle |
|---|---|---|---|
| `SUPABASE_URL` | ✅ | — | URL du projet Supabase EU |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅ | — | Clé secrète serveur. **Depuis la bascule des clés (cf. ADR-009 WATIIR), utiliser la clé `sb_secret_*`** — pas la legacy JWT (rejetée par GoTrue). Le worker n'appelle que PostgREST/Edge Functions/Storage. |
| `POLL_INTERVAL_IDLE_S` | — | `30` | Sleep entre polls quand la file est vide |
| `MAX_VIDEO_DURATION_S` | — | `600` | Hard cap durée vidéo (rejet au-delà) |
| `REQUEST_TIMEOUT_S` | — | `30.0` | Timeout HTTP vers Supabase |
| `WORK_DIR` | — | `/tmp/watiir-redact-work` | Dossier temp (volume persistant conseillé) |
| `MODELS_DIR` | — | `/app/models` (image) | Dossier des poids ML |
| `FACE_MODEL_PATH` / `PLATE_MODEL_PATH` | — | `<models>/model.pt` / `best.pt` | Modèles de floutage (`PLATE_MODEL_PATH=none` → mode face-only dégradé) |
| `SPOT_DETECTOR` | — | `stub` | Détecteur de places (rail 2) : `stub` (défaut) ou `yolov8` — cf. section dédiée |
| `SPOT_DETECTOR_CHECKPOINT` | si `yolov8` | — | Chemin du `.pt` (dans l'image : `/app/models/spot_detector.pt`) |
| `SPOT_DETECTOR_MODEL_VERSION` | — | dérivé du nom de fichier | Tracé jusqu'au staging pour audit (ex. `yolov8-spots-poc-v1`) |

## Déploiement on-demand (recommandation V4.0)

Le worker peut être lancé à la demande en réponse à une alerte sur la
profondeur de file, puis stoppé quand la file est vide. Procédure manuelle :

```bash
# 1. Provisionner une instance GPU-3070 ou L4
scw instance server create \
  type=GPU-3070-S \
  zone=fr-par-2 \
  image=ubuntu_jammy \
  name=watiir-redact-$(date +%Y%m%d-%H%M%S) \
  root-volume=l:50GB

# Attends que l'instance soit prête (~30s)
SERVER_ID=$(scw instance server list name=watiir-redact-* state=running -o json | jq -r '.[0].id')
SERVER_IP=$(scw instance server get "$SERVER_ID" -o json | jq -r '.public_ip.address')

# 2. SSH + run le container (NVIDIA drivers déjà sur les images Scaleway GPU)
ssh root@"$SERVER_IP" bash <<'EOSSH'
  docker login ghcr.io  # avec un PAT GitHub si repo privé (public ici : pas requis)
  docker pull ghcr.io/dynaminqs/watiir-video-redact:latest
  docker run -d \
    --name watiir-redact \
    --restart unless-stopped \
    --gpus all \
    -e SUPABASE_URL="$SUPABASE_URL" \
    -e SUPABASE_SERVICE_ROLE_KEY="$SUPABASE_SERVICE_ROLE_KEY" \
    -e POLL_INTERVAL_IDLE_S=30 \
    -e MAX_VIDEO_DURATION_S=600 \
    -v /var/lib/watiir-redact-work:/tmp/watiir-redact-work \
    ghcr.io/dynaminqs/watiir-video-redact:latest
EOSSH

# 3. Monitorer les logs
ssh root@"$SERVER_IP" "docker logs -f watiir-redact"

# 4. Quand la file est vide depuis > 1h, détruire l'instance
scw instance server delete "$SERVER_ID" with-ip=true with-volumes=all
```

⚠ **Secrets** : ne pas commiter `SUPABASE_SERVICE_ROLE_KEY` dans des fichiers
sur disque. Préférer Scaleway Secret Manager ou un wrapper qui injecte la
clé via `--env-file` chargé depuis un coffre. **Valeur attendue = la clé
`sb_secret_*`** (la legacy `service_role` JWT est rejetée par GoTrue depuis la
bascule des clés — cf. ADR-009 côté WATIIR).

## Détection de places (rail 2) — activer le modèle

L'étage **détection** (`claim_next_pending_detection` → `notify-detection-complete`)
tourne avec un détecteur **pluggable** (`SPOT_DETECTOR`) :

- **`stub` (défaut)** — `StubDetector`, sorties déterministes. Valide la plomberie
  E2E (création de candidats → modération) **sans modèle ML**. À garder tant
  qu'un checkpoint n'est pas validé.
- **`yolov8`** — `Yolov8SpotDetector`, modèle YOLOv8-OBB fine-tuné. Le binaire est
  **déjà dans l'image** (`models/download.py` le récupère au build depuis la
  GitHub Release, pinné par sha256 dans `manifest.json`) → présent à
  `/app/models/spot_detector.pt`.

Activation (ajouter au `docker run`) :

```bash
  -e SPOT_DETECTOR=yolov8 \
  -e SPOT_DETECTOR_CHECKPOINT=/app/models/spot_detector.pt \
  -e SPOT_DETECTOR_MODEL_VERSION=yolov8-spots-poc-v1 \
```

⚠ **Staging d'abord** : le 1er checkpoint (`spot-detector-poc-v1`, mAP50 val 0.771)
est un **POC** validé sur un petit jeu (chiffres bruités, pas « production »).
L'activer **en staging** d'abord, vérifier le bout-en-bout (candidats détectés →
modération humaine `pending_moderation`), surveiller le **rappel par classe**
(`recharge`/`livraison`/`pmr`) avant tout passage prod. Le `model_version` est
tracé jusqu'au staging pour corréler qualité ↔ version. Sans ces variables, le
worker reste sur `stub` (aucun risque qualité).

## Déploiement permanent (V4.1+ — volumes élevés)

Pour un worker permanent, packager le run via systemd sur l'instance, puis
ajouter un auto-scaler basé sur la profondeur de file
(`SELECT count(*) FROM video_imports WHERE status = 'uploaded'`) qui pop
de nouvelles instances Scaleway au-delà d'un seuil (ex: > 5 vidéos en attente
pendant > 5 min).

Voir l'issue tracker pour le script d'auto-scaling (non livré V4.0).

## Estimation coûts

Hypothèse V4.0 prudente : 100 vidéos/mois × 2 min compute moyen.

- Compute GPU : 100 × 2 min × (0.70 €/3600s) ≈ **2 €/mois**
- Storage Supabase : ~10 Go (vidéos floutées 30j max) → inclus dans le quota
  Supabase Pro EU
- Min facturation Scaleway (au moins 1h par lancement on-demand) : si on
  lance 10 fois/mois × 1h pleine = ~7 €
- **Total : ~10-50 €/mois selon le pattern de lancement**

À l'échelle V4.1+ (1000 vidéos/mois), passer en permanent + 2 workers =
~1000 €/mois compute, mais beaucoup de marge avant d'y arriver.

## Bascule de secours OVHcloud

Si Scaleway L4 indisponible (quota dépassé ou panne EU), pivoter sur OVHcloud :

```bash
# Pré-provisionner un compte OVH avec quota GPU L4 idem.
ovh-cli cloud server create \
  --image-id <Ubuntu 22.04 GPU> \
  --flavor-id <L4 EU> \
  --region GRA11 \
  --name watiir-redact-failover
# Même commande docker run que ci-dessus.
```

Documenter la bascule dans le runbook côté WATIIR (ARCHITECTURE.md section
Runbook) le jour où on l'exerce pour la 1re fois.

## Monitoring

- **Profondeur de file** : alert si > N pendant > 1h. Query :
  `SELECT count(*) FROM video_imports WHERE status='uploaded'` (à automatiser
  via pg_cron + Sentry capture ou Cloudflare KV trigger).
- **RTF du worker** : visible dans les logs Sentry — alert si RTF > 1.5×
  sur > 10 vidéos consécutives.
- **GPU utilization** : `nvidia-smi` côté instance Scaleway, ou métriques
  Scaleway console.
- **Audit log** : count(`action='video-import-redacted'`) vs
  count(`action='video-import-failed'`) → ratio d'échec à surveiller.

## Sécurité réseau

- L'instance Scaleway n'a PAS besoin de port ouvert (worker outbound-only
  vers Supabase). Configurer le Security Group pour bloquer tout inbound
  sauf SSH (port 22) restreint à l'IP du DevOps.
- Le worker ne sert AUCUN endpoint HTTP : pas de port à exposer, pas de
  Cloudflare/WAF en amont nécessaire.

## Procédure de rollback

Si une version cassée est déployée :

```bash
# Pin à une version SHA précédente
docker pull ghcr.io/dynaminqs/watiir-video-redact:<previous-sha>
docker stop watiir-redact && docker rm watiir-redact
docker run -d ... ghcr.io/dynaminqs/watiir-video-redact:<previous-sha>
```

Toute version est taggée `:latest` + `:<sha>` par le CI. Voir l'historique
sur https://github.com/dynaminqs/watiir-video-redact/pkgs/container/watiir-video-redact.
