"""Télécharge + vérifie l'intégrité des modèles ML pinnés dans manifest.json.

Lancé AU BUILD Docker (cf. Dockerfile) — pas au runtime. Le container final
est auto-suffisant et peut tourner sans accès Internet sortant.

Workflow :
  1. Pour chaque modèle du manifest, fetch selon sa `source` :
     - `huggingface` → huggingface_hub.hf_hub_download (revision épinglée) ;
     - `github_release` / autre avec `download_url` → téléchargement direct
       (httpx streaming) — utilisé par nos modèles internes fine-tunés
       (ex. `spot_detector`, publié en GitHub Release).
  2. Calcule SHA-256 et compare au manifest.json.
  3. Si commit_hash ou sha256 est null dans le manifest → MODE BOOTSTRAP :
     on imprime les valeurs mesurées (à coller dans le manifest) puis on
     échoue volontairement le build pour forcer une fixation explicite.
  4. Si valeurs présentes mais mismatch → échoue le build.

Usage :
    python -m models.download                # mode strict (CI / Docker build)
    python -m models.download --bootstrap    # autorise null → écrit les valeurs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MANIFEST_PATH = Path(__file__).parent / "manifest.json"
MODELS_DIR = Path(__file__).parent


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_sha(
    spec_key: str, spec: dict, actual_sha: str, actual_size: int, bootstrap: bool
) -> dict:
    """Compare le sha256 mesuré au manifest. Retourne le spec (mis à jour en
    mode bootstrap si sha256 était absent)."""
    print(f"  size: {actual_size:,} bytes")
    print(f"  sha256: {actual_sha}")
    expected_sha = spec.get("sha256")
    if expected_sha is None:
        if not bootstrap:
            raise RuntimeError(
                f"[{spec_key}] sha256 absent du manifest et --bootstrap non utilisé. "
                f"Re-lance avec --bootstrap pour figer : sha256={actual_sha}"
            )
        print(f"  [bootstrap] sha256 enregistré : {actual_sha}")
        return {**spec, "sha256": actual_sha}
    if expected_sha != actual_sha:
        raise RuntimeError(
            f"[{spec_key}] MISMATCH sha256\n"
            f"  attendu : {expected_sha}\n"
            f"  reçu    : {actual_sha}\n"
            f"Le fichier a probablement changé. Vérifier manuellement avant "
            f"de mettre à jour le manifest."
        )
    print("  sha256 OK")
    return spec


def fetch_one(spec_key: str, spec: dict, bootstrap: bool) -> dict:
    """Fetch un modèle HuggingFace et vérifie son intégrité."""
    repo_id = spec["repo_id"]
    filename = spec["filename"]
    revision = spec.get("commit_hash") or spec.get("revision") or "main"

    print(f"[{spec_key}] fetch {repo_id}/{filename} @ {revision}...", flush=True)
    local_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=str(MODELS_DIR),
        )
    )
    return _verify_sha(spec_key, spec, sha256_of(local_path), local_path.stat().st_size, bootstrap)


def fetch_url(spec_key: str, spec: dict, bootstrap: bool) -> dict:
    """Fetch un binaire depuis une URL directe (ex. GitHub Release : nos modèles
    internes fine-tunés) en streaming, puis vérifie son intégrité."""
    import httpx

    url = spec["download_url"]
    dest = MODELS_DIR / spec["filename"]
    print(f"[{spec_key}] download {url} ...", flush=True)
    h = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_bytes(1 << 20):
                f.write(chunk)
                h.update(chunk)
    return _verify_sha(spec_key, spec, h.hexdigest(), dest.stat().st_size, bootstrap)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch + verify ML models per manifest.")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Autorise les champs sha256 absents et les écrit dans manifest.json.",
    )
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"[fatal] {MANIFEST_PATH} introuvable.", file=sys.stderr)
        return 2

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    updated = False

    # Dispatch par source : HuggingFace (face/plate) ou URL directe / release
    # GitHub (spot_detector interne). Les entrées sans modèle téléchargeable
    # (tracker, runtime, $schema, version, description) sont ignorées.
    for key, spec in manifest.items():
        if not isinstance(spec, dict):
            continue
        source = spec.get("source")
        if source == "huggingface":
            new_spec = fetch_one(key, spec, bootstrap=args.bootstrap)
        elif spec.get("download_url"):
            new_spec = fetch_url(key, spec, bootstrap=args.bootstrap)
        else:
            continue
        if new_spec != spec:
            manifest[key] = new_spec
            updated = True

    if updated and args.bootstrap:
        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\n[bootstrap] manifest mis à jour : {MANIFEST_PATH}")

    print("\nAll models OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
