"""Télécharge + vérifie l'intégrité des modèles ML pinnés dans manifest.json.

Lancé AU BUILD Docker (cf. Dockerfile) — pas au runtime. Le container final
est auto-suffisant et peut tourner sans accès Internet sortant.

Workflow :
  1. Pour chaque modèle du manifest, fetch via huggingface_hub.hf_hub_download
     en epinglant `revision` (commit_hash si présent, sinon main).
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


def fetch_one(spec_key: str, spec: dict, bootstrap: bool) -> dict:
    """Fetch un modèle et vérifie son intégrité. Retourne le dict mis à jour
    (utile en mode bootstrap pour ré-écrire le manifest)."""
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
    actual_size = local_path.stat().st_size
    actual_sha = sha256_of(local_path)

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
        spec = {**spec, "sha256": actual_sha}
    elif expected_sha != actual_sha:
        raise RuntimeError(
            f"[{spec_key}] MISMATCH sha256\n"
            f"  attendu : {expected_sha}\n"
            f"  reçu    : {actual_sha}\n"
            f"Le fichier sur HF a probablement changé. Vérifier manuellement avant "
            f"de mettre à jour le manifest."
        )
    else:
        print("  sha256 ✓")

    return spec


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
    keys = ["face_detector", "license_plate_detector"]
    updated = False

    for key in keys:
        if key not in manifest:
            print(f"[skip] {key} absent du manifest.", file=sys.stderr)
            continue
        spec = manifest[key]
        new_spec = fetch_one(key, spec, bootstrap=args.bootstrap)
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
