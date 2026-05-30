"""Fixtures pytest partagées.

Stratégie : on mocke YOLO et Supabase pour rester < 2 min en CI sans GPU et
sans accès Internet. Les tests d'intégration GPU sont marqués `e2e` et skippés
par défaut (run avec WATIIR_REDACT_E2E=1).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _skip_e2e_by_default(request: pytest.FixtureRequest) -> None:
    """Skip auto tous les tests marqués `e2e` sauf si WATIIR_REDACT_E2E=1."""
    if "e2e" in request.keywords and not os.environ.get("WATIIR_REDACT_E2E"):
        pytest.skip("WATIIR_REDACT_E2E non défini — skip test e2e GPU.")


@pytest.fixture
def tmp_work_dir(tmp_path: Path) -> Iterator[Path]:
    """Work directory éphémère pour le worker."""
    work = tmp_path / "work"
    work.mkdir()
    yield work


@pytest.fixture
def fake_yolo_result_factory():
    """Construit un faux résultat ultralytics (list[Results]) avec des boxes
    + ids contrôlés. Utilisé par tests qui veulent simuler une inference."""

    def make(boxes_xyxy: list[list[float]], track_ids: list[int] | None = None):
        import numpy as np

        if not boxes_xyxy:
            boxes = None
        else:
            xyxy_np = np.array(boxes_xyxy, dtype=float)
            ids_np = np.array(track_ids, dtype=float) if track_ids is not None else None
            boxes = MagicMock()
            boxes.xyxy = MagicMock()
            boxes.xyxy.cpu = lambda: MagicMock(numpy=lambda: xyxy_np)
            if ids_np is not None:
                boxes.id = MagicMock()
                boxes.id.cpu = lambda: MagicMock(numpy=lambda: ids_np)
            else:
                boxes.id = None
            # Surcharge .xyxy.cpu().numpy()
            inner_cpu = MagicMock()
            inner_cpu.numpy = lambda: xyxy_np
            boxes.xyxy.cpu = lambda: inner_cpu

        result = MagicMock()
        result.boxes = boxes
        return [result]

    return make
