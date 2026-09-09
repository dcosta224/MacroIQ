"""Unit tests for hull geometry (no DB)."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hull_geometry import (  # noqa: E402
    TargetBox,
    hull_delaunay_2d,
    iter_conical_mixes,
    point_in_hull,
    region_intersects_hull,
)


def _two_food_M():
    # Food A: pure protein (4 kcal/g protein)
    # Food B: pure carb
    # M rows: protein_g, fat_g, carb_g, other per gram food
    return np.array(
        [
            [1.0, 0.0],  # protein g / g
            [0.0, 0.0],  # fat
            [0.0, 1.0],  # carb
            [0.0, 0.0],
        ],
        dtype=float,
    )


def test_conical_mixes_span_protein_carb_edge():
    M = _two_food_M()
    samples = iter_conical_mixes(M, n_samples=500, seed=0)
    assert samples.shape[1] == 3
    # Pure A → protein fraction 1
    assert samples[:, 0].max() > 0.9
    assert samples[:, 1].max() > 0.9


def test_point_in_hull_vertices():
    M = _two_food_M()
    samples = iter_conical_mixes(M, n_samples=800, seed=1)
    tri = hull_delaunay_2d(samples)
    assert tri is not None
    assert point_in_hull(1.0, 0.0, 0.0, tri)
    assert point_in_hull(0.0, 1.0, 0.0, tri)
    assert point_in_hull(0.5, 0.5, 0.0, tri)
    # High fat alone not in hull
    assert not point_in_hull(0.0, 0.0, 1.0, tri)


def test_region_intersects_uses_lp_oracle():
    M = _two_food_M()
    box = TargetBox(0.4, 0.6, 0.4, 0.6, 0.0, 0.2)
    # kcal from 100g of 50/50 mix: protein 50g*4 + carb 50g*4 = 400
    out = region_intersects_hull(M, box, kcal_target=400.0, n_samples=1000)
    assert out["lp_feasible"] is True
    assert out["intersects"] is True
