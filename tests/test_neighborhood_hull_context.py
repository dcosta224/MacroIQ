"""Tests for per-neighbor hull stretch context."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recipe_opt_agent.neighborhood_hull_context import build_neighborhood_hull_context


def _synthetic_lines() -> pd.DataFrame:
    # Two recipes: pasta-heavy vs protein-heavy vertex
    rows = []
    for rid, items in {
        "r1": [(1, 100.0), (2, 20.0)],
        "r2": [(1, 80.0), (3, 40.0)],
    }.items():
        for idx, (fdc, g) in enumerate(items):
            rows.append(
                {
                    "recipe_nlg_id": rid,
                    "ingredient_idx": idx,
                    "fdc_id": fdc,
                    "gram_weight": g,
                    "fdc_description": f"food_{fdc}",
                }
            )
    return pd.DataFrame(rows)


def test_build_neighborhood_hull_context_excludes_shell(monkeypatch):
    per_g = {
        1: np.array([0.12, 0.02, 0.70, 3.5]),  # high carb
        2: np.array([0.08, 0.05, 0.65, 3.2]),
        3: np.array([0.35, 0.03, 0.05, 1.8]),  # high protein
    }
    monkeypatch.setattr(
        "recipe_opt_agent.neighborhood_hull_context._per_gram_macros_for_fdc_ids",
        lambda _ids: per_g,
    )
    box = {
        "protein_min": 0.28,
        "protein_max": 0.35,
        "carb_min": 0.30,
        "carb_max": 0.50,
        "fat_min": 0.20,
        "fat_max": 0.40,
    }
    ctx = build_neighborhood_hull_context(
        _synthetic_lines(),
        ["r1", "r2", "shell_1"],
        box,
        shell_recipe_ids=["shell_1"],
        starting_recipe_id="r1",
        n_hull_samples=400,
    )
    assert ctx["n_recipes"] == 2
    assert "target_stretch_level" in ctx
    assert ctx["frac_hull_intersects"] >= 0.0


def test_dietary_precheck_flags_pork():
    from recipe_opt_agent.final_evaluator import dietary_precheck
    from recipe_opt_agent.requirement_tags import RequirementTag

    tags = [
        RequirementTag(tag_id="no_pork", kind="dietary_restriction", polarity="forbid", source_text="no pork")
    ]
    pre = dietary_precheck(
        [{"label": "Pork, cured, bacon, cooked", "grams": 50.0}],
        tags,
    )
    assert pre["dietary_violation_flag"] is True
    assert pre["violations"][0]["tag_ids"] == ["no_pork"]
