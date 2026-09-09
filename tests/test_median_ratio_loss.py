"""Tests for median-centered share deviation ratio metric."""

from __future__ import annotations

from recipe_opt_agent.score_display import (
    extract_ratio_and_nutrient,
    mean_abs_dev_from_median_shares,
)


def test_mean_abs_dev_from_median_zero_at_median():
    samples = {"n1": [0.1, 0.2, 0.3], "n2": [0.4, 0.5, 0.6]}
    # shares 0.2 and 0.5 are exact medians if mass=100 → 20 and 50
    loss = mean_abs_dev_from_median_shares(
        x=[20.0, 50.0, 30.0],
        ingredient_basis=["n1", "n2", None],
        basis_samples=samples,
    )
    assert loss is not None
    assert abs(loss - 0.0) < 1e-9


def test_mean_abs_dev_penalizes_any_median_drift():
    samples = {"n1": [0.2, 0.2, 0.2]}
    loss = mean_abs_dev_from_median_shares(
        x=[40.0, 60.0],
        ingredient_basis=["n1", None],
        basis_samples=samples,
    )
    assert loss is not None
    assert abs(loss - 0.2) < 1e-9  # share 0.4 vs median 0.2


def test_extract_ratio_prefers_median_metric():
    payload = {
        "opt": {
            "x_opt": [20.0, 50.0, 30.0],
            "term_losses": {"FOODON_A": 0.9, "ratio_surrogate": 0.0},
            "pfc_after": {"protein": 0.25, "carbs": 0.4, "fat": 0.35},
        },
        "problem": {
            "ingredient_basis": ["FOODON_A", "FOODON_B", None],
            "basis_samples": {
                "FOODON_A": [0.1, 0.2, 0.3],
                "FOODON_B": [0.4, 0.5, 0.6],
            },
            "ratio_samples": [],
            "protein_min": 0.2,
            "protein_max": 0.3,
            "carb_min": 0.35,
            "carb_max": 0.45,
            "fat_min": 0.3,
            "fat_max": 0.4,
        },
        "chosen_recipe": {"ingredients": []},
    }
    ratio, src, nutrient, _ = extract_ratio_and_nutrient(payload)
    assert src == "mean_abs_dev_from_median"
    assert ratio is not None
    assert ratio < 0.01
    # nutrient may be None if box is not on the usual config paths; ratio is the focus
    assert src == "mean_abs_dev_from_median"
