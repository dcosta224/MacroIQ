"""Tests for neighborhood-mean macro target suggestions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recipe_opt_agent.macro_target_suggestions import (
    _rounded_box_pm,
    suggest_macro_targets,
)


def _synthetic_lines() -> pd.DataFrame:
    rows = []
    for rid, items in {
        "rep": [(1, 120.0), (2, 30.0)],
        "other": [(1, 80.0), (3, 50.0)],
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


def test_suggest_neighborhood_mean_only(monkeypatch):
    fake = {
        "rep": {"protein": 0.18, "carbs": 0.52, "fat": 0.30},
        "other": {"protein": 0.22, "carbs": 0.48, "fat": 0.30},
        "third": {"protein": 0.12, "carbs": 0.58, "fat": 0.30},
        "fourth": {"protein": 0.28, "carbs": 0.42, "fat": 0.30},
    }
    monkeypatch.setattr(
        "recipe_opt_agent.problem_loader._batch_recipe_pfc_from_lines",
        lambda _df: fake,
    )

    result = suggest_macro_targets(_synthetic_lines(), "rep")
    assert result["n_recipes"] == 4
    presets = result["presets"]
    assert "neighborhood_mean" in presets
    assert "neighborhood_coverage" in presets
    assert result["default_preset"] == "neighborhood_coverage"
    assert result["distribution"]["n_recipes"] == 4

    mean = presets["neighborhood_mean"]
    assert mean["pad_pct"] == 5
    for key in ("protein_min", "protein_max", "carb_min", "carb_max", "fat_min", "fat_max"):
        assert abs(mean["box"][key] * 100 - round(mean["box"][key] * 100)) < 1e-9
    assert round((mean["box"]["protein_max"] - mean["box"]["protein_min"]) * 100) == 10

    cov = presets["neighborhood_coverage"]
    box = cov["box"]
    assert box["protein_max"] + box["carb_max"] + box["fat_max"] >= 1.0 - 1e-9
    assert box["protein_min"] + box["carb_min"] + box["fat_min"] <= 1.0 + 1e-9
    assert 0.0 <= cov["coverage_frac"] <= 1.0
    # Each macro band is at most 10 percentage points wide.
    for axis in ("protein", "carb", "fat"):
        width = box[f"{axis}_max"] - box[f"{axis}_min"]
        assert width <= 0.10 + 1e-9
        assert width >= 0.0
    assert cov.get("width") == pytest.approx(0.10)


def test_best_fixed_width_band_picks_densest_10pp():
    from recipe_opt_agent.macro_target_suggestions import _best_fixed_width_band

    # Cluster tightly around 0.20–0.28; a 10pp band should land there, not on the outlier.
    vals = np.asarray([0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.55], dtype=float)
    lo, hi, n_in = _best_fixed_width_band(vals, width=0.10)
    assert hi - lo == pytest.approx(0.10)
    assert n_in >= 7
    assert lo <= 0.20 + 1e-9
    assert hi >= 0.26 - 1e-9
    assert not (lo <= 0.55 <= hi and n_in < 7)

def test_assess_macro_box_vs_distribution():
    from recipe_opt_agent.macro_target_suggestions import assess_macro_box_vs_distribution

    # 10 recipes clustered around ~18% protein / 48% carbs / 34% fat.
    protein_vals = [0.15, 0.16, 0.17, 0.18, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23]
    carbs_vals = [0.42, 0.44, 0.46, 0.47, 0.48, 0.49, 0.50, 0.52, 0.54, 0.55]
    fat_vals = [0.28, 0.30, 0.32, 0.33, 0.34, 0.35, 0.36, 0.38, 0.39, 0.40]
    dist = {
        "n_recipes": 10,
        "protein": {
            "median": 0.18,
            "p25": 0.15,
            "p75": 0.22,
            "values": protein_vals,
        },
        "carbs": {
            "median": 0.48,
            "p25": 0.42,
            "p75": 0.55,
            "values": carbs_vals,
        },
        "fat": {
            "median": 0.34,
            "p25": 0.28,
            "p75": 0.40,
            "values": fat_vals,
        },
    }
    typical = assess_macro_box_vs_distribution(
        {
            "protein_min": 0.16,
            "protein_max": 0.22,
            "carb_min": 0.44,
            "carb_max": 0.52,
            "fat_min": 0.30,
            "fat_max": 0.38,
        },
        dist,
        dish_title="Pad Thai",
    )
    assert typical["overall"] == "typical"
    assert "typical for Pad Thai" in typical["summary"]
    assert typical["axes"]["protein"]["status"] == "typical"
    assert typical["axes"]["protein"]["n_in_range"] >= 3

    # Slightly high protein: ≥80% of recipes below the band min, but >2 still in-range.
    # Band [0.20, 0.30]: values below 0.20 → 6/10=60% — not enough.
    # Band [0.215, 0.315]: below → 8/10=80%, in-range → 0.22,0.23 = 2 → that is VERY.
    # Need ≥80% below AND ≥3 in range: e.g. expand cloud.
    protein_slight = [0.10 + 0.005 * i for i in range(10)]  # 0.10..0.145
    # Add 3 recipes inside a high band [0.20, 0.30]
    protein_slight = protein_slight[:7] + [0.21, 0.22, 0.23]
    dist_slight = {
        "n_recipes": 10,
        "protein": {"median": 0.12, "p25": 0.10, "p75": 0.21, "values": protein_slight},
        "carbs": {"median": 0.48, "p25": 0.42, "p75": 0.55, "values": carbs_vals},
        "fat": {"median": 0.34, "p25": 0.28, "p75": 0.40, "values": fat_vals},
    }
    # 7/10 = 70% below 0.20 — bump to 8 below:
    protein_slight = [0.10 + 0.005 * i for i in range(8)] + [0.21, 0.22]
    dist_slight["protein"]["values"] = protein_slight
    slightly_high = assess_macro_box_vs_distribution(
        {
            "protein_min": 0.20,
            "protein_max": 0.30,
            "carb_min": 0.40,
            "carb_max": 0.50,
            "fat_min": 0.28,
            "fat_max": 0.38,
        },
        dist_slight,
        dish_title="Pad Thai",
    )
    # 8 below min, 2 in range → very_high wins over slightly
    assert slightly_high["axes"]["protein"]["status"] == "very_high"
    assert "protein is very high" in slightly_high["summary"]

    # True slightly_high: 8/10 below min, and 3+ in range.
    protein_slight2 = [0.10 + 0.004 * i for i in range(8)] + [0.205, 0.21, 0.22]
    # that's 11 values — trim to 10 with 8 below and 2 in... need 8 below + 3 in = 11.
    # Use n=15: 12 below (80%), 3 in range.
    protein_slight2 = [0.10 + 0.005 * i for i in range(12)] + [0.205, 0.21, 0.22]
    carbs15 = carbs_vals + carbs_vals[:5]
    fat15 = fat_vals + fat_vals[:5]
    dist_slight2 = {
        "n_recipes": 15,
        "protein": {"median": 0.13, "p25": 0.11, "p75": 0.20, "values": protein_slight2},
        "carbs": {"median": 0.48, "p25": 0.42, "p75": 0.55, "values": carbs15},
        "fat": {"median": 0.34, "p25": 0.28, "p75": 0.40, "values": fat15},
    }
    slightly = assess_macro_box_vs_distribution(
        {
            "protein_min": 0.20,
            "protein_max": 0.30,
            "carb_min": 0.40,
            "carb_max": 0.55,
            "fat_min": 0.28,
            "fat_max": 0.40,
        },
        dist_slight2,
        dish_title="Pad Thai",
    )
    assert slightly["axes"]["protein"]["n_below"] / 15 >= 0.8
    assert slightly["axes"]["protein"]["n_in_range"] >= 3
    assert slightly["axes"]["protein"]["status"] == "slightly_high"
    assert "protein is slightly high" in slightly["summary"]

    # Very low carbs: 0–2 recipes inside a low band.
    very_low = assess_macro_box_vs_distribution(
        {
            "protein_min": 0.15,
            "protein_max": 0.25,
            "carb_min": 0.05,
            "carb_max": 0.15,
            "fat_min": 0.28,
            "fat_max": 0.38,
        },
        dist,
        dish_title="Pad Thai",
    )
    assert very_low["axes"]["carb"]["n_in_range"] <= 2
    assert very_low["axes"]["carb"]["status"] == "very_low"
    assert "carbs is very low" in very_low["summary"]


def test_macro_ranges_are_dish_specific(monkeypatch):
    """Coverage bands must differ across dishes with different PFC clouds."""
    from recipe_opt_agent.macro_target_suggestions import suggest_macro_targets

    low_protein = {
        f"r{i}": {"protein": 0.12 + 0.005 * (i % 5), "carbs": 0.55, "fat": 0.30}
        for i in range(12)
    }
    high_protein = {
        f"r{i}": {"protein": 0.32 + 0.005 * (i % 5), "carbs": 0.35, "fat": 0.30}
        for i in range(12)
    }

    def _run(fake):
        monkeypatch.setattr(
            "recipe_opt_agent.problem_loader._batch_recipe_pfc_from_lines",
            lambda _df: fake,
        )
        return suggest_macro_targets(_synthetic_lines(), "rep")

    a = _run(low_protein)
    b = _run(high_protein)
    a_box = a["presets"]["neighborhood_coverage"]["box"]
    b_box = b["presets"]["neighborhood_coverage"]["box"]
    # Dish-specific: high-protein neighborhood's protein band sits higher.
    assert a_box["protein_min"] < b_box["protein_min"]
    assert a_box["protein_max"] < b_box["protein_max"]
    # Values are attached for count-based typicality on the client/server.
    assert len(a["distribution"]["protein"]["values"]) == 12
    assert len(b["distribution"]["protein"]["values"]) == 12
    # Same absolute box is typical for one dish and very high for the other.
    from recipe_opt_agent.macro_target_suggestions import assess_macro_box_vs_distribution

    stretch = {
        "protein_min": 0.30,
        "protein_max": 0.40,
        "carb_min": 0.30,
        "carb_max": 0.40,
        "fat_min": 0.25,
        "fat_max": 0.35,
    }
    a_assess = assess_macro_box_vs_distribution(stretch, a["distribution"], dish_title="Rice bowl")
    b_assess = assess_macro_box_vs_distribution(stretch, b["distribution"], dish_title="Chicken plate")
    assert a_assess["axes"]["protein"]["status"] in {"slightly_high", "very_high"}
    assert b_assess["axes"]["protein"]["status"] == "typical"

def test_high_protein_targets_from_mean():
    from recipe_opt_agent.macro_target_suggestions import high_protein_targets_from_mean

    hp = high_protein_targets_from_mean((0.18, 0.47, 0.35), pad_pct=2)
    mid = hp["midpoint"]
    # +10pp protein, −5pp carbs/fat → ~0.28 / 0.42 / 0.30
    assert abs(mid["protein"] - 0.28) <= 0.02
    assert abs(mid["carbs"] - 0.42) <= 0.02
    assert abs(mid["fat"] - 0.30) <= 0.02
    box = hp["box"]
    assert box["protein_max"] - box["protein_min"] == pytest.approx(0.04)
    assert box["protein_min"] < box["protein_max"]
    # Feasible simplex intersection
    assert box["protein_max"] + box["carb_max"] + box["fat_max"] >= 1.0 - 1e-9
    assert box["protein_min"] + box["carb_min"] + box["fat_min"] <= 1.0 + 1e-9


def test_rounded_box_pm():
    box = _rounded_box_pm((0.183, 0.476, 0.341), pad_pct=5)
    # 18/48/34 ± 5
    assert box["protein_min"] == 0.13
    assert box["protein_max"] == 0.23
    assert box["carb_min"] == 0.43
    assert box["carb_max"] == 0.53
    assert box["fat_min"] == 0.29
    assert box["fat_max"] == 0.39


def test_validate_macro_box():
    from recipe_opt_web.server import validate_macro_box

    # Valid default box.
    assert validate_macro_box(0.19, 0.23, 0.345, 0.545, 0.245, 0.445) == []
    # Maxes sum to 90% < 100% → infeasible.
    errs = validate_macro_box(0.10, 0.30, 0.10, 0.30, 0.10, 0.30)
    assert any("sum to 90%" in e for e in errs)
    # Mins sum to 120% > 100% → infeasible.
    errs = validate_macro_box(0.40, 0.60, 0.40, 0.60, 0.40, 0.60)
    assert any("120%" in e for e in errs)
    # min > max on one axis.
    errs = validate_macro_box(0.25, 0.20, 0.345, 0.545, 0.245, 0.445)
    assert any("protein min" in e for e in errs)
