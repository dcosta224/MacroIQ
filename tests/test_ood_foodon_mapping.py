"""Tests for OOD FoodOn mapping, embedding preload hook, and ingredient losses."""

from __future__ import annotations

from recipe_opt_agent.ood_foodon import (
    annotate_candidate_foodon,
    lookup_foodon_for_label,
)
from recipe_opt_agent.score_display import build_ingredient_display_rows, extract_ratio_and_nutrient


def test_chicken_maps_to_real_foodon_not_ood_placeholder():
    hit = lookup_foodon_for_label("Chicken breast, raw, skinless")
    assert hit is not None
    assert str(hit["foodon_leaf_id"]).startswith("FOODON_")
    assert hit["foodon_leaf_id"] != "ood_lean_protein"

    problem = {
        "basis_nodes": [],
        "ingredient_basis": [],
        "rollup_chains": {},
        "chosen_recipe": {"ingredients": []},
        "retrieval_context": {},
    }
    cand = annotate_candidate_foodon(
        {
            "label": "Chicken breast, raw, skinless",
            "fdc_id": 900001,
            "branch": "ood_protein",
            "meta": {"basis_node": "ood_lean_protein", "ood": True},
        },
        problem,
    )
    meta = cand["meta"]
    assert meta.get("basis_node") != "ood_lean_protein"
    assert str(meta.get("basis_node") or "").startswith("FOODON_")
    assert str(meta.get("foodon_leaf_id") or "").startswith("FOODON_")


def test_ingredient_rows_compute_share_loss_when_term_missing():
    rows = build_ingredient_display_rows(
        ingredients=[{"label": "chicken", "grams": 80}],
        problem={
            "ingredient_basis": ["FOODON_02020280"],
            "basis_samples": {"FOODON_02020280": [0.05, 0.08, 0.1, 0.12, 0.15]},
            "total_mass": 400.0,
            "M": [[0.3], [0.03], [0.0], [1.6]],
            "x_opt": [80.0],
        },
        opt={"x_opt": [80.0], "term_losses": {}},  # no per-node term
        foodon_report={
            "ingredients": [
                {
                    "label": "chicken",
                    "foodon_leaf_id": "FOODON_02020280",
                    "foodon_leaf_label": "chicken breast",
                    "basis_node_id": "FOODON_02020280",
                    "basis_node_label": "chicken breast",
                    "aggregation_levels": 0,
                }
            ]
        },
    )
    assert len(rows) == 1
    assert rows[0]["loss_contribution"] is not None
    assert rows[0]["loss_contribution"] > 0
    assert rows[0]["loss_label"] == "ratio loss"
    assert rows[0]["basis_node_id"] == "FOODON_02020280"


def test_empty_ratio_samples_do_not_report_perfect_zero():
    ratio, src, *_ = extract_ratio_and_nutrient(
        {
            "chosen": {"opt": {"term_losses": {"ratio_surrogate": 0.0}}},
            "problem": {"ratio_samples": []},
            "macro_targets": {
                "protein_min": 0.19,
                "protein_max": 0.23,
                "carb_min": 0.3,
                "carb_max": 0.5,
                "fat_min": 0.2,
                "fat_max": 0.4,
            },
        }
    )
    assert ratio is None
    assert src is None
