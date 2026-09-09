"""Unit tests for FoodOn leaf→basis aggregation + hit-count reports."""

from __future__ import annotations

from recipe_opt_agent.foodon_basis_report import (
    aggregation_levels,
    attach_foodon_basis_report,
    basis_hit_counts,
    build_foodon_basis_report,
)


def test_aggregation_levels_along_chain():
    chains = {
        "LEAF_A": ["LEAF_A", "MID", "BASIS", "ROOT"],
        "LEAF_B": ["LEAF_B", "BASIS"],
    }
    assert aggregation_levels("LEAF_A", "BASIS", chains) == 2
    assert aggregation_levels("LEAF_B", "BASIS", chains) == 1
    assert aggregation_levels("BASIS", "BASIS", {"BASIS": ["BASIS"]}) == 0
    assert aggregation_levels("LEAF_A", "MISSING", chains) is None
    assert aggregation_levels(None, "BASIS", chains) is None


def test_basis_hit_counts_from_samples():
    problem = {
        "basis_samples": {"N1": [0.1, 0.2, 0.3], "N2": [0.4]},
        "basis_nodes": ["N1", "N2", "N3"],
    }
    hits = basis_hit_counts(problem)
    assert hits["N1"] == 3
    assert hits["N2"] == 1
    assert hits["N3"] == 0


def test_build_foodon_basis_report_ingredient_and_basis_rows():
    problem = {
        "n_matches": 12,
        "basis_nodes": ["BASIS_EGG", "BASIS_PASTA", "BASIS_UNUSED"],
        "basis_samples": {
            "BASIS_EGG": [0.1] * 8,
            "BASIS_PASTA": [0.5] * 10,
            "BASIS_UNUSED": [0.2] * 3,
        },
        "ingredient_basis": ["BASIS_EGG", "BASIS_PASTA", "BASIS_EGG"],
        "ingredient_foodon_leaves": ["LEAF_YOLK", "LEAF_SPAG", "LEAF_WHOLE"],
        "rollup_chains": {
            "LEAF_YOLK": ["LEAF_YOLK", "MID_EGG", "BASIS_EGG"],
            "LEAF_SPAG": ["LEAF_SPAG", "BASIS_PASTA"],
            "LEAF_WHOLE": ["LEAF_WHOLE", "BASIS_EGG"],
        },
        "chosen_recipe": {
            "ingredients": [
                {"label": "egg yolk", "fdc_id": 1},
                {"label": "spaghetti", "fdc_id": 2},
                {"label": "egg", "fdc_id": 3},
            ]
        },
        "build_params": {"adaptive_min_basis_hits": 5},
    }
    report = build_foodon_basis_report(problem)
    assert report["n_ingredients"] == 3
    assert report["n_aggregated"] == 3  # all levels > 0
    assert report["n_unmapped"] == 0
    assert report["min_basis_hits_target"] == 5
    by_label = {r["label"]: r for r in report["ingredients"]}
    assert by_label["egg yolk"]["aggregation_levels"] == 2
    assert by_label["spaghetti"]["aggregation_levels"] == 1
    assert by_label["egg"]["aggregation_levels"] == 1
    assert len(report["aggregated_ingredients"]) == 3

    egg = next(r for r in report["basis_nodes"] if r["node_id"] == "BASIS_EGG")
    pasta = next(r for r in report["basis_nodes"] if r["node_id"] == "BASIS_PASTA")
    unused = next(r for r in report["basis_nodes"] if r["node_id"] == "BASIS_UNUSED")
    assert egg["n_hits"] == 8
    assert egg["in_current_recipe"] is True
    assert egg["n_ingredients_mapped"] == 2
    assert pasta["n_hits"] == 10
    assert unused["in_current_recipe"] is False
    assert unused["n_hits"] == 3


def test_attach_foodon_basis_report_mutates_problem_and_chosen():
    problem = {
        "ingredient_basis": ["B"],
        "ingredient_foodon_leaves": ["L"],
        "rollup_chains": {"L": ["L", "B"]},
        "basis_samples": {"B": [0.2, 0.3]},
        "basis_nodes": ["B"],
        "chosen_recipe": {"ingredients": [{"label": "x"}]},
    }
    report = attach_foodon_basis_report(problem)
    assert report is not None
    assert problem["foodon_basis_report"]["n_aggregated"] == 1
    assert problem["chosen_recipe"]["foodon_basis_report"]["ingredients"][0]["aggregation_levels"] == 1
