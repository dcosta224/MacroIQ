"""Tests for UI score bands, authoritative loss extraction, and OOD protein branch."""

from __future__ import annotations

from recipe_opt_agent.ood_branch import (
    build_ood_add_candidates,
    maybe_build_ood_branch,
    protein_demand_high,
)
from recipe_opt_agent.score_display import (
    band_for_holistic_0_10,
    band_for_loss,
    branch_path_family,
    build_display_scores,
    build_ingredient_display_rows,
    empty_display_scores,
    extract_ratio_and_nutrient,
    select_path_finalists,
)


def test_band_for_loss_thresholds():
    assert band_for_loss(0.0, good_max=0.015, warn_max=0.04) == "good"
    assert band_for_loss(0.02, good_max=0.015, warn_max=0.04) == "warn"
    assert band_for_loss(0.1, good_max=0.015, warn_max=0.04) == "bad"
    assert band_for_loss(None, good_max=0.015, warn_max=0.04) == "unknown"


def test_band_for_holistic():
    assert band_for_holistic_0_10(9) == "good"
    assert band_for_holistic_0_10(6) == "warn"
    assert band_for_holistic_0_10(2) == "bad"


def test_empty_scores_are_blank_not_perfect():
    blank = empty_display_scores()
    assert blank["ready"] is False
    assert blank["ratio_loss"]["value"] is None
    assert blank["ratio_loss"]["band"] == "unknown"
    assert blank["nutrient_loss"]["value"] is None
    assert blank["holistic_0_10"]["value"] is None


def test_missing_ratio_is_not_false_zero():
    """Telemetry without an explicit ratio source must not paint a perfect score."""
    final = {
        "status": "accepted",
        "run_telemetry": {
            "final_ratio_term": 0.0,  # stale / invented zero
            "final_nutrient_slack": 0.0,
        },
        "chosen": {"opt": {"term_losses": {}, "pfc_after": {"protein": 0.2, "carbs": 0.4, "fat": 0.4}}},
        "macro_targets": {
            "protein_min": 0.19,
            "protein_max": 0.23,
            "carb_min": 0.345,
            "carb_max": 0.545,
            "fat_min": 0.245,
            "fat_max": 0.445,
        },
    }
    ratio, ratio_src, nutrient, nutrient_src = extract_ratio_and_nutrient(final)
    assert ratio is None
    assert ratio_src is None
    assert nutrient == 0.0  # inside box → real zero slack
    assert nutrient_src == "pfc_box_slack"
    scores = build_display_scores(final)
    assert scores["ratio_loss"]["value"] is None
    assert scores["ratio_loss"]["band"] == "unknown"
    assert scores["nutrient_loss"]["value"] == 0.0
    assert scores["nutrient_loss"]["band"] == "good"


def test_real_ratio_surrogate_and_nutrient_slack_displayed():
    final = {
        "status": "accepted",
        "title": "HP Carbonara",
        "macro_targets": {
            "protein_min": 0.3,
            "protein_max": 0.4,
            "carb_min": 0.2,
            "carb_max": 0.4,
            "fat_min": 0.2,
            "fat_max": 0.4,
        },
        "chosen": {
            "opt": {
                "term_losses": {"ratio_surrogate": 0.042, "FOODON:pasta": 0.01},
                "pfc_after": {"protein": 0.18, "carbs": 0.5, "fat": 0.32},
                "x_opt": [200.0, 50.0],
            },
            "ingredients": [
                {"label": "spaghetti", "grams": 200},
                {"label": "egg", "grams": 50},
            ],
        },
        "problem": {
            "ingredient_basis": ["FOODON:pasta", "FOODON:egg"],
            "basis_samples": {
                "FOODON:pasta": [0.5, 0.55, 0.6, 0.58, 0.52],
                "FOODON:egg": [0.1, 0.12, 0.11, 0.13, 0.09],
            },
            "total_mass": 250.0,
            "M": [
                [0.12, 0.12],
                [0.01, 0.1],
                [0.7, 0.01],
                [3.5, 1.4],
            ],
            "chosen_recipe": {
                "ingredients": [
                    {"label": "spaghetti", "grams": 200},
                    {"label": "egg", "grams": 50},
                ]
            },
        },
        "foodon_basis_report": {
            "ingredients": [
                {
                    "label": "spaghetti",
                    "foodon_leaf_id": "FOODON:00001234",
                    "foodon_leaf_label": "pasta food product",
                    "basis_node_id": "FOODON:pasta",
                    "basis_node_label": "pasta (basis)",
                    "aggregation_levels": 2,
                },
                {
                    "label": "egg",
                    "foodon_leaf_id": "FOODON:00005678",
                    "foodon_leaf_label": "hen egg",
                    "basis_node_id": "FOODON:egg",
                    "basis_node_label": "egg (basis)",
                    "aggregation_levels": 0,
                },
            ]
        },
        "judge_result": {"winner_id": "a", "holistic_score_0_10": 8.5},
    }
    ratio, ratio_src, nutrient, nutrient_src = extract_ratio_and_nutrient(final)
    # Prefer recomputed median-share deviation when basis samples are present.
    assert ratio_src == "mean_abs_dev_from_median"
    assert ratio is not None and ratio > 0
    assert nutrient is not None and nutrient > 0
    assert nutrient_src == "pfc_box_slack"

    scores = build_display_scores(final)
    assert scores["ratio_loss"]["value"] == ratio
    assert scores["ratio_loss"]["band"] in {"good", "warn", "bad"}
    assert scores["ratio_loss"]["band_summary"]
    assert scores["nutrient_loss"]["value"] == nutrient
    assert scores["nutrient_loss"]["band"] in {"warn", "bad"}
    assert scores["holistic_0_10"]["value"] == 8.5
    assert scores["macros"]["protein"] == 18
    assert scores["macros"]["calories"] is not None
    assert len(scores["ingredients"]) == 2
    pasta = scores["ingredients"][0]
    assert pasta["foodon_leaf_label"] == "pasta food product"
    assert pasta["basis_node_label"] == "pasta (basis)"
    assert pasta["aggregation_levels"] == 2
    assert pasta["loss_contribution"] == 0.01
    assert pasta["calories"] is not None and pasta["calories"] > 0
    assert pasta["share_iqr"] is not None
    assert pasta["share_iqr"]["median"] is not None
    assert pasta["recipe_share"] is not None
    assert pasta["amount_display"].endswith("g") or pasta["amount_unit"]


def test_share_level_keys_are_not_used_as_ratio_loss():
    final = {
        "chosen": {
            "opt": {
                "term_losses": {
                    "FOODON:pasta__share": 0.55,
                    "FOODON:egg__share": 0.12,
                }
            }
        },
        "run_telemetry": {"final_ratio_term": 0.0},
    }
    ratio, src, _, _ = extract_ratio_and_nutrient(final)
    assert ratio is None
    assert src is None


def test_build_ingredient_display_rows_iqr_bounds():
    rows = build_ingredient_display_rows(
        ingredients=[{"label": "pasta", "grams": 100}],
        problem={
            "ingredient_basis": ["n1"],
            "basis_samples": {"n1": [0.1, 0.2, 0.3, 0.4, 0.5]},
            "total_mass": 100.0,
            "M": [[0.1], [0.05], [0.7], [4.0]],
            "x_opt": [100.0],
        },
        opt={"x_opt": [100.0], "term_losses": {"n1": 0.08}},
        foodon_report={
            "ingredients": [
                {
                    "label": "pasta",
                    "foodon_leaf_label": "pasta",
                    "basis_node_id": "n1",
                    "basis_node_label": "basis pasta",
                    "aggregation_levels": 1,
                }
            ]
        },
    )
    assert len(rows) == 1
    assert rows[0]["share_iqr"]["min"] == 0.1
    assert rows[0]["share_iqr"]["max"] == 0.5
    assert rows[0]["recipe_share"] == 1.0
    # share 1.0 is beyond Tukey fences for [0.1..0.5] → real outlier (red)
    assert rows[0]["loss_band"] == "bad"


def test_iqr_stats_uses_p15_p85_band():
    from recipe_opt_agent.score_display import _iqr_stats

    # 20 evenly spaced shares → P15/P85 should sit near 0.15/0.85 of the range.
    samples = [i / 19.0 for i in range(20)]
    stats = _iqr_stats(samples)
    assert stats is not None
    assert abs(stats["q1"] - stats["p15"]) < 1e-12
    assert abs(stats["q3"] - stats["p85"]) < 1e-12
    assert stats["p15"] < 0.25
    assert stats["p85"] > 0.75
    assert stats["p15"] < stats["median"] < stats["p85"]


def test_share_band_from_iqr_inside_outside_outlier():
    from recipe_opt_agent.score_display import share_band_from_iqr

    iqr = {"q1": 0.2, "q3": 0.4, "n": 12}  # width 0.2 → fences [-0.1, 0.7]
    assert share_band_from_iqr(0.3, iqr) == "good"
    assert share_band_from_iqr(0.15, iqr) == "warn"
    assert share_band_from_iqr(0.55, iqr) == "warn"
    assert share_band_from_iqr(0.8, iqr) == "bad"
    # On/near the band edge within 0.01 stays green.
    assert share_band_from_iqr(0.2, iqr) == "good"
    assert share_band_from_iqr(0.4, iqr) == "good"
    assert share_band_from_iqr(0.191, iqr) == "good"
    assert share_band_from_iqr(0.409, iqr) == "good"
    assert share_band_from_iqr(0.189, iqr) == "warn"
    assert share_band_from_iqr(0.411, iqr) == "warn"
    assert share_band_from_iqr(0.3, {**iqr, "n": 3}) == "unknown"
    assert share_band_from_iqr(None, iqr) == "unknown"


def test_protein_demand_triggers():
    ok, reason = protein_demand_high(protein_min=0.35)
    assert ok and "protein_min" in reason
    ok2, _ = protein_demand_high(
        protein_min=0.2,
        requirement_tags=[{"tag_id": "high_protein"}],
    )
    assert ok2
    no, _ = protein_demand_high(protein_min=0.2, pfc_after={"protein": 0.22})
    assert not no


def test_ood_candidates_have_macros():
    cands = build_ood_add_candidates(
        box_dict={
            "protein_min": 0.3,
            "protein_max": 0.4,
            "carb_min": 0.2,
            "carb_max": 0.4,
            "fat_min": 0.2,
            "fat_max": 0.4,
        }
    )
    assert any("chicken" in c["label"].lower() for c in cands)
    assert all((c.get("meta") or {}).get("macros_per_g") for c in cands)


def test_maybe_build_ood_branch_when_needed():
    problem = {
        "x0": [200.0, 50.0],
        "M": [[0.05, 0.25], [0.01, 0.33], [0.25, 0.01], [1.3, 4.0]],
        "ingredient_basis": ["pasta", "cheese"],
        "basis_samples": {"pasta": [0.5, 0.55], "cheese": [0.1, 0.12]},
        "ratio_samples": [4.0, 5.0],
        "marginal_nodes": ["pasta", "cheese"],
        "total_mass": 250.0,
        "kcal_target": 500.0,
        "chosen_recipe": {
            "ingredients": [
                {"label": "pasta", "grams": 200, "fdc_id": 1},
                {"label": "cheese", "grams": 50, "fdc_id": 2},
            ]
        },
        "retrieval_context": {"fdc_macros": {}, "fdc_catalog": []},
    }
    box = {
        "protein_min": 0.32,
        "protein_max": 0.4,
        "carb_min": 0.2,
        "carb_max": 0.5,
        "fat_min": 0.2,
        "fat_max": 0.4,
    }
    info = maybe_build_ood_branch(
        problem,
        box_dict=box,
        diagnosis={"binding_macros": ["protein_min"]},
        opt={"pfc_after": {"protein": 0.18, "carbs": 0.5, "fat": 0.32}},
        requirement_tags=[],
        id_bundles=[],
    )
    assert info["needed"] is True
    assert info["ood_candidates"]
    assert any(c.get("branch") == "ood_protein" for c in info["ood_candidates"])


def test_branch_path_family_mapping():
    assert branch_path_family("in_distribution") == "in_distribution"
    assert branch_path_family("moderate") == "in_distribution"
    assert branch_path_family("ood_protein") == "ood"
    assert branch_path_family("hybrid") == "ood"


def test_select_path_finalists_picks_id_and_ood_champions():
    state = {
        "config": {
            "protein_min": 0.2,
            "protein_max": 0.4,
            "carb_min": 0.2,
            "carb_max": 0.5,
            "fat_min": 0.2,
            "fat_max": 0.4,
        },
        "title": "Carbonara",
        "interesting_candidates": [
            {
                "candidate_id": "bundle::id_swap",
                "branch": "in_distribution",
                "delta_L_star": -0.04,
                "edits": [{"action": "swap", "label": "whole wheat pasta"}],
                "ingredients": [{"label": "whole wheat pasta", "grams": 180}],
                "opt": {
                    "pfc_after": {"protein": 0.22, "carbs": 0.45, "fat": 0.33},
                    "term_losses": {"ratio_surrogate": 0.018},
                },
            },
            {
                "candidate_id": "bundle::ood_chicken",
                "branch": "ood_protein",
                "delta_L_star": -0.03,
                "edits": [{"action": "add", "label": "chicken breast"}],
                "ingredients": [
                    {"label": "pasta", "grams": 160},
                    {"label": "chicken breast", "grams": 90},
                ],
                "opt": {
                    "pfc_after": {"protein": 0.34, "carbs": 0.4, "fat": 0.26},
                    "term_losses": {"ratio_surrogate": 0.055},
                },
                "nutrient_slack": 0.0,
            },
        ],
        "chosen_recipe": {"ingredients": [{"label": "pasta", "grams": 200}]},
        "problem": {
            "ingredient_basis": ["n1"],
            "basis_samples": {"n1": [0.5, 0.55, 0.6, 0.58, 0.52]},
            "total_mass": 200.0,
            "M": [[0.1], [0.05], [0.7], [4.0]],
        },
    }
    paths = select_path_finalists(state, ood_handicap=0.015)
    assert paths["in_distribution"] is not None
    assert paths["ood"] is not None
    assert paths["in_distribution"]["path_label"] == "In-distribution"
    assert paths["ood"]["path_label"] == "OOD"
    assert paths["in_distribution"]["chosen"]["candidate_id"] == "bundle::id_swap"
    assert paths["ood"]["chosen"]["candidate_id"] == "bundle::ood_chicken"
    assert paths["in_distribution"]["display_scores"]["ratio_loss"]["value"] == 0.018
    assert paths["ood"]["display_scores"]["ratio_loss"]["value"] == 0.055
