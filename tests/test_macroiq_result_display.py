"""Portion display + MacroIQ result payload helpers."""

from __future__ import annotations

from recipe_opt_agent.problem_loader import _ingredient_rows
from recipe_opt_agent.score_display import (
    build_display_scores,
    scaled_portion_amount,
)
import pandas as pd


def test_scaled_portion_from_source_units():
    row = {
        "quantity": 2.0,
        "unit": "tablespoon",
        "original_grams": 27.0,
        "label": "olive oil",
    }
    out = scaled_portion_amount(row, grams=40.5)
    assert out["amount_source"] == "scaled_portion"
    assert abs(out["amount_value"] - 3.0) < 1e-9
    assert "tablespoon" in out["amount_display"]


def test_scaled_portion_falls_back_to_rounded_grams():
    out = scaled_portion_amount({"label": "pasta"}, grams=127.4)
    assert out["amount_source"] == "grams"
    assert out["amount_display"] == "127 g"


def test_ingredient_rows_preserve_quantity_unit():
    df = pd.DataFrame(
        [
            {
                "ingredient_idx": 0,
                "fdc_description": "Oil, olive",
                "ingredient": "1 tablespoon olive oil",
                "fdc_id": 171413,
                "gram_weight": 13.5,
                "quantity": 1.0,
                "unit": "tablespoon",
                "portion_label": "tbsp",
                "foodon_id": None,
            }
        ]
    )
    rows = _ingredient_rows(df)
    assert rows[0]["quantity"] == 1.0
    assert rows[0]["unit"] == "tablespoon"
    assert rows[0]["original_grams"] == 13.5


def test_display_scores_macros_and_labels():
    payload = {
        "opt": {
            "pfc_after": {"protein": 0.21, "carb": 0.44, "fat": 0.35},
            "x_opt": [100.0, 50.0],
            "term_losses": {"ratio_surrogate": 0.01},
        },
        "problem": {
            "M": [
                [0.2, 0.1],
                [0.1, 0.3],
                [0.05, 0.02],
                [1.5, 2.0],
            ],
            "ingredient_basis": ["a", "b"],
            "chosen_recipe": {
                "ingredients": [
                    {
                        "label": "FETTUCCINE PASTA",
                        "grams": 100,
                        "original_grams": 80,
                        "quantity": 8,
                        "unit": "ounce",
                    },
                    {
                        "label": "Butter, without salt",
                        "grams": 50,
                        "original_grams": 50,
                        "quantity": 4,
                        "unit": "tablespoon",
                    },
                ]
            },
        },
        "chosen": {},
    }
    scores = build_display_scores(payload)
    assert scores["ingredients"][0]["label"] == "FETTUCCINE PASTA"
    assert scores["ingredients"][0]["amount_source"] == "scaled_portion"
    assert "ounce" in scores["ingredients"][0]["amount_display"]
    assert scores["macros"] == {
        "protein": 21,
        "carb": 44,
        "fat": 35,
        "calories": scores["macros"]["calories"],
    }
    assert scores["macros"]["calories"] is not None
    assert scores["ratio_loss"]["band_summary"]


def test_progress_detail_edits_and_loss_improvements():
    from recipe_opt_agent.portion_display import build_progress_detail

    decide = build_progress_detail(
        "decide",
        {
            "decision": {
                "action": "apply_bundle",
                "edits": [
                    {"action": "add", "label": "Mayonnaise, regular"},
                    {"action": "swap", "label": "Turkey", "replace_label": "Ham, sliced"},
                ],
            }
        },
    )
    assert decide == "added mayonnaise; swapped ham"

    apply = build_progress_detail(
        "apply",
        {
            "last_applied_candidate": {
                "action": "add",
                "label": "Greek yogurt",
            }
        },
    )
    assert apply == "added greek yogurt"

    diagnose = build_progress_detail(
        "diagnose",
        {
            "decision_outcomes": [
                {
                    "decision": {
                        "edits": [{"action": "add", "label": "Mayonnaise, regular"}],
                    }
                }
            ],
            "score_history": [
                {"source": "diagnose", "ratio_loss": 0.20, "nutrient_loss": 0.10},
                {"source": "diagnose", "ratio_loss": 0.10, "nutrient_loss": 0.05},
            ],
            "live_scores": {
                "ratio_loss": {"value": 0.10},
                "nutrient_loss": {"value": 0.05},
            },
        },
    )
    assert "added mayonnaise" in diagnose
    assert "Improved cookability by 50%" in diagnose
    assert "Improved nutrient fit by 50%" in diagnose


def test_progress_detail_never_reports_regression():
    from recipe_opt_agent.portion_display import build_progress_detail

    detail = build_progress_detail(
        "diagnose",
        {
            "score_history": [
                {"source": "diagnose", "ratio_loss": 0.05, "nutrient_loss": 0.02},
                {"source": "diagnose", "ratio_loss": 0.12, "nutrient_loss": 0.08},
            ],
            "live_scores": {
                "ratio_loss": {"value": 0.12},
                "nutrient_loss": {"value": 0.08},
            },
        },
    )
    assert detail is None


def test_usda_portion_prefers_count_over_volume(monkeypatch):
    import sys
    import types

    from recipe_opt_agent import portion_display as pd

    stub = types.ModuleType("portion_gram")
    stub.classify_food_portion_row = lambda row: row["_kind"]
    monkeypatch.setitem(sys.modules, "portion_gram", stub)

    out = pd.kitchen_amount_from_usda_portions(
        60.0,
        [
            {
                "_kind": "volume",
                "gram_weight": 15.0,
                "amount": 1.0,
                "measure_unit_name": "tbsp",
                "id": 1,
            },
            {
                "_kind": "count",
                "gram_weight": 30.0,
                "amount": 1.0,
                "measure_unit_name": "piece",
                "id": 2,
            },
            {
                "_kind": "mass",
                "gram_weight": 28.0,
                "amount": 1.0,
                "measure_unit_name": "oz",
                "id": 3,
            },
        ],
    )
    assert out is not None
    assert out["amount_source"] == "usda_count"
    assert out["amount_unit"] == "piece"
    assert abs(out["amount_value"] - 2.0) < 1e-9


def test_consolidate_duplicate_ingredients_sums_portions():
    from recipe_opt_agent.portion_display import consolidate_duplicate_ingredients

    rows = [
        {
            "label": "Egg, whole, raw",
            "fdc_id": 171442,
            "grams": 50.0,
            "amount_value": 1.0,
            "amount_unit": "large",
            "amount_display": "1 large",
            "amount_source": "usda_count",
            "portion_gram_weight": 50.0,
            "recipe_share": 0.1,
            "calories": 70,
        },
        {
            "label": "Egg, whole, raw",
            "fdc_id": 171442,
            "grams": 50.0,
            "amount_value": 1.0,
            "amount_unit": "large",
            "amount_display": "1 large",
            "amount_source": "usda_count",
            "portion_gram_weight": 50.0,
            "recipe_share": 0.1,
            "calories": 70,
        },
        {
            "label": "Salt, table",
            "fdc_id": 173468,
            "grams": 2.0,
            "amount_value": 2.0,
            "amount_unit": "g",
            "amount_display": "2 g",
            "amount_source": "grams",
            "recipe_share": 0.01,
        },
    ]
    problem = {
        "M": [
            [0.12, 0.12, 0.0],
            [0.01, 0.01, 0.0],
            [0.1, 0.1, 0.0],
            [1.4, 1.4, 0.0],
        ],
        "ingredient_basis": ["egg", "egg", "salt"],
        "x0": [50.0, 50.0, 2.0],
    }
    merged, problem_out = consolidate_duplicate_ingredients(rows, problem=problem)
    assert len(merged) == 2
    egg = next(r for r in merged if r["fdc_id"] == 171442)
    assert abs(egg["grams"] - 100.0) < 1e-9
    assert egg["amount_unit"] == "large"
    assert abs(egg["amount_value"] - 2.0) < 1e-9
    assert "2" in egg["amount_display"]
    assert abs(egg["recipe_share"] - 0.2) < 1e-9
    assert egg["calories"] == 140
    assert egg["merged_from_count"] == 2
    assert problem_out is not None
    assert len(problem_out["x0"]) == 2
    assert abs(problem_out["x0"][0] - 100.0) < 1e-9
    assert len(problem_out["M"][0]) == 2


def test_prepare_browse_candidates_caps_at_four():
    from recipe_opt_agent.score_display import prepare_browse_candidates

    alts = []
    for i in range(6):
        alts.append(
            {
                "candidate_id": f"c{i}",
                "branch": "in_distribution",
                "ingredients": [{"label": f"food-{i}", "grams": 10 + i}],
                "opt": {
                    "pfc_after": {"protein": 0.2, "carbs": 0.5, "fat": 0.3},
                    "x_opt": [10.0 + i],
                    "term_losses": {"ratio_surrogate": 0.05 + 0.01 * i},
                },
            }
        )
    payload = {
        "chosen": {"source": "x", "entry": {"candidate_id": "c0", "branch": "in_distribution"}},
        "display_scores": {
            "macros": {"protein": 20, "carb": 50, "fat": 30, "calories": 700},
            "ingredients": [
                {
                    "label": "food-0",
                    "grams": 10,
                    "calories": 350,
                    "recipe_share": 0.5,
                    "share_iqr": {"q1": 0.4, "q3": 0.6},
                }
            ],
            "ratio_loss": {"value": 0.08, "band": "warn", "outside_iqr_calorie_frac": 0.5},
            "nutrient_loss": {"value": 0.0, "band": "good"},
        },
        "problem": {
            "M": [[0.2], [0.1], [0.05], [1.0]],
            "ingredient_basis": ["a"],
            "x0": [10.0],
            "kcal_target": 700,
            "user_kcal_target": 700,
            "chosen_recipe": {"ingredients": [{"label": "food-0", "grams": 10}]},
        },
        "macro_targets": {
            "protein_min": 0.15,
            "protein_max": 0.35,
            "carb_min": 0.3,
            "carb_max": 0.6,
            "fat_min": 0.15,
            "fat_max": 0.4,
        },
        "config": {
            "protein_min": 0.15,
            "protein_max": 0.35,
            "carb_min": 0.3,
            "carb_max": 0.6,
            "fat_min": 0.15,
            "fat_max": 0.4,
            "kcal_target": 700,
        },
        "alternatives": alts[1:],
        "scored_finalists": alts,
        "path_finals": {
            "in_distribution": {
                "display_scores": {
                    "macros": {"protein": 22, "carb": 48, "fat": 30, "calories": 700},
                    "ingredients": [
                        {
                            "label": "path-food",
                            "grams": 12,
                            "calories": 700,
                            "recipe_share": 0.5,
                            "share_iqr": {"q1": 0.4, "q3": 0.6},
                        }
                    ],
                    "ratio_loss": {"value": 0.02, "band": "good", "outside_iqr_calorie_frac": 0.0},
                    "nutrient_loss": {"value": 0.0, "band": "good"},
                },
                "problem": {"kcal_target": 700},
                "macro_targets": {},
            }
        },
        "judge_result": {"winner_id": "c0", "rationale": "c0 keeps identity with a cleaner macro box."},
    }
    cards = prepare_browse_candidates(payload, {"config": payload["config"]}, max_candidates=4)
    assert len(cards) <= 4
    # Best proportion quality first — path_in_distribution beats arbiter winner c0
    assert cards[0]["is_recommended"] is True
    assert cards[0]["candidate_id"] == "path_in_distribution"
    assert cards[0]["title"] == "Recommended"
    outs = [
        c["score_summary"].get("outside_iqr_calorie_frac")
        if c["score_summary"].get("outside_iqr_calorie_frac") is not None
        else 1.0 - float(c["score_summary"].get("iqr_in_band_frac") or 0)
        for c in cards
    ]
    assert outs == sorted(outs)


def test_prepare_browse_candidates_prefers_proportion_then_ratio():
    from recipe_opt_agent.score_display import prepare_browse_candidates

    def _card_display(ratio, shares_in_iqr, calories=700, *, kcal_per=100.0):
        ings = []
        for i in range(4):
            share = 0.5 if i < shares_in_iqr else 0.01
            ings.append(
                {
                    "label": f"ing-{i}",
                    "grams": 10,
                    "calories": kcal_per,
                    "basis_node_id": f"b{i}",
                    "recipe_share": share,
                    "share_iqr": {"q1": 0.4, "q3": 0.6},
                }
            )
        outside = max(0, 4 - shares_in_iqr) / 4.0
        return {
            "macros": {"protein": 20, "carb": 50, "fat": 30, "calories": calories},
            "ingredients": ings,
            "ratio_loss": {
                "value": ratio,
                "band": "good",
                "outside_iqr_calorie_frac": outside,
            },
            "nutrient_loss": {"value": 0.0, "band": "good"},
        }

    payload = {
        "chosen": {"source": "x", "entry": {"candidate_id": "bad", "branch": "in_distribution"}},
        "display_scores": _card_display(0.09, 1, calories=2100),
        "problem": {"kcal_target": 700, "user_kcal_target": 700, "M": [], "x0": [], "chosen_recipe": {"ingredients": []}},
        "config": {"kcal_target": 700},
        "macro_targets": {},
        "path_finals": {
            "in_distribution": {
                "display_scores": _card_display(0.03, 2, calories=700),
                "problem": {"kcal_target": 700},
            },
            "ood": {
                "display_scores": _card_display(0.03, 4, calories=700),
                "problem": {"kcal_target": 700},
            },
        },
        "judge_result": {"winner_id": "bad"},
    }
    cards = prepare_browse_candidates(payload, {"config": payload["config"]}, max_candidates=3)
    assert [c["candidate_id"] for c in cards] == [
        "path_ood",
        "path_in_distribution",
        "bad",
    ]
    assert cards[0]["score_summary"]["iqr_in_band_frac"] == 1.0
    assert cards[1]["score_summary"]["iqr_in_band_frac"] == 0.5
    assert cards[0]["score_summary"]["outside_iqr_calorie_frac"] == 0.0
    assert cards[1]["score_summary"]["outside_iqr_calorie_frac"] == 0.5


def test_prepare_browse_candidates_demotes_weird_last():
    from recipe_opt_agent.score_display import prepare_browse_candidates

    def _card_display(ratio, outside_cal, calories=700):
        return {
            "macros": {"protein": 20, "carb": 50, "fat": 30, "calories": calories},
            "ingredients": [
                {
                    "label": "ing",
                    "grams": 10,
                    "calories": 700,
                    "basis_node_id": "b0",
                    "recipe_share": 0.5,
                    "share_iqr": {"q1": 0.4, "q3": 0.6},
                }
            ],
            "ratio_loss": {
                "value": ratio,
                "band": "good",
                "outside_iqr_calorie_frac": outside_cal,
            },
            "nutrient_loss": {"value": 0.0, "band": "good"},
        }

    payload = {
        "chosen": {"source": "x", "entry": {"candidate_id": "weird_best_props", "branch": "in_distribution"}},
        "display_scores": _card_display(0.01, 0.0),
        "problem": {"kcal_target": 700, "user_kcal_target": 700, "M": [], "x0": [], "chosen_recipe": {"ingredients": []}},
        "config": {"kcal_target": 700},
        "macro_targets": {},
        "weird_candidate_ids": ["weird_best_props"],
        "weird_flags": {
            "weird_best_props": {
                "is_weird": True,
                "odd_ingredients": ["maple syrup"],
                "note": "Syrup does not belong here",
            }
        },
        "path_finals": {
            "in_distribution": {
                "display_scores": _card_display(0.05, 0.2),
                "problem": {"kcal_target": 700},
            },
            "ood": {
                "display_scores": _card_display(0.04, 0.1),
                "problem": {"kcal_target": 700},
            },
        },
    }
    cards = prepare_browse_candidates(payload, {"config": payload["config"]}, max_candidates=3)
    assert [c["candidate_id"] for c in cards] == [
        "path_ood",
        "path_in_distribution",
        "weird_best_props",
    ]
    assert cards[0]["is_weird"] is False
    assert cards[-1]["is_weird"] is True
    assert "unusual ingredients" in cards[-1]["title"].lower()


def test_weird_ids_from_judgment_clash_verdicts():
    from recipe_opt_agent.final_arbiter import weird_ids_from_judgment

    flags = weird_ids_from_judgment(
        {
            "verdicts": {
                "a": {"culinary_fit": "canonical", "odd_ingredients": []},
                "b": {"culinary_fit": "clash", "odd_ingredients": ["ricotta"], "note": "soft dairy clash"},
                "c": {"culinary_fit": "plausible_extension", "odd_ingredients": ["chicken"]},
            }
        }
    )
    assert set(flags) == {"b"}
    assert flags["b"]["odd_ingredients"] == ["ricotta"]


def test_filter_omits_zero_portion_without_distribution():
    from recipe_opt_agent.score_display import (
        filter_display_ingredients,
        proportion_typicality_from_ingredients,
        should_include_display_ingredient,
    )

    keep = {
        "label": "Pasta",
        "grams": 100,
        "calories": 350,
        "basis_node_id": "pasta",
        "recipe_share": 0.5,
        "share_iqr": {"q1": 0.4, "q3": 0.6, "n": 12},
    }
    omit_zero_no_iqr = {
        "label": "Ghost spice",
        "grams": 0,
        "calories": 0,
        "basis_node_id": "spice",
        "recipe_share": 0.0,
        "share_iqr": None,
    }
    omit_zero_sparse = {
        "label": "Trace curry",
        "grams": 0,
        "amount_value": 0,
        "basis_node_id": "curry",
        "recipe_share": 0.0,
        "share_iqr": {"q1": 0.0, "q3": 0.01, "n": 2},
    }
    keep_grey_but_mass = {
        "label": "Spices, curry powder",
        "grams": 16.5,
        "calories": 50,
        "basis_node_id": "curry",
        "recipe_share": 0.02,
        "share_iqr": {"q1": 0.0, "q3": 0.01, "n": 3},  # sparse → grey, but keep
    }
    assert should_include_display_ingredient(keep) is True
    assert should_include_display_ingredient(omit_zero_no_iqr) is False
    assert should_include_display_ingredient(omit_zero_sparse) is False
    assert should_include_display_ingredient(keep_grey_but_mass) is True

    filtered = filter_display_ingredients(
        [keep, omit_zero_no_iqr, omit_zero_sparse, keep_grey_but_mass]
    )
    assert [r["label"] for r in filtered] == ["Pasta", "Spices, curry powder"]

    # Zero / no-distribution lines must not pull the proportion band.
    tip = proportion_typicality_from_ingredients(
        [keep, omit_zero_no_iqr, omit_zero_sparse]
    )
    assert tip["known_count"] == 1
    assert tip["key"] == "very_typical"


def test_proportion_typicality_from_outside_iqr_share():
    from recipe_opt_agent.score_display import proportion_typicality_from_ingredients

    def _rows(n_out, n_in, *, out_cal=100.0, in_cal=100.0, invalid_iqr=0):
        rows = []
        for i in range(n_in):
            rows.append(
                {
                    "basis_node_id": f"in_{i}",
                    "recipe_share": 0.5,
                    "calories": in_cal,
                    "share_iqr": {"q1": 0.4, "q3": 0.6, "n": 12},
                }
            )
        for i in range(n_out):
            rows.append(
                {
                    "basis_node_id": f"out_{i}",
                    "recipe_share": 0.05,
                    "calories": out_cal,
                    "share_iqr": {"q1": 0.4, "q3": 0.6, "n": 12},
                }
            )
        for i in range(invalid_iqr):
            rows.append(
                {
                    "basis_node_id": f"bad_{i}",
                    "recipe_share": 0.01,
                    "calories": 500.0,
                    "loss_band": "unknown",
                    "share_iqr": {"q1": 0.4, "q3": 0.6, "n": 2},
                }
            )
        return rows

    assert proportion_typicality_from_ingredients(_rows(0, 10))["key"] == "very_typical"
    # ~11% calories outside → still very typical (<17%)
    assert proportion_typicality_from_ingredients(_rows(1, 9))["key"] == "very_typical"
    # 20% → mostly typical (17–25%)
    assert proportion_typicality_from_ingredients(_rows(2, 8))["key"] == "mostly_typical"
    # 30% → somewhat different (25–35%)
    assert proportion_typicality_from_ingredients(_rows(3, 7))["key"] == "somewhat_different"
    # Invalid / sparse IQR rows are ignored even if calorie-heavy / far outside.
    assert (
        proportion_typicality_from_ingredients(_rows(0, 10, invalid_iqr=5))["key"]
        == "very_typical"
    )
    # Calorie-weighted only: many outside ingredients with tiny calories stay typical.
    assert (
        proportion_typicality_from_ingredients(_rows(4, 6, out_cal=10.0, in_cal=100.0))["key"]
        == "very_typical"
    )
    # 40% of calories outside → substantially off (≥35%).
    assert (
        proportion_typicality_from_ingredients(_rows(4, 6, out_cal=100.0, in_cal=100.0))["key"]
        == "substantially_off"
    )

    # Ginger-scale spice: 60% of *ingredients* outside but ~2% of calories → still very typical.
    spice_rows = _rows(0, 2, in_cal=350.0)
    for i in range(3):
        spice_rows.append(
            {
                "basis_node_id": f"spice_{i}",
                "recipe_share": 0.01,
                "calories": 2.0 + i,
                "share_iqr": {"q1": 0.02, "q3": 0.08, "n": 12},
            }
        )
    tip = proportion_typicality_from_ingredients(spice_rows)
    assert tip["key"] == "very_typical"
    assert tip["outside_iqr_calorie_pct"] < 17

    # Sparse neighbor lines must not count against proportion fit.
    sparse_heavy = _rows(0, 2, in_cal=100.0)
    sparse_heavy.append(
        {
            "basis_node_id": "sparse_spice",
            "recipe_share": 0.9,
            "calories": 500.0,
            "grams": 20.0,
            "share_iqr": {"q1": 0.0, "q3": 0.05, "n": 3},
            "loss_band": "unknown",
        }
    )
    sparse_tip = proportion_typicality_from_ingredients(sparse_heavy)
    assert sparse_tip["known_count"] == 2
    assert sparse_tip["key"] == "very_typical"

    # Same basis on multiple lines must not be triple-counted.
    dup = [
        {
            "basis_node_id": "noodle",
            "recipe_share": 0.45,
            "calories": 200.0,
            "share_iqr": {"q1": 0.3, "q3": 0.5, "n": 12},
        },
        {
            "basis_node_id": "noodle",
            "recipe_share": 0.45,
            "calories": 150.0,
            "share_iqr": {"q1": 0.3, "q3": 0.5, "n": 12},
        },
        {
            "basis_node_id": "spice",
            "recipe_share": 0.01,
            "calories": 3.0,
            "share_iqr": {"q1": 0.02, "q3": 0.05, "n": 12},
        },
    ]
    assert proportion_typicality_from_ingredients(dup)["known_count"] == 2
    assert proportion_typicality_from_ingredients(dup)["key"] == "very_typical"

    # Zero-width / degenerate IQRs must not penalize typicality even if calorie-heavy.
    degenerate = _rows(0, 2, in_cal=100.0)
    degenerate.append(
        {
            "basis_node_id": "collapsed_band",
            "recipe_share": 0.05,
            "calories": 500.0,
            "grams": 80.0,
            "share_iqr": {"q1": 0.35, "q3": 0.35, "n": 12},
        }
    )
    deg_tip = proportion_typicality_from_ingredients(degenerate)
    assert deg_tip["known_count"] == 2
    assert deg_tip["key"] == "very_typical"


def test_scale_candidate_to_kcal_enforces_user_target():
    from recipe_opt_agent.kcal_utils import scale_candidate_to_kcal

    display = {
        "macros": {"calories": 2800, "protein": 30, "carb": 40, "fat": 30},
        "ingredients": [
            {"label": "a", "grams": 200.0, "calories": 1400.0, "recipe_share": 0.5},
            {"label": "b", "grams": 200.0, "calories": 1400.0, "recipe_share": 0.5},
        ],
        "ratio_loss": {"value": 0.04, "band": "warn"},
    }
    problem = {"x0": [200.0, 200.0], "x_opt": [200.0, 200.0], "total_mass": 400.0}
    problem2, display2 = scale_candidate_to_kcal(
        problem=problem, display=display, kcal_target=700.0, tol_frac=0.0
    )
    assert display2["macros"]["calories"] == 700
    assert abs(sum(r["calories"] for r in display2["ingredients"]) - 700) < 1e-6
    assert abs(sum(problem2["x0"]) - 100.0) < 1e-6
    # Shares preserved
    assert display2["ingredients"][0]["recipe_share"] == 0.5


def test_scale_candidate_to_kcal_default_tol_rescales_small_drift():
    """Default tol_frac=0 must rescale even ~1% calorie drift to the exact target."""
    from recipe_opt_agent.kcal_utils import scale_candidate_to_kcal

    display = {
        "macros": {"calories": 707, "protein": 20, "carb": 50, "fat": 30},
        "ingredients": [
            {"label": "a", "grams": 100.0, "calories": 353.5, "recipe_share": 0.5},
            {"label": "b", "grams": 100.0, "calories": 353.5, "recipe_share": 0.5},
        ],
    }
    problem = {"x0": [100.0, 100.0], "total_mass": 200.0}
    _problem2, display2 = scale_candidate_to_kcal(
        problem=problem, display=display, kcal_target=700.0
    )
    assert display2["macros"]["calories"] == 700
    assert abs(sum(r["calories"] for r in display2["ingredients"]) - 700) < 1e-6
