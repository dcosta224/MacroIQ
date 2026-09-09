"""Unit tests for agent-vs-GPT-5.5 win rules (ratio_loss / cookability / taste)."""

from __future__ import annotations

from tests.run_agent_vs_gpt55_eval import (
    _cookability_metrics,
    decide_winner,
)


BOX = {
    "protein_min": 0.30,
    "protein_max": 0.34,
    "carb_min": 0.37,
    "carb_max": 0.41,
    "fat_min": 0.27,
    "fat_max": 0.31,
}


def test_agent_wins_on_ratio_and_macro():
    agent = {
        "nutrient_loss": 0.0,
        "ratio_loss": 0.01,
        "holistic_0_10": 6.0,
        "dietary_violation_flag": False,
        "n_odd_ingredients": 0,
        "n_missing_high_hit": 1,
        "pfc_after": {"protein": 0.32, "carbs": 0.39, "fat": 0.29},
    }
    competitor = {
        "nutrient_loss": 0.2,
        "ratio_loss": 0.12,
        "holistic_0_10": 7.0,
        "dietary_violation_flag": False,
        "n_odd_ingredients": 0,
        "n_missing_high_hit": 1,
        "pfc_after": {"protein": 0.45, "carbs": 0.20, "fat": 0.35},
    }
    w = decide_winner(agent, competitor, box=BOX)
    assert w["winner"] == "agent"
    assert w["dimensions"]["ratio_loss"] == "agent"
    assert w["dimensions"]["macro_nutrient"] == "agent"
    assert w["ratio_loss_contextualizer"]["winner"] == "agent"


def test_dietary_safety_beats_holistic():
    agent = {
        "nutrient_loss": 0.05,
        "ratio_loss": 0.05,
        "holistic_0_10": 4.0,
        "dietary_violation_flag": False,
        "n_odd_ingredients": 1,
        "n_missing_high_hit": 2,
        "pfc_after": {"protein": 0.32, "carbs": 0.39, "fat": 0.29},
    }
    competitor = {
        "nutrient_loss": 0.0,
        "ratio_loss": 0.02,
        "holistic_0_10": 9.0,
        "dietary_violation_flag": True,
        "n_odd_ingredients": 0,
        "n_missing_high_hit": 0,
        "pfc_after": {"protein": 0.32, "carbs": 0.39, "fat": 0.29},
    }
    w = decide_winner(agent, competitor, box=BOX)
    assert w["dimensions"]["safety_dietary"] == "agent"


def test_holistic_margin_fallback():
    agent = {
        "nutrient_loss": 0.1,
        "ratio_loss": 0.1,
        "holistic_0_10": 8.0,
        "dietary_violation_flag": False,
        "n_odd_ingredients": 1,
        "n_missing_high_hit": 1,
        "pfc_after": {"protein": 0.5, "carbs": 0.3, "fat": 0.2},
    }
    competitor = {
        "nutrient_loss": 0.1,
        "ratio_loss": 0.1,
        "holistic_0_10": 6.5,
        "dietary_violation_flag": False,
        "n_odd_ingredients": 1,
        "n_missing_high_hit": 1,
        "pfc_after": {"protein": 0.5, "carbs": 0.3, "fat": 0.2},
    }
    w = decide_winner(agent, competitor, box=BOX)
    assert w["winner"] == "agent"
    assert "holistic" in w["reason"]


def test_suite_d_cookability_veto():
    agent = {
        "nutrient_loss": 0.1,
        "ratio_loss": 0.1,
        "holistic_0_10": 5.0,
        "dietary_violation_flag": False,
        "n_odd_ingredients": 2,
        "n_missing_high_hit": 2,
        "cookability_fail": False,
        "cookability_badness": 0.0,
        "pfc_after": {"protein": 0.5, "carbs": 0.3, "fat": 0.2},
    }
    competitor = {
        "nutrient_loss": 0.0,
        "ratio_loss": 0.05,
        "holistic_0_10": 7.0,
        "dietary_violation_flag": False,
        "n_odd_ingredients": 0,
        "n_missing_high_hit": 0,
        "cookability_fail": True,
        "cookability_badness": 3.0,
        "pfc_after": {"protein": 0.32, "carbs": 0.39, "fat": 0.29},
    }
    w = decide_winner(agent, competitor, box=BOX, suite="D")
    assert w["winner"] == "agent"
    assert "cookability_fail" in w["reason"]
    assert w["dimensions"]["cookability"] == "agent"


def test_suite_e_taste_adherence_dim():
    agent = {
        "nutrient_loss": 0.0,
        "ratio_loss": 0.05,
        "holistic_0_10": 6.0,
        "dietary_violation_flag": False,
        "taste_adherence": 1.0,
        "cookability_fail": False,
        "cookability_badness": 0.0,
        "pfc_after": {"protein": 0.32, "carbs": 0.39, "fat": 0.29},
    }
    competitor = {
        "nutrient_loss": 0.0,
        "ratio_loss": 0.12,
        "holistic_0_10": 6.0,
        "dietary_violation_flag": False,
        "taste_adherence": 0.0,
        "cookability_fail": False,
        "cookability_badness": 0.0,
        "pfc_after": {"protein": 0.32, "carbs": 0.39, "fat": 0.29},
    }
    w = decide_winner(agent, competitor, box=BOX, suite="E")
    assert w["dimensions"]["taste_adherence"] == "agent"
    assert w["dimensions"]["ratio_loss"] == "agent"
    assert w["winner"] == "agent"


def test_cookability_flags_nonsense_seasoning():
    payload = {
        "chosen_recipe": {
            "ingredients": [
                {"label": "Pork, fresh, loin", "grams": 200},
                {"label": "Spices, onion powder", "grams": 90},
            ]
        }
    }
    m = _cookability_metrics(payload, case={"require_protein_line": True})
    assert m["nonsense_seasoning_flag"] is True
    assert m["cookability_fail"] is True
    assert m["missing_protein_under_diet"] is False
