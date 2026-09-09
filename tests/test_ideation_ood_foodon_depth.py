"""Tests for FoodOn aggregation cap, ideation grounding, and fair OOD selection."""

from __future__ import annotations

from recipe_opt_agent.foodon_depth import (
    apply_foodon_aggregation_cap,
    capped_rollup_node,
    default_max_foodon_aggregation_levels,
    resolve_max_foodon_aggregation_levels,
)
from recipe_opt_agent.ideation import ground_ideas_to_candidates, ideate_ingredient_edits
from recipe_opt_agent.ood_branch import maybe_build_ood_branch, protein_demand_high
from recipe_opt_agent.telemetry import clear_favorite_bundle


def test_default_max_levels_is_positive_half_avg():
    levels = default_max_foodon_aggregation_levels()
    assert isinstance(levels, int)
    assert levels >= 1
    assert resolve_max_foodon_aggregation_levels(2) == 2
    assert resolve_max_foodon_aggregation_levels(None) == levels


def test_capped_rollup_never_exceeds_max_levels():
    chains = {"leaf": ["leaf", "a", "b", "c", "d", "e"]}
    active = {"e"}  # only deep basis
    # With max_levels=2, cannot reach e → stay at leaf
    assert capped_rollup_node("leaf", active, chains, max_levels=2) == "leaf"
    # With max_levels=4, still cannot reach e (index 4) wait: window = leaf,a,b,c,d = indices 0..4
    # e is index 5, so still leaf
    assert capped_rollup_node("leaf", active, chains, max_levels=4) == "leaf"
    # With max_levels=5, e is included
    assert capped_rollup_node("leaf", active, chains, max_levels=5) == "e"
    # Prefer nearer active node within window
    assert capped_rollup_node("leaf", {"b", "e"}, chains, max_levels=5) == "b"


def test_apply_aggregation_cap_updates_basis():
    problem = {
        "ingredient_basis": ["deep"],
        "ingredient_foodon_leaves": ["leaf"],
        "rollup_chains": {"leaf": ["leaf", "mid", "deep"]},
        "basis_nodes": ["deep", "mid"],
        "chosen_recipe": {"ingredients": [{"label": "x", "foodon_id": "leaf"}]},
        "build_params": {},
    }
    out = apply_foodon_aggregation_cap(problem, max_levels=1)
    assert out["ingredient_basis"][0] == "mid"
    assert out["build_params"]["max_foodon_aggregation_levels"] == 1


def test_ideation_heuristic_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = ideate_ingredient_edits(
        {
            "title": "Spaghetti Carbonara",
            "protein_min": 0.35,
            "diagnosis": {"binding_macros": ["protein_min"]},
            "current_ingredients": [{"label": "spaghetti", "grams": 100}],
            "macro_box": {"protein_min": 0.35, "protein_max": 0.4},
        }
    )
    assert out["ideas"]
    assert any(str(i.get("branch") or "").startswith("ood") for i in out["ideas"])
    assert out["neighborhood_search_queries"]
    assert all(" " in q or len(q) > 8 for q in out["neighborhood_search_queries"])


def test_ground_ideas_finds_ood_chicken():
    problem = {
        "chosen_recipe": {"ingredients": [{"label": "spaghetti", "grams": 100, "fdc_id": 1}]},
        "retrieval_context": {"fdc_catalog": [], "fdc_macros": {}},
    }
    cands = ground_ideas_to_candidates(
        [
            {
                "action": "add",
                "ingredient": "skinless chicken breast",
                "branch": "ood_protein",
                "role": "lean_protein",
                "rationale": "protein",
                "neighborhood_search_queries": ["creamy spaghetti with seared chicken breast"],
            }
        ],
        problem=problem,
        box_dict={"protein_min": 0.3, "protein_max": 0.4},
    )
    assert cands
    assert cands[0]["branch"] == "ood_protein"
    assert cands[0]["fdc_id"] is not None


def test_ood_clear_favorite_handicap_prefers_close_ood():
    id_b = {
        "bundle_id": "id",
        "branch": "in_distribution",
        "lp_evaluated": True,
        "delta_L_star": -0.02,
        "nutrient_slack": 0.04,
    }
    ood_b = {
        "bundle_id": "ood",
        "branch": "ood_protein",
        "lp_evaluated": True,
        "delta_L_star": -0.012,  # slightly worse raw LP
        "nutrient_slack": 0.0,
    }
    fav = clear_favorite_bundle(
        [id_b, ood_b],
        delta_eps=0.005,
        margin=0.005,
        ood_delta_handicap=0.015,
    )
    assert fav is not None
    assert fav["bundle_id"] == "ood"


def test_maybe_build_ood_branch_uses_ideation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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
        ideation_context={"title": "Spaghetti Carbonara"},
    )
    assert info["needed"] is True
    assert info["ood_candidates"]
    assert info.get("ideation") is not None
