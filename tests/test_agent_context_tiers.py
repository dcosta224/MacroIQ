"""Tests for clear-favorite auto-apply, curated context, identity, telemetry."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture(autouse=True)
def _force_heuristic(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_clear_favorite_and_uncertain():
    from recipe_opt_agent.telemetry import clear_favorite_bundle

    bundles = [
        {"bundle_id": "a", "lp_evaluated": True, "delta_L_star": -0.08, "edits": []},
        {"bundle_id": "b", "lp_evaluated": True, "delta_L_star": -0.03, "edits": []},
    ]
    fav = clear_favorite_bundle(bundles, delta_eps=0.01, margin=0.02)
    assert fav is not None and fav["bundle_id"] == "a"

    near = [
        {"bundle_id": "a", "lp_evaluated": True, "delta_L_star": -0.05},
        {"bundle_id": "b", "lp_evaluated": True, "delta_L_star": -0.04},
    ]
    assert clear_favorite_bundle(near, delta_eps=0.01, margin=0.02) is None


def test_curated_context_has_tradeoff_and_compact_hull():
    from recipe_opt_agent.context_builder import build_decision_context

    ctx = build_decision_context(
        {
            "fidelity_band": "must_retry",
            "diagnosis": {
                "diagnosis": "outside_iqr",
                "meaning": "test",
                "n_red": 1,
                "n_yellow": 0,
                "L_max_norm": 2.0,
                "L_total": 1.0,
                "terms": [{"name": "pasta", "zone": "RED", "L_norm": 2.0}],
                "retry_triggers": ["red_term"],
            },
            "hull": {
                "intersects": False,
                "distance": {"outside_score": 0.2, "interpretation": "outside"},
                "ingredient_pfc_vertices": [[0, 0, 0]],
            },
            "chosen_recipe": {"title": "T", "ingredients": [{"label": "pasta", "grams": 80}]},
            "opt": {"pfc_after": {"protein": 0.2, "carbs": 0.4, "fat": 0.4}, "term_losses": {}},
            "config": {"protein_min": 0.1, "protein_max": 0.3, "carb_min": 0.2, "carb_max": 0.5, "fat_min": 0.2, "fat_max": 0.5},
            "bundles": [
                {
                    "bundle_id": "b1",
                    "lp_evaluated": True,
                    "delta_L_star": -0.05,
                    "edits": [{"action": "add", "label": "chicken", "meta": {"slot_id": "s1"}}],
                    "next_problem": {"x0": [1]},
                }
            ],
            "planned_slots": [{"slot_id": "s1", "kind": "macro_gap", "reason": "protein"}],
            "identity_roles": ["pasta"],
            "iteration": 0,
            "decision_outcomes": [],
        }
    )
    assert "tradeoff_frame" in ctx
    assert ctx["hull"] == {"intersects": False, "outside_score": 0.2, "interpretation": "outside"}
    assert "ingredient_pfc_vertices" not in (ctx.get("hull") or {})
    assert ctx["bundles"][0].get("edit_annotations")
    assert "next_problem" not in ctx["bundles"][0]


def test_identity_carbonara_and_shakshuka():
    from recipe_opt_agent.identity_roles import resolve_identity_roles

    carb = resolve_identity_roles(title="Spaghetti Carbonara", request="classic", use_llm=False)
    assert "pasta" in carb and "egg" in carb and "cheese" in carb
    shak = resolve_identity_roles(title="Shakshuka", request="brunch eggs", use_llm=False)
    assert "egg" in shak and "tomato" in shak


def test_node_decide_auto_skips_llm(monkeypatch):
    from recipe_opt_agent.config import AgentConfig
    from recipe_opt_agent.graph import node_decide

    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("LLM should not be called for clear favorite")

    monkeypatch.setattr("recipe_opt_agent.graph.decide_action_llm", _boom)

    # Adds must come from LLM ideation to be auto-apply eligible (raw slot
    # retrieval adds always require an explicit LLM identity check).
    bundles = [
        {
            "bundle_id": "win",
            "lp_evaluated": True,
            "delta_L_star": -0.12,
            "edits": [
                {
                    "action": "add",
                    "label": "chicken",
                    "candidate_id": "idea_add_900001_0",
                    "meta": {"source": "llm_ideation"},
                }
            ],
            "next_problem": {"x0": [1.0]},
        },
        {
            "bundle_id": "runner",
            "lp_evaluated": True,
            "delta_L_star": -0.02,
            "edits": [
                {
                    "action": "add",
                    "label": "tofu",
                    "candidate_id": "idea_add_900003_1",
                    "meta": {"source": "llm_ideation"},
                }
            ],
            "next_problem": {"x0": [1.0]},
        },
    ]
    state = {
        "fidelity_band": "must_retry",
        "diagnosis": {"n_red": 1, "L_max_norm": 2.0, "L_total": 1.0, "terms": []},
        "bundles": bundles,
        "candidates": [],
        "iteration": 0,
        "history": [],
        "identity_roles": ["pasta"],
        "config": {
            "auto_apply_delta_eps": 0.01,
            "auto_apply_margin": 0.02,
            "model": "gpt-4o-mini",
            "model_escalate": "gpt-4.1-mini",
        },
        "opt": {"feasible": True, "pfc_after": {"protein": 0.2, "carbs": 0.4, "fat": 0.4}, "term_losses": {}},
        "chosen_recipe": {"title": "T", "ingredients": []},
    }
    update = node_decide(state)
    assert update["decision"]["action"] == "apply_bundle"
    assert update["decision"]["rationale"] == "auto_clear_favorite"
    assert update["tools_used"][0]["name"] == "decide_auto"
    assert calls["n"] == 0
    assert (update.get("run_telemetry") or {}).get("n_auto_applies", 0) >= 1


def test_agent_run_includes_telemetry():
    from recipe_opt_agent.config import AgentConfig
    from recipe_opt_agent.runner import run_recipe_opt_agent

    M = np.array([[0.25, 0.0], [0.0, 0.0], [0.0, 0.25], [0.0, 0.0]], dtype=float)
    problem = {
        "x0": [200.0, 200.0],
        "M": M.tolist(),
        "ingredient_basis": ["pasta", "cheese"],
        "basis_samples": {"pasta": [0.45, 0.5, 0.55], "cheese": [0.45, 0.5, 0.55]},
        "ratio_samples": [],
        "marginal_nodes": ["pasta", "cheese"],
        "kcal_target": 400.0,
        "total_mass": 400.0,
        "modification_candidates": [],
    }
    cfg = AgentConfig(
        protein_min=0.0,
        protein_max=1.0,
        carb_min=0.0,
        carb_max=1.0,
        fat_min=0.0,
        fat_max=1.0,
        max_iterations=1,
        F_accept=5.0,
        F_max=10.0,
    )
    result = run_recipe_opt_agent(problem=problem, taste_text="cheese pasta", title="Cheese Pasta", config=cfg)
    assert "run_telemetry" in result
    assert result["run_telemetry"].get("final_status") is not None
