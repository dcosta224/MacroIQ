"""Graph-level tests with a synthetic problem fixture (heuristic LLM, no API key)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

# Ensure heuristic path
os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture(autouse=True)
def _force_heuristic(monkeypatch):
    """Other test modules (e.g. the web server) may re-load .env and restore the key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _synthetic_problem(*, force_high_loss: bool = False) -> dict:
    # Two foods: protein + carb. Neighborhood samples centered near 0.5/0.5 shares.
    M = np.array(
        [
            [0.25, 0.0],
            [0.0, 0.0],
            [0.0, 0.25],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    x0 = np.array([200.0, 200.0])
    basis = ["pasta", "cheese"]
    samples = {
        "pasta": np.array([0.45, 0.5, 0.55, 0.48, 0.52]),
        "cheese": np.array([0.45, 0.5, 0.55, 0.48, 0.52]),
    }
    if force_high_loss:
        # Samples far from current equal mix → high L_norm
        samples = {
            "pasta": np.array([0.9, 0.92, 0.88, 0.91]),
            "cheese": np.array([0.05, 0.08, 0.1, 0.06]),
        }
    return {
        "x0": x0.tolist(),
        "M": M.tolist(),
        "ingredient_basis": basis,
        "basis_samples": samples,
        "ratio_samples": [],
        "marginal_nodes": ["pasta", "cheese"],
        "kcal_target": float(4 * 0.25 * 200 + 4 * 0.25 * 200),  # 400
        "total_mass": 400.0,
        "modification_candidates": [
            {
                "candidate_id": "add_chicken",
                "action": "add",
                "label": "chicken",
                "cooccurrence": 0.4,
                "geom_score": 1.0,
                "L_star": 0.05,
                "meta": {},
            }
        ],
    }


def test_agent_accepts_feasible_greenish():
    from recipe_opt_agent.config import AgentConfig
    from recipe_opt_agent.runner import run_recipe_opt_agent

    cfg = AgentConfig(
        protein_min=0.0,
        protein_max=1.0,
        carb_min=0.0,
        carb_max=1.0,
        fat_min=0.0,
        fat_max=1.0,
        max_iterations=2,
        F_accept=5.0,  # very permissive
        F_max=10.0,
    )
    result = run_recipe_opt_agent(
        problem=_synthetic_problem(force_high_loss=False),
        taste_text="cheese pasta",
        title="Cheese Pasta",
        config=cfg,
    )
    assert result.get("status") in {"accepted", "accepted_pool_best", "failed_or_best_effort"}
    assert "history" in result or result.get("chosen")


def test_apply_bundle_mutates_problem_arrays():
    """apply_bundle must replace the LP problem (M column count / x0 length change)."""
    from recipe_opt_agent.bundle_scoring import apply_edits_to_problem, score_bundles
    from recipe_opt_agent.graph import node_apply_or_expand

    problem = _synthetic_problem()
    problem["chosen_recipe"] = {
        "title": "Cheese Pasta",
        "ingredients": [
            {"label": "pasta", "grams": 200.0, "fdc_id": 1},
            {"label": "cheese", "grams": 200.0, "fdc_id": 2},
        ],
    }
    add_cand = {
        "candidate_id": "s1::add_10",
        "action": "add",
        "label": "chicken",
        "fdc_id": 10,
        "meta": {
            "slot_id": "s1",
            "macros_per_g": [0.31, 0.036, 0.0, 1.65],
            "delta_ratio_proxy": 0.0,
            "delta_nutrient_proxy": 0.1,
        },
    }
    box = {
        "protein_min": 0.0,
        "protein_max": 1.0,
        "carb_min": 0.0,
        "carb_max": 1.0,
        "fat_min": 0.0,
        "fat_max": 1.0,
    }
    bundles = score_bundles(problem, {"s1": [add_cand]}, box_dict=box)
    assert bundles and bundles[0]["next_problem"] is not None

    state = {
        "problem": problem,
        "decision": {"action": "apply_bundle", "chosen_bundle_id": bundles[0]["bundle_id"]},
        "bundles": bundles,
        "candidates": [add_cand],
        "iteration": 0,
        "config": {},
    }
    update = node_apply_or_expand(state)
    assert update["status"] == "modified"
    new_problem = update["problem"]
    assert len(new_problem["x0"]) == 3  # was 2, add appended a column
    assert len(np.asarray(new_problem["M"])[0]) == 3
    assert len(new_problem["ingredient_basis"]) == 3
    labels = [r["label"] for r in new_problem["chosen_recipe"]["ingredients"]]
    assert "chicken" in labels

    # Single-candidate path also materializes a real next_problem live.
    state_single = {
        "problem": problem,
        "decision": {"action": "add", "chosen_candidate_id": "s1::add_10"},
        "candidates": [add_cand],
        "iteration": 0,
        "config": {},
    }
    update2 = node_apply_or_expand(state_single)
    assert update2["status"] == "modified"
    assert len(update2["problem"]["x0"]) == 3

    # And apply_edits_to_problem is the same primitive both paths use.
    nxt = apply_edits_to_problem(problem, [add_cand])
    assert nxt is not None and len(nxt["x0"]) == 3


def test_propose_emits_slots_and_bundles():
    """node_propose puts planned_slots + bundles on state with staged tool events."""
    from recipe_opt_agent.graph import node_propose

    problem = _synthetic_problem()
    state = {
        "problem": problem,
        "diagnosis": {
            "diagnosis": "SINGLE_TERM_RED",
            "terms": [
                {"name": "pasta", "value": 0.9, "median": 0.5, "q25": 0.45, "q75": 0.55, "zone": "red", "L_norm": 3.0}
            ],
            "retry_triggers": [],
            "binding_macros": [],
        },
        "identity_critical": {},
        "config": {},
        "iteration": 0,
    }
    update = node_propose(state)
    assert update["planned_slots"], "diagnosis should produce at least one slot"
    tool_names = [t["name"] for t in update["tools_used"]]
    # Ideation (heuristic without API key) sits between plan and retrieve.
    for required in ("plan_slots", "retrieve_slots", "score_bundles"):
        assert required in tool_names
    assert tool_names.index("plan_slots") < tool_names.index("retrieve_slots") < tool_names.index(
        "score_bundles"
    )
    # Fixture candidates (no retrieval_context) flow through as the slot's pool.
    assert update["candidates"]
    # Bundles exposed to decide must not carry next_problem in tool events.
    score_event = update["tools_used"][tool_names.index("score_bundles")]
    for b in score_event["output"]["bundles"]:
        assert "next_problem" not in b


def test_agent_moderate_or_retry_with_candidates():
    from recipe_opt_agent.config import AgentConfig
    from recipe_opt_agent.runner import run_recipe_opt_agent

    cfg = AgentConfig(
        protein_min=0.4,
        protein_max=0.7,
        carb_min=0.3,
        carb_max=0.6,
        fat_min=0.0,
        fat_max=0.2,
        max_iterations=2,
        F_accept=0.3,
        F_max=0.8,
    )
    result = run_recipe_opt_agent(
        problem=_synthetic_problem(force_high_loss=True),
        taste_text="cheese pasta",
        title="Cheese Pizza",
        config=cfg,
    )
    assert result.get("status") is not None


def test_route_after_diagnose_blocks_accept_on_iter_0():
    from recipe_opt_agent.graph import _route_after_decide, _route_after_diagnose

    state0 = {
        "fidelity_band": "accept",
        "iteration": 0,
        "config": {"max_iterations": 3},
        "agent_mode": "neighborhood",
        "decision": {"action": "accept"},
    }
    assert _route_after_diagnose(state0) == "propose"
    assert _route_after_decide(state0) == "apply"

    state1 = {**state0, "iteration": 1}
    assert _route_after_diagnose(state1) == "finalize"
    assert _route_after_decide(state1) == "finalize"
