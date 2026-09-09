"""Eval harness: neighborhood vs creative cases (offline, no API/DB)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

os.environ.pop("OPENAI_API_KEY", None)

import numpy as np

from recipe_opt_agent.config import AgentConfig
from recipe_opt_agent.creative_loader import load_creative_problem
from recipe_opt_agent.requirement_tags import deduce_tags_from_text, tag_violations_for_ingredient
from recipe_opt_agent.runner import run_recipe_opt_agent


def _synthetic_neighborhood_problem() -> dict:
    M = np.array([[0.25, 0.0], [0.0, 0.0], [0.0, 0.25], [0.0, 0.0]], dtype=float)
    x0 = np.array([200.0, 200.0])
    return {
        "x0": x0.tolist(),
        "M": M.tolist(),
        "ingredient_basis": ["pasta", "cheese"],
        "basis_samples": {"pasta": [0.45, 0.5, 0.55], "cheese": [0.45, 0.5, 0.55]},
        "ratio_samples": [],
        "marginal_nodes": ["pasta", "cheese"],
        "kcal_target": 400.0,
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


EVAL_CASES = [
    {"name": "classic_carbonara_neighborhood", "mode": "neighborhood", "request": "carbonara", "canonical_id": None},
    {"name": "high_protein_carbonara_creative", "mode": "creative", "request": "40% protein carbonara", "canonical_id": None},
    {"name": "vegetarian_creative", "mode": "creative", "request": "vegetarian carbonara", "canonical_id": None},
]


def _tag_violations_in_result(result: dict, request: str) -> int:
    tags = deduce_tags_from_text(request)
    if not tags:
        return 0
    chosen = (result.get("chosen") or {}).get("entry") or result.get("chosen") or {}
    ings = chosen.get("ingredients") or []
    if not ings and "opt" in chosen:
        ings = []
    v = 0
    for row in ings:
        label = str(row.get("label") or row.get("name") or "")
        if tag_violations_for_ingredient(label, tags):
            v += 1
    return v


def run_eval_case(case: dict) -> dict:
    cfg = AgentConfig(
        max_iterations=2,
        F_accept=5.0,
        F_max=10.0,
        protein_min=0.0,
        protein_max=1.0,
        carb_min=0.0,
        carb_max=1.0,
        fat_min=0.0,
        fat_max=1.0,
    )
    if case["mode"] == "creative":
        problem = load_creative_problem(user_request=case["request"], offline=True)
        result = run_recipe_opt_agent(
            problem=problem,
            user_request=case["request"],
            title=case["request"],
            config=cfg,
            agent_mode="creative",
        )
    else:
        result = run_recipe_opt_agent(
            problem=_synthetic_neighborhood_problem(),
            taste_text="cheese pasta",
            title="Cheese Pasta",
            config=cfg,
            agent_mode="neighborhood",
        )
    scored = result.get("scored_finalists") or []
    winner = (result.get("chosen") or {}).get("entry") or {}
    metrics = winner.get("metrics") or {}
    return {
        "case": case["name"],
        "mode": case["mode"],
        "status": result.get("status"),
        "pool_size": len(result.get("alternatives") or []) + (1 if winner else 0) or len(scored),
        "tag_violations": _tag_violations_in_result(result, case.get("request", "")),
        "winner_composite": winner.get("composite"),
        "nutrient_dist": metrics.get("nutrient_dist"),
        "ratio_badness": metrics.get("ratio_badness"),
        "intent_gap": metrics.get("intent_gap"),
        "churn": metrics.get("churn"),
    }


def test_eval_cases_tag_violations_zero():
    rows = [run_eval_case(c) for c in EVAL_CASES]
    for row in rows:
        assert row["tag_violations"] == 0, row
        assert row["status"] is not None


if __name__ == "__main__":
    import json

    print(json.dumps([run_eval_case(c) for c in EVAL_CASES], indent=2))
