"""Tests for bundle enumeration, apply_edits_to_problem, and joint LP scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from recipe_opt_agent.bundle_scoring import (  # noqa: E402
    apply_edits_to_problem,
    enumerate_bundles,
    score_bundles,
)

BOX = {
    "protein_min": 0.0,
    "protein_max": 1.0,
    "carb_min": 0.0,
    "carb_max": 1.0,
    "fat_min": 0.0,
    "fat_max": 1.0,
}


def _problem() -> dict:
    # pasta (carb) + cheese (protein-ish/fat)
    M = np.array(
        [
            [0.05, 0.25],
            [0.01, 0.28],
            [0.75, 0.03],
            [3.5, 3.9],
        ],
        dtype=float,
    )
    return {
        "x0": [200.0, 100.0],
        "M": M.tolist(),
        "ingredient_basis": ["pasta_node", "cheese_node"],
        "basis_samples": {
            "pasta_node": [0.6, 0.65, 0.7],
            "cheese_node": [0.3, 0.35, 0.4],
        },
        "ratio_samples": [],
        "marginal_nodes": ["pasta_node", "cheese_node"],
        "kcal_target": 1090.0,
        "total_mass": 300.0,
        "chosen_recipe": {
            "title": "Test Pasta",
            "ingredients": [
                {"label": "pasta", "grams": 200.0, "fdc_id": 1},
                {"label": "cheese", "grams": 100.0, "fdc_id": 2},
            ],
        },
        "retrieval_context": {
            "starting_ingredients": [
                {"label": "pasta", "grams": 200.0, "fdc_id": 1},
                {"label": "cheese", "grams": 100.0, "fdc_id": 2},
            ],
            "starting_fdc": [1, 2],
            "starting_labels": ["pasta", "cheese"],
        },
    }


def _add_cand(cid="s1::add_10", label="chicken", fdc_id=10):
    return {
        "candidate_id": cid,
        "action": "add",
        "label": label,
        "fdc_id": fdc_id,
        "meta": {
            "slot_id": "s1",
            "macros_per_g": [0.31, 0.036, 0.0, 1.65],  # chicken-ish per gram
            "delta_ratio_proxy": 0.05,
            "delta_nutrient_proxy": 0.2,
        },
    }


def _remove_cand(idx=1, cid="s2::remove_1_2"):
    return {
        "candidate_id": cid,
        "action": "remove",
        "label": "cheese",
        "fdc_id": 2,
        "meta": {"slot_id": "s2", "remove_idx": idx, "delta_ratio_proxy": 0.1, "delta_nutrient_proxy": 0.0},
    }


def _swap_cand(idx=0, cid="s1::swap_0_10"):
    return {
        "candidate_id": cid,
        "action": "swap",
        "label": "quinoa",
        "fdc_id": 11,
        "meta": {
            "slot_id": "s1",
            "swap_out_idx": idx,
            "macros_per_g": [0.14, 0.06, 0.64, 3.68],
            "delta_ratio_proxy": 0.02,
            "delta_nutrient_proxy": 0.1,
        },
    }


# ---------- apply_edits_to_problem ----------


def test_apply_add_appends_column_and_conserves_mass():
    prob = _problem()
    nxt = apply_edits_to_problem(prob, [_add_cand()])
    assert nxt is not None
    assert len(nxt["x0"]) == 3
    assert np.asarray(nxt["M"]).shape == (4, 3)
    assert len(nxt["ingredient_basis"]) == 3
    assert abs(sum(nxt["x0"]) - prob["total_mass"]) < 1e-6
    labels = [r["label"] for r in nxt["chosen_recipe"]["ingredients"]]
    assert "chicken" in labels
    # retrieval context updated so next propose sees the new ingredient
    assert "chicken" in nxt["retrieval_context"]["starting_labels"]


def test_apply_remove_drops_column():
    prob = _problem()
    nxt = apply_edits_to_problem(prob, [_remove_cand(idx=1)])
    assert nxt is not None
    assert len(nxt["x0"]) == 1
    assert np.asarray(nxt["M"]).shape == (4, 1)
    labels = [r["label"] for r in nxt["chosen_recipe"]["ingredients"]]
    assert labels == ["pasta"]


def test_apply_swap_replaces_column():
    prob = _problem()
    nxt = apply_edits_to_problem(prob, [_swap_cand(idx=0)])
    assert nxt is not None
    assert len(nxt["x0"]) == 2
    labels = [r["label"] for r in nxt["chosen_recipe"]["ingredients"]]
    assert "quinoa" in labels and "pasta" not in labels


def test_apply_add_without_macros_returns_none():
    prob = _problem()
    cand = _add_cand()
    cand["meta"].pop("macros_per_g")
    assert apply_edits_to_problem(prob, [cand]) is None


def test_apply_conflicting_removes_returns_none():
    prob = _problem()
    assert (
        apply_edits_to_problem(prob, [_remove_cand(idx=1, cid="a"), _remove_cand(idx=1, cid="b")])
        is None
    )


# ---------- enumerate_bundles ----------


def test_enumerate_singletons_and_pairs():
    per_slot = {"s1": [_add_cand()], "s2": [_remove_cand()]}
    bundles = enumerate_bundles(per_slot)
    sizes = sorted(len(b) for b in bundles)
    assert sizes == [1, 1, 2]


def test_enumerate_prunes_same_line_conflicts():
    per_slot = {
        "s1": [_swap_cand(idx=1, cid="s1::swap_1_11")],
        "s2": [_remove_cand(idx=1)],
    }
    bundles = enumerate_bundles(per_slot)
    assert all(len(b) == 1 for b in bundles)  # pair removed as conflict


def test_enumerate_prunes_duplicate_add():
    per_slot = {
        "s1": [_add_cand(cid="s1::add_10")],
        "s2": [_add_cand(cid="s2::add_10")],
    }
    bundles = enumerate_bundles(per_slot)
    assert all(len(b) == 1 for b in bundles)


# ---------- score_bundles (real small LP) ----------


def test_score_bundles_reports_joint_lp_and_next_problem():
    prob = _problem()
    per_slot = {"s1": [_add_cand()], "s2": [_remove_cand()]}
    scored = score_bundles(prob, per_slot, box_dict=BOX)
    assert scored
    lp_rows = [b for b in scored if b["lp_evaluated"]]
    assert lp_rows, "at least one bundle should be LP-evaluated"
    for b in lp_rows:
        assert b["L_star_before"] is not None
        assert b["L_star_after"] is not None
        assert b["delta_L_star"] is not None
        assert b["next_problem"] is not None
        assert b["ratio_term"] is not None
        assert b["nutrient_slack"] is not None
    # sorted: best delta first among LP-evaluated
    deltas = [b["delta_L_star"] for b in lp_rows]
    assert deltas == sorted(deltas)


def test_score_bundles_size1_keeps_single_edit_shape():
    prob = _problem()
    scored = score_bundles(prob, {"s1": [_add_cand()]}, box_dict=BOX)
    assert scored
    assert scored[0]["size"] == 1
    assert scored[0]["edits"][0]["action"] == "add"
    assert scored[0]["edits"][0]["candidate_id"] == "s1::add_10"
