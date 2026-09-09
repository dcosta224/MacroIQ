"""Tests for hard culinary clash gates."""

from __future__ import annotations

from recipe_opt_agent.clash_gates import (
    clash_reason_for_label,
    filter_candidates_by_clash_gates,
    is_denylist_label,
)
from recipe_opt_agent.culinary_types import CONFLICTING_FAMILIES, families_for_text


def test_denylist_onion_rings_and_coffee():
    ok, _ = is_denylist_label("Fast foods, onion rings, breaded and fried")
    assert ok
    ok, _ = is_denylist_label("Beverages, coffee, brewed, prepared with tap water")
    assert ok
    ok, _ = is_denylist_label("Chicken breast, raw, skinless")
    assert not ok


def test_onion_vs_onion_rings_conflict():
    assert ("onion", "onion_rings") in CONFLICTING_FAMILIES
    assert "onion_rings" in families_for_text("Fast foods, onion rings, breaded and fried")
    assert "onion" not in families_for_text("Fast foods, onion rings, breaded and fried")


def test_yogurt_blocked_on_al_pastor():
    reason = clash_reason_for_label(
        "Yogurt, Greek, plain, nonfat",
        current_labels=["Pork Loin Boneless", "Pineapple, raw"],
        title="Al Pastor",
    )
    assert reason is not None
    assert "yogurt" in reason or "soft_protein" in reason


def test_rice_blocked_in_barbacoa():
    reason = clash_reason_for_label(
        "Rice, white, glutinous, unenriched, uncooked",
        current_labels=["BEEF", "Garlic, raw"],
        title="Barbacoa",
    )
    assert reason is not None
    assert "rice" in reason


def test_chicken_ood_allowed_through_filter():
    problem = {
        "chosen_recipe": {
            "ingredients": [
                {"label": "SPAGHETTI"},
                {"label": "Egg, whole, raw, fresh"},
                {"label": "Cheese, parmesan, grated"},
            ]
        }
    }
    cands = [
        {
            "action": "add",
            "label": "Chicken breast, raw, skinless",
            "branch": "ood_protein",
            "meta": {"ood": True},
        },
        {
            "action": "add",
            "label": "Yogurt, Greek, plain, nonfat",
            "branch": "in_distribution",
            "meta": {},
        },
        {
            "action": "add",
            "label": "Fast foods, onion rings, breaded and fried",
            "branch": "in_distribution",
            "meta": {},
        },
    ]
    kept, dropped = filter_candidates_by_clash_gates(
        cands, problem=problem, title="Spaghetti Carbonara"
    )
    kept_labels = {c["label"] for c in kept}
    assert "Chicken breast, raw, skinless" in kept_labels
    assert all("onion rings" not in (d["candidate"]["label"].lower()) for d in dropped) or any(
        "onion" in str(d.get("detail") or "").lower() or "denylist" in str(d.get("detail") or "").lower()
        for d in dropped
    )
    assert not any("Yogurt" in c["label"] for c in kept)
