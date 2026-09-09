"""Unit tests for diagnosis-driven slot planning."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from recipe_opt_agent.requirement_tags import deduce_tags_from_text  # noqa: E402
from recipe_opt_agent.slot_planner import MAX_SLOTS, plan_slots  # noqa: E402


def _diag(**over):
    base = {
        "diagnosis": "SINGLE_TERM_RED",
        "meaning": "one red term",
        "terms": [],
        "retry_triggers": [],
        "binding_macros": [],
        "recommended_action_class": "modify",
    }
    base.update(over)
    return base


def test_red_over_term_noncritical_becomes_remove_outlier():
    diag = _diag(
        terms=[
            {"name": "bacon", "value": 0.5, "median": 0.2, "q25": 0.15, "q75": 0.25, "zone": "red", "L_norm": 3.0},
        ]
    )
    slots = plan_slots(diag)
    assert len(slots) == 1
    s = slots[0]
    assert s.kind == "remove_outlier"
    assert s.preferred_actions[0] == "remove"
    assert s.basis_term == "bacon"
    assert s.direction == "over"


def test_red_over_identity_critical_prefers_dilute_add():
    diag = _diag(
        terms=[
            {"name": "pasta", "value": 0.8, "median": 0.4, "q25": 0.35, "q75": 0.45, "zone": "red", "L_norm": 3.0},
        ]
    )
    slots = plan_slots(diag, identity_critical={"pasta": True})
    assert slots[0].kind == "fix_share"
    assert "remove" not in slots[0].preferred_actions
    assert slots[0].constraints.get("identity_critical") is True


def test_under_red_term_prefers_add():
    diag = _diag(
        terms=[
            {"name": "cheese", "value": 0.02, "median": 0.2, "q25": 0.15, "q75": 0.25, "zone": "red", "L_norm": 3.0},
        ]
    )
    slots = plan_slots(diag)
    assert slots[0].kind == "fix_share"
    assert slots[0].direction == "under"
    assert slots[0].preferred_actions[0] == "add"


def test_hull_and_macro_triggers_make_slots():
    diag = _diag(
        diagnosis="OUTSIDE_HULL",
        retry_triggers=[
            {"metric": "hull_intersects", "reason": "", "current_value": False, "threshold_to_clear": True, "clearance": "", "primary": True},
            {"metric": "macros_feasible", "reason": "", "current_value": False, "threshold_to_clear": True, "clearance": "", "primary": False},
        ],
        binding_macros=["protein_min"],
    )
    slots = plan_slots(diag)
    kinds = [s.kind for s in slots]
    assert kinds[0] == "open_hull"
    assert "macro_gap" in kinds
    assert slots[kinds.index("macro_gap")].macro_axis == "protein_min"


def test_dietary_swap_outranks_everything_and_caps_at_two():
    diag = _diag(
        diagnosis="MULTI_TERM_RED",
        retry_triggers=[
            {"metric": "hull_intersects", "reason": "", "current_value": False, "threshold_to_clear": True, "clearance": "", "primary": True},
        ],
        terms=[
            {"name": "bacon", "value": 0.5, "median": 0.2, "q25": 0.15, "q75": 0.25, "zone": "red", "L_norm": 3.0},
            {"name": "cream", "value": 0.4, "median": 0.1, "q25": 0.05, "q75": 0.15, "zone": "red", "L_norm": 3.0},
        ],
    )
    tags = deduce_tags_from_text("vegetarian carbonara")
    slots = plan_slots(
        diag,
        requirement_tags=tags,
        current_ingredients=[{"label": "bacon", "grams": 50.0}, {"label": "spaghetti", "grams": 200.0}],
    )
    assert len(slots) <= MAX_SLOTS
    assert slots[0].kind == "dietary_swap"
    assert slots[0].target_line_label == "bacon"


def test_ok_diagnosis_yields_improve_fallback():
    slots = plan_slots(_diag(diagnosis="OK", terms=[], retry_triggers=[]))
    assert len(slots) == 1
    assert slots[0].kind == "improve"


def test_ratio_pseudo_term_skipped():
    diag = _diag(
        terms=[
            {"name": "ratio", "value": 5.0, "median": 1.0, "q25": 0.5, "q75": 1.5, "zone": "red", "L_norm": 4.0},
        ]
    )
    slots = plan_slots(diag)
    assert slots[0].kind == "improve"
