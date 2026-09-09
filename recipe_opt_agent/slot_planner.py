"""Diagnosis-driven slot planning for multi-ingredient edit retrieval.

A Slot is a structured gap the agent should try to fill with one edit
(add / swap / remove). Slots are derived from the optimizer diagnosis
(IQR term zones, retry triggers, binding macros) plus identity roles and
dietary requirement tags. Max two slots per iteration keeps bundle
enumeration tractable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from recipe_opt_agent.requirement_tags import (
    RequirementTag,
    tag_violations_for_ingredient,
)

MAX_SLOTS = 2

# Priority: lower sorts first when capping to MAX_SLOTS.
_KIND_PRIORITY = {
    "dietary_swap": 0,
    "open_hull": 1,
    "macro_gap": 2,
    "fix_share": 3,
    "remove_outlier": 4,
    "improve": 9,
}


@dataclass
class Slot:
    slot_id: str
    kind: str  # open_hull | fix_share | macro_gap | dietary_swap | remove_outlier | improve
    preferred_actions: tuple[str, ...]
    reason: str
    basis_term: str | None = None  # labeled share-term name (e.g. "pasta")
    macro_axis: str | None = None  # e.g. "protein_min"
    target_line_label: str | None = None  # for swaps/removes: which current line
    direction: str | None = None  # "over" | "under" for fix_share
    priority: int = 9
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _red_terms(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for t in diagnosis.get("terms") or []:
        zone = str(t.get("zone") or "").lower()
        if zone == "red":
            out.append(t)
    return out


def _trigger_metrics(diagnosis: dict[str, Any]) -> set[str]:
    return {str(t.get("metric") or "") for t in (diagnosis.get("retry_triggers") or [])}


def plan_slots(
    diagnosis: dict[str, Any],
    *,
    identity_critical: dict[str, bool] | None = None,
    requirement_tags: list[RequirementTag] | None = None,
    current_ingredients: list[dict[str, Any]] | None = None,
    max_slots: int = MAX_SLOTS,
) -> list[Slot]:
    """Derive at most ``max_slots`` structured edit slots from the diagnosis."""
    identity_critical = identity_critical or {}
    requirement_tags = requirement_tags or []
    current_ingredients = current_ingredients or []
    slots: list[Slot] = []
    seen_ids: set[str] = set()

    def _add(slot: Slot) -> None:
        if slot.slot_id in seen_ids:
            return
        seen_ids.add(slot.slot_id)
        slots.append(slot)

    # 1) Dietary swaps: current lines violating forbid tags must be replaced.
    forbid_tags = [t for t in requirement_tags if t.polarity == "forbid" or t.kind == "dietary_restriction"]
    if forbid_tags:
        for row in current_ingredients:
            label = str(row.get("label") or row.get("name") or "")
            if not label:
                continue
            vios = tag_violations_for_ingredient(label, requirement_tags)
            if vios:
                vio_ids = [v.get("tag_id") if isinstance(v, dict) else str(v) for v in vios]
                _add(
                    Slot(
                        slot_id=f"dietary_swap::{label.lower()}",
                        kind="dietary_swap",
                        preferred_actions=("swap", "remove"),
                        reason=f"'{label}' violates dietary tags: {vio_ids}",
                        target_line_label=label,
                        priority=_KIND_PRIORITY["dietary_swap"],
                        constraints={"tag_violations": vios},
                    )
                )

    metrics = _trigger_metrics(diagnosis)

    # 2) Open hull: target box unreachable with current ingredient vertices.
    if "hull_intersects" in metrics or str(diagnosis.get("diagnosis") or "") == "OUTSIDE_HULL":
        _add(
            Slot(
                slot_id="open_hull",
                kind="open_hull",
                preferred_actions=("add", "swap"),
                reason="Target macro box does not intersect the ingredient hull; need a new PFC vertex",
                priority=_KIND_PRIORITY["open_hull"],
            )
        )

    # 3) Macro gap: LP infeasible or binding macro bounds.
    binding = list(diagnosis.get("binding_macros") or [])
    if "macros_feasible" in metrics or binding:
        axis = binding[0] if binding else None
        _add(
            Slot(
                slot_id=f"macro_gap::{axis or 'general'}",
                kind="macro_gap",
                preferred_actions=("add", "swap"),
                reason=(
                    f"Macro bound binding at {axis}" if axis else "Optimizer cannot satisfy macro bounds"
                ),
                macro_axis=axis,
                priority=_KIND_PRIORITY["macro_gap"],
            )
        )

    # 4) Fix share: RED IQR terms (skip the ratio pseudo-term).
    for term in _red_terms(diagnosis):
        name = str(term.get("name") or "")
        if not name or name.lower() == "ratio":
            continue
        value = float(term.get("value") or 0.0)
        q75 = float(term.get("q75") or 0.0)
        q25 = float(term.get("q25") or 0.0)
        over = value > q75
        critical = bool(identity_critical.get(name))
        if over and not critical:
            actions: tuple[str, ...] = ("remove", "swap", "add")
            kind = "remove_outlier"
        elif over:
            actions = ("add", "swap")  # dilute; cannot remove identity role
            kind = "fix_share"
        else:
            actions = ("add", "swap")
            kind = "fix_share"
        _add(
            Slot(
                slot_id=f"{kind}::{name.lower()}",
                kind=kind,
                preferred_actions=actions,
                reason=(
                    f"Share term '{name}' is RED ({value:.3f} vs IQR [{q25:.3f}, {q75:.3f}]); "
                    + ("over-represented" if over else "under-represented")
                ),
                basis_term=name,
                direction="over" if over else "under",
                target_line_label=name if kind == "remove_outlier" else None,
                priority=_KIND_PRIORITY[kind],
                constraints={"identity_critical": critical},
            )
        )

    # 5) Fallback: generic improve slot so propose always has one.
    if not slots:
        _add(
            Slot(
                slot_id="improve",
                kind="improve",
                preferred_actions=("add", "swap", "remove"),
                reason=diagnosis.get("meaning") or "General improvement pass",
                priority=_KIND_PRIORITY["improve"],
            )
        )

    slots.sort(key=lambda s: (s.priority, s.slot_id))
    return slots[: max(1, int(max_slots))]


def slots_to_dicts(slots: list[Slot]) -> list[dict[str, Any]]:
    return [s.to_dict() for s in slots]
