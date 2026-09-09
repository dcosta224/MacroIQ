"""Hard culinary-clash gates for add/swap candidates before apply.

Suite A failures were dominated by identity clashes the prompts already forbid
(dairy stacking, onion rings, coffee-in-bread, rice-in-barbacoa-style noise).
This module enforces those rules numerically so auto-apply / decide cannot
buy a tiny LP gain with a clash ingredient.
"""

from __future__ import annotations

import re
from typing import Any

from recipe_opt_agent.culinary_types import families_for_text

# Absolute FDC / catalog junk that must never enter a savory dish via edit.
_DENYLIST_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\bonion\s+rings?\b",
        r"\bfast\s+foods?\b",
        r"\bbreaded\s+and\s+fried\b",
        r"\bbrewed\s+coffee\b",
        r"\bcoffee,\s*brewed\b",
        r"\bespresso\b",
        r"\bspaghetti,?\s+with\s+meatballs\b",
        r"\bcanned\s+spaghetti\b",
        r"\bice\s+cream\b",
        r"\bfrozen\s+dessert\b",
    )
)

# Culinary families that should not be stacked when already present.
_STACK_FAMILIES: frozenset[str] = frozenset(
    {
        "cheese",
        "yogurt",
        "milk",
        "egg",
        "butter",
        "cream",
    }
)

# Soft protein hacks that clash unless the dish already uses them / OOD+structure.
_SOFT_PROTEIN_CLASH: frozenset[str] = frozenset({"yogurt", "tofu", "milk"})


def _labels_from_problem(problem: dict[str, Any] | None) -> list[str]:
    labels: list[str] = []
    for row in ((problem or {}).get("chosen_recipe") or {}).get("ingredients") or []:
        lab = str(row.get("label") or row.get("name") or "").strip()
        if lab:
            labels.append(lab)
    for row in (problem or {}).get("grounded_r0") or []:
        lab = str(row.get("label") or "").strip()
        if lab:
            labels.append(lab)
    return labels


def is_denylist_label(label: str) -> tuple[bool, str]:
    lab = (label or "").strip()
    if not lab:
        return False, ""
    for pat in _DENYLIST_PATTERNS:
        if pat.search(lab):
            return True, f"denylist:{pat.pattern}"
    return False, ""


def families_in_recipe(labels: list[str]) -> set[str]:
    hit: set[str] = set()
    for lab in labels:
        hit |= families_for_text(lab)
    return hit


def clash_reason_for_label(
    label: str,
    *,
    current_labels: list[str],
    allow_ood: bool = False,
    title: str | None = None,
) -> str | None:
    """Return a short reason if ``label`` must be blocked, else None."""
    lab = (label or "").strip()
    if not lab:
        return "empty_label"

    denied, detail = is_denylist_label(lab)
    if denied:
        return detail

    new_fams = families_for_text(lab)
    present = families_in_recipe(current_labels)

    # Onion rings / fried-onion snacks must not match plain onion ideation.
    if "onion_rings" in new_fams:
        return "clash_family:onion_rings"

    # Coffee does not belong in bread/dough dishes.
    title_l = (title or "").lower()
    if "coffee" in new_fams and any(
        t in title_l for t in ("focaccia", "bread", "pizza", "dough", "bagel", "naan")
    ):
        return "clash_family:coffee_in_bread"

    # Dairy / egg stacking: second member of the same soft family.
    for fam in _STACK_FAMILIES:
        if fam in new_fams and fam in present:
            # Swapping within family is fine; adding another hit is not.
            return f"family_already_represented:{fam}"

    # Soft protein hacks into meat dishes (Suite A: yogurt in al pastor / baba ganoush stacking).
    if not allow_ood and (new_fams & _SOFT_PROTEIN_CLASH):
        if present & {"cheese", "egg", "yogurt", "milk"} and present & {
            "pork_rib",
            "ground_beef",
            "turkey",
            "chicken",
            "beef",
            "pork",
        }:
            return "soft_protein_hack_on_rich_dish"
        if "yogurt" in new_fams and ("cheese" in present or "yogurt" in present):
            return "family_already_represented:yogurt"
        meaty_title = any(
            t in title_l
            for t in (
                "al pastor",
                "barbacoa",
                "biryani",
                "arroz",
                "pollo",
                "carbonara",
                "rib",
                "bourguignon",
                "enchilada",
            )
        )
        if meaty_title and (new_fams & {"yogurt", "tofu", "milk"}):
            return "soft_protein_hack_on_meat_dish"
        # Dip / puree dishes: yogurt is ok alone, but not stacked with egg/tofu protein hacks.
        if "yogurt" in new_fams and present & {"egg", "tofu"}:
            return "yogurt_stacked_with_protein_hack"
        if "tofu" in new_fams and present & {"egg", "yogurt", "cheese"}:
            return "tofu_stacked_on_dairy_egg_dish"

    # Rice / butter into meat stews titled barbacoa / al pastor / carnitas-style.
    meaty = any(
        t in title_l
        for t in ("barbacoa", "al pastor", "carnitas", "birria", "pulled pork", "bbq rib")
    )
    if meaty and "rice" in new_fams:
        return "clash_structure:rice_in_meat_stew"
    if meaty and "butter" in new_fams and "butter" not in present:
        return "clash_structure:butter_in_meat_stew"

    return None


def filter_candidates_by_clash_gates(
    candidates: list[dict[str, Any]],
    *,
    problem: dict[str, Any] | None,
    title: str | None = None,
    min_nutrient_gain_to_allow_ood: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop add/swap candidates that fail hard clash gates."""
    current = _labels_from_problem(problem)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for c in candidates:
        action = str(c.get("action") or "")
        if action not in {"add", "swap"}:
            kept.append(c)
            continue
        meta = dict(c.get("meta") or {})
        if meta.get("allow_clash"):
            kept.append(c)
            continue
        label = str(c.get("label") or c.get("name") or "")
        branch = str(c.get("branch") or meta.get("branch") or "")
        ood = bool(meta.get("ood") or str(branch).startswith("ood"))
        # Large explicit nutrient rescue can keep a single OOD lean protein.
        nutrient_gain = meta.get("expected_nutrient_gain")
        allow_ood = False
        if ood and nutrient_gain is not None:
            try:
                allow_ood = float(nutrient_gain) >= float(min_nutrient_gain_to_allow_ood)
            except (TypeError, ValueError):
                allow_ood = False
        # Structured OOD lean poultry/meat is allowed through family stack checks
        # but never through denylist.
        denied, _ = is_denylist_label(label)
        if denied:
            reason = clash_reason_for_label(label, current_labels=current, title=title)
            dropped.append({"candidate": c, "reason": "clash_gate", "detail": reason or "denylist"})
            continue
        reason = clash_reason_for_label(
            label,
            current_labels=current,
            allow_ood=ood or allow_ood,
            title=title,
        )
        if reason is None:
            kept.append({**c, "clash_gate": "pass"})
            continue
        # OOD lean poultry/beef adds that only trip soft_protein are ok when structured.
        if ood and reason.startswith("soft_protein") and not denied:
            kept.append({**c, "clash_gate": f"ood_override:{reason}"})
            continue
        dropped.append({"candidate": c, "reason": "clash_gate", "detail": reason})
    return kept, dropped


def bundle_clash_detail(
    bundle: dict[str, Any] | None,
    *,
    problem: dict[str, Any] | None,
    title: str | None = None,
) -> list[dict[str, Any]]:
    """Return clash details for each add/swap edit in a scored bundle."""
    if not bundle:
        return []
    current = _labels_from_problem(problem)
    hits: list[dict[str, Any]] = []
    for e in bundle.get("edits") or []:
        if str(e.get("action") or "") not in {"add", "swap"}:
            continue
        label = str(e.get("label") or e.get("name") or "")
        meta = e.get("meta") or {}
        ood = bool(meta.get("ood") or str(e.get("branch") or "").startswith("ood"))
        reason = clash_reason_for_label(
            label, current_labels=current, allow_ood=ood, title=title
        )
        if reason:
            hits.append({"label": label, "detail": reason, "edit": e})
    return hits
