"""Multi-signal resolution plans for ingredient → gram conversion."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Literal

from amount_kind import (
    COUNT_UNITS,
    classify_amount_kind,
    infer_count_query,
    is_micro_volume_unit,
    missing_quantity,
    normalize_count_unit,
)
from recipe_parse_rules import normalize_unit
from unit_convert import UnitConversionError, unit_kind
from usda_volume_units import MASS_PATTERN

ResolutionPath = Literal[
    "embedded_mass",
    "explicit_mass",
    "explicit_volume",
    "count_portion",
    "parenthetical_mass_override",
]

PlanFlag = Literal[
    "unresolvable_serving_only",
    "vague_amount",
    "micro_amount",
    "no_quantity_specified",
    "ambiguous_quantity_accepted",
    "compound_ingredient",
    "negligible_calorie_compound",
]

SIZE_TOKENS = ("small", "medium", "large", "extra-large", "extra large", "jumbo")

PAREN_MASS_RE = re.compile(
    r"\(\s*(?:about|approx(?:\.|imately)?|around)?\s*"
    r"([\d]+(?:\.\d+)?(?:\s+\d+/\d+)?|\d+/\d+)\s*"
    r"(oz|ounce|ounces|lb|lbs|pound|pounds|g|gram|grams|kg|kilogram|kilograms)\s*\.?\s*\)",
    re.IGNORECASE,
)

VAGUE_QUANT_RE = re.compile(
    r"\b(a bit|some|few|handful|pinch|dash)\s+of\b",
    re.IGNORECASE,
)

COMPOUND_RE = re.compile(
    r"\b(?:fresh\s+)?\w[\w\s-]*\s+and\s+\w[\w\s-]+\b",
    re.IGNORECASE,
)

AMBIGUOUS_CONTAINER_RE = re.compile(
    r"\b(small|medium|large)\s+(box|bag|package|pkg|can|jar|bottle)\b",
    re.IGNORECASE,
)

SERVING_ONLY_LABELS = frozenset(
    {
        "serving",
        "servings",
        "racc",
        "undetermined",
    }
)


def _parse_qty_text(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    mixed = re.match(r"^(\d+)\s+(\d+)/(\d+)$", text)
    if mixed:
        return float(mixed.group(1)) + float(Fraction(int(mixed.group(2)), int(mixed.group(3))))
    frac = re.match(r"^(\d+)/(\d+)$", text)
    if frac:
        return float(Fraction(int(frac.group(1)), int(frac.group(2))))
    try:
        return float(text)
    except ValueError:
        return None


def extract_parenthetical_mass(ingredient_raw: str) -> tuple[float | None, str | None]:
    """Extract mass from parentheses, e.g. '(8 oz.)' or '(about 8 ounces)'."""
    match = PAREN_MASS_RE.search(ingredient_raw or "")
    if not match:
        return None, None
    qty = _parse_qty_text(match.group(1))
    unit_raw = match.group(2).lower().rstrip(".")
    unit = normalize_unit(unit_raw) or unit_raw
    if qty is None:
        return None, None
    return qty, unit


def ingredient_has_mass_reference(
    ingredient_raw: str | None,
    *,
    embedded_mass_qty: float | None = None,
    embedded_mass_unit: str | None = None,
) -> bool:
    if embedded_mass_qty is not None and embedded_mass_unit:
        return True
    if not ingredient_raw:
        return False
    if PAREN_MASS_RE.search(ingredient_raw):
        return True
    return bool(MASS_PATTERN.search(ingredient_raw))


def extract_size_tokens(
    ingredient_raw: str | None,
    size: str | None = None,
) -> list[str]:
    found: list[str] = []
    for tok in SIZE_TOKENS:
        if size and tok in str(size).lower():
            found.append(tok.replace(" ", "-") if tok == "extra large" else tok)
    if ingredient_raw:
        low = ingredient_raw.lower()
        for tok in SIZE_TOKENS:
            key = tok if tok != "extra large" else r"extra[- ]large"
            if tok == "extra large":
                if re.search(r"extra[- ]large", low):
                    found.append("extra-large")
            elif re.search(rf"\b{re.escape(tok)}\b", low):
                found.append(tok)
    return list(dict.fromkeys(found))


def extract_count_unit_tokens(
    unit: str | None,
    name: str | None,
    ingredient_raw: str | None = None,
) -> list[str]:
    tokens = infer_count_query(unit, name)
    if ingredient_raw:
        low = ingredient_raw.lower()
        for u in COUNT_UNITS:
            if re.search(rf"\b{re.escape(u)}s?\b", low) and u not in tokens:
                tokens.append(u)
    return tokens


def needs_line_enrichment(
    ingredient_raw: str,
    parse_fields: dict[str, Any],
    plan: "ResolutionPlan",
) -> bool:
    if "micro_amount" in plan.flags:
        return False
    if plan.flags:
        return any(
            f in plan.flags
            for f in ("vague_amount", "compound_ingredient", "ambiguous_quantity_accepted")
        )
    if parse_fields.get("amount_kind") == "unknown":
        return True
    if COMPOUND_RE.search(ingredient_raw):
        return True
    if VAGUE_QUANT_RE.search(ingredient_raw):
        return True
    if AMBIGUOUS_CONTAINER_RE.search(ingredient_raw):
        return True
    if plan.embedded_mass_qty and plan.resolution_paths == ["count_portion", "parenthetical_mass_override"]:
        return True
    return False


@dataclass
class ResolutionPlan:
    resolution_paths: list[ResolutionPath] = field(default_factory=list)
    flags: list[PlanFlag] = field(default_factory=list)
    quantity: float | None = None
    unit: str | None = None
    embedded_mass_qty: float | None = None
    embedded_mass_unit: str | None = None
    parenthetical_mass_qty: float | None = None
    parenthetical_mass_unit: str | None = None
    authoritative_mass_is_total: bool = False
    count_size_tokens: list[str] = field(default_factory=list)
    count_unit_tokens: list[str] = field(default_factory=list)
    authoritative_source: str = "parse"
    is_compound: bool = False
    components: list[str] = field(default_factory=list)
    negligible_calories: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def primary_amount_kind(self) -> str:
        if self.resolution_paths:
            first = self.resolution_paths[0]
            if first in ("embedded_mass", "parenthetical_mass_override", "explicit_mass"):
                return "mass"
            if first == "explicit_volume":
                return "volume"
            if first == "count_portion":
                return "count"
        if self.flags:
            if "compound_ingredient" in self.flags:
                return "unmeasurable"
            if "no_quantity_specified" in self.flags:
                return "unmeasurable"
            if "vague_amount" in self.flags:
                return "unknown"
        return "unknown"

    @property
    def quantity_specified(self) -> bool:
        return not missing_quantity(self.quantity)

    def count_query_tokens(self) -> list[str]:
        unit_tokens = _as_str_list(self.count_unit_tokens)
        size_tokens = _as_str_list(self.count_size_tokens)
        return list(
            dict.fromkeys(unit_tokens + size_tokens + infer_count_query(self.unit, None))
        )


def _normalize_plan_dict(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    for key in ("resolution_paths", "flags", "count_size_tokens", "count_unit_tokens", "components"):
        if key in out:
            out[key] = _as_str_list(out[key])
    return out


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return [str(v) for v in value.tolist()]
    except ImportError:
        pass
    return [str(value)]


def build_resolution_plan(
    parse_fields: dict[str, Any],
    *,
    ingredient_raw: str | None = None,
    enrichment: dict[str, Any] | None = None,
) -> ResolutionPlan:
    """Build ordered resolution paths from rules parse + optional LLM enrichment."""
    line = ingredient_raw or parse_fields.get("ingredient") or ""
    qty = parse_fields.get("quantity")
    unit = parse_fields.get("unit")
    name = parse_fields.get("name")
    size = parse_fields.get("size")

    plan = ResolutionPlan(
        quantity=float(qty) if qty is not None else None,
        unit=str(unit) if unit is not None else None,
        count_size_tokens=extract_size_tokens(line, size),
        count_unit_tokens=extract_count_unit_tokens(unit, name, line),
    )

    paren_qty, paren_unit = extract_parenthetical_mass(line)
    if paren_qty is not None and paren_unit:
        plan.parenthetical_mass_qty = paren_qty
        plan.parenthetical_mass_unit = paren_unit
        plan.embedded_mass_qty = paren_qty
        plan.embedded_mass_unit = paren_unit
        plan.authoritative_source = "rules_paren"

    if missing_quantity(qty):
        if VAGUE_QUANT_RE.search(line):
            plan.flags.append("vague_amount")
        elif "no_quantity_specified" not in plan.flags:
            plan.flags.append("no_quantity_specified")

    if COMPOUND_RE.search(line) and " and " in line.lower():
        plan.flags.append("compound_ingredient")

    if AMBIGUOUS_CONTAINER_RE.search(line):
        plan.flags.append("ambiguous_quantity_accepted")

    if is_micro_volume_unit(str(unit) if unit is not None else None):
        if "micro_amount" not in plan.flags:
            plan.flags.append("micro_amount")

    amount_kind = classify_amount_kind(
        qty,
        unit,
        name,
        ingredient_raw=line or None,
        parse_status=parse_fields.get("parse_status"),
    )

    paths: list[ResolutionPath] = []

    if plan.parenthetical_mass_qty is not None and amount_kind == "count":
        paths.append("parenthetical_mass_override")
        plan.authoritative_mass_is_total = True

    if plan.embedded_mass_qty is not None and "parenthetical_mass_override" not in paths:
        paths.append("embedded_mass")

    if amount_kind == "mass":
        paths.append("explicit_mass")
    elif amount_kind == "volume":
        paths.append("explicit_volume")
    elif amount_kind == "count":
        paths.append("count_portion")

    if not paths and amount_kind == "unknown":
        if plan.embedded_mass_qty:
            paths.append("embedded_mass")
        elif qty is not None and unit:
            try:
                k = unit_kind(normalize_count_unit(unit) or str(unit))
                if k == "mass":
                    paths.append("explicit_mass")
                elif k == "volume":
                    paths.append("explicit_volume")
            except UnitConversionError:
                paths.append("count_portion")

    plan.resolution_paths = list(dict.fromkeys(paths))

    if enrichment:
        _apply_enrichment(plan, enrichment)

    return plan


def _apply_enrichment(plan: ResolutionPlan, enrichment: dict[str, Any]) -> None:
    if enrichment.get("resolution_paths"):
        plan.resolution_paths = list(enrichment["resolution_paths"])
    for flag in enrichment.get("flags") or []:
        if flag not in plan.flags:
            plan.flags.append(flag)
    if enrichment.get("quantity") is not None:
        plan.quantity = float(enrichment["quantity"])
    if enrichment.get("unit") is not None:
        plan.unit = enrichment.get("unit")
    if enrichment.get("embedded_mass_qty") is not None:
        plan.embedded_mass_qty = float(enrichment["embedded_mass_qty"])
    if enrichment.get("embedded_mass_unit"):
        plan.embedded_mass_unit = enrichment["embedded_mass_unit"]
    if enrichment.get("parenthetical_mass_qty") is not None:
        plan.parenthetical_mass_qty = float(enrichment["parenthetical_mass_qty"])
        plan.parenthetical_mass_unit = enrichment.get("parenthetical_mass_unit")
    if enrichment.get("count_size_tokens"):
        plan.count_size_tokens = list(enrichment["count_size_tokens"])
    if enrichment.get("count_unit_tokens"):
        plan.count_unit_tokens = list(enrichment["count_unit_tokens"])
    if enrichment.get("authoritative_mass_is_total"):
        plan.authoritative_mass_is_total = True
    if enrichment.get("is_compound"):
        plan.is_compound = True
        if "compound_ingredient" not in plan.flags:
            plan.flags.append("compound_ingredient")
    if enrichment.get("components"):
        plan.components = list(enrichment["components"])
    if enrichment.get("negligible_calories"):
        plan.negligible_calories = True
        if "negligible_calorie_compound" not in plan.flags:
            plan.flags.append("negligible_calorie_compound")
    plan.authoritative_source = "llm"


def plan_from_parsed_row(row: dict[str, Any] | Any) -> ResolutionPlan:
    if hasattr(row, "get"):
        get = row.get
    else:
        get = lambda k, d=None: getattr(row, k, d)

    preset = get("resolution_plan")
    if isinstance(preset, ResolutionPlan):
        return preset
    if isinstance(preset, dict):
        fields = _normalize_plan_dict(preset)
        plan = ResolutionPlan(**{k: v for k, v in fields.items() if k in ResolutionPlan.__dataclass_fields__})
        return plan

    parse_fields = {
        "quantity": get("quantity"),
        "unit": get("unit"),
        "name": get("name"),
        "size": get("size"),
        "parse_status": get("parse_status"),
        "ingredient": get("ingredient") or get("ingredient_raw"),
        "amount_kind": get("amount_kind_final") or get("amount_kind"),
    }
    enrichment = get("line_enrichment")
    return build_resolution_plan(
        parse_fields,
        ingredient_raw=str(get("ingredient") or get("ingredient_raw") or ""),
        enrichment=enrichment,
    )
