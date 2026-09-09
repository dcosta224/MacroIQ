"""Classify parsed ingredient amounts as mass, volume, count, or unmeasurable."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from parse_recipe_ingredient import is_measurable
from recipe_parse_rules import UNIT_ALIASES, normalize_unit
from unit_convert import UnitConversionError, unit_kind

AmountKind = Literal["mass", "volume", "count", "unmeasurable", "unknown"]

# Count/container units from recipe_parse_rules that are not mass or volume.
COUNT_UNITS = frozenset(
    {
        "can",
        "package",
        "box",
        "jar",
        "bottle",
        "stick",
        "slice",
        "piece",
        "clove",
        "bunch",
        "head",
        "stalk",
        "sprig",
        "each",
    }
)

# Tiny imprecise volume amounts (treated as volume, often negligible calories).
MICRO_VOLUME_UNITS = frozenset({"pinch", "dash"})

# Vague count units that are not reliably resolvable to grams.
VAGUE_COUNT_UNITS = frozenset({"bunch", "sprig"})


def missing_quantity(quantity: Any) -> bool:
    """True when the parsed line has no numeric amount (None or NaN)."""
    if quantity is None:
        return True
    try:
        if pd.isna(quantity):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _singularize(token: str) -> str:
    token = token.lower().strip()
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 3:
        base = token[:-2]
        if base.endswith(("s", "x", "z", "ch", "sh")):
            return base
    if token.endswith("s") and len(token) > 2 and not token.endswith("ss"):
        return token[:-1]
    return token


def _name_count_tokens(name: str | None) -> list[str]:
    if not name or not str(name).strip():
        return []
    tokens: list[str] = []
    for raw in str(name).lower().replace(",", " ").split():
        tok = raw.strip(".,;")
        if not tok or tok in {"and", "or", "of", "the", "a", "an"}:
            continue
        tokens.append(tok)
        sing = _singularize(tok)
        if sing != tok:
            tokens.append(sing)
    return list(dict.fromkeys(tokens))


def normalize_count_unit(unit: str | None) -> str | None:
    if unit is None or not str(unit).strip():
        return None
    return normalize_unit(str(unit))


def infer_count_query(unit: str | None, name: str | None) -> list[str]:
    """Tokens to match against food_portion count labels."""
    queries: list[str] = []
    norm = normalize_count_unit(unit)
    if norm:
        if norm in COUNT_UNITS:
            queries.append(norm)
            if norm == "each":
                queries.extend(_name_count_tokens(name))
            elif norm.endswith("s"):
                queries.append(_singularize(norm))
    elif name:
        queries.extend(_name_count_tokens(name))
    return list(dict.fromkeys(q for q in queries if q))


def _normalize_parsed_unit(unit: str | None) -> str | None:
    norm = normalize_count_unit(unit)
    if norm:
        return norm
    if unit is None:
        return None
    raw = str(unit).strip().lower().replace(".", "")
    if raw.endswith("s") and len(raw) > 2:
        return normalize_count_unit(raw[:-1]) or raw[:-1]
    return raw or None


def classify_amount_kind(
    quantity: Any,
    unit: str | None,
    name: str | None = None,
    *,
    ingredient_raw: str | None = None,
    parse_status: str | None = None,
) -> AmountKind:
    """Return mass | volume | count | unmeasurable | unknown for a parsed line."""
    if ingredient_raw:
        parse_result = {
            "quantity": quantity,
            "unit": unit,
            "name": name,
            "parse_status": parse_status or "ok",
        }
        if not is_measurable(parse_result, ingredient_raw=ingredient_raw):
            return "unmeasurable"

    if missing_quantity(quantity):
        return "unmeasurable"

    norm = _normalize_parsed_unit(unit)
    if norm:
        if norm in MICRO_VOLUME_UNITS:
            return "volume"
        try:
            kind = unit_kind(norm)
            return kind  # mass or volume
        except UnitConversionError:
            if norm in COUNT_UNITS:
                if norm in VAGUE_COUNT_UNITS:
                    return "unknown"
                return "count"
            return "unknown"

    # Implicit count: quantity present, no unit, food name looks like discrete item.
    if name and str(name).strip():
        tokens = _name_count_tokens(name)
        if tokens:
            return "count"

    return "unknown"


def is_micro_volume_unit(unit: str | None) -> bool:
    """True for dash/pinch-style micro volume units."""
    norm = _normalize_parsed_unit(unit)
    return norm in MICRO_VOLUME_UNITS if norm else False


def classify_from_parsed_row(row: dict[str, Any] | Any) -> AmountKind:
    """Classify from a parsed ingredient row (Series or dict)."""
    if hasattr(row, "get"):
        get = row.get
    else:
        get = lambda k, d=None: getattr(row, k, d)

    preset = get("amount_kind_final") or get("amount_kind_override")
    if preset in ("mass", "volume", "count", "unmeasurable", "unknown"):
        return preset  # type: ignore[return-value]

    ingredient = get("ingredient") or get("ingredient_raw")
    return classify_amount_kind(
        get("quantity"),
        get("unit"),
        get("name"),
        ingredient_raw=str(ingredient) if ingredient else None,
        parse_status=get("parse_status"),
    )
