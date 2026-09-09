"""Rule-based ingredient line parser (ported from kadin-dev portion-test.ipynb)."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from parse_recipe_ingredient import PARSE_FIELDS
from unit_aliases import ALLOWED_UNITS, UNIT_ALIASES, normalize_unit

SIZE_WORDS = {
    "small",
    "medium",
    "large",
    "jumbo",
    "extra-large",
    "extra",
    "whole",
}

PREP_WORDS = {
    "chopped",
    "diced",
    "minced",
    "sliced",
    "crushed",
    "grated",
    "shredded",
    "beaten",
    "melted",
    "softened",
    "drained",
    "rinsed",
    "peeled",
    "cooked",
    "uncooked",
    "fresh",
    "frozen",
    "dry",
    "dried",
    "ground",
}

TEXT_AMOUNTS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
}

RULE_PARSE_METHOD = "rule"


def parse_quantity(qty_text: str | None) -> float | None:
    if qty_text is None:
        return None

    qty_text = str(qty_text).strip().lower()
    if qty_text in TEXT_AMOUNTS:
        return float(TEXT_AMOUNTS[qty_text])

    try:
        if " " in qty_text:
            parts = qty_text.split()
            return float(Fraction(parts[0])) + float(Fraction(parts[1]))
        return float(Fraction(qty_text))
    except Exception:
        return None


def _map_rule_status(status: str) -> str:
    if status == "parsed":
        return "ok"
    return status


def simple_rule_parse(line: str) -> dict[str, Any]:
    """Parse one ingredient line with deterministic rules."""
    line = str(line).strip()
    line = re.sub(r"\s+", " ", line)

    pattern = (
        r"^((?:\d+\s+\d+/\d+)|(?:\d+/\d+)|(?:\d+(?:\.\d+)?)|"
        r"a|an|one|two|three|four)\s+([A-Za-z.]+)?\s*(.*)$"
    )
    match = re.match(pattern, line, flags=re.IGNORECASE)

    if not match:
        return {
            "ingredient_raw": line,
            "quantity": None,
            "unit": None,
            "food_name": line,
            "size": None,
            "preparation": None,
            "parse_source": RULE_PARSE_METHOD,
            "parse_status": "no_quantity",
            "confidence": 0.35,
        }

    qty_raw, possible_unit, rest = match.groups()
    quantity = parse_quantity(qty_raw)
    possible_unit_clean = possible_unit.lower().replace(".", "") if possible_unit else None
    unit = normalize_unit(possible_unit)
    size = None

    if unit is None and possible_unit_clean in SIZE_WORDS:
        size = possible_unit_clean
        unit = "each"
        food_name = rest.strip()
        status = "parsed_size_as_count"
        confidence = 0.80
    elif unit is None:
        unit = "each"
        food_name = f"{possible_unit or ''} {rest}".strip()
        status = "quantity_no_known_unit"
        confidence = 0.60
    else:
        food_name = rest.strip()
        status = "parsed"
        confidence = 0.90

    tokens = food_name.lower().replace(",", "").split()
    prep_found = [t for t in tokens if t in PREP_WORDS]
    preparation = ", ".join(sorted(set(prep_found))) if prep_found else None

    clean_food_tokens = [
        t for t in food_name.split() if t.lower().strip(",") not in PREP_WORDS
    ]
    clean_food_name = " ".join(clean_food_tokens).strip(" ,")
    if clean_food_name == "":
        clean_food_name = food_name

    return {
        "ingredient_raw": line,
        "quantity": quantity,
        "unit": unit,
        "food_name": clean_food_name,
        "size": size,
        "preparation": preparation,
        "parse_source": RULE_PARSE_METHOD,
        "parse_status": status,
        "confidence": confidence,
    }


def rule_parse_fields(text: str) -> dict[str, Any]:
    """Return unified parse fields aligned with parse_recipe_ingredient.PARSE_FIELDS."""
    empty = {field: None for field in PARSE_FIELDS}
    empty["parse_status"] = "empty"
    empty["parse_method"] = None

    if text is None or not str(text).strip():
        return {**empty, "confidence": None}

    raw = simple_rule_parse(str(text).strip())
    status = _map_rule_status(raw["parse_status"])

    return {
        "quantity": raw["quantity"],
        "quantity_max": raw["quantity"],
        "unit": raw["unit"],
        "unit_raw": raw["unit"],
        "amount_text": None,
        "name": raw["food_name"],
        "size": raw["size"],
        "preparation": raw["preparation"],
        "parse_status": status,
        "parse_method": RULE_PARSE_METHOD,
        "confidence": raw["confidence"],
    }


def rules_accepted(result: dict[str, Any], *, threshold: float = 0.80) -> bool:
    """True when rule parse is high-confidence enough to skip LLM."""
    status = result.get("parse_status")
    confidence = result.get("confidence")
    if confidence is None or confidence < threshold:
        return False
    return status in ("ok", "parsed_size_as_count")
