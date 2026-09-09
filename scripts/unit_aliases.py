"""Exhaustive recipe unit synonym → canonical maps (single source of truth).

Canonical volume/mass names align with ``unit_convert.VOLUME_TO_ML`` /
``MASS_TO_GRAM`` keys (e.g. ``fluid_ounce``, ``tablespoon``, ``cup``).

Used by rule parsing, gram resolution, dequant embedding text, and
``unit_convert`` alias lookup.
"""

from __future__ import annotations

import re
from typing import Final

# Match before bare mass ``oz`` in combined strings like ``fl oz``.
FLUID_OUNCE_TEXT_RE = re.compile(r"\bfl\.?\s*oz\b|\bfluid\s+ounces?\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Canonical unit names (values in synonym maps)
# ---------------------------------------------------------------------------

VOLUME_UNITS: Final[frozenset[str]] = frozenset(
    {
        "teaspoon",
        "tablespoon",
        "fluid_ounce",
        "cup",
        "pint",
        "quart",
        "gallon",
        "milliliter",
        "liter",
        "cubic_inch",
    }
)

MASS_UNITS: Final[frozenset[str]] = frozenset(
    {
        "gram",
        "kilogram",
        "ounce",
        "pound",
    }
)

COUNT_UNITS: Final[frozenset[str]] = frozenset(
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
        "pinch",
        "dash",
        "each",
    }
)

ALL_CANONICAL_UNITS: Final[frozenset[str]] = VOLUME_UNITS | MASS_UNITS | COUNT_UNITS


def _expand_synonym_groups(groups: dict[str, list[str]]) -> dict[str, str]:
    """Map every synonym variant (lowered, no trailing period) → canonical."""
    out: dict[str, str] = {}
    for canonical, variants in groups.items():
        out[canonical] = canonical
        for variant in variants:
            key = variant.lower().replace(".", "").strip()
            if not key:
                continue
            out[key] = canonical
    return out


VOLUME_SYNONYM_GROUPS: dict[str, list[str]] = {
    "cup": [
        "cup",
        "cups",
        "c",
        "c.",
        "C",
        "C.",
    ],
    "tablespoon": [
        "tablespoon",
        "tablespoons",
        "tbsp",
        "tbsp.",
        "tbspn",
        "tbspn.",
        "tbs",
        "tbs.",
        "tb",
        "tb.",
        "tbsps",
        "tbsps.",
        "T",
        "T.",
        "Tb",
        "Tb.",
        "TB",
        "TB.",
        "Tbl",
        "Tbl.",
        "Tbls",
        "Tbls.",
        "tbls",
        "tbls.",
        "Tbsp",
        "Tbsp.",
        "Tbs",
        "Tbs.",
    ],
    "teaspoon": [
        "teaspoon",
        "teaspoons",
        "tsp",
        "tsp.",
        "tspn",
        "tspn.",
        "tsps",
        "tsps.",
        "t",
        "t.",
        "Ts",
        "Ts.",
        "Tsp",
        "Tsp.",
    ],
    "fluid_ounce": [
        "fluid ounce",
        "fluid ounces",
        "fluid oz",
        "fluid oz.",
        "fl oz",
        "fl. oz",
        "fl. oz.",
        "floz",
        "floz.",
        "fl-oz",
        "oz fl",
    ],
    "pint": [
        "pint",
        "pints",
        "pt",
        "pt.",
        "pts",
        "pts.",
    ],
    "quart": [
        "quart",
        "quarts",
        "qt",
        "qt.",
        "qts",
        "qts.",
    ],
    "gallon": [
        "gallon",
        "gallons",
        "gal",
        "gal.",
        "gals",
        "gals.",
    ],
    "milliliter": [
        "milliliter",
        "milliliters",
        "millilitre",
        "millilitres",
        "ml",
        "ml.",
        "mL",
        "cc",
        "cc.",
        "ccs",
        "cubic centimeter",
        "cubic centimeters",
        "cubic centimetre",
        "cubic centimetres",
        "cubic cm",
        "cu cm",
        "cu. cm",
    ],
    "liter": [
        "liter",
        "liters",
        "litre",
        "litres",
        "l",
        "l.",
        "L",
        "L.",
    ],
    "cubic_inch": [
        "cubic inch",
        "cubic inches",
        "cu in",
        "cu. in.",
        "cu in.",
        "cubic in",
    ],
}

MASS_SYNONYM_GROUPS: dict[str, list[str]] = {
    "gram": [
        "gram",
        "grams",
        "g",
        "g.",
        "gm",
        "gm.",
        "gms",
        "gr",
        "gr.",
    ],
    "kilogram": [
        "kilogram",
        "kilograms",
        "kg",
        "kg.",
        "kilo",
        "kilos",
        "kgs",
    ],
    "ounce": [
        "ounce",
        "ounces",
        "oz",
        "oz.",
        "OZ",
        "OZ.",
    ],
    "pound": [
        "pound",
        "pounds",
        "lb",
        "lb.",
        "lbs",
        "lbs.",
        "LB",
        "LB.",
        "LBS",
        "#",
    ],
}

COUNT_SYNONYM_GROUPS: dict[str, list[str]] = {
    "can": ["can", "cans"],
    "package": [
        "package",
        "packages",
        "pkg",
        "pkg.",
        "pkgs",
        "pkgs.",
        "pack",
        "packs",
    ],
    "box": ["box", "boxes"],
    "jar": ["jar", "jars"],
    "bottle": ["bottle", "bottles"],
    "stick": ["stick", "sticks"],
    "slice": ["slice", "slices"],
    "piece": ["piece", "pieces", "pc", "pc.", "pcs", "pcs."],
    "clove": ["clove", "cloves"],
    "bunch": ["bunch", "bunches"],
    "head": ["head", "heads"],
    "stalk": ["stalk", "stalks"],
    "sprig": ["sprig", "sprigs"],
    "pinch": ["pinch", "pinches"],
    "dash": ["dash", "dashes"],
    "each": ["each", "ea", "ea."],
}

VOLUME_SYNONYMS: dict[str, str] = _expand_synonym_groups(VOLUME_SYNONYM_GROUPS)
MASS_SYNONYMS: dict[str, str] = _expand_synonym_groups(MASS_SYNONYM_GROUPS)
COUNT_SYNONYMS: dict[str, str] = _expand_synonym_groups(COUNT_SYNONYM_GROUPS)

# Flat map for rule parsing / amount_kind (volume + mass + count).
UNIT_ALIASES: dict[str, str] = {
    **VOLUME_SYNONYMS,
    **MASS_SYNONYMS,
    **COUNT_SYNONYMS,
}

ALLOWED_UNITS: list[str] = sorted(set(UNIT_ALIASES.values()))

# Multi-word volume phrases → canonical (longest / most specific first).
MULTIWORD_UNIT_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bfluid\s+ounces?\b", re.IGNORECASE), "fluid_ounce"),
    (re.compile(r"\bfluid\s+oz\.?\b", re.IGNORECASE), "fluid_ounce"),
    (FLUID_OUNCE_TEXT_RE, "fluid_ounce"),
    (re.compile(r"\bcubic\s+centimet(?:er|re)s?\b", re.IGNORECASE), "milliliter"),
    (re.compile(r"\bcubic\s+centimeters?\b", re.IGNORECASE), "milliliter"),
    (re.compile(r"\bcubic\s+cm\b", re.IGNORECASE), "milliliter"),
    (re.compile(r"\bcu\.?\s*cm\b", re.IGNORECASE), "milliliter"),
    (re.compile(r"\bcubic\s+inches?\b", re.IGNORECASE), "cubic_inch"),
    (re.compile(r"\bcu\.?\s*in\.?\b", re.IGNORECASE), "cubic_inch"),
)

# Case-sensitive single-char / short tokens (checked before lowercasing).
_CASE_SENSITIVE_UNIT: dict[str, str] = {
    "T": "tablespoon",
    "T.": "tablespoon",
    "L": "liter",
    "L.": "liter",
}


def _unit_lookup_key(raw: str) -> str | None:
    text = str(raw).strip()
    if not text:
        return None
    if text in _CASE_SENSITIVE_UNIT:
        return _CASE_SENSITIVE_UNIT[text]
    if FLUID_OUNCE_TEXT_RE.search(text):
        return "fluid_ounce"
    return text.lower().replace(".", "").strip()


def normalize_unit(unit_raw: str | None) -> str | None:
    """Map a parsed unit token to a canonical unit name, or None if unknown."""
    if unit_raw is None:
        return None
    text = str(unit_raw).strip()
    if not text:
        return None
    if text in _CASE_SENSITIVE_UNIT:
        return _CASE_SENSITIVE_UNIT[text]
    if FLUID_OUNCE_TEXT_RE.search(text):
        return "fluid_ounce"
    key = text.lower().replace(".", "").strip()
    return UNIT_ALIASES.get(key)


def normalize_units_in_text(text: str) -> str:
    """Replace recognized unit tokens in free text with canonical unit names.

    Used for dequantified ingredient lines so ``c. milk``, ``cup milk``, and
    ``cups milk`` all become ``cup milk`` before embedding / lexical match.
    """
    if not text or not str(text).strip():
        return text

    out = re.sub(r"\s+", " ", str(text).strip())
    for pattern, canonical in MULTIWORD_UNIT_REPLACEMENTS:
        out = pattern.sub(canonical, out)

    tokens = out.split()
    normalized: list[str] = []
    for tok in tokens:
        canon = normalize_unit(tok)
        normalized.append(canon if canon is not None else tok)
    return " ".join(normalized)


def volume_synonyms_for_unit_convert() -> dict[str, str]:
    """Alias map keyed like ``unit_convert._clean_unit_token`` output."""
    extra: dict[str, str] = {}
    for key, canonical in VOLUME_SYNONYMS.items():
        extra[key] = canonical
    # ``unit_convert`` uses spaced ``fluid ounce`` as the cleaned fl-oz token.
    extra["fluid ounce"] = "fluid_ounce"
    extra["fluid ounces"] = "fluid_ounce"
    # Case-sensitive tokens preserved by ``_clean_unit_token``.
    extra["T"] = "tablespoon"
    return extra


def mass_synonyms_for_unit_convert() -> dict[str, str]:
    return dict(MASS_SYNONYMS)
