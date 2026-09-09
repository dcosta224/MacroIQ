"""Convert recipe portion quantities between volume and mass units.

Uses US customary definitions for kitchen units (cup, tbsp, tsp, pint, quart,
fluid ounce, gallon) and SI for metric (mL, L, g, kg). Mass ounces are
avoirdupois (weight oz), not fluid ounces — use ``fl oz`` / ``fluid ounce``
for volume.

Primary entry points:
  convert_volume(quantity, from_unit, to_unit)
  convert_mass(quantity, from_unit, to_unit)
  convert_unit_quantity(quantity, from_unit, to_unit)  # auto-detect dimension

Example:
  convert_volume(2, "cups", "mL")          # → 473.176...
  convert_mass(1, "lb", "g")               # → 453.592...
  convert_unit_quantity(3, "tbsp", "tsp")   # → 9.0
"""

from __future__ import annotations

import re
from numbers import Real
from typing import Literal

from unit_aliases import FLUID_OUNCE_TEXT_RE, mass_synonyms_for_unit_convert, volume_synonyms_for_unit_convert

UnitKind = Literal["volume", "mass"]

# US customary + metric; values are multipliers to the base unit (mL or g).
VOLUME_TO_ML: dict[str, float] = {
    "teaspoon": 4.92892159375,
    "tablespoon": 14.7867647813,
    "fluid_ounce": 29.5735295625,
    "cup": 236.5882365,
    "pint": 473.176473,
    "quart": 946.352946,
    "gallon": 3785.411784,
    "milliliter": 1.0,
    "liter": 1000.0,
    "cubic_inch": 16.387064,
}

MASS_TO_GRAM: dict[str, float] = {
    "gram": 1.0,
    "kilogram": 1000.0,
    "ounce": 28.349523125,
    "pound": 453.59237,
}

VOLUME_ALIASES: dict[str, str] = volume_synonyms_for_unit_convert()

MASS_ALIASES: dict[str, str] = mass_synonyms_for_unit_convert()


class UnitConversionError(ValueError):
    """Raised for unknown units or incompatible unit dimensions."""


def _clean_unit_token(unit: str) -> str:
    text = str(unit).strip()
    if not text:
        raise UnitConversionError("Unit string is empty")
    if text == "T":
        return "T"
    lowered = text.lower()
    if FLUID_OUNCE_TEXT_RE.search(lowered):
        return "fluid ounce"
    normalized = lowered.replace(".", "")
    return re.sub(r"\s+", " ", normalized).strip()


def canonical_volume_token(raw: str) -> str | None:
    """Map a volume token (from regex or USDA text) to a canonical unit, or None."""
    if not raw or not str(raw).strip():
        return None
    try:
        return normalize_volume_unit(str(raw))
    except UnitConversionError:
        return None


def normalize_volume_unit(unit: str) -> str:
    """Map a volume unit string to a canonical name."""
    token = _clean_unit_token(unit)
    canonical = VOLUME_ALIASES.get(token, token)
    if canonical not in VOLUME_TO_ML:
        raise UnitConversionError(f"Unknown volume unit: {unit!r}")
    return canonical


def normalize_mass_unit(unit: str) -> str:
    """Map a mass unit string to a canonical name."""
    token = _clean_unit_token(unit)
    canonical = MASS_ALIASES.get(token, token)
    if canonical not in MASS_TO_GRAM:
        raise UnitConversionError(f"Unknown mass unit: {unit!r}")
    return canonical


def unit_kind(unit: str) -> UnitKind:
    """Return ``volume`` or ``mass`` for a unit string."""
    if FLUID_OUNCE_TEXT_RE.search(str(unit)):
        return "volume"
    token = _clean_unit_token(unit)
    if token in VOLUME_ALIASES or token in VOLUME_TO_ML:
        return "volume"
    if token in MASS_ALIASES or token in MASS_TO_GRAM:
        return "mass"
    raise UnitConversionError(f"Unknown unit (not volume or mass): {unit!r}")


def _convert(
    quantity: Real,
    from_unit: str,
    to_unit: str,
    *,
    factors: dict[str, float],
    normalize,
) -> float:
    q = float(quantity)
    src = normalize(from_unit)
    dst = normalize(to_unit)
    if src == dst:
        return q
    base = q * factors[src]
    return base / factors[dst]


def convert_volume(quantity: Real, from_unit: str, to_unit: str) -> float:
    """Convert ``quantity`` from ``from_unit`` to ``to_unit`` (volume only)."""
    return _convert(
        quantity,
        from_unit,
        to_unit,
        factors=VOLUME_TO_ML,
        normalize=normalize_volume_unit,
    )


def convert_mass(quantity: Real, from_unit: str, to_unit: str) -> float:
    """Convert ``quantity`` from ``from_unit`` to ``to_unit`` (mass only)."""
    return _convert(
        quantity,
        from_unit,
        to_unit,
        factors=MASS_TO_GRAM,
        normalize=normalize_mass_unit,
    )


def convert_unit_quantity(quantity: Real, from_unit: str, to_unit: str) -> float:
    """
    Convert ``quantity`` between two units of the same dimension.

    Dispatches to volume or mass conversion based on the unit strings.
    Raises ``UnitConversionError`` if units differ in dimension (e.g. cup → g).
    """
    from_kind = unit_kind(from_unit)
    to_kind = unit_kind(to_unit)
    if from_kind != to_kind:
        raise UnitConversionError(
            f"Cannot convert {from_kind} unit {from_unit!r} to {to_kind} unit {to_unit!r}. "
            "Use food density for volume ↔ mass."
        )
    if from_kind == "volume":
        return convert_volume(quantity, from_unit, to_unit)
    return convert_mass(quantity, from_unit, to_unit)
