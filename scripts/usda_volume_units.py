"""Shared volume-unit detection for USDA portion rows (used by EDA + export)."""

from __future__ import annotations

import re

# Word-boundary matching: cup/cups are whole words only (not cupcake, coconut, cupboard).
# Includes abbreviated and full unit names searched in modifier, portion_description,
# and measure_unit.name.
VOLUME_PATTERN = re.compile(
    r"\b("
    r"cups?|"
    r"tsp|teaspoon|teaspoons|"
    r"tbsp|tablespoon|tablespoons|"
    r"fl\.?\s*oz|fluid\s+ounces?|"
    r"liter|litre|liters|litres|"
    r"ml|milliliter|milliliters|"
    r"pint|pints|quart|quarts|gallon|gallons|"
    r"cubic\s*inch|cubic\s*inches|cubic\s*centimeter|cubic\s*centimeters|cubic\s*cm|"
    r"cc"
    r")\b",
    re.IGNORECASE,
)

MASS_PATTERN = re.compile(
    r"\b(oz|ounce|ounces|lb|lbs|pound|pounds|g|gram|grams|kg|kilogram|mg|milligram)s?\b",
    re.IGNORECASE,
)


def text_has_volume(*parts: str) -> bool:
    combined = " ".join(p for p in parts if p)
    return bool(VOLUME_PATTERN.search(combined))


def text_has_fluid_ounce(*parts: str) -> bool:
    from unit_convert import FLUID_OUNCE_TEXT_RE

    combined = " ".join(p for p in parts if p)
    return bool(FLUID_OUNCE_TEXT_RE.search(combined))


def classify_modifier_text(modifier: str) -> str:
    text = (modifier or "").strip()
    if not text:
        return "other"
    if text.isdigit():
        return "other"
    if text_has_fluid_ounce(text):
        return "volume"
    has_mass = bool(MASS_PATTERN.search(text))
    has_volume = bool(VOLUME_PATTERN.search(text))
    if has_mass and not has_volume:
        return "mass"
    if has_volume and not has_mass:
        return "volume"
    return "other"
