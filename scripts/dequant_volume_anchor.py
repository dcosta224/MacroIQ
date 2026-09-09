"""Volume-stem dequant_norm cache keys and cross-unit portion lookup."""

from __future__ import annotations

from typing import Any

from unit_convert import UnitConversionError, unit_kind

VOLUME_STEM_PREFIX = "vol:"


def split_volume_dequant_norm(dequant_norm: str) -> tuple[str, str] | None:
    """Return (recipe_unit, ingredient_stem) when norm is volume-prefixed."""
    tokens = str(dequant_norm).strip().lower().split()
    if len(tokens) < 2:
        return None
    unit_token = tokens[0]
    try:
        if unit_kind(unit_token) != "volume":
            return None
    except UnitConversionError:
        return None
    stem = " ".join(tokens[1:]).strip()
    if not stem:
        return None
    return unit_token, stem


def volume_stem(dequant_norm: str) -> str | None:
    split = split_volume_dequant_norm(dequant_norm)
    return split[1] if split else None


def leading_volume_unit(dequant_norm: str) -> str | None:
    split = split_volume_dequant_norm(dequant_norm)
    return split[0] if split else None


def volume_stem_cache_key(stem: str) -> str:
    return f"{VOLUME_STEM_PREFIX}{stem}"


def is_volume_stem_cache_key(key: str) -> bool:
    return str(key).startswith(VOLUME_STEM_PREFIX)


def lookup_dequant_cache_entry(
    entries: dict[str, dict[str, Any]],
    dequant_norm: str,
) -> dict[str, Any] | None:
    """Exact dequant_norm match, then volume-stem anchor for cross-unit portions."""
    entry = entries.get(dequant_norm)
    if entry is not None:
        return entry
    stem = volume_stem(dequant_norm)
    if not stem:
        return None
    anchor = entries.get(volume_stem_cache_key(stem))
    if anchor is not None and anchor.get("volume_portion_anchor"):
        if anchor.get("curator_fixed_scale") or anchor.get("curator_scale_quantity") is not None:
            return None
        if anchor.get("curator_manual_volume"):
            return None
        return anchor
    return None


def sibling_volume_dequant_norms(
    dequant_norm: str,
    candidates: list[dict[str, Any]],
) -> list[str]:
    """All volume-prefixed norms sharing the same ingredient stem."""
    stem = volume_stem(dequant_norm)
    if not stem:
        return [dequant_norm]
    siblings = [
        str(item["dequant_norm"])
        for item in candidates
        if volume_stem(str(item["dequant_norm"])) == stem
    ]
    return sorted(set(siblings)) or [dequant_norm]
