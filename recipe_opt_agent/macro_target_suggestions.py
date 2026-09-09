"""Suggest macro target boxes from resolved neighborhood PFC profiles.

Presets:
  - neighborhood_coverage: per-macro 10pp bands placed to cover the most
    neighborhood recipes on that axis (reachable, compact targets).
  - neighborhood_mean: legacy mean ± pad (kept for compatibility).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_simplex(p: float, c: float, f: float) -> tuple[float, float, float]:
    p, c, f = max(0.0, float(p)), max(0.0, float(c)), max(0.0, float(f))
    s = p + c + f
    if s <= 1e-12:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return p / s, c / s, f / s


def _recipe_pfc_arrays(lines_df) -> dict[str, np.ndarray] | None:
    """Per-recipe protein/carbs/fat calorie fractions as parallel arrays."""
    from recipe_opt_agent.problem_loader import _batch_recipe_pfc_from_lines

    pfc_by_rid = _batch_recipe_pfc_from_lines(lines_df)
    if not pfc_by_rid:
        return None
    ps = np.asarray([float(v.get("protein", 0.0)) for v in pfc_by_rid.values()], dtype=float)
    cs = np.asarray([float(v.get("carbs", 0.0)) for v in pfc_by_rid.values()], dtype=float)
    fs = np.asarray([float(v.get("fat", 0.0)) for v in pfc_by_rid.values()], dtype=float)
    # Renormalize each recipe onto the simplex (guards float drift).
    for i in range(ps.size):
        p, c, f = _normalize_simplex(float(ps[i]), float(cs[i]), float(fs[i]))
        ps[i], cs[i], fs[i] = p, c, f
    return {"protein": ps, "carbs": cs, "fat": fs}


def _axis_stats(vals: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def neighborhood_pfc_distribution(lines_df) -> dict[str, Any] | None:
    """Distribution of resolved PFC calorie fractions across neighborhood recipes.

    Each axis includes summary stats plus the raw per-recipe ``values`` list so
    callers can score a user box against dish-specific coverage counts.
    """
    arrays = _recipe_pfc_arrays(lines_df)
    if arrays is None:
        return None
    n = int(arrays["protein"].size)
    return {
        "n_recipes": n,
        "protein": {**_axis_stats(arrays["protein"]), "values": arrays["protein"].tolist()},
        "carbs": {**_axis_stats(arrays["carbs"]), "values": arrays["carbs"].tolist()},
        "fat": {**_axis_stats(arrays["fat"]), "values": arrays["fat"].tolist()},
    }


def _axis_values(stats: dict[str, Any] | None) -> np.ndarray | None:
    if not stats:
        return None
    raw = stats.get("values")
    if raw is None:
        return None
    vals = np.asarray(raw, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    return vals if vals.size else None


def _assess_axis_vs_values(
    vals: np.ndarray,
    lo: float,
    hi: float,
    *,
    slightly_frac: float = 0.80,
    very_max_in_range: int = 2,
) -> dict[str, Any]:
    """Classify one macro range as typical / slightly_* / very_* for a dish.

    Rules (dish-specific recipe cloud for this nutrient only):
      - very high/low: 0, 1, or 2 recipes fall inside [lo, hi]
      - slightly high: ≥ ``slightly_frac`` of recipes fall below ``lo``
      - slightly low:  ≥ ``slightly_frac`` of recipes fall above ``hi``
      - otherwise typical

    ``very_*`` wins over ``slightly_*`` when both would apply.
    """
    n = int(vals.size)
    lo_f, hi_f = float(lo), float(hi)
    if hi_f < lo_f:
        lo_f, hi_f = hi_f, lo_f
    in_range = int(np.sum((vals + 1e-12 >= lo_f) & (vals - 1e-12 <= hi_f)))
    n_below = int(np.sum(vals < lo_f - 1e-12))
    n_above = int(np.sum(vals > hi_f + 1e-12))
    frac_below = (n_below / n) if n else 0.0
    frac_above = (n_above / n) if n else 0.0
    median = float(np.median(vals))
    mid = 0.5 * (lo_f + hi_f)

    status = "typical"
    if in_range <= int(very_max_in_range):
        # Direction from where the dish cloud sits relative to the band.
        if median < lo_f - 1e-12 or (n_below > n_above and not (median > hi_f + 1e-12)):
            status = "very_high"
        elif median > hi_f + 1e-12 or n_above > n_below:
            status = "very_low"
        else:
            # Split / median inside a nearly empty band — treat as stretch by mid.
            status = "very_high" if mid >= median else "very_low"
    elif frac_below >= float(slightly_frac):
        status = "slightly_high"
    elif frac_above >= float(slightly_frac):
        status = "slightly_low"

    return {
        "status": status,
        "n_recipes": n,
        "n_in_range": in_range,
        "n_below": n_below,
        "n_above": n_above,
        "frac_below": frac_below,
        "frac_above": frac_above,
        "midpoint": mid,
        "neighborhood_median": median,
    }


def assess_macro_box_vs_distribution(
    box: dict[str, float] | None,
    distribution: dict[str, Any] | None,
    *,
    dish_title: str | None = None,
) -> dict[str, Any]:
    """Compare a user macro box to the dish neighborhood recipe cloud.

    Per nutrient (protein / carbs / fat independently):
      - **slightly high/low** only when ≥80% of recipes fall below the min /
        above the max
      - **very high/low** when 0–2 recipes fall inside that nutrient's range
    """
    dish = (dish_title or "this dish").strip() or "this dish"
    if not box or not distribution:
        return {
            "summary": "",
            "overall": "unknown",
            "axes": {},
        }
    axes_out: dict[str, Any] = {}
    highs: list[str] = []
    lows: list[str] = []
    for axis, label, dist_key in (
        ("protein", "Protein", "protein"),
        ("carb", "Carbs", "carbs"),
        ("fat", "Fat", "fat"),
    ):
        stats = distribution.get(dist_key) or {}
        lo = box.get(f"{axis}_min")
        hi = box.get(f"{axis}_max")
        vals = _axis_values(stats)
        if lo is None or hi is None or vals is None:
            # Fall back: without per-recipe values we cannot apply count rules.
            axes_out[axis] = {"status": "unknown", "label": label}
            continue
        assessed = _assess_axis_vs_values(vals, float(lo), float(hi))
        status = assessed["status"]
        q1 = stats.get("p25")
        q3 = stats.get("p75")
        axes_out[axis] = {
            **assessed,
            "label": label,
            "neighborhood_iqr": (
                [float(q1), float(q3)] if q1 is not None and q3 is not None else None
            ),
        }
        pretty = label.lower()
        if status in {"slightly_high", "very_high"}:
            highs.append((pretty, status))
        elif status in {"slightly_low", "very_low"}:
            lows.append((pretty, status))

    def _phrase(items: list[tuple[str, str]], side: str) -> str:
        # side is "high" or "low"
        parts: list[str] = []
        for name, status in items:
            intensity = "very" if status.startswith("very_") else "slightly"
            parts.append(f"{name} is {intensity} {side}")
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    if not highs and not lows:
        med_p = (distribution.get("protein") or {}).get("median")
        med_c = (distribution.get("carbs") or {}).get("median")
        med_f = (distribution.get("fat") or {}).get("median")
        if med_p is not None and med_c is not None and med_f is not None:
            summary = (
                f"Macros look typical for {dish} "
                f"(neighborhood median ~{round(med_p * 100)}% P / "
                f"{round(med_c * 100)}% C / {round(med_f * 100)}% F)."
            )
        else:
            summary = f"Macros look typical for {dish}."
        overall = "typical"
    else:
        bits: list[str] = []
        if highs:
            bits.append(_phrase(highs, "high"))
        if lows:
            bits.append(_phrase(lows, "low"))
        summary = f"For {dish}, {' · '.join(bits)} versus typical recipes of this type."
        overall = "atypical"
    return {"summary": summary, "overall": overall, "axes": axes_out}


def _neighborhood_mean_pfc(lines_df) -> tuple[float, float, float] | None:
    """Mean resolved PFC (calorie-share) across every neighborhood recipe."""
    arrays = _recipe_pfc_arrays(lines_df)
    if arrays is None:
        return None
    return _normalize_simplex(
        float(np.mean(arrays["protein"])),
        float(np.mean(arrays["carbs"])),
        float(np.mean(arrays["fat"])),
    )


def _rounded_box_pm(mid: tuple[float, float, float], pad_pct: int = 5) -> dict[str, float]:
    """±pad_pct box around each macro, midpoint rounded to nearest percent."""
    p, c, f = (round(x * 100) for x in mid)
    box_pct = {
        "protein_min": p - pad_pct,
        "protein_max": p + pad_pct,
        "carb_min": c - pad_pct,
        "carb_max": c + pad_pct,
        "fat_min": f - pad_pct,
        "fat_max": f + pad_pct,
    }
    return {k: max(0.0, min(1.0, v / 100.0)) for k, v in box_pct.items()}


def _round_box_pct(box: dict[str, float]) -> dict[str, float]:
    """Round each bound to the nearest percent and clamp to [0, 1]."""
    out: dict[str, float] = {}
    for key, val in box.items():
        pct = int(round(float(val) * 100))
        pct = max(0, min(100, pct))
        out[key] = pct / 100.0
    # Keep min ≤ max after rounding.
    for axis in ("protein", "carb", "fat"):
        lo_k, hi_k = f"{axis}_min", f"{axis}_max"
        if out[lo_k] > out[hi_k]:
            mid = round((out[lo_k] + out[hi_k]) * 50) / 100.0
            out[lo_k] = mid
            out[hi_k] = mid
    return out


def _cap_axis_widths(box: dict[str, float], *, max_width: float = 0.10) -> dict[str, float]:
    """Ensure each macro interval is at most ``max_width`` (default 10pp)."""
    b = dict(box)
    width_pct = int(round(float(max_width) * 100))
    width_pct = max(1, min(100, width_pct))
    for axis in ("protein", "carb", "fat"):
        lo_k, hi_k = f"{axis}_min", f"{axis}_max"
        lo_pct = int(round(float(b[lo_k]) * 100))
        hi_pct = int(round(float(b[hi_k]) * 100))
        if hi_pct - lo_pct <= width_pct:
            b[lo_k] = max(0, min(100, lo_pct)) / 100.0
            b[hi_k] = max(0, min(100, hi_pct)) / 100.0
            continue
        mid_pct = int(round(0.5 * (lo_pct + hi_pct)))
        lo_pct = mid_pct - width_pct // 2
        hi_pct = lo_pct + width_pct
        if lo_pct < 0:
            lo_pct = 0
            hi_pct = width_pct
        if hi_pct > 100:
            hi_pct = 100
            lo_pct = 100 - width_pct
        b[lo_k] = lo_pct / 100.0
        b[hi_k] = hi_pct / 100.0
    return b


def _ensure_box_feasible(
    box: dict[str, float],
    *,
    max_width: float | None = None,
) -> dict[str, float]:
    """Make the box intersect the PFC simplex.

    When ``max_width`` is set, shift whole bands (keep width) instead of widening.
    """
    b = dict(box)
    if max_width is not None:
        b = _cap_axis_widths(b, max_width=max_width)
    for _ in range(80):
        s_max = b["protein_max"] + b["carb_max"] + b["fat_max"]
        s_min = b["protein_min"] + b["carb_min"] + b["fat_min"]
        if s_max >= 1.0 - 1e-9 and s_min <= 1.0 + 1e-9:
            return b
        if s_max < 1.0 - 1e-9:
            room = {
                "protein": 1.0 - b["protein_max"],
                "carb": 1.0 - b["carb_max"],
                "fat": 1.0 - b["fat_max"],
            }
            axis = max(room, key=room.get)
            if room[axis] <= 1e-12:
                break
            if max_width is not None:
                w = b[f"{axis}_max"] - b[f"{axis}_min"]
                b[f"{axis}_max"] = min(1.0, b[f"{axis}_max"] + 0.01)
                b[f"{axis}_min"] = max(0.0, b[f"{axis}_max"] - w)
            else:
                b[f"{axis}_max"] = min(1.0, b[f"{axis}_max"] + 0.01)
            continue
        if s_min > 1.0 + 1e-9:
            room = {
                "protein": b["protein_min"],
                "carb": b["carb_min"],
                "fat": b["fat_min"],
            }
            axis = max(room, key=room.get)
            if room[axis] <= 1e-12:
                break
            if max_width is not None:
                w = b[f"{axis}_max"] - b[f"{axis}_min"]
                b[f"{axis}_min"] = max(0.0, b[f"{axis}_min"] - 0.01)
                b[f"{axis}_max"] = min(1.0, b[f"{axis}_min"] + w)
            else:
                b[f"{axis}_min"] = max(0.0, b[f"{axis}_min"] - 0.01)
                if b[f"{axis}_min"] > b[f"{axis}_max"]:
                    b[f"{axis}_min"] = b[f"{axis}_max"]
            continue
        break
    if max_width is not None:
        b = _cap_axis_widths(b, max_width=max_width)
    return b


def _best_fixed_width_band(
    vals: np.ndarray,
    *,
    width: float = 0.10,
) -> tuple[float, float, int]:
    """Return ``(lo, hi, n_inside)`` for the width-capped band covering the most points.

    Searches lo at 1pp steps so the band is exactly ``width`` after percent rounding.
    Ties break toward the axis median.
    """
    vals = np.asarray(vals, dtype=float).ravel()
    if vals.size == 0:
        return 0.0, float(width), 0
    width_pct = max(1, min(100, int(round(float(width) * 100))))
    max_lo_pct = 100 - width_pct
    median = float(np.median(vals))
    best_lo_pct = 0
    best_count = -1
    best_mid_dist = 1e9
    for lo_pct in range(0, max_lo_pct + 1):
        lo = lo_pct / 100.0
        hi = (lo_pct + width_pct) / 100.0
        count = int(np.sum((vals + 1e-12 >= lo) & (vals - 1e-12 <= hi)))
        mid = 0.5 * (lo + hi)
        mid_dist = abs(mid - median)
        if count > best_count or (count == best_count and mid_dist < best_mid_dist - 1e-15):
            best_count = count
            best_lo_pct = lo_pct
            best_mid_dist = mid_dist
    lo = best_lo_pct / 100.0
    hi = (best_lo_pct + width_pct) / 100.0
    return lo, hi, max(0, best_count)


def _coverage_box_from_arrays(
    arrays: dict[str, np.ndarray],
    *,
    width: float = 0.10,
    lo_pct: float | None = None,
    hi_pct: float | None = None,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    """Per-macro fixed-width bands covering the most recipes; return box + frac + meta.

    Legacy ``lo_pct``/``hi_pct`` args are ignored — coverage always uses max-count
    fixed-width bands (default 10pp).
    """
    _ = lo_pct, hi_pct
    width = float(width) if width is not None else 0.10
    width = min(1.0, max(0.01, width))

    p_lo, p_hi, p_n = _best_fixed_width_band(arrays["protein"], width=width)
    c_lo, c_hi, c_n = _best_fixed_width_band(arrays["carbs"], width=width)
    f_lo, f_hi, f_n = _best_fixed_width_band(arrays["fat"], width=width)
    raw = {
        "protein_min": p_lo,
        "protein_max": p_hi,
        "carb_min": c_lo,
        "carb_max": c_hi,
        "fat_min": f_lo,
        "fat_max": f_hi,
    }
    box = _ensure_box_feasible(_round_box_pct(raw), max_width=width)
    box = _cap_axis_widths(box, max_width=width)

    n = int(arrays["protein"].size)
    inside = 0
    for i in range(n):
        p, c, f = float(arrays["protein"][i]), float(arrays["carbs"][i]), float(arrays["fat"][i])
        if (
            box["protein_min"] - 1e-9 <= p <= box["protein_max"] + 1e-9
            and box["carb_min"] - 1e-9 <= c <= box["carb_max"] + 1e-9
            and box["fat_min"] - 1e-9 <= f <= box["fat_max"] + 1e-9
        ):
            inside += 1
    coverage = (inside / n) if n else 0.0
    meta = {
        "width": width,
        "axis_counts": {"protein": p_n, "carbs": c_n, "fat": f_n},
        "axis_n": n,
    }
    return box, float(coverage), meta


def suggest_macro_targets(
    lines_df,
    starting_recipe_id: str | None = None,
    *,
    pad_pct: int = 5,
    coverage_width: float = 0.10,
    coverage_lo_pct: float = 10.0,
    coverage_hi_pct: float = 90.0,
    **_ignored: Any,
) -> dict[str, Any]:
    """Return coverage + mean macro boxes and neighborhood PFC distribution."""
    arrays = _recipe_pfc_arrays(lines_df)
    if arrays is None:
        return {"error": "no_neighborhood_pfc", "n_recipes": 0, "presets": {}}

    n_recipes = int(arrays["protein"].size)
    distribution = neighborhood_pfc_distribution(lines_df) or {}
    mean_pfc = _normalize_simplex(
        float(np.mean(arrays["protein"])),
        float(np.mean(arrays["carbs"])),
        float(np.mean(arrays["fat"])),
    )
    mean_box = _rounded_box_pm(mean_pfc, pad_pct=pad_pct)
    coverage_box, coverage_frac, cov_meta = _coverage_box_from_arrays(
        arrays,
        width=coverage_width,
        lo_pct=coverage_lo_pct,
        hi_pct=coverage_hi_pct,
    )
    width_pp = int(round(float(cov_meta.get("width") or coverage_width) * 100))

    presets = {
        "neighborhood_coverage": {
            "kind": "neighborhood_coverage",
            "label": (
                f"Per-macro {width_pp}pp bands placed to cover the most resolved "
                f"neighborhood recipes (~{round(100 * coverage_frac)}% fully inside)."
            ),
            "box": coverage_box,
            "width": float(cov_meta.get("width") or coverage_width),
            "axis_counts": cov_meta.get("axis_counts"),
            "coverage_frac": coverage_frac,
            "n_recipes": n_recipes,
            # Legacy keys for older UI copy.
            "lo_pct": coverage_lo_pct,
            "hi_pct": coverage_hi_pct,
        },
        "neighborhood_mean": {
            "kind": "neighborhood_mean",
            "label": (
                f"Mean nutrient profile of the neighborhood, with a ±{pad_pct}% box "
                "around each macro (rounded to the nearest percent)."
            ),
            "box": mean_box,
            "midpoint": {
                "protein": mean_pfc[0],
                "carbs": mean_pfc[1],
                "fat": mean_pfc[2],
            },
            "n_recipes": n_recipes,
            "pad_pct": pad_pct,
        },
    }
    return {
        "n_recipes": n_recipes,
        "starting_recipe_id": str(starting_recipe_id) if starting_recipe_id else None,
        "neighborhood_mean_pfc": {
            "protein": mean_pfc[0],
            "carbs": mean_pfc[1],
            "fat": mean_pfc[2],
        },
        "distribution": distribution,
        "presets": presets,
        "default_preset": "neighborhood_coverage",
        "note": (
            f"Coverage box uses {width_pp}pp per-macro bands on {n_recipes} resolved "
            f"neighborhood recipes, each placed to cover the most recipes on that axis "
            f"(~{round(100 * coverage_frac)}% of recipes fall fully inside)."
        ),
    }


def suggest_macro_targets_for_canonical(
    canonical_id: int,
    *,
    half_widths: dict[str, float] | None = None,
    fast_neighborhood: bool = True,
    require_cache: bool = True,
    coverage_lo_pct: float = 10.0,
    coverage_hi_pct: float = 90.0,
) -> dict[str, Any]:
    from canonical_optimization import CanonicalNeighborhood

    nb = CanonicalNeighborhood.build(
        int(canonical_id),
        fast=fast_neighborhood,
        use_cache=True,
        require_cache=require_cache,
    )
    # half_widths unused; kept for API compatibility with the server request model.
    _ = half_widths
    result = suggest_macro_targets(
        nb.lines_df,
        str(nb.starting_recipe_id),
        pad_pct=5,
        coverage_width=0.10,
        coverage_lo_pct=coverage_lo_pct,
        coverage_hi_pct=coverage_hi_pct,
    )
    title = getattr(nb, "title", None) or getattr(nb, "canonical_title", None)
    result["canonical_id"] = int(canonical_id)
    result["title"] = title
    result["neighborhood_from_cache"] = bool(getattr(nb, "from_cache", False))
    result["n_neighborhood_recipes"] = len(nb.recipe_ids or [])
    if result.get("distribution"):
        # Attach a default assessment for the coverage box midpoint.
        cov = (result.get("presets") or {}).get("neighborhood_coverage") or {}
        result["coverage_assessment"] = assess_macro_box_vs_distribution(
            cov.get("box"),
            result.get("distribution"),
            dish_title=str(title) if title else None,
        )
    return result


def high_protein_targets_from_mean(
    mean_pfc: tuple[float, float, float],
    *,
    protein_delta: float = 0.10,
    carb_delta: float = -0.05,
    fat_delta: float = -0.05,
    pad_pct: int = 2,
) -> dict[str, Any]:
    """Shift neighborhood mean PFC toward higher protein, then ±pad_pct box.

    Default: protein +10pp, carbs −5pp, fat −5pp (sum preserved), then
    renormalize onto the simplex and round midpoint to the nearest percent.
    """
    p0, c0, f0 = mean_pfc
    raw = (p0 + protein_delta, c0 + carb_delta, f0 + fat_delta)
    mid = _normalize_simplex(*raw)
    # Clamp each mid so ±pad leaves a non-empty [0,1] interval after rounding.
    mid_pct = [round(x * 100) for x in mid]
    # Re-normalize rounded percents so they sum to 100 when possible.
    s = sum(mid_pct)
    if s > 0 and s != 100:
        # Adjust the largest component so percents sum to 100.
        adj = 100 - s
        i_max = int(np.argmax(mid_pct))
        mid_pct[i_max] = max(pad_pct, min(100 - 2 * pad_pct, mid_pct[i_max] + adj))
    mid = (mid_pct[0] / 100.0, mid_pct[1] / 100.0, mid_pct[2] / 100.0)
    box = _rounded_box_pm(mid, pad_pct=pad_pct)
    return {
        "midpoint": {"protein": mid[0], "carbs": mid[1], "fat": mid[2]},
        "box": box,
        "protein_delta": protein_delta,
        "carb_delta": carb_delta,
        "fat_delta": fat_delta,
        "pad_pct": pad_pct,
        "neighborhood_mean_pfc": {
            "protein": float(p0),
            "carbs": float(c0),
            "fat": float(f0),
        },
    }


def suggest_high_protein_targets_for_canonical(
    canonical_id: int,
    *,
    protein_delta: float = 0.10,
    carb_delta: float = -0.05,
    fat_delta: float = -0.05,
    pad_pct: int = 2,
    fast_neighborhood: bool = True,
) -> dict[str, Any]:
    """Neighborhood-mean PFC → high-protein target box for one canonical dish."""
    from canonical_optimization import CanonicalNeighborhood

    nb = CanonicalNeighborhood.build(
        int(canonical_id),
        fast=fast_neighborhood,
        use_cache=True,
    )
    mean_pfc = _neighborhood_mean_pfc(nb.lines_df)
    if mean_pfc is None:
        return {"error": "no_neighborhood_pfc", "canonical_id": int(canonical_id)}
    hp = high_protein_targets_from_mean(
        mean_pfc,
        protein_delta=protein_delta,
        carb_delta=carb_delta,
        fat_delta=fat_delta,
        pad_pct=pad_pct,
    )
    return {
        "canonical_id": int(canonical_id),
        "title": getattr(nb, "title", None),
        "starting_recipe_id": str(nb.starting_recipe_id),
        "n_neighborhood_recipes": len(nb.recipe_ids or []),
        "neighborhood_from_cache": bool(getattr(nb, "from_cache", False)),
        **hp,
        "note": (
            f"Protein +{protein_delta:.0%} / carbs {carb_delta:+.0%} / fat {fat_delta:+.0%} "
            f"vs neighborhood mean PFC, then ±{pad_pct}% box (rounded)."
        ),
    }
