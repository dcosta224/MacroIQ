"""Neighborhood hull stretch: how far the macro target sits outside typical recipes.

Uses the *original* Jaccard neighborhood only (not query-expanded shell recipes).
For each neighbor NLG recipe we build its ingredient conical hull and measure whether
the user's target macro box intersects it and the outside_score distance. The
aggregate tells the final GPT-4o evaluator how much fidelity slack the macro
demand justifies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hull_geometry import TargetBox, region_intersects_hull


def _per_gram_macros_for_fdc_ids(fdc_ids: list[int]) -> dict[int, np.ndarray]:
    from recipe_opt_agent.problem_loader import _per_gram_macros_for_fdc_ids as _loader_per_g

    return _loader_per_g(fdc_ids)


def _macro_inputs_from_lines(sub, per_g: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    x0_list: list[float] = []
    cols: list[np.ndarray] = []
    for row in sub.itertuples(index=False):
        fid = getattr(row, "fdc_id", None)
        grams = float(getattr(row, "gram_weight", 0.0) or 0.0)
        if fid is None or grams <= 0:
            continue
        fid_i = int(fid)
        col = per_g.get(fid_i)
        if col is None:
            continue
        x0_list.append(grams)
        cols.append(np.asarray(col, dtype=float).ravel()[:4])
    if not x0_list:
        return None
    x0 = np.asarray(x0_list, dtype=float)
    M = np.column_stack(cols)  # (4, n): rows protein/fat/carbs/kcal per gram
    return x0, M


def _atwater_kcal(x0: np.ndarray, M: np.ndarray) -> float:
    if x0.size == 0 or M.ndim != 2:
        return 0.0
    return float(M[3] @ x0) if M.shape[0] >= 4 else float(np.sum(x0))


def compute_neighbor_hull_row(
    sub,
    per_g: dict[int, np.ndarray],
    box: TargetBox,
    *,
    recipe_id: str,
    n_hull_samples: int = 1200,
) -> dict[str, Any] | None:
    built = _macro_inputs_from_lines(sub, per_g)
    if built is None:
        return None
    x0, M = built
    if M.shape[1] < 1:
        return None
    kcal = _atwater_kcal(x0, M)
    if kcal <= 0:
        kcal = max(float(x0.sum()), 1.0)
    hull = region_intersects_hull(M, box, kcal_target=kcal, n_samples=n_hull_samples)
    dist = hull.get("distance") or {}
    outside = dist.get("outside_score")
    return {
        "recipe_nlg_id": str(recipe_id),
        "n_ingredients": int(M.shape[1]),
        "kcal": float(kcal),
        "hull_intersects": bool(hull.get("intersects")),
        "geometric_intersects": bool(hull.get("geometric_intersects")),
        "lp_feasible": bool(hull.get("lp_feasible")),
        "outside_score": float(outside) if outside is not None else None,
        "min_distance_box_to_hull": dist.get("min_distance_box_to_hull"),
        "fraction_box_points_in_hull": dist.get("fraction_box_points_in_hull"),
        "interpretation": dist.get("interpretation"),
    }


def _stretch_level(frac_intersects: float, median_outside: float | None) -> tuple[str, float]:
    """Return (level, fidelity_forgiveness 0–0.35 for LLM guidance)."""
    med = float(median_outside or 0.0)
    if frac_intersects >= 0.65 and med < 0.025:
        return "in_hull", 0.0
    if frac_intersects >= 0.35 or med < 0.06:
        return "edge", 0.15
    return "outside_hull", 0.30


def build_neighborhood_hull_context(
    lines_df,
    recipe_ids: list[str],
    box_dict: dict[str, float],
    *,
    shell_recipe_ids: list[str] | None = None,
    starting_recipe_id: str | None = None,
    n_hull_samples: int = 1200,
) -> dict[str, Any]:
    """Aggregate hull-vs-target stats over the original neighborhood recipes."""
    shell = {str(x) for x in (shell_recipe_ids or [])}
    core_ids = [str(r) for r in recipe_ids if str(r) not in shell]
    if lines_df is None or getattr(lines_df, "empty", True) or not core_ids:
        return {"n_recipes": 0, "error": "no_neighborhood_lines"}

    fdc_ids = sorted({int(x) for x in lines_df["fdc_id"].dropna().astype(int).tolist()})
    per_g = _per_gram_macros_for_fdc_ids(fdc_ids)
    box = TargetBox(
        protein_min=float(box_dict["protein_min"]),
        protein_max=float(box_dict["protein_max"]),
        carb_min=float(box_dict["carb_min"]),
        carb_max=float(box_dict["carb_max"]),
        fat_min=float(box_dict["fat_min"]),
        fat_max=float(box_dict["fat_max"]),
    )

    rows: list[dict[str, Any]] = []
    for rid in core_ids:
        sub = lines_df.loc[lines_df["recipe_nlg_id"] == rid]
        if sub.empty:
            continue
        row = compute_neighbor_hull_row(
            sub, per_g, box, recipe_id=rid, n_hull_samples=n_hull_samples
        )
        if row:
            rows.append(row)

    if not rows:
        return {"n_recipes": 0, "error": "no_hull_rows"}

    outside_vals = [float(r["outside_score"]) for r in rows if r.get("outside_score") is not None]
    n_hit = sum(1 for r in rows if r.get("hull_intersects"))
    n = len(rows)
    frac = float(n_hit / n) if n else 0.0
    median_out = float(np.median(outside_vals)) if outside_vals else None
    level, forgiveness = _stretch_level(frac, median_out)

    start_row = next((r for r in rows if r["recipe_nlg_id"] == str(starting_recipe_id)), None)
    sorted_out = sorted(rows, key=lambda r: -(float(r.get("outside_score") or 0.0)))
    med_s = f"{median_out:.4f}" if median_out is not None else "n/a"

    return {
        "target_box": box_dict,
        "n_recipes": n,
        "n_hull_intersects": n_hit,
        "frac_hull_intersects": round(frac, 4),
        "median_outside_score": median_out,
        "mean_outside_score": float(np.mean(outside_vals)) if outside_vals else None,
        "p75_outside_score": float(np.percentile(outside_vals, 75)) if outside_vals else None,
        "min_outside_score": float(min(outside_vals)) if outside_vals else None,
        "max_outside_score": float(max(outside_vals)) if outside_vals else None,
        "target_stretch_level": level,
        "fidelity_forgiveness_hint": forgiveness,
        "interpretation": (
            f"{n_hit}/{n} original neighborhood recipes have a conical ingredient hull that "
            f"intersects the user's macro box (median outside_score={med_s}). "
            f"Stretch level={level}: the evaluator may be up to ~{int(forgiveness * 100)}% "
            f"more forgiving on ingredient-fidelity tradeoffs, but strange ingredients still "
            f"require strong justification."
        ),
        "starting_recipe_hull": start_row,
        "worst_outside_recipes": sorted_out[:5],
        "note": (
            "Computed from original Jaccard neighborhood NLG recipes only — query-expanded "
            "shell recipes are excluded."
        ),
    }
