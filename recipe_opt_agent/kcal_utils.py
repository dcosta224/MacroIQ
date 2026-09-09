"""Calorie-target helpers shared by the web UI and agent candidate paths."""

from __future__ import annotations

from typing import Any


def resolve_kcal_target(
    *sources: Any,
    default: float | None = None,
) -> float | None:
    """First positive kcal from problem dicts, configs, or raw floats."""
    for src in sources:
        if src is None:
            continue
        if isinstance(src, (int, float)):
            val = float(src)
            if val > 0:
                return val
            continue
        if isinstance(src, dict):
            for key in ("user_kcal_target", "kcal_target"):
                raw = src.get(key)
                if raw is None:
                    continue
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    return val
            continue
        raw = getattr(src, "kcal_target", None)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val
    return default


def resolve_user_kcal_target(*sources: Any) -> float | None:
    """Calorie target the user explicitly requested (not ambient recipe Atwater kcal)."""
    for src in sources:
        if src is None:
            continue
        if isinstance(src, (int, float)):
            val = float(src)
            if val > 0:
                return val
            continue
        if isinstance(src, dict):
            raw = src.get("user_kcal_target")
            if raw is not None:
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    val = None
                if val is not None and val > 0:
                    return val
            # AgentConfig / request snapshots store the UI value as kcal_target.
            if "protein_min" in src or "max_iterations" in src or "shadow_draft_model" in src:
                raw = src.get("kcal_target")
                if raw is not None:
                    try:
                        val = float(raw)
                    except (TypeError, ValueError):
                        val = None
                    if val is not None and val > 0:
                        return val
            continue
        raw = getattr(src, "kcal_target", None)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val
    return None


def apply_kcal_target(problem: dict[str, Any], kcal_target: float | None) -> dict[str, Any]:
    """Override problem kcal (and scale mass/x0) when a calorie target is set.

    Marks ``user_kcal_target`` so later grounding/rebuilds can restore the intent
    even if ``kcal_target`` is temporarily overwritten by draft Atwater kcal.
    """
    if kcal_target is None:
        return problem
    target = float(kcal_target)
    if target <= 0:
        return problem
    problem = dict(problem)
    old = float(problem.get("kcal_target") or 0.0)
    if old <= 0:
        from weighted_empirical_opt import atwater_kcal
        import numpy as np

        x0 = np.asarray(problem.get("x0") or [], dtype=float)
        M = np.asarray(problem.get("M") or [], dtype=float)
        if x0.size and M.size:
            old = float(atwater_kcal(x0, M))
    scale = (target / old) if old > 1e-9 else 1.0
    problem["kcal_target"] = target
    problem["user_kcal_target"] = target
    if scale != 1.0 and scale > 0:
        x0 = list(problem.get("x0") or [])
        problem["x0"] = [float(v) * scale for v in x0]
        problem["total_mass"] = float(problem.get("total_mass") or sum(x0)) * scale
        if problem.get("x_opt"):
            problem["x_opt"] = [float(v) * scale for v in list(problem.get("x_opt") or [])]
        chosen = dict(problem.get("chosen_recipe") or {})
        ings = []
        for i, row in enumerate(chosen.get("ingredients") or []):
            row = dict(row)
            if i < len(problem["x0"]):
                row["grams"] = float(problem["x0"][i])
            ings.append(row)
        if ings:
            chosen["ingredients"] = ings
            problem["chosen_recipe"] = chosen
    return problem


def restore_kcal_target(
    problem: dict[str, Any],
    *sources: Any,
) -> dict[str, Any]:
    """Re-apply the best-known user kcal onto a rebuilt problem."""
    user = resolve_user_kcal_target(problem, *sources)
    target = user if user is not None else resolve_kcal_target(problem, *sources)
    if target is None:
        return problem
    return apply_kcal_target(problem, target)


def _current_recipe_kcal(
    *,
    problem: dict[str, Any] | None,
    display: dict[str, Any] | None,
    ingredients: list[dict[str, Any]] | None = None,
) -> float | None:
    """Best available Atwater / display calorie total for a candidate."""
    display = display or {}
    macros = display.get("macros") if isinstance(display.get("macros"), dict) else {}
    try:
        val = float(macros.get("calories"))
    except (TypeError, ValueError):
        val = None
    if val is not None and val > 1e-6:
        return val

    ings = list(ingredients if ingredients is not None else (display.get("ingredients") or []))
    total = 0.0
    n = 0
    for row in ings:
        try:
            cal = float(row.get("calories"))
        except (TypeError, ValueError):
            continue
        if cal >= 0:
            total += cal
            n += 1
    if n > 0 and total > 1e-6:
        return float(total)

    problem = problem or {}
    try:
        import numpy as np
        from weighted_empirical_opt import atwater_kcal

        x = np.asarray(problem.get("x_opt") or problem.get("x0") or [], dtype=float)
        M = np.asarray(problem.get("M") or [], dtype=float)
        if x.size and M.ndim == 2 and M.shape[1] >= x.size:
            val = float(atwater_kcal(x, M[:, : x.size]))
            if val > 1e-6:
                return val
    except Exception:
        pass
    return None


def scale_candidate_to_kcal(
    *,
    problem: dict[str, Any] | None,
    display: dict[str, Any] | None,
    kcal_target: float | None,
    tol_frac: float = 0.0,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Uniformly scale grams so the candidate hits ``kcal_target`` (post-process).

    Share ratios / IQR positions are preserved under a uniform scale. Returns
    updated ``(problem, display)`` copies when processed.

    Default ``tol_frac=0`` always rescales when off-target so display calories
    match the user target even after optimizer / LLM drafting drift.
    """
    if kcal_target is None:
        return problem, display
    target = float(kcal_target)
    if target <= 0:
        return problem, display

    display = dict(display or {})
    problem = dict(problem or {}) if problem is not None else {}
    ings = [dict(r) for r in (display.get("ingredients") or [])]
    current = _current_recipe_kcal(problem=problem, display=display, ingredients=ings)
    if current is None or current <= 1e-6:
        if problem is not None:
            problem["kcal_target"] = target
            problem["user_kcal_target"] = target
        macros = dict(display.get("macros") or {})
        macros["calories"] = int(round(target))
        display["macros"] = macros
        return (problem or None), display

    if abs(current - target) / target <= float(tol_frac):
        macros = dict(display.get("macros") or {})
        macros["calories"] = int(round(target))
        display["macros"] = macros
        problem["kcal_target"] = target
        problem["user_kcal_target"] = target
        return problem, display

    scale = target / current
    if scale <= 0:
        return problem, display

    for row in ings:
        try:
            g = float(row.get("grams"))
        except (TypeError, ValueError):
            g = None
        if g is not None:
            row["grams"] = float(g) * scale
            if row.get("grams_rounded") is not None:
                row["grams_rounded"] = int(round(row["grams"]))
        try:
            cal = float(row.get("calories"))
        except (TypeError, ValueError):
            cal = None
        if cal is not None:
            row["calories"] = float(cal) * scale
    display["ingredients"] = ings

    macros = dict(display.get("macros") or {})
    macros["calories"] = int(round(target))
    display["macros"] = macros

    try:
        from recipe_opt_agent.score_display import proportion_typicality_from_ingredients

        typicality = proportion_typicality_from_ingredients(ings)
        ratio = dict(display.get("ratio_loss") or {})
        ratio["band"] = typicality.get("band") or ratio.get("band")
        ratio["band_summary"] = typicality.get("summary") or ratio.get("band_summary")
        ratio["proportion_key"] = typicality.get("key")
        ratio["proportion_css"] = typicality.get("css")
        ratio["outside_iqr_frac"] = typicality.get("outside_iqr_frac")
        ratio["outside_iqr_pct"] = typicality.get("outside_iqr_pct")
        ratio["outside_iqr_calorie_frac"] = typicality.get("outside_iqr_calorie_frac")
        ratio["outside_iqr_calorie_pct"] = typicality.get("outside_iqr_calorie_pct")
        display["ratio_loss"] = ratio
    except Exception:
        pass

    for key in ("x0", "x_opt"):
        vec = problem.get(key)
        if isinstance(vec, list) and vec:
            problem[key] = [float(v) * scale for v in vec]
    if problem.get("total_mass") is not None:
        try:
            problem["total_mass"] = float(problem["total_mass"]) * scale
        except (TypeError, ValueError):
            pass
    problem["kcal_target"] = target
    problem["user_kcal_target"] = target

    chosen = dict(problem.get("chosen_recipe") or {})
    chosen_ings = list(chosen.get("ingredients") or [])
    if chosen_ings and ings:
        n = min(len(chosen_ings), len(ings))
        for i in range(n):
            row = dict(chosen_ings[i])
            if ings[i].get("grams") is not None:
                row["grams"] = float(ings[i]["grams"])
            chosen_ings[i] = row
        chosen["ingredients"] = chosen_ings
        problem["chosen_recipe"] = chosen

    return problem, display
