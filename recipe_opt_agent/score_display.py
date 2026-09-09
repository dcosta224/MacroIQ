"""Contextualized score bands + per-ingredient display for the Web UI.

Ratio loss = ``ratio_surrogate`` from the *chosen* recipe's ``term_losses``
(not a raw pasta∶egg ratio, and not a silent 0 when samples are missing).

Nutrient loss = PFC box slack recomputed from the chosen recipe's ``pfc_after``
vs the run's macro targets (not an unchecked telemetry zero).

Missing / unavailable signals render as ``None`` with band ``unknown`` so the
UI shows blanks instead of a false perfect score.
"""

from __future__ import annotations

from typing import Any

import numpy as np


RATIO_LOSS_BANDS = {
    "good_max": 0.015,
    "warn_max": 0.040,
    "prior_p50": 0.012,
    "prior_p75": 0.028,
    "prior_p90": 0.055,
}

NUTRIENT_LOSS_BANDS = {
    # Solver slack noise below 0.05% renders as "0.000" — treat it as green.
    "good_max": 0.0005,
    "warn_max": 0.025,
    "prior_p50": 0.0,
    "prior_p75": 0.02,
    "prior_p90": 0.06,
}

RATIO_BAND_SUMMARY = {
    "good": "Proportions are very typical.",
    "warn": "Proportions are somewhat different.",
    "bad": "Proportions are substantially off.",
    "unknown": "Typicality could not be scored for this recipe.",
}

# Outside-central-band calorie share → user-facing typicality (primary MacroIQ copy).
# Thresholds are on calorie-weighted fraction of basis groups outside the P15–P85 band.
PROPORTION_TYPICALITY = {
    "very_typical": {
        "max_outside": 0.17,
        "summary": "Proportions are very typical.",
        "band": "good",
        "css": "very-typical",
    },
    "mostly_typical": {
        "max_outside": 0.25,
        "summary": "Proportions are mostly typical.",
        "band": "good",
        "css": "mostly-typical",
    },
    "somewhat_different": {
        "max_outside": 0.35,
        "summary": "Proportions are somewhat different.",
        "band": "warn",
        "css": "somewhat-different",
    },
    # Legacy alias kept so older payloads / CSS still resolve.
    "somewhat_unusual": {
        "max_outside": 0.35,
        "summary": "Proportions are somewhat different.",
        "band": "warn",
        "css": "somewhat-different",
    },
    "substantially_off": {
        "max_outside": 1.01,
        "summary": "Proportions are substantially off.",
        "band": "bad",
        "css": "substantially-off",
    },
    "unknown": {
        "max_outside": None,
        "summary": "Typicality could not be scored for this recipe.",
        "band": "unknown",
        "css": "unknown",
    },
}

NUTRIENT_BAND_SUMMARY = {
    "good": "Protein, carbs, and fat land inside your target ranges.",
    "warn": "Macros are close to your targets, with a small miss.",
    "bad": "Macros sit meaningfully outside your target ranges.",
    "unknown": "Macro fit could not be scored for this recipe.",
}


def band_for_loss(value: float | None, *, good_max: float, warn_max: float) -> str:
    if value is None:
        return "unknown"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if v <= good_max:
        return "good"
    if v <= warn_max:
        return "warn"
    return "bad"


def band_for_holistic_0_10(score: float | None) -> str:
    if score is None:
        return "unknown"
    try:
        v = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if v >= 8.0:
        return "good"
    if v >= 5.0:
        return "warn"
    return "bad"


def empty_display_scores() -> dict[str, Any]:
    """Blank score cards before any agent iteration."""
    return {
        "ready": False,
        "iteration": None,
        "ratio_loss": {
            "value": None,
            "band": "unknown",
            "label": "Ratio loss",
            "direction": "lower_better",
            "unit": "surrogate",
            "explanation": "Neighborhood pasta∶egg ratio surrogate (lower better).",
            "band_summary": RATIO_BAND_SUMMARY["unknown"],
            "thresholds": dict(RATIO_LOSS_BANDS),
            "source": None,
        },
        "nutrient_loss": {
            "value": None,
            "band": "unknown",
            "label": "Nutrient loss",
            "direction": "lower_better",
            "unit": "pfc_slack",
            "explanation": "L1 calorie-fraction distance outside the protein/carb/fat box.",
            "band_summary": NUTRIENT_BAND_SUMMARY["unknown"],
            "thresholds": dict(NUTRIENT_LOSS_BANDS),
            "source": None,
        },
        "holistic_0_10": {
            "value": None,
            "band": "unknown",
            "label": "Holistic",
            "direction": "higher_better",
            "unit": "0_10",
            "source": None,
            "explanation": "LLM judge score 0–10 when available.",
        },
        "macros": {
            "protein": None,
            "carb": None,
            "fat": None,
            "calories": None,
        },
        "pfc_after": None,
        "cookability": {
            "improved": False,
            "improved_pct": None,
            "summary": None,
        },
        "applied_edits": [],
        "ingredients": [],
        "score_history": [],
        "status": None,
        "title": None,
    }


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _macro_box_from_payload(payload: dict[str, Any]) -> dict[str, float] | None:
    cfg = payload.get("config") or {}
    tel = payload.get("run_telemetry") or {}
    box = payload.get("macro_targets") or tel.get("macro_targets") or {}
    if not box and cfg:
        box = {
            "protein_min": cfg.get("protein_min"),
            "protein_max": cfg.get("protein_max"),
            "carb_min": cfg.get("carb_min"),
            "carb_max": cfg.get("carb_max"),
            "fat_min": cfg.get("fat_min"),
            "fat_max": cfg.get("fat_max"),
        }
    keys = ("protein_min", "protein_max", "carb_min", "carb_max", "fat_min", "fat_max")
    if not all(box.get(k) is not None for k in keys):
        return None
    return {k: float(box[k]) for k in keys}


def _pfc_box_slack(pfc: dict[str, Any] | None, box: dict[str, float] | None) -> float | None:
    if not pfc or not box:
        return None
    try:
        p = float(pfc["protein"])
        c = float(pfc["carbs"] if "carbs" in pfc else pfc.get("carb"))
        f = float(pfc["fat"])
    except (KeyError, TypeError, ValueError):
        return None

    def _axis(v: float, lo: float, hi: float) -> float:
        return max(lo - v, 0.0) + max(v - hi, 0.0)

    return (
        _axis(p, box["protein_min"], box["protein_max"])
        + _axis(c, box["carb_min"], box["carb_max"])
        + _axis(f, box["fat_min"], box["fat_max"])
    )


def resolve_chosen_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Pick opt / problem / ingredients for the recipe that scores are about."""
    payload = payload or {}
    chosen = payload.get("chosen") or {}
    entry = chosen.get("entry") if isinstance(chosen.get("entry"), dict) else None
    # Creative scored finalist nests pool entry under metrics.raw.entry
    nested = None
    if entry and isinstance(entry.get("metrics"), dict):
        nested = (entry.get("metrics") or {}).get("raw", {}).get("entry")
    pool_entry = nested if isinstance(nested, dict) else entry

    opt = None
    if isinstance(pool_entry, dict):
        opt = pool_entry.get("opt")
    if not opt and isinstance(chosen, dict):
        opt = chosen.get("opt")
    if not opt:
        opt = payload.get("opt")
    # Pool-entry opts are often slim (objective/pfc only). Prefer a richer opt that
    # carries term_losses when the top-level payload has one.
    top_opt = payload.get("opt") if isinstance(payload.get("opt"), dict) else None
    if isinstance(top_opt, dict) and top_opt.get("term_losses"):
        if not isinstance(opt, dict) or not opt.get("term_losses"):
            opt = top_opt
        else:
            # Merge missing diagnostic fields from top-level opt.
            merged = dict(opt)
            for k, v in top_opt.items():
                if k not in merged or merged.get(k) in (None, {}, []):
                    merged[k] = v
            opt = merged

    problem = payload.get("problem") or {}
    if isinstance(pool_entry, dict) and pool_entry.get("next_problem"):
        problem = pool_entry["next_problem"]

    ings = None
    for cand in (
        chosen.get("ingredients") if isinstance(chosen, dict) else None,
        (pool_entry or {}).get("ingredients") if isinstance(pool_entry, dict) else None,
        (problem.get("chosen_recipe") or {}).get("ingredients"),
        payload.get("ingredients"),
    ):
        if isinstance(cand, list) and cand:
            ings = cand
            break

    foodon = (
        payload.get("foodon_basis_report")
        or (chosen.get("foodon_basis_report") if isinstance(chosen, dict) else None)
        or (pool_entry or {}).get("foodon_basis_report")
        or (problem.get("foodon_basis_report") if isinstance(problem, dict) else None)
        or ((problem.get("chosen_recipe") or {}).get("foodon_basis_report") if isinstance(problem, dict) else None)
    )
    return {
        "opt": opt or {},
        "problem": problem or {},
        "ingredients": ings or [],
        "foodon_basis_report": foodon if isinstance(foodon, dict) else {},
        "pool_entry": pool_entry,
        "pfc_after": (opt or {}).get("pfc_after") if isinstance(opt, dict) else None,
    }


def _ratio_loss_from_term_losses(
    tl: dict[str, Any] | None,
    *,
    ratio_samples_n: int = 0,
) -> tuple[float | None, str | None]:
    """Return (value, source). Never invent 0 from missing keys / empty samples."""
    if not isinstance(tl, dict) or not tl:
        return None, None
    # Prefer explicit median-centered Wasserstein when the optimizer recorded it.
    med = _as_float(tl.get("mean_abs_dev_from_median"))
    if med is not None:
        return med, "mean_abs_dev_from_median"
    if "ratio_surrogate" in tl and tl["ratio_surrogate"] is not None:
        v = _as_float(tl["ratio_surrogate"])
        # Empty ratio_samples force the optimizer to report 0 — treat as missing.
        if ratio_samples_n <= 0 and v == 0.0:
            return None, None
        return v, "ratio_surrogate"
    for key in ("ratio_loss", "ratio"):
        if key in tl and tl[key] is not None:
            v = _as_float(tl[key])
            return v, key
    return None, None


def mean_abs_dev_from_median_shares(
    *,
    x: list[float] | Any,
    ingredient_basis: list[str | None],
    basis_samples: dict[str, Any],
) -> float | None:
    """Mean |mass_share − neighborhood_median| over mapped basis nodes (W1 to median)."""
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or arr.sum() <= 0:
        return None
    total = float(arr.sum())
    losses: list[float] = []
    seen: set[str] = set()
    for i, nid in enumerate(ingredient_basis):
        if not nid or nid in seen or i >= arr.size:
            continue
        samples_raw = basis_samples.get(nid)
        if samples_raw is None or isinstance(samples_raw, str):
            continue
        try:
            samples = np.asarray(samples_raw, dtype=float)
        except (TypeError, ValueError):
            continue
        if samples.size == 0:
            continue
        seen.add(str(nid))
        grams = float(
            sum(float(arr[j]) for j, n2 in enumerate(ingredient_basis) if n2 == nid and j < arr.size)
        )
        z = grams / total
        med = float(np.median(samples))
        losses.append(abs(z - med))
    if not losses:
        return None
    return float(np.mean(losses))


def extract_ratio_and_nutrient(
    payload: dict[str, Any],
) -> tuple[float | None, str | None, float | None, str | None]:
    """Authoritative ratio + nutrient from the chosen recipe context."""
    ctx = resolve_chosen_context(payload)
    opt = ctx["opt"]
    problem = ctx.get("problem") or {}
    tl = opt.get("term_losses") if isinstance(opt, dict) else None
    raw_rs = problem.get("ratio_samples")
    if isinstance(raw_rs, str) or raw_rs is None:
        # Artifact omission sentinel (e.g. "<omitted shape=list[0]>") or missing.
        ratio_n = 0
    else:
        try:
            ratio_n = len(raw_rs)
        except TypeError:
            ratio_n = 0

    # Prefer recompute of median-centered share deviation when samples are present.
    ratio: float | None = None
    ratio_src: str | None = None
    samples = problem.get("basis_samples")
    x_opt = (opt or {}).get("x_opt") if isinstance(opt, dict) else None
    basis = problem.get("ingredient_basis")
    if isinstance(samples, dict) and x_opt is not None and isinstance(basis, list):
        med_loss = mean_abs_dev_from_median_shares(
            x=x_opt,
            ingredient_basis=list(basis),
            basis_samples=samples,
        )
        if med_loss is not None:
            ratio, ratio_src = med_loss, "mean_abs_dev_from_median"

    if ratio is None:
        ratio, ratio_src = _ratio_loss_from_term_losses(tl, ratio_samples_n=ratio_n)
    # Fallback: sum of per-basis share losses (still a fidelity / ratio-family signal)
    if ratio is None and isinstance(tl, dict):
        share_parts = []
        for k, v in tl.items():
            ks = str(k)
            if ks.endswith("__share") or ks.endswith("__abs_dev_median") or ks in {
                "ratio_surrogate",
                "ratio_value",
                "ratio_loss",
                "ratio",
                "mean_abs_dev_from_median",
            }:
                continue
            fv = _as_float(v)
            if fv is not None:
                share_parts.append(fv)
        if share_parts:
            ratio, ratio_src = float(sum(share_parts)), "share_losses_sum"

    # Prefer recomputed slack from chosen pfc + box over stale telemetry.
    box = _macro_box_from_payload(payload)
    pfc = ctx.get("pfc_after") or (opt.get("pfc_after") if isinstance(opt, dict) else None)
    nutrient = _pfc_box_slack(pfc, box)
    nutrient_src = "pfc_box_slack" if nutrient is not None else None
    if nutrient is None:
        tel = payload.get("run_telemetry") or {}
        nutrient = _as_float(tel.get("final_nutrient_slack"))
        if nutrient is not None:
            nutrient_src = "telemetry_final_nutrient_slack"

    # Telemetry fallback only when marked as coming from ratio_surrogate (not share sums).
    if ratio is None:
        tel = payload.get("run_telemetry") or {}
        if tel.get("final_ratio_source") in {
            "ratio_surrogate",
            "ratio_loss",
            "ratio",
            "mean_abs_dev_from_median",
        }:
            tr = _as_float(tel.get("final_ratio_term"))
            if tr is not None:
                ratio, ratio_src = tr, f"telemetry_{tel.get('final_ratio_source')}"

    return ratio, ratio_src, nutrient, nutrient_src


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float(sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w)


def _iqr_stats(samples: list[float] | None) -> dict[str, float] | None:
    """Neighborhood share band for MacroIQ boxplots / typicality.

    ``q1``/``q3`` are the **P15 / P85** edges of the central band (not classical
    quartiles). Tukey fences still use 1.5× that band width. Sparse samples
    (n < 5) are marked via ``n`` and excluded from proportion fit elsewhere.
    """
    if not samples:
        return None
    vals = sorted(float(x) for x in samples if x is not None)
    if not vals:
        return None
    p15 = _percentile(vals, 0.15)
    p85 = _percentile(vals, 0.85)
    return {
        "min": float(vals[0]),
        "q1": float(p15),
        "p15": float(p15),
        "median": _percentile(vals, 0.50),
        "q3": float(p85),
        "p85": float(p85),
        "max": float(vals[-1]),
        "n": float(len(vals)),
    }


def share_band_from_iqr(
    recipe_share: float | None,
    iqr: dict[str, Any] | None,
    *,
    whisker: float = 1.5,
    min_n: float = 5,
    edge_eps: float = 0.01,
) -> str:
    """Color band from share vs neighborhood central band (P15–P85) + Tukey fences.

    - good: inside [p15, p85], or within ``edge_eps`` of either edge
    - warn: strictly outside that softened band but within Tukey fences
    - bad: beyond the fences (boxplot outliers)
    - unknown: missing / too-sparse neighborhood stats (does not affect proportion fit)
    """
    share = _as_float(recipe_share)
    if share is None or not isinstance(iqr, dict):
        return "unknown"
    q1 = _as_float(iqr.get("q1"))
    q3 = _as_float(iqr.get("q3"))
    if q1 is None or q3 is None:
        return "unknown"
    n = _as_float(iqr.get("n"))
    if n is not None and n < min_n:
        return "unknown"
    width = max(float(q3) - float(q1), 1e-9)
    fence_lo = float(q1) - whisker * width
    fence_hi = float(q3) + whisker * width
    # Soften the box edges so floating shares sitting on q1/q3 stay green.
    box_lo = float(q1) - float(edge_eps)
    box_hi = float(q3) + float(edge_eps)
    if box_lo <= share <= box_hi:
        return "good"
    if fence_lo <= share <= fence_hi:
        return "warn"
    return "bad"


def _calories_for_column(M: np.ndarray, i: int, grams: float) -> float | None:
    if M.ndim != 2 or i < 0 or i >= M.shape[1]:
        return None
    col = M[:, i]
    # Prefer Atwater from rows 0–2 (matches LP); fall back to energy row.
    if col.size >= 3:
        atw = float(grams) * (4.0 * float(col[0]) + 9.0 * float(col[1]) + 4.0 * float(col[2]))
        if atw > 0:
            return atw
    if col.size >= 4 and float(col[3]) > 0:
        return float(grams) * float(col[3])
    return atw if col.size >= 3 else None


def _format_amount(value: float, unit: str) -> str:
    """Human amount string; whole numbers when close, else one decimal."""
    unit = (unit or "g").strip()
    if abs(value - round(value)) < 0.05:
        qty_s = str(int(round(value)))
        whole = int(round(value))
    else:
        qty_s = f"{value:.1f}".rstrip("0").rstrip(".")
        whole = None
    if unit in {"g", "gram", "grams"}:
        return f"{qty_s} g"
    unit_out = unit
    if whole is not None and abs(whole) != 1:
        # Light pluralization for common cooking units.
        if unit.endswith(("spoon", "ounce", "cup", "clove", "slice", "piece")) and not unit.endswith("s"):
            unit_out = unit + "s"
        elif unit in {"tbsp", "tsp", "oz", "lb"}:
            unit_out = unit
    return f"{qty_s} {unit_out}"


def scaled_portion_amount(
    row: dict[str, Any],
    *,
    grams: float | None,
) -> dict[str, Any]:
    """Scale source quantity/unit by LP gram change; fall back to whole grams."""
    orig_g = _as_float(row.get("original_grams"))
    if orig_g is None:
        orig_g = _as_float(row.get("gram_weight"))
    qty = _as_float(row.get("quantity"))
    unit = str(row.get("unit") or "").strip() or None
    if (
        grams is not None
        and qty is not None
        and unit
        and unit.lower() not in {"g", "gram", "grams"}
        and orig_g is not None
        and orig_g > 1e-9
    ):
        scaled = float(qty) * (float(grams) / float(orig_g))
        return {
            "amount_value": scaled,
            "amount_unit": unit,
            "amount_display": _format_amount(scaled, unit),
            "amount_source": "scaled_portion",
        }
    if grams is not None:
        rounded = float(round(float(grams)))
        return {
            "amount_value": rounded,
            "amount_unit": "g",
            "amount_display": _format_amount(rounded, "g"),
            "amount_source": "grams",
        }
    return {
        "amount_value": None,
        "amount_unit": None,
        "amount_display": "—",
        "amount_source": "missing",
    }


def build_ingredient_display_rows(
    *,
    ingredients: list[dict[str, Any]],
    problem: dict[str, Any],
    opt: dict[str, Any],
    foodon_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Per-ingredient FoodOn, basis, levels, loss band, kcal, IQR sparkline stats."""
    foodon_report = foodon_report or {}
    fo_rows = list(foodon_report.get("ingredients") or [])
    basis_list = list(problem.get("ingredient_basis") or [])
    samples_map = problem.get("basis_samples") or {}
    M_raw = problem.get("M") or []
    try:
        M = np.asarray(M_raw, dtype=float)
        if M.ndim != 2:
            M = np.zeros((0, 0), dtype=float)
    except (TypeError, ValueError):
        M = np.zeros((0, 0), dtype=float)
    x_raw = opt.get("x_opt") or problem.get("x_opt") or problem.get("x0") or []
    try:
        x_opt = np.asarray(x_raw, dtype=float)
    except (TypeError, ValueError):
        x_opt = np.zeros(0, dtype=float)
    total_mass = float(problem.get("total_mass") or (float(x_opt.sum()) if x_opt.size else 0.0) or 0.0)
    tl = opt.get("term_losses") or {}

    # Recipe share per basis node
    share_by_basis: dict[str, float] = {}
    if total_mass > 0 and x_opt.size and basis_list:
        n = min(len(basis_list), x_opt.size)
        for i in range(n):
            nid = basis_list[i]
            if not nid:
                continue
            share_by_basis[str(nid)] = share_by_basis.get(str(nid), 0.0) + float(x_opt[i]) / total_mass

    out: list[dict[str, Any]] = []
    for i, row in enumerate(ingredients):
        fo = fo_rows[i] if i < len(fo_rows) else {}
        label = row.get("label") or row.get("name") or fo.get("label") or "?"
        grams = _as_float(row.get("grams") if row.get("grams") is not None else row.get("gram_weight"))
        if grams is None and i < x_opt.size:
            grams = float(x_opt[i])
        portion = scaled_portion_amount(row, grams=grams)
        basis = (
            fo.get("basis_node_id")
            or row.get("basis_node_id")
            or (basis_list[i] if i < len(basis_list) else None)
        )
        basis = str(basis) if basis else None
        leaf = fo.get("foodon_leaf_id") or row.get("foodon_id") or row.get("foodon_leaf_id")
        leaf = str(leaf) if leaf else None
        levels = fo.get("aggregation_levels")
        if levels is None:
            levels = row.get("aggregation_levels")
        if levels is not None:
            try:
                levels = int(levels)
            except (TypeError, ValueError):
                levels = None

        loss_contrib = _as_float(tl.get(basis)) if basis else None
        samples = None
        n_hits = 0
        if basis and basis in samples_map:
            try:
                samples = [float(x) for x in samples_map[basis]]
                n_hits = len(samples)
            except Exception:
                samples = None
        # Line-item mass share (what the user is editing). Basis-aggregate share is
        # still available via share_by_basis for loss terms.
        line_share = None
        if grams is not None and total_mass > 0:
            line_share = float(grams) / total_mass
        basis_share = share_by_basis.get(basis) if basis else None
        recipe_share = line_share if line_share is not None else basis_share
        if loss_contrib is None and recipe_share is not None and samples:
            # Mean absolute deviation vs neighborhood shares (same as empirical CDF L1).
            arr = np.asarray(samples, dtype=float)
            # Compare basis role when available; else the line share.
            compare_share = basis_share if basis_share is not None else float(recipe_share)
            loss_contrib = float(np.mean(np.abs(arr - float(compare_share))))
        elif loss_contrib is None and basis and str(basis) not in {"ood_lean_protein", "None"}:
            # Mapped but no neighborhood hits yet — mark unknown, not zero.
            loss_contrib = None

        iqr = _iqr_stats(samples)
        if iqr is None and isinstance(row.get("share_iqr"), dict):
            # Recompute payloads may omit basis_samples; keep the UI IQR.
            iqr = dict(row["share_iqr"])
            n_hits = int(_as_float(iqr.get("n")) or n_hits or 0)
        # Color by this line's share vs the basis neighborhood IQR.
        loss_band = share_band_from_iqr(recipe_share, iqr)

        kcal = None
        if grams is not None and M.ndim == 2 and i < M.shape[1]:
            kcal = _calories_for_column(M, i, grams)
        if kcal is None:
            # Prefer density scaling when the client already had calories for a
            # previous gram weight (common when M is omitted from the UI problem).
            prev_cal = _as_float(row.get("calories"))
            prev_g = _as_float(row.get("_prev_grams"))
            if (
                prev_cal is not None
                and prev_g is not None
                and prev_g > 1e-9
                and grams is not None
            ):
                kcal = float(prev_cal) * (float(grams) / float(prev_g))
            elif prev_cal is not None and grams is not None:
                # Same grams snapshot — keep prior calories.
                kcal = float(prev_cal)

        out.append(
            {
                "index": i,
                "label": label,
                "grams": grams,
                "grams_rounded": None if grams is None else int(round(float(grams))),
                "amount_value": portion["amount_value"],
                "amount_unit": portion["amount_unit"],
                "amount_display": portion["amount_display"],
                "amount_source": portion["amount_source"],
                "quantity": row.get("quantity"),
                "unit": row.get("unit"),
                "original_grams": row.get("original_grams") or row.get("gram_weight"),
                "source_text": row.get("source_text"),
                "fdc_id": row.get("fdc_id"),
                "foodon_leaf_id": leaf,
                "foodon_leaf_label": fo.get("foodon_leaf_label") or leaf,
                "basis_node_id": basis,
                "basis_node_label": fo.get("basis_node_label") or basis,
                "aggregation_levels": levels,
                "loss_contribution": loss_contrib,
                "loss_band": loss_band,
                "loss_label": "ratio loss",
                "basis_n_hits": n_hits,
                "calories": None if kcal is None else int(round(float(kcal))),
                "recipe_share": recipe_share,
                "share_iqr": iqr,
            }
        )
    return filter_display_ingredients(out)


def _extract_holistic_0_10(payload: dict[str, Any]) -> tuple[float | None, str]:
    feval = payload.get("final_evaluation") or {}
    if isinstance(feval, dict) and feval.get("overall_score_0_10") is not None:
        v = _as_float(feval["overall_score_0_10"])
        if v is not None:
            return max(0.0, min(10.0, v)), "final_evaluator"

    judge = payload.get("judge_result") or (payload.get("chosen") or {}).get("judge") or {}
    if isinstance(judge, dict):
        for key in ("holistic_score_0_10", "holistic_0_10", "score_0_10"):
            if judge.get(key) is not None:
                v = _as_float(judge[key])
                if v is not None:
                    return max(0.0, min(10.0, v)), "llm_judge"
        scores = judge.get("scores_0_10") or {}
        wid = judge.get("winner_id")
        if wid and isinstance(scores, dict) and scores.get(wid) is not None:
            v = _as_float(scores[wid])
            if v is not None:
                return max(0.0, min(10.0, v)), "llm_judge"

    tel = payload.get("run_telemetry") or {}
    if tel.get("final_holistic") is not None:
        v = _as_float(tel["final_holistic"])
        if v is not None:
            return max(0.0, min(10.0, round(v * 10.0, 1))), "intent_overlap"

    scored = payload.get("scored_finalists") or []
    winner_id = None
    chosen = payload.get("chosen") or {}
    if isinstance(chosen.get("entry"), dict):
        winner_id = chosen["entry"].get("candidate_id")
    if not winner_id and scored:
        winner_id = scored[0].get("candidate_id")
    for s in scored:
        if s.get("candidate_id") == winner_id and s.get("composite") is not None:
            v = _as_float(s["composite"])
            if v is not None:
                return max(0.0, min(10.0, round(v * 10.0, 1))), "composite"
    return None, "missing"


def build_display_scores(payload: dict[str, Any] | None, *, ready: bool = True) -> dict[str, Any]:
    """Compact payload for live + final score UI."""
    if not payload:
        return empty_display_scores()
    if not ready and not payload.get("opt") and not payload.get("chosen"):
        return empty_display_scores()

    ratio, ratio_src, nutrient, nutrient_src = extract_ratio_and_nutrient(payload)
    holistic, holistic_source = _extract_holistic_0_10(payload)
    ctx = resolve_chosen_context(payload)
    ingredients = build_ingredient_display_rows(
        ingredients=ctx["ingredients"],
        problem=ctx["problem"],
        opt=ctx["opt"],
        foodon_report=ctx["foodon_basis_report"],
    )

    # Do not color a missing signal as "good".
    ratio_band = band_for_loss(
        ratio, good_max=RATIO_LOSS_BANDS["good_max"], warn_max=RATIO_LOSS_BANDS["warn_max"]
    )
    nutrient_band = band_for_loss(
        nutrient, good_max=NUTRIENT_LOSS_BANDS["good_max"], warn_max=NUTRIENT_LOSS_BANDS["warn_max"]
    )

    pfc = ctx.get("pfc_after")
    if not isinstance(pfc, dict):
        pfc = (ctx.get("opt") or {}).get("pfc_after") if isinstance(ctx.get("opt"), dict) else None
    protein = _as_float((pfc or {}).get("protein") if isinstance(pfc, dict) else None)
    carb = _as_float(
        (pfc or {}).get("carb") if isinstance(pfc, dict) else None
    )
    if carb is None and isinstance(pfc, dict):
        carb = _as_float(pfc.get("carbs"))
    fat = _as_float((pfc or {}).get("fat") if isinstance(pfc, dict) else None)
    total_kcal = 0.0
    kcal_n = 0
    for row in ingredients:
        kcal = _as_float(row.get("calories"))
        if kcal is not None:
            total_kcal += float(kcal)
            kcal_n += 1
    if kcal_n == 0:
        # Fall back to Atwater from M · x when ingredient kcal rows are blank.
        M_raw = (ctx.get("problem") or {}).get("M") or []
        x_raw = (
            (ctx.get("opt") or {}).get("x_opt")
            or (ctx.get("problem") or {}).get("x_opt")
            or (ctx.get("problem") or {}).get("x0")
            or []
        )
        try:
            M = np.asarray(M_raw, dtype=float)
            x_opt = np.asarray(x_raw, dtype=float)
        except (TypeError, ValueError):
            M = np.zeros((0, 0), dtype=float)
            x_opt = np.zeros(0, dtype=float)
        if M.ndim == 2 and x_opt.size and M.shape[1] == x_opt.size:
            for i in range(x_opt.size):
                kcal = _calories_for_column(M, i, float(x_opt[i]))
                if kcal is not None:
                    total_kcal += float(kcal)
                    kcal_n += 1
    macros = {
        "protein": None if protein is None else round(protein * 100),
        "carb": None if carb is None else round(carb * 100),
        "fat": None if fat is None else round(fat * 100),
        "calories": None if kcal_n == 0 else int(round(total_kcal)),
    }

    from recipe_opt_agent.portion_display import (
        annotate_ingredient_edits,
        collect_applied_edits,
        cookability_from_score_history,
        mark_novel_ingredients,
    )

    applied_edits = collect_applied_edits(payload)
    ingredients = annotate_ingredient_edits(ingredients, applied_edits)
    original = (
        payload.get("original_ingredients")
        or ((payload.get("problem") or {}).get("starting_ingredients"))
        or ((payload.get("problem") or {}).get("chosen_recipe") or {}).get("original_ingredients")
        or []
    )
    # Prefer frozen init snapshot when present on the run payload.
    if not original and isinstance(payload.get("chosen_recipe"), dict):
        # no-op; keep empty
        pass
    ingredients = mark_novel_ingredients(
        ingredients,
        original_ingredients=list(original) if isinstance(original, list) else [],
    )
    cookability = cookability_from_score_history(
        list(payload.get("score_history") or []),
        final_ratio=ratio,
    )

    typicality = proportion_typicality_from_ingredients(ingredients)

    return {
        "ready": True,
        "iteration": payload.get("iteration"),
        "ratio_loss": {
            "value": ratio,
            "band": typicality.get("band") or ratio_band,
            "label": "Ratio loss",
            "direction": "lower_better",
            "unit": "surrogate",
            "source": ratio_src,
            "explanation": (
                "Typicality uses ingredients with a valid neighborhood IQR only. "
                "A worse band requires BOTH the %-of-those-ingredients outside IQR and the "
                "%-of-their-calories outside IQR to exceed the threshold "
                "(<17% very typical; <25% mostly typical; <35% somewhat different; 35%+ substantially off)."
            ),
            "band_summary": typicality.get("summary")
            or RATIO_BAND_SUMMARY.get(ratio_band, RATIO_BAND_SUMMARY["unknown"]),
            "proportion_key": typicality.get("key"),
            "proportion_css": typicality.get("css"),
            "outside_iqr_frac": typicality.get("outside_iqr_frac"),
            "outside_iqr_pct": typicality.get("outside_iqr_pct"),
            "outside_iqr_calorie_frac": typicality.get("outside_iqr_calorie_frac"),
            "outside_iqr_calorie_pct": typicality.get("outside_iqr_calorie_pct"),
            "thresholds": dict(RATIO_LOSS_BANDS),
        },
        "nutrient_loss": {
            "value": nutrient,
            "band": nutrient_band,
            "label": "Nutrient loss",
            "direction": "lower_better",
            "unit": "pfc_slack",
            "source": nutrient_src,
            "explanation": (
                "L1 calorie-fraction distance outside the protein/carb/fat box "
                "(0 = inside the box). Recomputed from the chosen recipe PFC."
            ),
            "band_summary": NUTRIENT_BAND_SUMMARY.get(nutrient_band, NUTRIENT_BAND_SUMMARY["unknown"]),
            "thresholds": dict(NUTRIENT_LOSS_BANDS),
        },
        "holistic_0_10": {
            "value": holistic,
            "band": band_for_holistic_0_10(holistic),
            "label": "Holistic",
            "direction": "higher_better",
            "unit": "0_10",
            "source": holistic_source,
            "explanation": "LLM judge 0–10 when available; otherwise intent/composite × 10.",
        },
        "macros": macros,
        "pfc_after": (
            {"protein": protein, "carb": carb, "fat": fat}
            if protein is not None or carb is not None or fat is not None
            else None
        ),
        "cookability": cookability,
        "applied_edits": applied_edits,
        "ingredients": ingredients,
        "score_history": list(payload.get("score_history") or []),
        "L_max_norm": _as_float((payload.get("diagnosis") or {}).get("L_max_norm"))
        or _as_float(((ctx.get("opt") or {}).get("term_losses") or {}).get("L_max_norm")),
        "status": payload.get("status"),
        "title": payload.get("title"),
    }


def live_scores_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Build a live score snapshot after diagnose / propose (best current sample)."""
    cfg = state.get("config") or {}
    payload = {
        "opt": state.get("opt"),
        "problem": state.get("problem"),
        "chosen": {"ingredients": (state.get("chosen_recipe") or {}).get("ingredients"), "opt": state.get("opt")},
        "foodon_basis_report": state.get("foodon_basis_report")
        or (state.get("problem") or {}).get("foodon_basis_report"),
        "config": cfg,
        "macro_targets": {
            "protein_min": cfg.get("protein_min"),
            "protein_max": cfg.get("protein_max"),
            "carb_min": cfg.get("carb_min"),
            "carb_max": cfg.get("carb_max"),
            "fat_min": cfg.get("fat_min"),
            "fat_max": cfg.get("fat_max"),
        },
        "iteration": state.get("iteration"),
        "diagnosis": state.get("diagnosis"),
        "run_telemetry": state.get("run_telemetry"),
        "score_history": state.get("score_history"),
        "title": state.get("title"),
        "status": state.get("status"),
    }
    return build_display_scores(payload, ready=True)


def best_branch_scores_from_bundles(
    bundles: list[dict[str, Any]],
    *,
    iteration: int,
    box: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """One history point per branch from the best (lowest delta_L_star) LP bundle."""
    best: dict[str, dict[str, Any]] = {}
    for b in bundles or []:
        if not b.get("lp_evaluated") or b.get("oscillation_blocked"):
            continue
        branch = str(b.get("branch") or "in_distribution")
        d = _as_float(b.get("delta_L_star"))
        prev = best.get(branch)
        if prev is None or (d is not None and (prev.get("delta_L_star") is None or d < prev["delta_L_star"])):
            nutrient = _as_float(b.get("nutrient_slack"))
            if nutrient is None and box and b.get("pfc_after"):
                nutrient = _pfc_box_slack(b.get("pfc_after"), box)
            # Bundle ratio_term may be surrogate or value; prefer next_problem term_losses when present
            ratio = None
            np_ = b.get("next_problem") if isinstance(b.get("next_problem"), dict) else None
            # next_problem usually stripped in SSE; use stored ratio_term if >0 or always store
            ratio = _as_float(b.get("ratio_surrogate"))
            if ratio is None:
                # If ratio_term looks like a raw ratio (typically >1), don't use as loss
                rt = _as_float(b.get("ratio_term"))
                if rt is not None and rt < 1.5:
                    ratio = rt
            best[branch] = {
                "iteration": iteration,
                "branch": branch,
                "delta_L_star": d,
                "ratio_loss": ratio,
                "nutrient_loss": nutrient,
                "bundle_id": b.get("bundle_id"),
            }
    return list(best.values())


ID_PATH_BRANCHES = frozenset(
    {"in_distribution", "moderate", "must_retry_feasible", "accept_polish", "current"}
)
OOD_PATH_BRANCHES = frozenset({"ood_protein", "ood_other", "hybrid"})


def branch_path_family(branch: str | None) -> str | None:
    """Map a candidate branch tag to ``in_distribution`` or ``ood`` path family."""
    b = str(branch or "in_distribution")
    if b in ID_PATH_BRANCHES:
        return "in_distribution"
    if b in OOD_PATH_BRANCHES or b.startswith("ood"):
        return "ood"
    if b == "hybrid":
        return "ood"
    if b.startswith("in_"):
        return "in_distribution"
    return None


def _candidate_rank_key(candidate: dict[str, Any], *, ood_handicap: float = 0.015) -> tuple:
    """Lower is better. Prefer LP ``delta_L_star``, then diagnosis norms."""
    d = _as_float(candidate.get("delta_L_star"))
    if d is not None:
        fam = branch_path_family(candidate.get("branch"))
        if fam == "ood":
            nutrient = _as_float(candidate.get("nutrient_slack"))
            extra = 0.0
            if nutrient is not None and nutrient <= 1e-6:
                extra = 0.5 * float(ood_handicap)
            d = d - float(ood_handicap) - extra
        return (0, d)

    lmn = _as_float(candidate.get("L_max_norm"))
    if lmn is None and isinstance(candidate.get("diagnosis_full"), dict):
        lmn = _as_float(candidate["diagnosis_full"].get("L_max_norm"))
    if lmn is not None:
        n_red = int(candidate.get("n_red") or (candidate.get("diagnosis_full") or {}).get("n_red") or 99)
        obj = _as_float(candidate.get("objective"))
        if obj is None:
            obj = _as_float((candidate.get("opt") or {}).get("objective"))
        return (1, lmn, n_red, obj if obj is not None else 99.0)

    obj = _as_float((candidate.get("opt") or {}).get("objective"))
    if obj is not None:
        return (2, obj)
    return (3, 99.0)


def _candidate_has_recipe_signal(candidate: dict[str, Any]) -> bool:
    if candidate.get("ingredients"):
        return True
    if isinstance(candidate.get("next_problem"), dict):
        cr = (candidate["next_problem"].get("chosen_recipe") or {}).get("ingredients")
        if cr:
            return True
    opt = candidate.get("opt") or {}
    if opt.get("x_opt"):
        return True
    if candidate.get("x_opt"):
        return True
    return False


def _normalize_candidate_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    if not row.get("candidate_id") and row.get("bundle_id"):
        row["candidate_id"] = f"bundle::{row['bundle_id']}"
    return row


def _candidate_to_final_payload(
    candidate: dict[str, Any],
    state: dict[str, Any],
    *,
    path_key: str,
) -> dict[str, Any]:
    """Build a mini final payload for one path-family champion."""
    cfg = state.get("config") or {}
    candidate = _normalize_candidate_row(candidate)
    next_problem = candidate.get("next_problem") if isinstance(candidate.get("next_problem"), dict) else None
    problem = next_problem or state.get("problem") or {}

    opt = dict(candidate.get("opt") or {})
    if not opt.get("term_losses"):
        tl: dict[str, float] = {}
        rs = _as_float(candidate.get("ratio_surrogate"))
        if rs is not None:
            tl["ratio_surrogate"] = rs
        else:
            rt = _as_float(candidate.get("ratio_term"))
            if rt is not None and rt < 1.5:
                tl["ratio_surrogate"] = rt
        if tl:
            opt["term_losses"] = tl
    if not opt.get("pfc_after") and candidate.get("pfc_after"):
        opt["pfc_after"] = candidate["pfc_after"]
    if not opt.get("x_opt") and candidate.get("x_opt"):
        opt["x_opt"] = candidate["x_opt"]

    ingredients = candidate.get("ingredients")
    if not ingredients and next_problem:
        ingredients = (next_problem.get("chosen_recipe") or {}).get("ingredients")
    if not ingredients:
        ingredients = (state.get("chosen_recipe") or {}).get("ingredients")

    foodon = (
        candidate.get("foodon_basis_report")
        or problem.get("foodon_basis_report")
        or state.get("foodon_basis_report")
        or ((problem.get("chosen_recipe") or {}).get("foodon_basis_report"))
    )

    chosen = {
        "source": f"path_{path_key}",
        "entry": candidate,
        "opt": opt,
        "ingredients": ingredients,
        "edits": candidate.get("edits"),
        "branch": candidate.get("branch") or path_key,
        "candidate_id": candidate.get("candidate_id"),
        "delta_L_star": candidate.get("delta_L_star"),
        "foodon_basis_report": foodon,
    }
    payload: dict[str, Any] = {
        "path_key": path_key,
        "path_label": "In-distribution" if path_key == "in_distribution" else "OOD",
        "chosen": chosen,
        "problem": problem,
        "opt": opt,
        "config": cfg,
        "macro_targets": {
            "protein_min": cfg.get("protein_min"),
            "protein_max": cfg.get("protein_max"),
            "carb_min": cfg.get("carb_min"),
            "carb_max": cfg.get("carb_max"),
            "fat_min": cfg.get("fat_min"),
            "fat_max": cfg.get("fat_max"),
        },
        "foodon_basis_report": foodon,
        "title": state.get("title"),
        "iteration": candidate.get("iteration") if candidate.get("iteration") is not None else state.get("iteration"),
        "score_history": state.get("score_history") or [],
    }
    payload["display_scores"] = build_display_scores(payload)
    return payload


def select_path_finalists(
    state: dict[str, Any],
    *,
    ood_handicap: float = 0.015,
) -> dict[str, Any | None]:
    """Pick the best in-distribution and OOD path champions for dual final display."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(raw: dict[str, Any] | None) -> None:
        if not isinstance(raw, dict):
            return
        row = _normalize_candidate_row(raw)
        cid = str(row.get("candidate_id") or row.get("bundle_id") or "")
        if not cid:
            cid = f"anon::{len(seen)}"
        if cid in seen:
            return
        if not _candidate_has_recipe_signal(row):
            return
        seen.add(cid)
        candidates.append(row)

    for row in state.get("interesting_candidates") or []:
        _add(row)
    for row in state.get("candidate_pool") or []:
        _add(row)
    for b in state.get("bundles") or []:
        _add(b)
    for s in state.get("scored_finalists") or []:
        entry = s.get("entry") if isinstance(s.get("entry"), dict) else None
        nested = None
        if entry and isinstance(entry.get("metrics"), dict):
            nested = (entry.get("metrics") or {}).get("raw", {}).get("entry")
        pool_entry = nested if isinstance(nested, dict) else entry
        if isinstance(pool_entry, dict):
            merged = {**pool_entry, "candidate_id": s.get("candidate_id") or pool_entry.get("candidate_id")}
            if not merged.get("branch"):
                merged["branch"] = s.get("branch")
            _add(merged)

    last = state.get("last_applied_candidate") or {}
    current_branch = last.get("branch") or "in_distribution"
    _add(
        {
            "candidate_id": "current::final",
            "branch": current_branch,
            "opt": state.get("opt"),
            "diagnosis_full": state.get("diagnosis"),
            "L_max_norm": (state.get("diagnosis") or {}).get("L_max_norm"),
            "n_red": (state.get("diagnosis") or {}).get("n_red"),
            "objective": (state.get("opt") or {}).get("objective"),
            "ingredients": (state.get("chosen_recipe") or {}).get("ingredients"),
            "foodon_basis_report": state.get("foodon_basis_report")
            or (state.get("problem") or {}).get("foodon_basis_report"),
            "source": "current_state",
            "iteration": state.get("iteration"),
        }
    )

    by_family: dict[str, list[dict[str, Any]]] = {"in_distribution": [], "ood": []}
    for row in candidates:
        fam = branch_path_family(row.get("branch"))
        if fam:
            by_family[fam].append(row)

    out: dict[str, Any | None] = {"in_distribution": None, "ood": None}
    for fam in ("in_distribution", "ood"):
        pool = by_family[fam]
        if not pool:
            continue
        best = min(pool, key=lambda c: _candidate_rank_key(c, ood_handicap=ood_handicap))
        out[fam] = _candidate_to_final_payload(best, state, path_key=fam)
    return out


def extract_judge_rationale(payload: dict[str, Any] | None) -> str | None:
    """Best-available LLM/heuristic rationale comparing final candidates."""
    payload = payload or {}
    for path in (
        (payload.get("judge_result") or {}).get("rationale"),
        (payload.get("final_judgment") or {}).get("rationale"),
        ((payload.get("chosen") or {}).get("arbiter_rationale")),
        ((payload.get("chosen") or {}).get("judge") or {}).get("rationale"),
    ):
        text = str(path or "").strip()
        if text and text.lower() not in {"none", "null"}:
            return text
    return None


def _polish_candidate_display(
    *,
    display_scores: dict[str, Any],
    problem: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """USDA enrich + consolidate duplicates; optionally remap problem columns."""
    from recipe_opt_agent.portion_display import (
        consolidate_duplicate_ingredients,
        enrich_ingredients_with_usda_portions,
    )

    display = dict(display_scores or {})
    ings = list(display.get("ingredients") or [])
    try:
        ings = enrich_ingredients_with_usda_portions(ings)
    except Exception:
        pass
    merged, problem_update = consolidate_duplicate_ingredients(
        ings,
        problem=problem if isinstance(problem, dict) else None,
    )
    display["ingredients"] = filter_display_ingredients(merged)
    return display, problem_update


def prepare_browse_candidates(
    final_payload: dict[str, Any],
    state: dict[str, Any] | None = None,
    *,
    max_candidates: int = 4,
) -> list[dict[str, Any]]:
    """Build ≤N MacroIQ-browsable candidates ranked best→worst.

    Primary order is proportion quality. LLM-flagged weird recipes are demoted
    to the end (still ordered among themselves by proportion).
    """
    state = state or {}
    max_n = max(1, int(max_candidates))
    recommended_id = None
    chosen = final_payload.get("chosen") or {}
    if isinstance(chosen.get("entry"), dict):
        recommended_id = chosen["entry"].get("candidate_id") or chosen.get("arbiter_winner_id")
    recommended_id = (
        recommended_id
        or chosen.get("candidate_id")
        or (final_payload.get("judge_result") or {}).get("winner_id")
        or (final_payload.get("final_judgment") or {}).get("winner_id")
        or "recommended"
    )

    weird_by_id: dict[str, dict[str, Any]] = {}
    raw_weird = final_payload.get("weird_flags") or {}
    if isinstance(raw_weird, dict):
        weird_by_id = {
            str(k): v for k, v in raw_weird.items() if isinstance(v, dict) and v.get("is_weird")
        }
    for cid in final_payload.get("weird_candidate_ids") or []:
        weird_by_id.setdefault(str(cid), {"is_weird": True, "odd_ingredients": [], "note": ""})

    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    from recipe_opt_agent.kcal_utils import resolve_user_kcal_target, scale_candidate_to_kcal

    kcal_target = resolve_user_kcal_target(
        final_payload.get("config"),
        state.get("config"),
        final_payload.get("problem"),
        state.get("problem"),
    )

    def _push(
        *,
        candidate_id: str,
        title: str,
        branch: str | None,
        display: dict[str, Any],
        problem: dict[str, Any] | None,
        macro_targets: dict[str, Any] | None,
        is_recommended: bool,
        edits: list[dict[str, Any]] | None = None,
        source: str | None = None,
    ) -> None:
        cid = str(candidate_id or title)
        if cid in seen:
            return
        seen.add(cid)
        problem_scaled, display_scaled = scale_candidate_to_kcal(
            problem=problem if isinstance(problem, dict) else {},
            display=display or {},
            kcal_target=kcal_target,
        )
        scores = display_scaled or {}
        problem_out = problem_scaled or problem or {}
        macros = scores.get("macros") or {}
        iqr_frac, iqr_in, iqr_known = _iqr_in_band_fraction(scores)
        ratio_val = (scores.get("ratio_loss") or {}).get("value")
        outside_cal = (scores.get("ratio_loss") or {}).get("outside_iqr_calorie_frac")
        if outside_cal is None:
            try:
                outside_cal = _iqr_alignment_stats(list(scores.get("ingredients") or [])).get(
                    "outside_iqr_calorie_frac"
                )
            except Exception:
                outside_cal = None
        weird_meta = weird_by_id.get(cid) or {}
        cards.append(
            {
                "candidate_id": cid,
                "title": title,
                "branch": branch or "in_distribution",
                "is_recommended": bool(is_recommended),
                "source": source,
                "display_scores": scores,
                "problem": problem_out,
                "macro_targets": macro_targets or final_payload.get("macro_targets") or {},
                "edits": list(edits or scores.get("applied_edits") or []),
                "is_weird": bool(weird_meta.get("is_weird")),
                "weird_note": weird_meta.get("note") or "",
                "odd_ingredients": list(weird_meta.get("odd_ingredients") or []),
                "score_summary": {
                    "macros": macros,
                    "ratio_loss": ratio_val,
                    "ratio_band": (scores.get("ratio_loss") or {}).get("band"),
                    "nutrient_loss": (scores.get("nutrient_loss") or {}).get("value"),
                    "nutrient_band": (scores.get("nutrient_loss") or {}).get("band"),
                    "cookability": (scores.get("cookability") or {}).get("summary"),
                    "holistic_0_10": (scores.get("holistic_0_10") or {}).get("value"),
                    "iqr_in_band_frac": iqr_frac,
                    "iqr_in_band_count": iqr_in,
                    "iqr_known_count": iqr_known,
                    "outside_iqr_calorie_frac": outside_cal,
                    "calories": macros.get("calories"),
                    "is_weird": bool(weird_meta.get("is_weird")),
                },
            }
        )

    # 1) Current enriched final (arbiter/judge pick) — may not stay first after sort
    rec_display = dict(final_payload.get("display_scores") or {})
    rec_problem = (
        final_payload.get("problem")
        if isinstance(final_payload.get("problem"), dict)
        else state.get("problem")
    )
    _push(
        candidate_id=str(recommended_id),
        title="Recommended",
        branch=(chosen.get("branch") or (chosen.get("entry") or {}).get("branch")),
        display=rec_display,
        problem=rec_problem,
        macro_targets=final_payload.get("macro_targets"),
        is_recommended=True,
        edits=rec_display.get("applied_edits"),
        source=chosen.get("source") or "recommended",
    )

    # 2) Creative scored finalists / alternatives
    for raw in list(final_payload.get("scored_finalists") or []) + list(
        final_payload.get("alternatives") or []
    ):
        if not isinstance(raw, dict):
            continue
        entry = raw.get("entry") if isinstance(raw.get("entry"), dict) else raw
        if isinstance(entry.get("metrics"), dict):
            nested = (entry.get("metrics") or {}).get("raw", {}).get("entry")
            if isinstance(nested, dict):
                entry = {**nested, "candidate_id": raw.get("candidate_id") or nested.get("candidate_id")}
        cid = str(raw.get("candidate_id") or entry.get("candidate_id") or "")
        if not cid or cid in seen:
            continue
        mini = _candidate_to_final_payload(
            {**entry, "candidate_id": cid},
            {**state, "config": final_payload.get("config") or state.get("config") or {}},
            path_key=branch_path_family(entry.get("branch")) or "in_distribution",
        )
        display, problem_update = _polish_candidate_display(
            display_scores=mini.get("display_scores") or {},
            problem=mini.get("problem"),
        )
        problem = problem_update or mini.get("problem")
        _push(
            candidate_id=cid,
            title="Alternative",
            branch=entry.get("branch"),
            display=display,
            problem=problem,
            macro_targets=mini.get("macro_targets"),
            is_recommended=False,
            edits=entry.get("edits") or display.get("applied_edits"),
            source="scored_finalist",
        )

    # 3) Path finals (ID / OOD champions)
    for fam, payload in (final_payload.get("path_finals") or {}).items():
        if not isinstance(payload, dict):
            continue
        cid = f"path_{fam}"
        if cid in seen:
            continue
        display = dict(payload.get("display_scores") or {})
        problem = payload.get("problem") if isinstance(payload.get("problem"), dict) else None
        display, problem_update = _polish_candidate_display(display_scores=display, problem=problem)
        problem = problem_update or problem
        _push(
            candidate_id=cid,
            title="In-distribution" if fam == "in_distribution" else "Out-of-distribution",
            branch=fam,
            display=display,
            problem=problem,
            macro_targets=payload.get("macro_targets"),
            is_recommended=False,
            edits=(payload.get("chosen") or {}).get("edits") or display.get("applied_edits"),
            source="path_final",
        )

    # 4) Arbiter / pool leftovers
    for raw in list(state.get("candidate_pool") or []) + list(
        final_payload.get("interesting_candidates") or []
    ):
        if not isinstance(raw, dict):
            continue
        cid = str(raw.get("candidate_id") or "")
        if not cid or cid in seen:
            continue
        if not _candidate_has_recipe_signal(raw):
            continue
        mini = _candidate_to_final_payload(
            raw,
            {**state, "config": final_payload.get("config") or state.get("config") or {}},
            path_key=branch_path_family(raw.get("branch")) or "in_distribution",
        )
        display, problem_update = _polish_candidate_display(
            display_scores=mini.get("display_scores") or {},
            problem=mini.get("problem"),
        )
        _push(
            candidate_id=cid,
            title="Alternative",
            branch=raw.get("branch"),
            display=display,
            problem=problem_update or mini.get("problem"),
            macro_targets=mini.get("macro_targets"),
            is_recommended=False,
            edits=raw.get("edits") or display.get("applied_edits"),
            source="pool",
        )

    cards.sort(key=lambda c: _browse_rank_key(c, kcal_target=kcal_target))
    cards = cards[:max_n]

    # Best proportion quality first (weird demoted); relabel Recommended + Option N
    for i, card in enumerate(cards):
        card["is_recommended"] = i == 0
        card["proportion_rank"] = i + 1
        branch = str(card.get("branch") or "")
        suffix = ""
        if card.get("is_weird"):
            suffix = " · unusual ingredients"
        elif branch.startswith("ood"):
            suffix = " · stretch"
        elif branch == "hybrid":
            suffix = " · hybrid"
        if i == 0:
            card["title"] = "Recommended"
        else:
            card["title"] = f"Option {i + 1}{suffix}"
    return cards


def _iqr_in_band_fraction(display: dict[str, Any] | None) -> tuple[float, int, int]:
    """Fraction of ingredients whose recipe share falls in the neighborhood IQR."""
    ings = list((display or {}).get("ingredients") or [])
    stats = _iqr_alignment_stats(ings)
    return float(stats["in_iqr_frac"]), int(stats["in_iqr_count"]), int(stats["known_count"])


def _iqr_width_usable(q1: float, q3: float) -> bool:
    """False for zero-width / degenerate central bands (do not score or color)."""
    lo, hi = (q1, q3) if q1 <= q3 else (q3, q1)
    width = float(hi) - float(lo)
    if width <= 1e-12:
        return False
    # Near-zero relative width (collapsed P15≈P85) is also unusable.
    scale = max(abs(float(lo)), abs(float(hi)), 1e-9)
    if width / scale <= 1e-6:
        return False
    return True


def _ingredient_has_valid_iqr(row: dict[str, Any] | None) -> bool:
    """True when neighborhood IQR stats are usable for typicality scoring.

    Sparse (n < 5) and degenerate (zero / near-zero width) bands are excluded so
    they neither penalize proportion typicality nor drive UI outside-band tones.
    """
    if not isinstance(row, dict):
        return False
    iqr = row.get("share_iqr") if isinstance(row.get("share_iqr"), dict) else None
    if not iqr:
        return False
    q1 = _as_float(iqr.get("q1"))
    q3 = _as_float(iqr.get("q3"))
    if q1 is None or q3 is None:
        return False
    n = _as_float(iqr.get("n"))
    if n is not None and n < 5:
        return False
    if not _iqr_width_usable(float(q1), float(q3)):
        return False
    return True


def _is_zero_portion(row: dict[str, Any] | None) -> bool:
    """True when the line has no meaningful mass / kitchen amount.

    Explicit 0 grams/amount counts as zero. Missing amount fields still count as
    non-zero when calories are present (common in typicality unit tests / rollups).
    """
    if not isinstance(row, dict):
        return True
    grams = _as_float(row.get("grams"))
    if grams is not None:
        return abs(float(grams)) <= 1e-9
    amount = _as_float(row.get("amount_value"))
    if amount is not None:
        return abs(float(amount)) <= 1e-9
    cal = _as_float(row.get("calories"))
    if cal is not None and float(cal) > 0:
        return False
    return True


def _has_distribution_data(row: dict[str, Any] | None) -> bool:
    """Neighborhood share distribution usable for coloring / typicality."""
    return _ingredient_has_valid_iqr(row)


def should_include_display_ingredient(row: dict[str, Any] | None) -> bool:
    """Drop zero-amount lines that also lack neighborhood distribution data.

    Grey (unknown) rows with real mass stay visible; empty phantom spices with no
    IQR do not.
    """
    if not isinstance(row, dict):
        return False
    label = str(row.get("label") or row.get("name") or "").strip()
    if not label:
        return False
    if _is_zero_portion(row) and not _has_distribution_data(row):
        return False
    return True


def filter_display_ingredients(ingredients: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Omit zero-portion + no-distribution ingredients from MacroIQ lists."""
    return [dict(r) for r in (ingredients or []) if should_include_display_ingredient(r)]


def _row_weight_kcal(row: dict[str, Any]) -> float | None:
    """Calorie (or mass) weight for typicality; None if the row should be skipped."""
    if _is_zero_portion(row):
        return None
    cal = _as_float(row.get("calories"))
    if cal is not None and cal > 0:
        return float(cal)
    grams = _as_float(row.get("grams"))
    if grams is not None and grams > 0:
        return float(grams)
    return None


def _iqr_counts_from_ingredients(ingredients: list[dict[str, Any]] | None) -> tuple[float, int, int]:
    """Return (in_band_frac, in_band_count, known_count) for ingredients with valid IQR."""
    stats = _iqr_alignment_stats(ingredients)
    return float(stats["in_iqr_frac"]), int(stats["in_iqr_count"]), int(stats["known_count"])


def _iqr_alignment_stats(ingredients: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Calorie-weighted IQR alignment, deduped by FoodOn basis node.

    Neighborhood IQRs are basis-level. Counting every line item separately (or
    equal-weighting spices) makes a 2 kcal ginger tweak look like a recipe-wide
    proportion failure. We group by ``basis_node_id``, sum calories, and compare
    the group's mass share to that basis IQR once.

    Zero-portion rows and rows without usable distribution data are ignored.
    """
    groups: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(ingredients or []):
        if _is_zero_portion(row):
            continue
        if not _ingredient_has_valid_iqr(row):
            continue
        share = _as_float(row.get("recipe_share"))
        if share is None:
            continue
        iqr = row.get("share_iqr") or {}
        q1 = _as_float(iqr.get("q1"))
        q3 = _as_float(iqr.get("q3"))
        if q1 is None or q3 is None:
            continue
        weight = _row_weight_kcal(row)
        if weight is None:
            continue
        basis = row.get("basis_node_id") or row.get("basis_node_label")
        key = str(basis) if basis else f"__row_{idx}"
        g = groups.get(key)
        if g is None:
            groups[key] = {
                "shares": [],
                "weight": 0.0,
                "q1": float(q1),
                "q3": float(q3),
            }
            g = groups[key]
        g["shares"].append(float(share))
        g["weight"] = float(g["weight"]) + float(weight)

    known = 0
    in_band = 0
    cal_known = 0.0
    cal_outside = 0.0
    for g in groups.values():
        shares = list(g["shares"])
        if not shares:
            continue
        # Rows that already store the aggregate basis share are identical; take
        # one. Otherwise sum line-item shares for the basis group.
        if len(shares) > 1 and max(shares) - min(shares) <= 1e-9:
            share = shares[0]
        else:
            share = float(sum(shares))
        q1 = float(g["q1"])
        q3 = float(g["q3"])
        lo, hi = (q1, q3) if q1 <= q3 else (q3, q1)
        outside = not (lo <= share <= hi)
        known += 1
        if not outside:
            in_band += 1
        w = float(g["weight"])
        cal_known += w
        if outside:
            cal_outside += w

    in_frac = (in_band / known) if known else 0.0
    outside_count = known - in_band
    outside_count_frac = (outside_count / known) if known else 0.0
    outside_cal_frac = (cal_outside / cal_known) if cal_known > 1e-9 else 0.0
    return {
        "in_iqr_frac": float(in_frac),
        "in_iqr_count": int(in_band),
        "known_count": int(known),
        "outside_count": int(outside_count),
        "outside_iqr_frac": float(outside_count_frac),
        "outside_iqr_calorie_frac": float(outside_cal_frac),
        "outside_iqr_calories": float(cal_outside),
        "known_iqr_calories": float(cal_known),
    }


def proportion_typicality_from_ingredients(
    ingredients: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Map central-band misalignment → MacroIQ proportion copy.

    Severity is the **calorie fraction** of basis groups whose share sits outside
    the neighborhood P15–P85 band. Sparse neighbor lines (n < 5) and zero-portion
    rows are ignored — they do not count against the proportion fit band.
    """
    stats = _iqr_alignment_stats(ingredients)
    known = int(stats["known_count"])
    if known <= 0 or float(stats.get("known_iqr_calories") or 0) <= 1e-9:
        meta = PROPORTION_TYPICALITY["unknown"]
        return {
            "key": "unknown",
            "summary": meta["summary"],
            "band": meta["band"],
            "css": meta["css"],
            "outside_iqr_frac": None,
            "outside_iqr_pct": None,
            "outside_iqr_calorie_frac": None,
            "outside_iqr_calorie_pct": None,
            "in_iqr_count": 0,
            "known_count": 0,
        }
    count_pct = 100.0 * float(stats["outside_iqr_frac"])
    cal_pct = 100.0 * float(stats["outside_iqr_calorie_frac"])
    severity_pct = cal_pct
    # Sparse / unknown-distribution lines are already excluded from ``stats``.
    if severity_pct < 17.0:
        key = "very_typical"
    elif severity_pct < 25.0:
        key = "mostly_typical"
    elif severity_pct < 35.0:
        key = "somewhat_different"
    else:
        key = "substantially_off"
    meta = PROPORTION_TYPICALITY[key]
    return {
        "key": key,
        "summary": meta["summary"],
        "band": meta["band"],
        "css": meta["css"],
        "outside_iqr_frac": float(stats["outside_iqr_frac"]),
        "outside_iqr_pct": int(round(count_pct)),
        "outside_iqr_calorie_frac": float(stats["outside_iqr_calorie_frac"]),
        "outside_iqr_calorie_pct": int(round(cal_pct)),
        "severity_pct": float(severity_pct),
        "in_iqr_count": int(stats["in_iqr_count"]),
        "known_count": known,
    }


def _browse_rank_key(
    card: dict[str, Any],
    *,
    kcal_target: float | None = None,
) -> tuple:
    """Lower is better: weird demotion, then proportion quality, ratio, kcal.

    Proportion quality = calorie-weighted fraction of basis groups outside the
    neighborhood IQR (same metric as MacroIQ's proportion verdict). LLM flags
    only push obviously-bad recipes after all normal ones.
    """
    summary = card.get("score_summary") or {}
    scores = card.get("display_scores") or {}
    weird_pen = 1 if (card.get("is_weird") or summary.get("is_weird")) else 0

    outside_cal = _as_float(summary.get("outside_iqr_calorie_frac"))
    if outside_cal is None:
        outside_cal = _as_float((scores.get("ratio_loss") or {}).get("outside_iqr_calorie_frac"))
    if outside_cal is None:
        # Derive from ingredients when polish hasn't stamped the field yet.
        ings = list(scores.get("ingredients") or [])
        if ings:
            try:
                outside_cal = float(_iqr_alignment_stats(ings).get("outside_iqr_calorie_frac") or 0.0)
            except Exception:
                outside_cal = None
    if outside_cal is None:
        # Unknown IQR → fall back to 1 − in-band count fraction.
        iqr_frac = _as_float(summary.get("iqr_in_band_frac"))
        if iqr_frac is None:
            iqr_frac, _, known = _iqr_in_band_fraction(scores)
            if known <= 0:
                outside_cal = 1.0
            else:
                outside_cal = 1.0 - float(iqr_frac)
        else:
            outside_cal = 1.0 - float(iqr_frac)

    ratio = _as_float(summary.get("ratio_loss"))
    if ratio is None:
        ratio = _as_float((scores.get("ratio_loss") or {}).get("value"))
    if ratio is None:
        ratio = 99.0

    kcal_pen = 0.0
    if kcal_target is not None and float(kcal_target) > 0:
        macros = summary.get("macros") or scores.get("macros") or {}
        cal = _as_float(macros.get("calories") if isinstance(macros, dict) else None)
        if cal is not None and cal > 0:
            kcal_pen = abs(float(cal) - float(kcal_target)) / float(kcal_target)
        else:
            kcal_pen = 2.0
    return (int(weird_pen), float(outside_cal), float(ratio), float(kcal_pen))


def recompute_recipe_at_grams(
    *,
    problem: dict[str, Any],
    ingredients: list[dict[str, Any]],
    grams: list[float],
    macro_targets: dict[str, Any] | None = None,
    score_history: list[dict[str, Any]] | None = None,
    baseline_ratio: float | None = None,
) -> dict[str, Any]:
    """Recompute macros, share bands, and cookability after a user gram edit."""
    from recipe_opt_agent.portion_display import (
        _format_qty,
        cookability_from_score_history,
    )
    from weighted_empirical_opt import pfc_fractions_from_portions

    problem = dict(problem or {})
    ings = [dict(r) for r in (ingredients or [])]
    n = min(len(ings), len(grams))
    x = np.asarray([float(grams[i]) for i in range(n)], dtype=float)
    for i in range(n):
        prev_g = _as_float(ings[i].get("grams"))
        if prev_g is not None:
            ings[i]["_prev_grams"] = float(prev_g)
        ings[i]["grams"] = float(x[i])

    M_raw = problem.get("M") or []
    try:
        M = np.asarray(M_raw, dtype=float)
    except (TypeError, ValueError):
        M = np.zeros((0, 0), dtype=float)

    pfc = None
    if M.ndim == 2 and M.shape[1] >= n and x.size == n and n > 0:
        try:
            p, c, f = pfc_fractions_from_portions(x, M[:, :n])
            pfc = {"protein": float(p), "carb": float(c), "fat": float(f)}
        except Exception:
            pfc = None

    opt = {"x_opt": x.tolist(), "pfc_after": pfc, "term_losses": {}}
    problem_local = {
        **problem,
        "x_opt": x.tolist(),
        "total_mass": float(x.sum()) if x.size else problem.get("total_mass"),
        "chosen_recipe": {
            **(problem.get("chosen_recipe") or {}),
            "ingredients": ings,
        },
    }
    display_rows = build_ingredient_display_rows(
        ingredients=ings,
        problem=problem_local,
        opt=opt,
        foodon_report=problem.get("foodon_basis_report"),
    )

    for i, row in enumerate(display_rows):
        src = ings[i] if i < len(ings) else {}
        portion = scaled_portion_amount(src, grams=row.get("grams"))
        if portion.get("amount_source") == "scaled_portion":
            row.update(portion)
            continue
        gw = _as_float(src.get("portion_gram_weight"))
        unit = src.get("amount_unit") or src.get("unit")
        if gw and gw > 0 and unit and row.get("grams") is not None:
            qty = float(row["grams"]) / float(gw)
            row["amount_value"] = qty
            row["amount_unit"] = unit
            row["amount_display"] = _format_qty(qty, str(unit))
            row["amount_source"] = src.get("amount_source") or "usda_scaled"
            row["portion_gram_weight"] = gw

    # Keep a stable 1:1 ingredient list for interactive edits (no merge/split).
    total_kcal = 0.0
    kcal_n = 0
    for row in display_rows:
        if row.get("calories") is not None:
            total_kcal += float(row["calories"])
            kcal_n += 1

    nutrient = _pfc_box_slack(pfc, macro_targets or {})
    nutrient_band = band_for_loss(
        nutrient, good_max=NUTRIENT_LOSS_BANDS["good_max"], warn_max=NUTRIENT_LOSS_BANDS["warn_max"]
    )

    basis = list(problem.get("ingredient_basis") or [])
    samples = problem.get("basis_samples") or {}
    ratio = mean_abs_dev_from_median_shares(
        x=x.tolist(),
        ingredient_basis=basis,
        basis_samples=samples if isinstance(samples, dict) else {},
    )
    ratio_band = band_for_loss(
        ratio, good_max=RATIO_LOSS_BANDS["good_max"], warn_max=RATIO_LOSS_BANDS["warn_max"]
    )
    cookability = cookability_from_score_history(
        list(score_history or []),
        final_ratio=ratio,
    )
    if not cookability.get("improved") and baseline_ratio is not None and ratio is not None:
        try:
            base = float(baseline_ratio)
            if base > 1e-12 and ratio < base - 1e-12:
                pct_i = int(round((base - ratio) / base * 100.0))
                if pct_i >= 1:
                    cookability = {
                        "improved": True,
                        "improved_pct": pct_i,
                        "summary": f"Improved cookability by {pct_i}%",
                        "first_ratio_loss": base,
                        "final_ratio_loss": ratio,
                    }
        except (TypeError, ValueError):
            pass

    typicality = proportion_typicality_from_ingredients(display_rows)
    display_rows = filter_display_ingredients(display_rows)

    return {
        "macros": {
            "protein": None if not pfc else round(float(pfc["protein"]) * 100),
            "carb": None if not pfc else round(float(pfc["carb"]) * 100),
            "fat": None if not pfc else round(float(pfc["fat"]) * 100),
            "calories": None if kcal_n == 0 else int(round(total_kcal)),
        },
        "pfc_after": pfc,
        "ratio_loss": {
            "value": ratio,
            "band": typicality.get("band") or ratio_band,
            "band_summary": typicality.get("summary")
            or RATIO_BAND_SUMMARY.get(ratio_band, RATIO_BAND_SUMMARY["unknown"]),
            "proportion_key": typicality.get("key"),
            "proportion_css": typicality.get("css"),
            "outside_iqr_frac": typicality.get("outside_iqr_frac"),
            "outside_iqr_pct": typicality.get("outside_iqr_pct"),
            "outside_iqr_calorie_frac": typicality.get("outside_iqr_calorie_frac"),
            "outside_iqr_calorie_pct": typicality.get("outside_iqr_calorie_pct"),
        },
        "nutrient_loss": {
            "value": nutrient,
            "band": nutrient_band,
            "band_summary": NUTRIENT_BAND_SUMMARY.get(nutrient_band, NUTRIENT_BAND_SUMMARY["unknown"]),
        },
        "cookability": cookability,
        "ingredients": display_rows,
    }
