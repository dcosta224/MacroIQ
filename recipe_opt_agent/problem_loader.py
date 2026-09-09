"""Helpers to load a problem for the recipe opt agent (fixture or live DB)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "recipe_opt" / "synthetic_problem.json"


def load_fixture_problem(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_FIXTURE
    data = json.loads(p.read_text())
    if "title" not in data:
        data["title"] = p.stem
    # Synthetic fixture: treat title as the "semantic" / selected recipe label.
    data.setdefault("taste_text", data.get("title") or p.stem)
    data.setdefault(
        "chosen_recipe",
        {
            "source": "fixture",
            "title": data.get("title"),
            "ingredients": [
                {"label": lab, "grams": float(g)}
                for lab, g in zip(data.get("ingredient_basis") or [], data.get("x0") or [])
            ],
            "selection_note": "Offline fixture; neighborhood is baked into the problem JSON.",
        },
    )
    data.setdefault("neighborhood_recipes", [])
    return data


def list_canonical_dishes(
    *,
    limit: int | None = None,
    min_neighborhood: int = 5,
    q: str | None = None,
) -> list[dict[str, Any]]:
    """Canonical dishes ordered by match count (n_neighborhood) descending.

    Uses local recipe store by default (``RECIPE_DATA_SOURCE=local``).
    """
    from recipe_data_access import data_source, get_store

    try:
        return get_store().list_canonical_dishes(
            min_neighborhood=min_neighborhood, limit=limit, q=q
        )
    except FileNotFoundError:
        if data_source() == "local":
            raise

    from db import connect

    multi_exclusion = """
  AND cm.recipe_nlg_id NOT IN (
      SELECT recipe_nlg_id
      FROM recipe.canonical_matches
      GROUP BY recipe_nlg_id
      HAVING COUNT(DISTINCT canonical_recipe_id) > 1
  )
"""
    title_filter = ""
    params: list[Any] = []
    if q and q.strip():
        title_filter = " AND cr.title ILIKE %s"
        params.append(f"%{q.strip()}%")
    params.append(min_neighborhood)

    limit_clause = ""
    if limit is not None:
        limit_clause = " LIMIT %s"
        params.append(int(limit))

    sql = f"""
SELECT cr.id AS canonical_recipe_id,
       cr.title,
       COUNT(DISTINCT cm.recipe_nlg_id) AS n_neighborhood
FROM recipe.canonical_recipes cr
JOIN recipe.canonical_matches cm ON cm.canonical_recipe_id = cr.id
JOIN recipe.resolved_recipes rr ON rr.recipe_id = cm.recipe_nlg_id
WHERE rr.fdc_id IS NOT NULL
  AND rr.gram_weight IS NOT NULL
  {multi_exclusion}
  {title_filter}
GROUP BY cr.id, cr.title
HAVING COUNT(DISTINCT cm.recipe_nlg_id) >= %s
ORDER BY n_neighborhood DESC, cr.title ASC, cr.id ASC
{limit_clause}
"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for row in rows:
        rec = dict(zip(cols, row))
        out.append(
            {
                "canonical_id": int(rec["canonical_recipe_id"]),
                "title": str(rec["title"]),
                "n_matches": int(rec["n_neighborhood"]),
            }
        )
    return out


def count_canonical_dishes(*, min_neighborhood: int = 5, q: str | None = None) -> int:
    """Count canonical dishes matching the same filters as list_canonical_dishes."""
    from recipe_data_access import data_source, get_store

    try:
        return get_store().count_canonical_dishes(min_neighborhood=min_neighborhood, q=q)
    except FileNotFoundError:
        if data_source() == "local":
            raise

    from db import connect

    multi_exclusion = """
  AND cm.recipe_nlg_id NOT IN (
      SELECT recipe_nlg_id
      FROM recipe.canonical_matches
      GROUP BY recipe_nlg_id
      HAVING COUNT(DISTINCT canonical_recipe_id) > 1
  )
"""
    title_filter = ""
    params: list[Any] = []
    if q and q.strip():
        title_filter = " AND cr.title ILIKE %s"
        params.append(f"%{q.strip()}%")
    params.append(min_neighborhood)

    sql = f"""
SELECT COUNT(*) FROM (
  SELECT cr.id
  FROM recipe.canonical_recipes cr
  JOIN recipe.canonical_matches cm ON cm.canonical_recipe_id = cr.id
  JOIN recipe.resolved_recipes rr ON rr.recipe_id = cm.recipe_nlg_id
  WHERE rr.fdc_id IS NOT NULL
    AND rr.gram_weight IS NOT NULL
    {multi_exclusion}
    {title_filter}
  GROUP BY cr.id, cr.title
  HAVING COUNT(DISTINCT cm.recipe_nlg_id) >= %s
) sub
"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
            return int(row[0]) if row else 0
    finally:
        conn.close()


def search_canonical_dishes(
    q: str,
    *,
    min_neighborhood: int = 5,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Search canonical recipes by title over the full catalog."""
    return list_canonical_dishes(
        limit=max(1, int(limit)),
        min_neighborhood=min_neighborhood,
        q=q,
    )


def _ingredient_rows(ingredients) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if ingredients is None or getattr(ingredients, "empty", True):
        return rows
    for row in ingredients.itertuples(index=False):
        leaf = getattr(row, "foodon_id", None) or getattr(row, "foodon_leaf_id", None)
        grams = float(getattr(row, "gram_weight", 0.0) or 0.0)
        qty = getattr(row, "quantity", None)
        unit = getattr(row, "unit", None)
        try:
            qty_f = float(qty) if qty is not None and str(qty).strip() != "" else None
        except (TypeError, ValueError):
            qty_f = None
        unit_s = str(unit).strip() if unit is not None and str(unit).strip() else None
        source_text = getattr(row, "ingredient", None) or getattr(row, "name", None)
        rows.append(
            {
                "ingredient_idx": int(getattr(row, "ingredient_idx", 0) or 0),
                "label": str(
                    getattr(row, "fdc_description", None)
                    or source_text
                    or ""
                ),
                "source_text": str(source_text).strip() if source_text else None,
                "fdc_id": int(row.fdc_id) if getattr(row, "fdc_id", None) is not None else None,
                "grams": grams,
                # Preserve resolved source portions so the product UI can show
                # cups/tbsp/etc. scaled after LP gram reweighting.
                "original_grams": grams,
                "quantity": qty_f,
                "unit": unit_s,
                "portion_label": (
                    str(getattr(row, "portion_label", "") or "").strip() or None
                ),
                "foodon_id": str(leaf) if leaf else None,
            }
        )
    return rows


def _pfc_distance_to_box(
    protein_g: float,
    fat_g: float,
    carbs_g: float,
    *,
    protein_min: float,
    protein_max: float,
    carb_min: float,
    carb_max: float,
    fat_min: float,
    fat_max: float,
) -> float:
    """0 if inside the target box; else L1 distance to the nearest point in the box (calorie fractions)."""
    from canonical_optimization import macro_calorie_fractions_from_grams

    p, c, f = macro_calorie_fractions_from_grams(protein_g, fat_g, carbs_g)
    p_t = min(max(p, protein_min), protein_max)
    c_t = min(max(c, carb_min), carb_max)
    f_t = min(max(f, fat_min), fat_max)
    # Renormalize target onto simplex for a fair distance
    s = p_t + c_t + f_t
    if s > 0:
        p_t, c_t, f_t = p_t / s, c_t / s, f_t / s
    return float(abs(p - p_t) + abs(c - c_t) + abs(f - f_t))


def _per_gram_macros_for_fdc_ids(fdc_ids: list[int]) -> dict[int, np.ndarray]:
    """One nutrient query → per-gram macro vector [protein_g, fat_g, carbs_g, kcal] per fdc_id."""
    from mvp_data import fetch_food_nutrients_for_recipe
    from recipe_macro_optimizer import NUTRIENT_IDS

    if not fdc_ids:
        return {}
    nutrients = fetch_food_nutrients_for_recipe(None, fdc_ids)
    pivot = (
        nutrients.pivot_table(index="fdc_id", columns="nutrient_id", values="amount", aggfunc="first")
        if not nutrients.empty
        else None
    )
    per_g: dict[int, np.ndarray] = {}
    for fid in fdc_ids:
        if pivot is None or fid not in pivot.index:
            per_g[fid] = np.zeros(4, dtype=float)
            continue
        row = pivot.loc[fid]
        # NUTRIENT_IDS amounts are per 100g → convert to per-gram.
        per_g[fid] = np.array(
            [float(row.get(nid, 0.0) or 0.0) / 100.0 for nid in NUTRIENT_IDS],
            dtype=float,
        )
    return per_g


def _batch_recipe_pfc_from_lines(lines_df) -> dict[str, dict[str, float]]:
    """One nutrient query → resolved PFC calorie fractions per recipe_nlg_id."""
    from canonical_optimization import macro_calorie_fractions_from_grams

    fdc_ids = sorted({int(x) for x in lines_df["fdc_id"].dropna().astype(int).tolist()})
    if not fdc_ids:
        return {}
    per_g = _per_gram_macros_for_fdc_ids(fdc_ids)

    out: dict[str, dict[str, float]] = {}
    for rid, sub in lines_df.groupby("recipe_nlg_id"):
        totals = np.zeros(4, dtype=float)
        for row in sub.itertuples(index=False):
            fid = int(getattr(row, "fdc_id"))
            grams = float(getattr(row, "gram_weight") or 0.0)
            totals += per_g.get(fid, np.zeros(4)) * grams
        protein_g, fat_g, carbs_g, energy = map(float, totals)
        if energy <= 0:
            energy = 4.0 * protein_g + 9.0 * fat_g + 4.0 * carbs_g
        if energy <= 0:
            continue
        p, c, f = macro_calorie_fractions_from_grams(protein_g, fat_g, carbs_g)
        out[str(rid)] = {
            "protein": float(p),
            "carbs": float(c),
            "fat": float(f),
            "energy_kcal": energy,
        }
    return out


def _box_midpoint(
    *,
    protein_min: float,
    protein_max: float,
    carb_min: float,
    carb_max: float,
    fat_min: float,
    fat_max: float,
) -> tuple[float, float, float]:
    p = 0.5 * (float(protein_min) + float(protein_max))
    c = 0.5 * (float(carb_min) + float(carb_max))
    f = 0.5 * (float(fat_min) + float(fat_max))
    s = p + c + f
    if s <= 1e-12:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return p / s, c / s, f / s


def _pfc_in_macro_box(
    p: float,
    c: float,
    f: float,
    *,
    protein_min: float,
    protein_max: float,
    carb_min: float,
    carb_max: float,
    fat_min: float,
    fat_max: float,
) -> bool:
    return (
        float(protein_min) - 1e-9 <= float(p) <= float(protein_max) + 1e-9
        and float(carb_min) - 1e-9 <= float(c) <= float(carb_max) + 1e-9
        and float(fat_min) - 1e-9 <= float(f) <= float(fat_max) + 1e-9
    )


def _l1_to_box_projection(
    p: float,
    c: float,
    f: float,
    *,
    protein_min: float,
    protein_max: float,
    carb_min: float,
    carb_max: float,
    fat_min: float,
    fat_max: float,
) -> float:
    """L1 distance from PFC to its projection onto the axis-aligned macro box.

    For points outside the box this is distance to the nearest face/edge/corner
    of the target (after a light simplex renormalization of the projection).
    """
    p_t = min(max(float(p), float(protein_min)), float(protein_max))
    c_t = min(max(float(c), float(carb_min)), float(carb_max))
    f_t = min(max(float(f), float(fat_min)), float(fat_max))
    s = p_t + c_t + f_t
    if s > 0:
        p_t, c_t, f_t = p_t / s, c_t / s, f_t / s
    return float(abs(float(p) - p_t) + abs(float(c) - c_t) + abs(float(f) - f_t))


def _l1_to_midpoint(
    p: float,
    c: float,
    f: float,
    *,
    protein_min: float,
    protein_max: float,
    carb_min: float,
    carb_max: float,
    fat_min: float,
    fat_max: float,
) -> float:
    mp, mc, mf = _box_midpoint(
        protein_min=protein_min,
        protein_max=protein_max,
        carb_min=carb_min,
        carb_max=carb_max,
        fat_min=fat_min,
        fat_max=fat_max,
    )
    return float(abs(float(p) - mp) + abs(float(c) - mc) + abs(float(f) - mf))


def pick_neighborhood_recipe_for_macro_box(
    pfc_by_rid: dict[str, dict[str, float]],
    recipe_ids: list[Any],
    *,
    protein_min: float,
    protein_max: float,
    carb_min: float,
    carb_max: float,
    fat_min: float,
    fat_max: float,
    default_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Pick a human neighborhood recipe as the nutrition reference for the target box.

    - If ≥1 resolved recipe sits inside the box → closest (L1) to the box midpoint.
    - Else → closest (L1) to the nearest point on the box boundary (edge/face).
    """
    mid = _box_midpoint(
        protein_min=protein_min,
        protein_max=protein_max,
        carb_min=carb_min,
        carb_max=carb_max,
        fat_min=fat_min,
        fat_max=fat_max,
    )
    in_box: list[dict[str, Any]] = []
    out_box: list[dict[str, Any]] = []
    ranked: list[dict[str, Any]] = []

    for rid in recipe_ids:
        rid_s = str(rid)
        pfc = pfc_by_rid.get(rid_s)
        if not pfc:
            ranked.append({"recipe_nlg_id": rid_s, "error": "no_pfc"})
            continue
        p, c, f = float(pfc["protein"]), float(pfc["carbs"]), float(pfc["fat"])
        inside = _pfc_in_macro_box(
            p,
            c,
            f,
            protein_min=protein_min,
            protein_max=protein_max,
            carb_min=carb_min,
            carb_max=carb_max,
            fat_min=fat_min,
            fat_max=fat_max,
        )
        if inside:
            dist = _l1_to_midpoint(
                p,
                c,
                f,
                protein_min=protein_min,
                protein_max=protein_max,
                carb_min=carb_min,
                carb_max=carb_max,
                fat_min=fat_min,
                fat_max=fat_max,
            )
            row = {
                "recipe_nlg_id": rid_s,
                "pfc": pfc,
                "in_box": True,
                "distance_to_midpoint": dist,
                "distance_to_target_box": 0.0,
            }
            in_box.append(row)
        else:
            dist = _l1_to_box_projection(
                p,
                c,
                f,
                protein_min=protein_min,
                protein_max=protein_max,
                carb_min=carb_min,
                carb_max=carb_max,
                fat_min=fat_min,
                fat_max=fat_max,
            )
            row = {
                "recipe_nlg_id": rid_s,
                "pfc": pfc,
                "in_box": False,
                "distance_to_midpoint": _l1_to_midpoint(
                    p,
                    c,
                    f,
                    protein_min=protein_min,
                    protein_max=protein_max,
                    carb_min=carb_min,
                    carb_max=carb_max,
                    fat_min=fat_min,
                    fat_max=fat_max,
                ),
                "distance_to_target_box": dist,
            }
            out_box.append(row)
        ranked.append(row)

    if in_box:
        pool = in_box
        mode = "closest_to_midpoint_in_box"
        key = "distance_to_midpoint"
    elif out_box:
        pool = out_box
        mode = "closest_to_box_edge"
        key = "distance_to_target_box"
    else:
        fallback = str(default_id or (recipe_ids[0] if recipe_ids else ""))
        return fallback, {
            "method": "l1_pfc_to_target_box",
            "selection_mode": "fallback_empty",
            "chosen_recipe_nlg_id": fallback,
            "distance_to_target_box": None,
            "distance_to_midpoint": None,
            "chosen_pfc": None,
            "n_in_box": 0,
            "target_midpoint": {"protein": mid[0], "carbs": mid[1], "fat": mid[2]},
            "scored": ranked[:20],
            "n_scored": 0,
        }

    pool.sort(key=lambda r: (float(r[key]), str(r["recipe_nlg_id"])))
    best = pool[0]
    best_id = str(best["recipe_nlg_id"])
    ranked_sorted = sorted(
        [r for r in ranked if key in r],
        key=lambda r: (float(r.get(key, 99.0)), str(r["recipe_nlg_id"])),
    )
    return best_id, {
        "method": "l1_pfc_to_target_box",
        "selection_mode": mode,
        "chosen_recipe_nlg_id": best_id,
        "distance_to_target_box": best.get("distance_to_target_box"),
        "distance_to_midpoint": best.get("distance_to_midpoint"),
        "chosen_pfc": best.get("pfc"),
        "n_in_box": len(in_box),
        "n_out_box": len(out_box),
        "target_midpoint": {"protein": mid[0], "carbs": mid[1], "fat": mid[2]},
        "scored": ranked_sorted[:20],
        "n_scored": len(ranked_sorted),
    }


def _pick_start_l1_pfc(
    nb,
    *,
    protein_min: float,
    protein_max: float,
    carb_min: float,
    carb_max: float,
    fat_min: float,
    fat_max: float,
) -> tuple[str, dict[str, Any]]:
    """Pick neighborhood recipe closest to the target box midpoint (in-box) or edge."""
    pfc_by_rid = _batch_recipe_pfc_from_lines(nb.lines_df)
    best_id, meta = pick_neighborhood_recipe_for_macro_box(
        pfc_by_rid,
        list(nb.recipe_ids or []),
        protein_min=protein_min,
        protein_max=protein_max,
        carb_min=carb_min,
        carb_max=carb_max,
        fat_min=fat_min,
        fat_max=fat_max,
        default_id=str(nb.starting_recipe_id),
    )
    meta["default_build_start"] = str(nb.starting_recipe_id)
    meta["switched_from_default"] = best_id != str(nb.starting_recipe_id)
    return best_id, meta


def _loss_projection_in_box(
    points_p: list[float],
    points_c: list[float],
    points_f: list[float],
    hard_loss: list[float | None],
    hard_feasible: list[bool],
    *,
    protein_min: float,
    protein_max: float,
    carb_min: float,
    carb_max: float,
    fat_min: float,
    fat_max: float,
) -> float | None:
    """Min hard_loss among lattice points inside the target box (feasible only)."""
    best: float | None = None
    for p, c, f, loss, ok in zip(points_p, points_c, points_f, hard_loss, hard_feasible):
        if not ok or loss is None:
            continue
        try:
            lv = float(loss)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(lv):
            continue
        if not (protein_min <= float(p) <= protein_max):
            continue
        if not (carb_min <= float(c) <= carb_max):
            continue
        if not (fat_min <= float(f) <= fat_max):
            continue
        if best is None or lv < best:
            best = lv
    return best


def _pick_start_loss_projection(
    nb,
    *,
    protein_min: float,
    protein_max: float,
    carb_min: float,
    carb_max: float,
    fat_min: float,
    fat_max: float,
) -> tuple[str, dict[str, Any]]:
    """Pick start by min precomputed loss projection inside the target box.

    Reads ``recipe.recipe_loss_fields`` (local store by default). Falls back to L1
    if no rows are available.
    """
    rows: list[tuple[Any, ...]] = []
    cols: list[str] = []
    try:
        from recipe_data_access import get_store

        df = get_store().recipe_loss_fields(canonical_id=int(nb.canonical_recipe_id))
        want = [
            "recipe_nlg_id",
            "points_p",
            "points_c",
            "points_f",
            "hard_loss",
            "hard_feasible",
            "protein_frac",
            "carb_frac",
            "fat_frac",
            "grid_n",
        ]
        cols = [c for c in want if c in df.columns]
        if cols and not df.empty:
            rows = [tuple(r) for r in df[cols].itertuples(index=False, name=None)]
    except Exception as exc:
        chosen, meta = _pick_start_l1_pfc(
            nb,
            protein_min=protein_min,
            protein_max=protein_max,
            carb_min=carb_min,
            carb_max=carb_max,
            fat_min=fat_min,
            fat_max=fat_max,
        )
        meta = {
            **meta,
            "method": "l1_pfc_to_target_box",
            "requested_method": "loss_projection",
            "fallback_reason": f"loss_fields_query_failed: {exc}",
        }
        return chosen, meta

    if not rows:
        chosen, meta = _pick_start_l1_pfc(
            nb,
            protein_min=protein_min,
            protein_max=protein_max,
            carb_min=carb_min,
            carb_max=carb_max,
            fat_min=fat_min,
            fat_max=fat_max,
        )
        meta = {
            **meta,
            "method": "l1_pfc_to_target_box",
            "requested_method": "loss_projection",
            "fallback_reason": "no_rows_in_recipe_loss_fields",
        }
        return chosen, meta

    # Prefer densest grid when multiple configs exist per recipe.
    by_rid: dict[str, dict[str, Any]] = {}
    for row in rows:
        rec = dict(zip(cols, row))
        rid = str(rec["recipe_nlg_id"])
        prev = by_rid.get(rid)
        if prev is None or int(rec.get("grid_n") or 0) >= int(prev.get("grid_n") or 0):
            by_rid[rid] = rec

    ranked: list[dict[str, Any]] = []
    best_id = str(nb.starting_recipe_id)
    best_loss = float("inf")
    for rid, rec in by_rid.items():
        proj = _loss_projection_in_box(
            list(rec["points_p"] or []),
            list(rec["points_c"] or []),
            list(rec["points_f"] or []),
            list(rec["hard_loss"] or []),
            list(rec["hard_feasible"] or []),
            protein_min=protein_min,
            protein_max=protein_max,
            carb_min=carb_min,
            carb_max=carb_max,
            fat_min=fat_min,
            fat_max=fat_max,
        )
        entry = {
            "recipe_nlg_id": rid,
            "loss_projection": proj,
            "chosen_pfc": {
                "protein": float(rec["protein_frac"]) if rec.get("protein_frac") is not None else None,
                "carbs": float(rec["carb_frac"]) if rec.get("carb_frac") is not None else None,
                "fat": float(rec["fat_frac"]) if rec.get("fat_frac") is not None else None,
            },
            "grid_n": rec.get("grid_n"),
        }
        ranked.append(entry)
        if proj is not None and (
            proj < best_loss - 1e-12 or (abs(proj - best_loss) <= 1e-12 and rid < best_id)
        ):
            best_loss = proj
            best_id = rid

    in_box = [r for r in ranked if r.get("loss_projection") is not None]
    if not in_box:
        chosen, meta = _pick_start_l1_pfc(
            nb,
            protein_min=protein_min,
            protein_max=protein_max,
            carb_min=carb_min,
            carb_max=carb_max,
            fat_min=fat_min,
            fat_max=fat_max,
        )
        meta = {
            **meta,
            "method": "l1_pfc_to_target_box",
            "requested_method": "loss_projection",
            "fallback_reason": "no_feasible_lattice_points_in_target_box",
            "n_loss_field_recipes": len(ranked),
        }
        return chosen, meta

    ranked.sort(key=lambda r: (r.get("loss_projection") is None, r.get("loss_projection", 1e9)))
    return best_id, {
        "method": "loss_projection_in_target_box",
        "chosen_recipe_nlg_id": best_id,
        "loss_projection": best_loss if best_loss < float("inf") else None,
        "scored": ranked[:20],
        "n_scored": len(ranked),
        "n_in_box": len(in_box),
        "default_build_start": str(nb.starting_recipe_id),
        "switched_from_default": best_id != str(nb.starting_recipe_id),
    }


def _pick_nutrition_start(
    nb,
    *,
    protein_min: float,
    protein_max: float,
    carb_min: float,
    carb_max: float,
    fat_min: float,
    fat_max: float,
    start_metric: str = "l1_pfc",
    max_candidates: int = 12,
) -> tuple[str, dict[str, Any]]:
    """Pick starting NLG recipe by ``start_metric``: ``l1_pfc`` or ``loss_projection``."""
    metric = (start_metric or "l1_pfc").strip().lower()
    if metric in {"loss_projection", "loss", "projection", "loss_field"}:
        return _pick_start_loss_projection(
            nb,
            protein_min=protein_min,
            protein_max=protein_max,
            carb_min=carb_min,
            carb_max=carb_max,
            fat_min=fat_min,
            fat_max=fat_max,
        )
    # Default / legacy: fast batch L1 (replaces per-recipe load_recipe_macro_problem loop).
    return _pick_start_l1_pfc(
        nb,
        protein_min=protein_min,
        protein_max=protein_max,
        carb_min=carb_min,
        carb_max=carb_max,
        fat_min=fat_min,
        fat_max=fat_max,
    )


def _rebuild_start(nb, recipe_nlg_id: str) -> None:
    """Point neighborhood start at a specific NLG recipe (mutates nb)."""
    from canonical_optimization import load_recipe_macro_problem

    index = None
    try:
        from canonical_optimization import _get_index

        index = _get_index()
    except Exception:
        pass

    ingredients, x0, M = load_recipe_macro_problem(recipe_nlg_id)
    leaf_by_fdc_idx: dict[tuple[int, int], str] = {}
    for _, row in nb.lines_df.loc[nb.lines_df["recipe_nlg_id"] == recipe_nlg_id].iterrows():
        leaf_by_fdc_idx[(int(row["ingredient_idx"]), int(row["fdc_id"]))] = str(row["foodon_id"])

    def rollup_to_active(leaf_id: str | None) -> str | None:
        if not leaf_id:
            return None
        for node_id in nb.rollup_chains.get(str(leaf_id), (str(leaf_id),)):
            if node_id in nb.basis_nodes:
                return node_id
        return None

    ingredient_basis: list[str | None] = []
    for row in ingredients.itertuples(index=False):
        leaf = leaf_by_fdc_idx.get((int(row.ingredient_idx), int(row.fdc_id)))
        ingredient_basis.append(rollup_to_active(leaf))

    nb.starting_recipe_id = str(recipe_nlg_id)
    nb.starting_ingredients = ingredients
    nb.x0 = x0
    nb.M = M
    nb.ingredient_basis = ingredient_basis
    recipe_basis_nodes = {nid for nid in ingredient_basis if nid is not None}
    if index is not None:
        nb.basis_index = sorted(
            recipe_basis_nodes, key=lambda nid: index.labels.get(nid, nid).lower()
        )
    else:
        nb.basis_index = sorted(recipe_basis_nodes)
    nb.basis_samples = {
        nid: nb.basis_share_df.loc[nb.basis_share_df["basis_node_id"] == nid, "share"].to_numpy(
            dtype=float
        )
        for nid in nb.basis_index
    }


def _build_modification_candidates_from_problem(
    problem: dict[str, Any],
    *,
    box_dict: dict[str, float],
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Live retrieval shortlist from neighborhood context stored on the problem."""
    from augmentation_retrieve import rank_add_candidates
    from hull_geometry import TargetBox

    ctx = problem.get("retrieval_context") or {}
    starting_ingredients = list(ctx.get("starting_ingredients") or [])
    starting_fdc = {int(x) for x in (ctx.get("starting_fdc") or []) if x is not None}
    starting_labels = {str(x).lower() for x in (ctx.get("starting_labels") or []) if x}
    neighbor_sets = [set(s) for s in (ctx.get("neighbor_label_sets") or [])]
    catalog = list(ctx.get("fdc_catalog") or [])

    pool: list[dict[str, Any]] = []
    for row in catalog:
        fdc_id = row.get("fdc_id")
        label = str(row.get("fdc_description") or fdc_id or "")
        if fdc_id is not None and int(fdc_id) in starting_fdc:
            continue
        if label.lower() in starting_labels:
            continue
        pool.append(
            {
                "id": f"add_fdc_{fdc_id}",
                "label": label,
                "fdc_id": fdc_id,
                "in_basis": False,
                "pfc": None,
            }
        )

    box = TargetBox(**box_dict)
    kcal_target = float(problem.get("kcal_target") or 500.0)
    M = np.asarray(problem.get("M") or [], dtype=float)
    if M.ndim != 2 or M.size == 0:
        M = np.zeros((4, 1), dtype=float)

    ranked = rank_add_candidates(
        pool=pool[:80],
        core_labels=starting_labels,
        neighbor_sets=neighbor_sets,
        M=M,
        box=box,
        kcal_target=kcal_target,
        p_B_star=None,
        p_T_star=(
            0.5 * (box.protein_min + box.protein_max),
            0.5 * (box.carb_min + box.carb_max),
            0.5 * (box.fat_min + box.fat_max),
        ),
        top_k_cooc=top_k * 2,
        top_k_lp=top_k,
        evaluate_fn=None,
    )
    out = [m.to_dict() for m in ranked]
    for row in starting_ingredients:
        out.append(
            {
                "candidate_id": f"remove_{row.get('ingredient_idx')}_{row.get('fdc_id')}",
                "action": "remove",
                "label": row.get("label"),
                "fdc_id": row.get("fdc_id"),
                "cooccurrence": 0.0,
                "geom_score": 0.0,
                "L_star": None,
                "meta": {"grams": row.get("grams")},
            }
        )
    return out[: top_k + len(starting_ingredients)]


def _candidate_pool_from_context(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Catalog rows → slot-retrieval pool items with macros/pfc/basis when available."""
    starting_fdc = {int(x) for x in (ctx.get("starting_fdc") or []) if x is not None}
    starting_labels = {str(x).lower() for x in (ctx.get("starting_labels") or []) if x}
    fdc_macros = {str(k): v for k, v in (ctx.get("fdc_macros") or {}).items()}
    fdc_basis = {str(k): v for k, v in (ctx.get("fdc_basis") or {}).items()}

    pool: list[dict[str, Any]] = []
    for row in ctx.get("fdc_catalog") or []:
        fdc_id = row.get("fdc_id")
        label = str(row.get("fdc_description") or fdc_id or "")
        if fdc_id is not None and int(fdc_id) in starting_fdc:
            continue
        if label.lower() in starting_labels:
            continue
        macros = fdc_macros.get(str(fdc_id))
        pfc = None
        if macros is not None:
            protein_g, fat_g, carbs_g = float(macros[0]), float(macros[1]), float(macros[2])
            pk, ck, fk = 4.0 * protein_g, 4.0 * carbs_g, 9.0 * fat_g
            total = pk + ck + fk
            if total > 0:
                pfc = [pk / total, ck / total, fk / total]
        pool.append(
            {
                "id": f"fdc_{fdc_id}",
                "label": label,
                "fdc_id": fdc_id,
                "in_basis": False,
                "pfc": pfc,
                "macros_per_g": macros,
                "basis_node": fdc_basis.get(str(fdc_id)),
            }
        )
    return pool


def _build_slot_candidates_from_problem(
    problem: dict[str, Any],
    slots: list[dict[str, Any]],
    *,
    box_dict: dict[str, float],
    top_k_per_slot: int = 6,
) -> dict[str, list[dict[str, Any]]]:
    """Per-slot candidate shortlists (proxy cards attached, no LP)."""
    from augmentation_retrieve import retrieve_for_slot
    from hull_geometry import TargetBox
    from weighted_empirical_opt import pfc_fractions_from_portions

    ctx = problem.get("retrieval_context") or {}
    starting_ingredients = list(ctx.get("starting_ingredients") or [])
    starting_labels = {str(x).lower() for x in (ctx.get("starting_labels") or []) if x}
    neighbor_sets = [set(map(str.lower, map(str, s))) for s in (ctx.get("neighbor_label_sets") or [])]
    pool = _candidate_pool_from_context(ctx)

    box = TargetBox(**box_dict)
    x0 = np.asarray(problem.get("x0") or [], dtype=float)
    M = np.asarray(problem.get("M") or [], dtype=float)
    if M.ndim != 2 or M.size == 0:
        M = np.zeros((4, max(1, x0.size)), dtype=float)
    kcal_target = float(problem.get("kcal_target") or 500.0)
    total_mass = float(problem.get("total_mass") or (x0.sum() if x0.size else 0.0))
    ingredient_basis = list(problem.get("ingredient_basis") or [None] * x0.size)
    basis_samples = {
        k: np.asarray(v, dtype=float) for k, v in (problem.get("basis_samples") or {}).items()
    }
    p_current = None
    try:
        if x0.size and M.shape[1] == x0.size:
            p_current = pfc_fractions_from_portions(x0, M)
    except Exception:
        p_current = None

    out: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        cands = retrieve_for_slot(
            slot,
            pool=pool[:120],
            core_labels=starting_labels,
            neighbor_sets=neighbor_sets,
            current_ingredients=starting_ingredients,
            x0=x0,
            M=M,
            box=box,
            kcal_target=kcal_target,
            total_mass=total_mass,
            ingredient_basis=ingredient_basis,
            basis_samples=basis_samples,
            p_current=p_current,
            top_k=top_k_per_slot,
        )
        out[str(slot.get("slot_id"))] = [c.to_dict() for c in cands]
    return out


def _build_modification_candidates(nb, *, box_dict: dict[str, float], top_k: int = 8) -> list[dict[str, Any]]:
    """Neighborhood ingredient adds not already in the starting recipe (from live CanonicalNeighborhood)."""
    ingredients = _ingredient_rows(nb.starting_ingredients)
    starting_fdc = {int(r["fdc_id"]) for r in ingredients if r.get("fdc_id") is not None}
    starting_labels = {str(r["label"]).lower() for r in ingredients if r.get("label")}
    neighbor_sets: list[set[str]] = []
    for _, sub in nb.lines_df.groupby("recipe_nlg_id"):
        neighbor_sets.append({str(x).lower() for x in sub["fdc_description"].dropna().tolist()})
    catalog = []
    if nb.fdc_catalog is not None and not nb.fdc_catalog.empty:
        for row in nb.fdc_catalog.itertuples(index=False):
            catalog.append({"fdc_id": int(row.fdc_id), "fdc_description": str(row.fdc_description or "")})
    problem_stub = {
        "M": nb.M.tolist(),
        "kcal_target": float(
            __import__("canonical_optimization", fromlist=["atwater_kcal"]).atwater_kcal(nb.x0, nb.M)
        ),
        "retrieval_context": {
            "starting_ingredients": ingredients,
            "starting_fdc": list(starting_fdc),
            "starting_labels": list(starting_labels),
            "neighbor_label_sets": [list(s) for s in neighbor_sets],
            "fdc_catalog": catalog,
        },
    }
    return _build_modification_candidates_from_problem(problem_stub, box_dict=box_dict, top_k=top_k)


def load_canonical_problem(
    canonical_id: int,
    *,
    protein_min: float = 0.19,
    protein_max: float = 0.23,
    carb_min: float = 0.345,
    carb_max: float = 0.545,
    fat_min: float = 0.245,
    fat_max: float = 0.445,
    prefer_nutrition_start: bool = True,
    start_metric: str = "l1_pfc",
    fast_neighborhood: bool = True,
    max_foodon_aggregation_levels: int | None = None,
    require_cache: bool = False,
) -> dict[str, Any]:
    from canonical_optimization import CanonicalNeighborhood
    from weighted_empirical_opt import (
        EGG_YOLK_BASIS_NODE,
        MARGINAL_COLUMN_NODES,
        SPAGHETTI_BASIS_NODE,
        atwater_kcal,
        neighborhood_ratio_samples,
    )

    from recipe_opt_agent.config import save_neighborhood_cache_from_env

    nb = CanonicalNeighborhood.build(
        canonical_id,
        fast=fast_neighborhood,
        use_cache=True,
        save_cache=save_neighborhood_cache_from_env(),
        require_cache=require_cache,
    )
    selection_meta: dict[str, Any] = {
        "method": "default_foodon_hit_start",
        "chosen_recipe_nlg_id": str(nb.starting_recipe_id),
        "switched_from_default": False,
        "start_metric": start_metric,
        "fast_neighborhood": fast_neighborhood,
        "neighborhood_from_cache": bool(getattr(nb, "from_cache", False)),
    }
    if prefer_nutrition_start:
        chosen_id, selection_meta = _pick_nutrition_start(
            nb,
            protein_min=protein_min,
            protein_max=protein_max,
            carb_min=carb_min,
            carb_max=carb_max,
            fat_min=fat_min,
            fat_max=fat_max,
            start_metric=start_metric,
        )
        selection_meta["start_metric"] = start_metric
        selection_meta["fast_neighborhood"] = fast_neighborhood
        selection_meta["neighborhood_from_cache"] = bool(getattr(nb, "from_cache", False))
        if chosen_id != str(nb.starting_recipe_id):
            _rebuild_start(nb, chosen_id)

    ratio_samples: list[float] = []
    if nb.basis_share_df is not None and not nb.basis_share_df.empty:
        ratio_samples = neighborhood_ratio_samples(
            nb.basis_share_df, SPAGHETTI_BASIS_NODE, EGG_YOLK_BASIS_NODE
        ).tolist()
    basis_samples = {k: list(map(float, v)) for k, v in nb.basis_samples.items()}
    basis_sample_weights = {
        k: list(map(float, v)) for k, v in (getattr(nb, "basis_sample_weights", {}) or {}).items()
    }

    ingredients = _ingredient_rows(nb.starting_ingredients)
    neighborhood_recipes = [
        {
            "recipe_nlg_id": str(rid),
            "n_lines": int((nb.lines_df["recipe_nlg_id"] == rid).sum()),
            "is_start": str(rid) == str(nb.starting_recipe_id),
        }
        for rid in nb.recipe_ids
    ]

    box_dict = {
        "protein_min": protein_min,
        "protein_max": protein_max,
        "carb_min": carb_min,
        "carb_max": carb_max,
        "fat_min": fat_min,
        "fat_max": fat_max,
    }
    # Do NOT precompute modification_candidates here — propose retrieves them live.
    starting_fdc = [int(r["fdc_id"]) for r in ingredients if r.get("fdc_id") is not None]
    starting_labels = [str(r["label"]).lower() for r in ingredients if r.get("label")]
    neighbor_sets: list[list[str]] = []
    for _, sub in nb.lines_df.groupby("recipe_nlg_id"):
        neighbor_sets.append([str(x).lower() for x in sub["fdc_description"].dropna().tolist()])
    catalog = []
    if nb.fdc_catalog is not None and not nb.fdc_catalog.empty:
        for row in nb.fdc_catalog.itertuples(index=False):
            catalog.append({"fdc_id": int(row.fdc_id), "fdc_description": str(row.fdc_description or "")})

    # Per-fdc macro columns + basis-node rollups for slot/bundle scoring (best effort).
    fdc_macros: dict[str, list[float]] = {}
    fdc_basis: dict[str, str] = {}
    try:
        cat_fdc_ids = sorted({int(c["fdc_id"]) for c in catalog})
        per_g = _per_gram_macros_for_fdc_ids(cat_fdc_ids)
        fdc_macros = {str(fid): [float(v) for v in vec] for fid, vec in per_g.items()}
    except Exception:
        fdc_macros = {}
    try:
        rollup = getattr(nb, "rollup_chains", {}) or {}
        active = set(getattr(nb, "basis_nodes", set()) or set())
        leaf_by_fdc: dict[int, str] = {}
        for row in nb.lines_df.itertuples(index=False):
            fid = getattr(row, "fdc_id", None)
            leaf = getattr(row, "foodon_id", None)
            if fid is None or leaf is None:
                continue
            leaf_by_fdc.setdefault(int(fid), str(leaf))
        for fid, leaf in leaf_by_fdc.items():
            for node_id in rollup.get(leaf, (leaf,)):
                if node_id in active:
                    fdc_basis[str(fid)] = str(node_id)
                    break
    except Exception:
        fdc_basis = {}

    title = str(nb.title)
    from_cache = bool(getattr(nb, "from_cache", False))
    if from_cache:
        build_note = (
            "Neighborhood geometry loaded from recipe.canonical_neighborhood_cache "
            "(precomputed full Jaccard)."
        )
    else:
        build_note = (
            "Neighborhood built live with fast basis (Jaccard cache miss; "
            "skips combinatorial antichain search)."
        )
    from recipe_opt_agent.foodon_basis_report import (
        attach_foodon_basis_report,
        foodon_geometry_from_neighborhood,
    )

    foodon_geom = foodon_geometry_from_neighborhood(nb)
    leaves = list(foodon_geom.get("ingredient_foodon_leaves") or [])
    for i, row in enumerate(ingredients):
        leaf = leaves[i] if i < len(leaves) else None
        if leaf and not row.get("foodon_id"):
            row["foodon_id"] = leaf
    problem = {
        "x0": nb.x0.tolist(),
        "M": nb.M.tolist(),
        "ingredient_basis": list(nb.ingredient_basis),
        "basis_samples": basis_samples,
        "basis_sample_weights": basis_sample_weights,
        "ratio_samples": ratio_samples,
        "marginal_nodes": [nid for _, nid in MARGINAL_COLUMN_NODES],
        "kcal_target": float(atwater_kcal(nb.x0, nb.M)),
        "total_mass": float(nb.x0.sum()),
        "title": title,
        "shell_recipe_ids": [str(x) for x in (getattr(nb, "shell_recipe_ids", []) or [])],
        "expansion_meta": dict(getattr(nb, "expansion_meta", {}) or {}),
        # Dropdown selection stands in for semantic / taste input.
        "taste_text": title,
        "canonical_id": int(canonical_id),
        "n_matches": int(nb.n_recipes),
        "neighborhood_from_cache": from_cache,
        **foodon_geom,
        "chosen_recipe": {
            "source": "canonical",
            "canonical_id": int(canonical_id),
            "title": title,
            "recipe_nlg_id": str(nb.starting_recipe_id),
            "ingredients": ingredients,
            "selection": selection_meta,
            "selection_note": (
                "Canonical dish chosen from match-ranked search (defines the FoodOn neighborhood). "
                "Starting NLG recipe is the neighborhood recipe closest to the macro target: "
                "midpoint if any recipe is in-box, otherwise nearest box edge "
                f"(start_metric={start_metric}). "
                + build_note
            ),
        },
        "neighborhood_recipes": neighborhood_recipes,
        "modification_candidates": [],
        "retrieval_context": {
            "starting_ingredients": ingredients,
            "starting_fdc": starting_fdc,
            "starting_labels": starting_labels,
            "neighbor_label_sets": neighbor_sets,
            "fdc_catalog": catalog,
            "fdc_macros": fdc_macros,
            "fdc_basis": fdc_basis,
            "target_box": box_dict,
            "rollup_chains": foodon_geom.get("rollup_chains"),
            "basis_nodes": foodon_geom.get("basis_nodes"),
        },
    }
    from recipe_opt_agent.foodon_depth import apply_foodon_aggregation_cap

    problem = apply_foodon_aggregation_cap(
        problem,
        max_levels=max_foodon_aggregation_levels,
    )
    attach_foodon_basis_report(problem)
    try:
        from recipe_opt_agent.neighborhood_hull_context import build_neighborhood_hull_context

        problem["neighborhood_hull_context"] = build_neighborhood_hull_context(
            nb.lines_df,
            [str(r) for r in nb.recipe_ids],
            box_dict,
            shell_recipe_ids=[str(x) for x in (getattr(nb, "shell_recipe_ids", []) or [])],
            starting_recipe_id=str(nb.starting_recipe_id),
        )
    except Exception as exc:
        problem["neighborhood_hull_context"] = {"error": str(exc), "n_recipes": 0}
    try:
        from recipe_opt_agent.macro_target_suggestions import _neighborhood_mean_pfc

        mean_pfc = _neighborhood_mean_pfc(nb.lines_df)
        if mean_pfc is not None:
            problem["neighborhood_mean_pfc"] = {
                "protein": float(mean_pfc[0]),
                "carbs": float(mean_pfc[1]),
                "fat": float(mean_pfc[2]),
            }
    except Exception:
        pass
    try:
        from recipe_opt_agent.example_recipe import (
            attach_example_recipe_to_problem,
            example_from_problem_start,
        )

        example = example_from_problem_start(problem)
        if example:
            problem = attach_example_recipe_to_problem(problem, example)
    except Exception:
        pass
    return problem
