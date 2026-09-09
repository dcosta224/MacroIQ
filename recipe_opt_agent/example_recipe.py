"""Pick a neighborhood example recipe near the user's macro target.

The reference recipe for LLM drafting is the human neighborhood recipe whose
resolved PFC is closest to:
  - the box midpoint, when at least one neighborhood recipe falls inside the box
  - the nearest box edge/face, when the target sits outside the neighborhood cloud

Optional light semantic scoring is only a tie-break among equal nutrition scores.
"""

from __future__ import annotations

import re
from typing import Any


def _tokenize(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 2}


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb) / len(ta | tb))


def _ingredient_rows_for_recipe(lines_df, recipe_id: str) -> list[dict[str, Any]]:
    sub = lines_df.loc[lines_df["recipe_nlg_id"].astype(str) == str(recipe_id)]
    if sub.empty:
        return []
    rows: list[dict[str, Any]] = []
    for row in sub.itertuples(index=False):
        label = str(
            getattr(row, "fdc_description", None)
            or getattr(row, "ingredient_text", None)
            or getattr(row, "name", None)
            or "?"
        )
        grams = float(getattr(row, "gram_weight", 0.0) or 0.0)
        if grams <= 0:
            continue
        rows.append(
            {
                "label": label,
                "grams": round(grams, 1),
                "fdc_id": int(getattr(row, "fdc_id")) if getattr(row, "fdc_id", None) is not None else None,
                "role": None,
            }
        )
    return rows


def pick_example_recipe_near_targets(
    *,
    lines_df,
    recipe_ids: list[str],
    query: str,
    target_mid: dict[str, float] | None = None,
    target_box: dict[str, float] | None = None,
    titles_by_id: dict[str, str] | None = None,
    semantic_weight: float = 0.0,
    nutrition_weight: float = 1.0,
    top_k_semantic: int = 12,
) -> dict[str, Any] | None:
    """Return the best example recipe dict, or None.

    Nutrition policy (primary):
      - if any recipe is inside ``target_box`` → closest L1 to the box midpoint
      - else → closest L1 to the box edge/face

    ``target_mid`` alone builds a tiny ±1pp box when ``target_box`` is omitted.
    ``query`` / semantic weights are optional tie-breaks only.
    """
    from recipe_opt_agent.problem_loader import (
        _batch_recipe_pfc_from_lines,
        pick_neighborhood_recipe_for_macro_box,
    )

    if lines_df is None or getattr(lines_df, "empty", True) or not recipe_ids:
        return None
    pfc_by_rid = _batch_recipe_pfc_from_lines(lines_df)
    if not pfc_by_rid:
        return None

    box = dict(target_box or {})
    if not box and target_mid:
        pad = 0.01
        box = {
            "protein_min": max(0.0, float(target_mid.get("protein", 0.0)) - pad),
            "protein_max": min(1.0, float(target_mid.get("protein", 0.0)) + pad),
            "carb_min": max(0.0, float(target_mid.get("carbs", 0.0)) - pad),
            "carb_max": min(1.0, float(target_mid.get("carbs", 0.0)) + pad),
            "fat_min": max(0.0, float(target_mid.get("fat", 0.0)) - pad),
            "fat_max": min(1.0, float(target_mid.get("fat", 0.0)) + pad),
        }
    if not box:
        return None

    best_id, meta = pick_neighborhood_recipe_for_macro_box(
        pfc_by_rid,
        list(map(str, recipe_ids)),
        protein_min=float(box["protein_min"]),
        protein_max=float(box["protein_max"]),
        carb_min=float(box["carb_min"]),
        carb_max=float(box["carb_max"]),
        fat_min=float(box["fat_min"]),
        fat_max=float(box["fat_max"]),
        default_id=str(recipe_ids[0]),
    )

    # Optional semantic re-rank only among nutrition-tied near-equals.
    titles_by_id = titles_by_id or {}
    scored = list(meta.get("scored") or [])
    mode = str(meta.get("selection_mode") or "")
    key = (
        "distance_to_midpoint"
        if mode == "closest_to_midpoint_in_box"
        else "distance_to_target_box"
    )
    if scored and semantic_weight > 0 and query:
        best_dist = float(scored[0].get(key) or 0.0)
        near = [r for r in scored if abs(float(r.get(key) or 0.0) - best_dist) <= 1e-6]
        if len(near) > 1:
            label_text: dict[str, str] = {}
            for rid, sub in lines_df.groupby(lines_df["recipe_nlg_id"].astype(str)):
                descs = []
                if "fdc_description" in sub.columns:
                    descs = [str(x) for x in sub["fdc_description"].dropna().tolist()[:24]]
                title = titles_by_id.get(str(rid), "")
                label_text[str(rid)] = f"{title} {' '.join(descs)}".strip()

            def _sem(r: dict[str, Any]) -> float:
                rid = str(r["recipe_nlg_id"])
                text = label_text.get(rid) or titles_by_id.get(rid, "")
                return max(
                    _token_jaccard(query, text),
                    _token_jaccard(query, titles_by_id.get(rid, "")),
                )

            near.sort(
                key=lambda r: (-_sem(r), float(r.get(key) or 0.0), str(r["recipe_nlg_id"]))
            )
            best_id = str(near[0]["recipe_nlg_id"])
            meta = {
                **meta,
                "chosen_recipe_nlg_id": best_id,
                "chosen_pfc": near[0].get("pfc"),
            }

    ingredients = _ingredient_rows_for_recipe(lines_df, best_id)
    if not ingredients:
        return None
    pfc = meta.get("chosen_pfc") or pfc_by_rid.get(best_id) or {}
    title = titles_by_id.get(best_id, "") or best_id
    return {
        "recipe_nlg_id": best_id,
        "title": title,
        "ingredients": ingredients,
        "pfc": {
            "protein": float(pfc.get("protein", 0.0)),
            "carbs": float(pfc.get("carbs", 0.0)),
            "fat": float(pfc.get("fat", 0.0)),
        },
        "semantic_score": None,
        "pfc_l1_to_target": meta.get("distance_to_target_box")
        if mode == "closest_to_box_edge"
        else meta.get("distance_to_midpoint"),
        "selection": {
            "query": query,
            "target_mid": target_mid or meta.get("target_midpoint"),
            "target_box": box,
            "selection_mode": mode,
            "n_in_box": meta.get("n_in_box"),
            "semantic_weight": semantic_weight,
            "nutrition_weight": nutrition_weight,
            "top_k_semantic": top_k_semantic,
        },
    }


def example_from_problem_start(problem: dict[str, Any]) -> dict[str, Any] | None:
    """Build an LLM example payload from the problem's chosen/start recipe."""
    chosen = problem.get("chosen_recipe") or {}
    ings = list(chosen.get("ingredients") or [])
    if not ings:
        ings = list((problem.get("retrieval_context") or {}).get("starting_ingredients") or [])
    if not ings:
        return None
    sel = chosen.get("selection") or {}
    pfc = sel.get("chosen_pfc")
    return {
        "recipe_nlg_id": chosen.get("recipe_nlg_id") or sel.get("chosen_recipe_nlg_id"),
        "title": chosen.get("title") or problem.get("title"),
        "ingredients": [
            {
                "label": r.get("label") or r.get("name"),
                "grams": r.get("grams"),
                "fdc_id": r.get("fdc_id"),
                "role": r.get("role"),
            }
            for r in ings
            if (r.get("label") or r.get("name")) and (r.get("grams") is not None)
        ],
        "pfc": pfc,
        "pfc_l1_to_target": sel.get("distance_to_midpoint")
        if sel.get("selection_mode") == "closest_to_midpoint_in_box"
        else sel.get("distance_to_target_box"),
        "selection": sel,
    }


def attach_example_recipe_to_problem(
    problem: dict[str, Any],
    example: dict[str, Any] | None,
) -> dict[str, Any]:
    """Store example on the problem so creative draft can surface it to the LLM."""
    problem = dict(problem)
    if example:
        problem["example_recipe"] = example
        ctx = dict(problem.get("retrieval_context") or {})
        ctx["example_recipe"] = {
            "recipe_nlg_id": example.get("recipe_nlg_id"),
            "title": example.get("title"),
            "pfc": example.get("pfc"),
            "semantic_score": example.get("semantic_score"),
            "pfc_l1_to_target": example.get("pfc_l1_to_target"),
            "n_ingredients": len(example.get("ingredients") or []),
            "selection_mode": (example.get("selection") or {}).get("selection_mode"),
        }
        problem["retrieval_context"] = ctx
    return problem
