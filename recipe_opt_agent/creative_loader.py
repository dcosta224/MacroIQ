"""Load creative-mode problems: neighborhood context + offline grounding stub."""

from __future__ import annotations

from typing import Any


def load_creative_problem(
    *,
    user_request: str,
    canonical_id: int | None = None,
    protein_min: float = 0.19,
    protein_max: float = 0.23,
    carb_min: float = 0.345,
    carb_max: float = 0.545,
    fat_min: float = 0.245,
    fat_max: float = 0.445,
    offline: bool = False,
    require_cache: bool = True,
) -> dict[str, Any]:
    """Build a problem stub for creative mode.

    When canonical_id is set, loads neighborhood FDC catalog + basis samples from DB.
    When offline=True, skips DB and uses heuristic grounding only.
    When a dish is selected, ``require_cache`` (default True) refuses live Jaccard rebuilds.
    """
    if offline or canonical_id is None:
        return {
            "creative_offline": True,
            "grounding_offline": True,
            "title": user_request[:80],
            "taste_text": user_request,
            "user_request": user_request,
            "retrieval_context": {
                "fdc_catalog": _offline_fdc_catalog(),
                "starting_ingredients": [],
                "starting_fdc": [],
                "starting_labels": [],
                "neighbor_label_sets": [],
                "target_box": {
                    "protein_min": protein_min,
                    "protein_max": protein_max,
                    "carb_min": carb_min,
                    "carb_max": carb_max,
                    "fat_min": fat_min,
                    "fat_max": fat_max,
                },
            },
            "basis_samples": {
                "pasta": [0.45, 0.5, 0.55],
                "egg": [0.1, 0.12, 0.15],
                "cheese": [0.08, 0.1, 0.12],
                "cured_pork": [0.05, 0.08, 0.1],
                "protein": [0.15, 0.2, 0.25],
            },
            "ratio_samples": [4.0, 5.0, 6.0],
        }

    from recipe_opt_agent.example_recipe import (
        attach_example_recipe_to_problem,
        example_from_problem_start,
    )
    from recipe_opt_agent.problem_loader import load_canonical_problem

    base = load_canonical_problem(
        int(canonical_id),
        protein_min=protein_min,
        protein_max=protein_max,
        carb_min=carb_min,
        carb_max=carb_max,
        fat_min=fat_min,
        fat_max=fat_max,
        # Pick the neighborhood recipe closest to the macro target (midpoint if
        # any recipe is in-box, else nearest box edge) so the LLM draft has a
        # human reference that already needs few nutrition edits.
        prefer_nutrition_start=True,
        start_metric="l1_pfc",
        fast_neighborhood=True,
        require_cache=require_cache,
    )
    base["user_request"] = user_request
    base["taste_text"] = user_request
    base["creative_offline"] = False
    example = example_from_problem_start(base)
    if example is None:
        # Fallback: rebuild example from lines if chosen_recipe lacked rows.
        try:
            from canonical_optimization import CanonicalNeighborhood
            from recipe_opt_agent.example_recipe import pick_example_recipe_near_targets

            nb = CanonicalNeighborhood.build(
                int(canonical_id), fast=True, use_cache=True, require_cache=require_cache
            )
            example = pick_example_recipe_near_targets(
                lines_df=nb.lines_df,
                recipe_ids=list(map(str, nb.recipe_ids or [])),
                query=user_request or str(base.get("title") or ""),
                target_box={
                    "protein_min": protein_min,
                    "protein_max": protein_max,
                    "carb_min": carb_min,
                    "carb_max": carb_max,
                    "fat_min": fat_min,
                    "fat_max": fat_max,
                },
                titles_by_id={},
            )
        except Exception:
            example = None
    return attach_example_recipe_to_problem(base, example)


def _offline_fdc_catalog() -> list[dict[str, Any]]:
    return [
        {"fdc_id": 1001, "fdc_description": "Spaghetti, cooked, enriched"},
        {"fdc_id": 1002, "fdc_description": "Egg, whole, raw, fresh"},
        {"fdc_id": 1003, "fdc_description": "Cheese, parmesan, grated"},
        {"fdc_id": 1004, "fdc_description": "Pork, bacon, cooked"},
        {"fdc_id": 1005, "fdc_description": "Chicken breast, raw, skinless"},
        {"fdc_id": 1006, "fdc_description": "Oil, olive, salad or cooking"},
        {"fdc_id": 1007, "fdc_description": "Rice, white, cooked"},
        {"fdc_id": 1008, "fdc_description": "Mushrooms, white, raw"},
        {"fdc_id": 1009, "fdc_description": "Tofu, firm, prepared with calcium sulfate"},
        {"fdc_id": 1010, "fdc_description": "Pecorino romano cheese"},
        {"fdc_id": 1011, "fdc_description": "Guanciale, cured pork jowl"},
        {"fdc_id": 1012, "fdc_description": "Egg white, raw, fresh"},
        {"fdc_id": 1013, "fdc_description": "Pasta, dry, enriched"},
    ]
