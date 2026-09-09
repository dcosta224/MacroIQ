"""GPT-5.5 holistic evaluation of the final selected recipe.

Runs after candidate arbitration. Uses neighborhood hull stretch (how far the macro
target sits outside typical neighbor hulls) to calibrate how strict fidelity should
be, while still penalizing odd ingredients. Also raises a hard flag when dietary
restrictions are violated (deterministic pre-check + LLM confirmation).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

FINAL_EVALUATOR_SYSTEM_PROMPT = """You evaluate the FINAL selected recipe from an optimization agent run.

You receive:
- user_request, title, identity_roles, requirement_tags (hard dietary constraints)
- macro_target_context: how far the user's macro box sits OUTSIDE the typical conical
  ingredient hulls of original neighborhood recipes (not expanded shell recipes).
  target_stretch_level is one of:
    * in_hull — most neighbor recipes could reach this macro box with their ingredients;
      hold fidelity to a normal version of the dish.
    * edge — target is near the neighborhood hull boundary; allow modest ingredient
      stretches if they clearly help macros without wrecking the dish.
    * outside_hull — target is outside what most neighbor recipes can reach; you may be
      SLIGHTLY more forgiving on fidelity (see fidelity_forgiveness_hint, 0–0.35) BUT
      you must still penalize strange or identity-clashing ingredients. Macro pressure
      does not excuse ricotta-in-carbonara-style clashes or catalog macro hacks.
- final_recipe: ingredients, diff vs original, optimizer losses, PFC vs macro box
- dietary_precheck: deterministic tag scan (trust violations listed there)
- neighborhood_evidence for added ingredients (co-occurrence in expanded neighborhood)
- taste_preference (optional): an explicit sensory/lifestyle ask from the user
  (e.g. lighter, smokier, brighter lemon, weeknight-simpler). When present, judge
  whether the OUTPUT actually reflects that preference.

Score dimensions (0–10 each, then overall_score_0_10):
1. ingredients_make_sense — culinary plausibility for this dish given stretch level
   (also penalize kitchen-implausible quantities: e.g. 80g spice powders)
2. meets_user_needs — macros + request + identity roles + taste_preference when given
3. fidelity_vs_stretch — ratio/IQR losses interpreted WITH hull stretch context
4. overall_score_0_10 — holistic, not a naive average of losses

Rules:
- If dietary_precheck.all_restrictions_met is false, set dietary_violation_flag=true
  and reflect the violation in scores and concerns; do not apply a fixed numeric score cap.
- Small loss improvements (e.g. 1–5% better ratio/nutrient) NEVER justify clash
  ingredients when stretch is in_hull or edge.
- When stretch is outside_hull, a plausible_extension (e.g. lean poultry in pasta) can
  score well if neighborhood evidence supports it; clash ingredients still score poorly.
- Comment on whether the OUTPUT actually satisfies the macro box (nutrient_slack).
- When taste_preference is present, set taste_preference_met to yes|partially|no and
  explain briefly in concerns/strengths. Do not award "yes" if the preference is ignored.

Return ONLY JSON:
{
  "overall_score_0_10": 0-10,
  "ingredients_make_sense_score_0_10": 0-10,
  "meets_user_needs_score_0_10": 0-10,
  "fidelity_vs_stretch_score_0_10": 0-10,
  "ingredients_make_sense": "yes"|"mostly"|"no",
  "meets_user_needs": "yes"|"partially"|"no",
  "taste_preference_met": "yes"|"partially"|"no"|null,
  "dietary_restrictions_met": true|false,
  "dietary_violation_flag": true|false,
  "dietary_violations": [{"label": "...", "tag_ids": ["..."]}],
  "odd_ingredients": ["labels you judge as strange for this dish"],
  "plausible_extensions": ["labels that are non-traditional but justified"],
  "fidelity_assessment": "1-2 sentences referencing stretch level and losses",
  "macro_target_assessment": "1-2 sentences on box fit and hull stretch",
  "summary_markdown": "3-5 sentences for the user",
  "strengths": ["..."],
  "concerns": ["..."]
}
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))


def _tags_from_state(state: dict[str, Any]) -> list[Any]:
    from recipe_opt_agent.requirement_tags import RequirementTag

    out: list[RequirementTag] = []
    for r in state.get("requirement_tags") or []:
        if isinstance(r, RequirementTag):
            out.append(r)
        elif isinstance(r, dict):
            out.append(
                RequirementTag(
                    tag_id=str(r.get("tag_id") or ""),
                    kind=str(r.get("kind") or "preference"),
                    polarity=str(r.get("polarity") or "require"),
                    source_text=str(r.get("source_text") or ""),
                )
            )
    return out


def dietary_precheck(
    ingredients: list[dict[str, Any]],
    tags: list[Any],
) -> dict[str, Any]:
    from recipe_opt_agent.requirement_tags import tag_violations_for_ingredient

    dietary = [t for t in tags if getattr(t, "kind", None) == "dietary_restriction" or str(getattr(t, "tag_id", "")) in {
        "vegetarian", "vegan", "no_pork", "no_beef", "no_dairy", "gluten_free"
    }]
    violations: list[dict[str, Any]] = []
    for row in ingredients or []:
        label = str(row.get("label") or row.get("name") or "")
        if not label:
            continue
        vios = tag_violations_for_ingredient(
            label, dietary or tags, fdc_description=row.get("fdc_description")
        )
        if vios:
            violations.append({"label": label, "tag_ids": vios, "grams": row.get("grams")})
    return {
        "all_restrictions_met": len(violations) == 0,
        "dietary_violation_flag": len(violations) > 0,
        "violations": violations,
        "requirement_tags": [
            {
                "tag_id": getattr(t, "tag_id", None),
                "kind": getattr(t, "kind", None),
                "polarity": getattr(t, "polarity", None),
            }
            for t in tags
        ],
    }


def _ingredient_diff(
    ingredients: list[dict[str, Any]],
    original: list[dict[str, Any]],
) -> dict[str, list[str]]:
    cur = {str(r.get("label") or "").lower(): str(r.get("label") or "") for r in ingredients or []}
    orig = {str(r.get("label") or "").lower(): str(r.get("label") or "") for r in original or []}
    return {
        "added": [v for k, v in cur.items() if k and k not in orig],
        "removed": [v for k, v in orig.items() if k and k not in cur],
    }


def _neighborhood_evidence(added: list[str], problem: dict[str, Any]) -> dict[str, Any]:
    from recipe_opt_agent.final_arbiter import _neighborhood_evidence as _ev

    return _ev(added, problem)


def build_final_eval_briefing(
    state: dict[str, Any],
    final_payload: dict[str, Any],
) -> dict[str, Any]:
    cfg = state.get("config") or {}
    problem = state.get("problem") or {}
    chosen = final_payload.get("chosen") or {}
    entry = chosen.get("entry") if isinstance(chosen.get("entry"), dict) else {}
    ingredients = (
        chosen.get("ingredients")
        or entry.get("ingredients")
        or (problem.get("chosen_recipe") or {}).get("ingredients")
        or []
    )
    original = list(state.get("original_ingredients") or [])
    tags = _tags_from_state(state)
    diff = _ingredient_diff(ingredients, original)
    display = final_payload.get("display_scores") or {}
    opt = chosen.get("opt") or entry.get("opt") or state.get("opt") or {}
    pfc = opt.get("pfc_after") or {}
    box = {
        k: cfg.get(k)
        for k in ("protein_min", "protein_max", "carb_min", "carb_max", "fat_min", "fat_max")
    }

    hull_ctx = dict(
        problem.get("neighborhood_hull_context")
        or state.get("neighborhood_hull_context")
        or {}
    )
    # Suite A / creative paths sometimes omit stretch; infer from macro gap vs neighborhood mean.
    if not hull_ctx.get("target_stretch_level") or hull_ctx.get("error"):
        hull_ctx = dict(hull_ctx)
        mean_pfc = (
            problem.get("neighborhood_mean_pfc")
            or (problem.get("macro_target_meta") or {}).get("neighborhood_mean_pfc")
            or {}
        )
        try:
            p_lo = float(box.get("protein_min") or 0.0)
            p_mean = float(mean_pfc.get("protein") or 0.0)
            delta = p_lo - p_mean if p_mean > 0 else None
        except (TypeError, ValueError):
            delta = None
        if delta is None:
            # No neighborhood mean available: treat aggressive protein floors as outside_hull.
            if p_lo >= 0.28:
                level, forg, delta = "outside_hull", 0.30, p_lo
            elif p_lo >= 0.22:
                level, forg, delta = "edge", 0.15, p_lo
            else:
                level, forg, delta = "in_hull", 0.0, p_lo
        elif delta >= 0.08:
            level, forg = "outside_hull", 0.30
        elif delta >= 0.04:
            level, forg = "edge", 0.15
        else:
            level, forg = "in_hull", 0.0
        hull_ctx.setdefault("target_stretch_level", level)
        hull_ctx.setdefault("fidelity_forgiveness_hint", forg)
        hull_ctx["stretch_inferred_from"] = "protein_box_vs_neighborhood_mean"
        hull_ctx["protein_delta_vs_mean"] = delta

    return {
        "title": state.get("title"),
        "user_request": state.get("user_request") or state.get("taste_text"),
        "identity_roles": state.get("identity_roles") or [],
        "macro_target_context": {
            "macro_box": box,
            "neighborhood_hull": hull_ctx,
            "fidelity_forgiveness_hint": hull_ctx.get("fidelity_forgiveness_hint"),
            "target_stretch_level": hull_ctx.get("target_stretch_level"),
            "frac_neighbor_hulls_intersecting_target": hull_ctx.get("frac_hull_intersects"),
            "median_neighbor_outside_score": hull_ctx.get("median_outside_score"),
            "stretch_note": (
                "Interpret ratio/IQR losses WITH this stretch level. "
                "outside_hull allows modest plausible_extension (e.g. lean poultry) but still "
                "penalize clash ingredients (yogurt/milk/ricotta stacks, onion rings, coffee-in-bread)."
            ),
        },
        "final_recipe": {
            "source": chosen.get("source"),
            "status": final_payload.get("status"),
            "ingredients": [
                {"label": r.get("label") or r.get("name"), "grams": r.get("grams")}
                for r in ingredients
            ],
            "diff_vs_original": diff,
            "display_scores": display,
            "pfc_after": pfc,
            "nutrient_slack": display.get("nutrient_loss", {}).get("value")
            if isinstance(display.get("nutrient_loss"), dict)
            else None,
            "ratio_loss": display.get("ratio_loss", {}).get("value")
            if isinstance(display.get("ratio_loss"), dict)
            else None,
            "arbiter_rationale": chosen.get("arbiter_rationale")
            or (state.get("final_judgment") or {}).get("rationale"),
        },
        "dietary_precheck": dietary_precheck(ingredients, tags),
        "neighborhood_evidence": _neighborhood_evidence(diff["added"], problem),
        "taste_preference": (
            state.get("taste_preference")
            or final_payload.get("taste_preference")
            or problem.get("taste_preference")
        ),
    }


def evaluate_final_recipe(
    state: dict[str, Any],
    final_payload: dict[str, Any],
    *,
    model: str = "gpt-5.5",
) -> dict[str, Any] | None:
    """Return evaluation dict with scores, flags, summary_markdown, _llm_trace."""
    briefing = build_final_eval_briefing(state, final_payload)
    if not briefing["final_recipe"]["ingredients"]:
        return None

    messages = [
        {"role": "system", "content": FINAL_EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(briefing, indent=2, default=str)},
    ]

    pre = briefing["dietary_precheck"]

    def _heuristic() -> dict[str, Any]:
        stretch = (briefing.get("macro_target_context") or {}).get("target_stretch_level")
        odd = list(diff.get("added") or []) if (diff := briefing["final_recipe"]["diff_vs_original"]) else []
        score = 7.0 if pre["all_restrictions_met"] else 3.0
        if odd and stretch == "in_hull":
            score -= 1.0
        return {
            "overall_score_0_10": score,
            "ingredients_make_sense": "mostly" if not odd else "no",
            "meets_user_needs": "partially",
            "dietary_restrictions_met": pre["all_restrictions_met"],
            "dietary_violation_flag": pre["dietary_violation_flag"],
            "dietary_violations": pre["violations"],
            "summary_markdown": "Heuristic evaluation (no API key).",
            "fidelity_assessment": f"Stretch level: {stretch}.",
            "macro_target_assessment": "See display_scores in briefing.",
        }

    if not os.environ.get("OPENAI_API_KEY"):
        out = _heuristic()
        out["briefing"] = briefing
        out["_llm_trace"] = {"mode": "heuristic", "model": None, "messages": messages}
        return out

    try:
        from recipe_opt_agent.observability import get_openai_client

        client = get_openai_client()
        create_kwargs: dict[str, Any] = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        # gpt-5.x rejects non-default temperature; omit the param for that family.
        if not str(model).startswith("gpt-5"):
            create_kwargs["temperature"] = 0.2
        resp = client.chat.completions.create(**create_kwargs)
        content = resp.choices[0].message.content or "{}"
        data = _extract_json(content)
    except Exception as exc:
        out = _heuristic()
        out["briefing"] = briefing
        out["_llm_trace"] = {
            "mode": "heuristic_fallback",
            "model": model,
            "error": str(exc),
            "messages": messages,
        }
        return out

    # Merge deterministic dietary flag (authoritative)
    if pre["dietary_violation_flag"]:
        data["dietary_restrictions_met"] = False
        data["dietary_violation_flag"] = True
        if not data.get("dietary_violations"):
            data["dietary_violations"] = pre["violations"]
    data["briefing"] = briefing
    data["_llm_trace"] = {
        "mode": "openai",
        "model": model,
        "messages": messages,
        "raw_response": content,
        "usage": {
            "prompt_tokens": getattr(getattr(resp, "usage", None), "prompt_tokens", None),
            "completion_tokens": getattr(getattr(resp, "usage", None), "completion_tokens", None),
        },
    }
    return data
