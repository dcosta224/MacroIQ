"""LLM ideation for ingredient edits, then FDC grounding + numeric verification.

Nutrition/co-occurrence retrieval alone tends to dominate with odd catalog hits
(ricotta, milk, …). Instead the LLM proposes 5–10 contextual ideas; we ground
each to FDC and score with ratio/nutrition proxies + joint LP.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

IDEATION_SYSTEM_PROMPT = """You propose ingredient edits for a recipe optimizer.

Your job is IDEATION only. Downstream code will ground your ideas to USDA FDC
foods and verify them with ratio-fidelity and nutrition LP scoring. Prefer
plausible culinary ideas over nutrient-max hacks.

Rules:
1. Propose 5–10 ideas. Mix of swap / add / remove as appropriate.
2. Prefer staying inside the dish's typical ingredient neighborhood when that
   can reach the macro box without wrecking identity or pasta∶egg-style ratios.
3. Only after arguing that in-neighborhood edits would force a large semantic
   or ratio/nutrition sacrifice may you propose out-of-distribution (OOD) adds
   (e.g. lean poultry in a carbonara-style pasta when protein demand is high).
   OOD ideas are allowed and often desirable in that case — chicken carbonara
   is a reasonable dish — but mark them clearly.
4. For each OOD or neighborhood-stretching idea, also propose 1–3
   neighborhood_search_queries: short phrases that would retrieve related
   recipes (titles/ingredient lists) whose co-occurrence should inform the
   expanded neighborhood. Prefer culinary phrases like
   "creamy pasta with seared chicken breast" or "poultry carbonara-style
   spaghetti" — not bare token dumps like "pasta chicken".
5. CAPABILITY HANDSHAKE for neighborhood search: the SYSTEM verifies
   retrieved recipes by FoodOn gram shares. When stretch_role="accent",
   anchors must be the dominant mass component (e.g. rice-primary fried rice
   with chicken accent — NOT chicken with a rice side). Write queries for
   dishes where the anchor is the base. Declare dish_structure so the system
   can gate harvest correctly:
     anchor_ingredients = mass/identity base foods (rice, pasta, …)
     stretch_ingredient = the underrepresented / new ingredient
     stretch_role = "accent" (stretch is secondary) or "co_main" (both primary)
6. Never remove the last filler of a critical identity role (pasta, egg, hard
   cheese, cured pork for carbonara unless a dietary tag forbids it).
7. Respect hard requirement_tags (vegetarian, no_pork, …).
8. MINIMAL EDITS. One well-chosen add beats stacking several. Each add must
   earn its place; a finished dish with one clean edit is the goal, not a
   pantry pile-on. Propose at most ONE new OOD ingredient across all ideas
   (variants of the same ingredient are fine).
9. Do not add an ingredient whose culinary family is already well represented
   (e.g. no ricotta, milk, or Greek yogurt into a dish already rich in cheese and egg —
   that flattens the flavor profile and hurts the dish more than the macros gain).
   A macro gap is better closed by scaling or one distinct lean ingredient.
10. HIGH-PROTEIN PLAYBOOK (when the macro box needs more protein):
    (a) First propose swap/scale of existing meat, egg, or legume lines toward leaner
        or higher-protein cuts already in the dish (or attested in the neighborhood).
    (b) Only then propose ONE OOD lean animal protein that co-occurs with the dish's
        anchors (e.g. chicken breast with pasta/rice), with dish_structure + search queries.
    (c) Never close a protein gap with yogurt, milk, ricotta, tofu, whey, or fried snacks
        when the dish is already savory meat/egg/cheese based.
    (d) Never ground "onion" to onion rings / fast-food items; never add coffee to bread.
11. TASTE PREFERENCE: When user_request (or taste_preference) asks for a sensory or
    lifestyle change — lighter/less oily, smokier, brighter/more lemon, milder heat,
    weeknight-simpler, more garlicky — prefer edits that advance THAT preference while
    still respecting macros and identity. Do not ignore the taste ask in favor of
    macros-only hacks.

Return ONLY JSON:
{
  "ideas": [
    {
      "action": "add"|"swap"|"remove",
      "ingredient": "human food name",
      "replace": "label being swapped/removed or null",
      "role": "short culinary role",
      "branch": "in_distribution"|"ood_protein"|"ood_other",
      "rationale": "one sentence",
      "neighborhood_search_queries": ["…"],
      "dish_structure": {
        "anchor_ingredients": ["pasta"],
        "stretch_ingredient": "chicken breast",
        "stretch_role": "accent"
      }
    }
  ],
  "ood_justification": "null or why OOD is needed after ID options looked costly",
  "neighborhood_search_queries": ["global queries for the adjustment being considered"],
  "dish_structure": {
    "anchor_ingredients": ["pasta"],
    "stretch_ingredient": "chicken breast",
    "stretch_role": "accent"
  }
}
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))


def _heuristic_ideas(context: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fallback when no API key / LLM failure."""
    diagnosis = context.get("diagnosis") or {}
    binding = list(diagnosis.get("binding_macros") or [])
    title = str(context.get("title") or "dish")
    ideas: list[dict[str, Any]] = []
    if any("protein" in str(b) for b in binding) or float(context.get("protein_min") or 0) >= 0.28:
        ideas.append(
            {
                "action": "add",
                "ingredient": "skinless chicken breast",
                "replace": None,
                "role": "lean_protein",
                "branch": "ood_protein",
                "rationale": "High protein demand; lean poultry is a classic carbonara-adjacent stretch.",
                "neighborhood_search_queries": [
                    f"{title} with chicken breast",
                    "creamy spaghetti with seared chicken",
                    "chicken carbonara-style pasta",
                ],
                "dish_structure": {
                    "anchor_ingredients": ["pasta", "spaghetti"],
                    "stretch_ingredient": "skinless chicken breast",
                    "stretch_role": "accent",
                },
            }
        )
        ideas.append(
            {
                "action": "add",
                "ingredient": "extra egg white",
                "replace": None,
                "role": "lean_protein",
                "branch": "in_distribution",
                "rationale": "Boost protein while staying inside egg identity.",
                "neighborhood_search_queries": [],
            }
        )
    # Generic ID swaps from current ingredients
    for row in list(context.get("current_ingredients") or [])[:3]:
        label = str(row.get("label") or row.get("name") or "")
        if not label:
            continue
        ideas.append(
            {
                "action": "swap",
                "ingredient": label,
                "replace": label,
                "role": "refine",
                "branch": "in_distribution",
                "rationale": f"Stay near neighborhood foods related to {label}.",
                "neighborhood_search_queries": [],
            }
        )
    structures = [
        idea.get("dish_structure")
        for idea in ideas
        if isinstance(idea.get("dish_structure"), dict)
    ]
    return {
        "ideas": ideas[:8],
        "ood_justification": "Heuristic: protein binding or high protein_min.",
        "neighborhood_search_queries": [
            q
            for idea in ideas
            for q in (idea.get("neighborhood_search_queries") or [])
        ],
        "dish_structure": structures[0] if structures else None,
    }


def ideate_ingredient_edits(
    context: dict[str, Any],
    *,
    model: str = "gpt-4.1-mini",
    n_ideas: int = 8,
) -> dict[str, Any]:
    """Return `{ideas, neighborhood_search_queries, ood_justification, _llm_trace}`."""
    n_ideas = max(5, min(10, int(n_ideas)))
    user = {
        "title": context.get("title"),
        "user_request": context.get("user_request") or context.get("taste_text"),
        "taste_preference": context.get("taste_preference"),
        "identity_roles": context.get("identity_roles"),
        "requirement_tags": context.get("requirement_tags"),
        "fidelity_band": context.get("fidelity_band"),
        "diagnosis": {
            "diagnosis": (context.get("diagnosis") or {}).get("diagnosis"),
            "binding_macros": (context.get("diagnosis") or {}).get("binding_macros"),
            "n_red": (context.get("diagnosis") or {}).get("n_red"),
            "L_max_norm": (context.get("diagnosis") or {}).get("L_max_norm"),
            "recommended_action_class": (context.get("diagnosis") or {}).get(
                "recommended_action_class"
            ),
        },
        "pfc_after": (context.get("opt") or {}).get("pfc_after"),
        "macro_box": context.get("macro_box") or context.get("macro_targets"),
        "protein_min": (context.get("macro_box") or {}).get("protein_min"),
        "current_ingredients": [
            {"label": r.get("label") or r.get("name"), "grams": r.get("grams")}
            for r in (context.get("current_ingredients") or [])[:16]
        ],
        "slots": context.get("slots") or [],
        "n_ideas": n_ideas,
        "note": (
            "Prefer in-distribution ideas first. Use OOD only when ID edits would "
            "badly hurt fidelity or still miss the macro box. Always include search "
            "queries AND dish_structure when proposing OOD so the system can retrieve "
            "analogous dishes and verify the stretch is an accent (or co-main) relative "
            "to the anchor via FoodOn gram shares — not a meat-primary dish with the "
            "anchor as a side."
        ),
    }
    messages = [
        {"role": "system", "content": IDEATION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user, indent=2, default=str)},
    ]

    if not os.environ.get("OPENAI_API_KEY"):
        data = _heuristic_ideas({**context, **user})
        data["_llm_trace"] = {
            "mode": "heuristic",
            "model": None,
            "messages": messages,
            "raw_response": json.dumps(data, indent=2, default=str),
        }
        return data

    try:
        from recipe_opt_agent.observability import get_openai_client

        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            temperature=0.4,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = resp.choices[0].message.content or "{}"
        data = _extract_json(content)
    except Exception as exc:
        data = _heuristic_ideas({**context, **user})
        data["_llm_trace"] = {
            "mode": "heuristic_fallback",
            "model": model,
            "error": str(exc),
            "messages": messages,
        }
        return data

    ideas = list(data.get("ideas") or [])[:n_ideas]
    global_q = list(data.get("neighborhood_search_queries") or [])
    for idea in ideas:
        for q in idea.get("neighborhood_search_queries") or []:
            if q and q not in global_q:
                global_q.append(q)
    dish_structure = data.get("dish_structure")
    if not isinstance(dish_structure, dict):
        for idea in ideas:
            if isinstance(idea.get("dish_structure"), dict):
                dish_structure = idea["dish_structure"]
                break
        else:
            dish_structure = None
    out = {
        "ideas": ideas,
        "ood_justification": data.get("ood_justification"),
        "neighborhood_search_queries": global_q[:8],
        "dish_structure": dish_structure,
        "_llm_trace": {
            "mode": "openai",
            "model": model,
            "messages": messages,
            "raw_response": content,
            "parsed": {k: v for k, v in data.items() if not str(k).startswith("_")},
            "usage": {
                "prompt_tokens": getattr(getattr(resp, "usage", None), "prompt_tokens", None),
                "completion_tokens": getattr(
                    getattr(resp, "usage", None), "completion_tokens", None
                ),
            },
        },
    }
    return out


def ground_ideas_to_candidates(
    ideas: list[dict[str, Any]],
    *,
    problem: dict[str, Any],
    requirement_tags: list[Any] | None = None,
    box_dict: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Map LLM ideas → FDC candidates with branch tags for bundle scoring."""
    from recipe_opt_agent.grounding import _search_catalog
    from recipe_opt_agent.ood_branch import OOD_PROTEIN_FOODS, ensure_ood_macros_on_problem
    from recipe_opt_agent.requirement_tags import RequirementTag, ingredient_passes_tags

    tags: list[RequirementTag] = []
    for r in requirement_tags or []:
        if isinstance(r, RequirementTag):
            tags.append(r)
        elif isinstance(r, dict):
            tags.append(
                RequirementTag(
                    tag_id=str(r.get("tag_id") or ""),
                    kind=str(r.get("kind") or "preference"),
                    polarity=str(r.get("polarity") or "require"),
                    source_text=str(r.get("source_text") or ""),
                )
            )

    problem = ensure_ood_macros_on_problem(dict(problem))
    ctx = problem.get("retrieval_context") or {}
    catalog = list(ctx.get("fdc_catalog") or [])
    # Ensure OOD synthetic foods are searchable by name
    for food in OOD_PROTEIN_FOODS:
        catalog.append(
            {
                "fdc_id": food["fdc_id"],
                "fdc_description": food["fdc_description"],
                "label": food["label"],
                "ood": True,
            }
        )

    current = {
        str(r.get("label") or "").lower(): r
        for r in ((problem.get("chosen_recipe") or {}).get("ingredients") or [])
    }
    box_dict = box_dict or {}
    out: list[dict[str, Any]] = []
    for i, idea in enumerate(ideas or []):
        action = str(idea.get("action") or "add").lower()
        name = str(idea.get("ingredient") or "").strip()
        if not name:
            continue
        branch = str(idea.get("branch") or "in_distribution")
        replace = idea.get("replace")
        hit, _score = _search_catalog(name, catalog, tags, min_score=0.35)
        if hit is None and branch.startswith("ood"):
            # Fuzzy match against OOD catalog labels
            for food in OOD_PROTEIN_FOODS:
                if name.lower() in food["label"].lower() or food["label"].lower() in name.lower():
                    hit = {
                        "fdc_id": food["fdc_id"],
                        "fdc_description": food["fdc_description"],
                        "ood": True,
                    }
                    break
            if hit is None:
                # Retry catalog with a softer floor for OOD-labeled catalog rows only
                soft_hit, soft_score = _search_catalog(name, catalog, tags, min_score=0.2)
                if soft_hit is not None and (soft_hit.get("ood") or soft_score >= 0.35):
                    hit = soft_hit
        if action == "remove":
            target = current.get(str(replace or name).lower())
            if not target:
                continue
            label = str(target.get("label") or name)
            if tags and not ingredient_passes_tags(label, tags):
                continue
            out.append(
                {
                    "candidate_id": f"idea_remove_{i}",
                    "action": "remove",
                    "label": label,
                    "fdc_id": target.get("fdc_id"),
                    "branch": branch,
                    "score": float(10 - i),
                    "meta": {
                        "source": "llm_ideation",
                        "role": idea.get("role"),
                        "rationale": idea.get("rationale"),
                        "ood": branch.startswith("ood"),
                        "neighborhood_search_queries": idea.get("neighborhood_search_queries") or [],
                        "delta_ratio_proxy": 0.0,
                        "delta_nutrient_proxy": 0.0,
                    },
                }
            )
            continue
        if hit is None:
            continue
        label = str(hit.get("fdc_description") or hit.get("label") or name)
        if tags and not ingredient_passes_tags(label, tags, fdc_description=label):
            continue
        fid = hit.get("fdc_id")
        macros = None
        if fid is not None:
            macros = (ctx.get("fdc_macros") or {}).get(str(fid))
        if action == "swap":
            repl = str(replace or "").lower()
            if repl and repl not in current:
                # allow swap against closest current label token overlap
                repl = next(iter(current), "")
            cand_action = "swap"
        else:
            cand_action = "add"
            repl = None
        out.append(
            {
                "candidate_id": f"idea_{cand_action}_{fid}_{i}",
                "action": cand_action,
                "label": label,
                "fdc_id": fid,
                "replace_label": replace,
                "branch": branch,
                "score": float(10 - i),
                "meta": {
                    "source": "llm_ideation",
                    "role": idea.get("role"),
                    "rationale": idea.get("rationale"),
                    "ood": branch.startswith("ood") or bool(hit.get("ood")),
                    "basis_node": hit.get("basis_node"),
                    "macros_per_g": macros,
                    "neighborhood_search_queries": idea.get("neighborhood_search_queries") or [],
                    # Soft OOD ratio prior — verification LP is authoritative
                    "delta_ratio_proxy": 0.02 if branch.startswith("ood") else 0.0,
                    "delta_nutrient_proxy": float(macros[0]) if macros else 0.0,
                    "protein_frac_mid": 0.5
                    * (float(box_dict.get("protein_min") or 0.2) + float(box_dict.get("protein_max") or 0.3)),
                },
            }
        )
    from recipe_opt_agent.ood_foodon import annotate_candidate_foodon

    return [annotate_candidate_foodon(c, problem) for c in out]
