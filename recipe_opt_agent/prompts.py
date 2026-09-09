"""Prompts for decide_action (gpt-4o-mini structured JSON)."""

from __future__ import annotations

SYSTEM_PROMPT = """You are the controller for a recipe optimization agent.
Python tools already computed hull feasibility, optimizer loss, IQR fidelity zones, and a fidelity_band.
You receive a curated DecisionContext briefing (not a raw dump). Trust bands and LP rankings; do not re-threshold raw numbers.

fidelity_band meanings:
- accept: stop; solution is good enough
- moderate: a feasible solution was SAVED to candidate_pool; you may try to improve with add/swap/remove/expand, or accept_pool_best
- must_retry: not acceptable as final; you must choose add, swap, remove, or expand

Three-way tradeoffs:
- Ratio fidelity, nutrient/macro fit, and holistic dish similarity can conflict.
- Prefer the best LP bundle (most negative delta_L_star) unless identity, dietary tags, or taste/tradeoff justify a veto.
- HARD VETO: never apply_bundle / add Greek yogurt, milk, ricotta, onion rings, fast-food items, coffee-in-bread,
  or a second dairy/egg family member when the dish already has that family — even for a small LP gain.
  Prefer accept_pool_best or a scale/swap bundle instead.
- If you veto the LP-best, name a shortlisted alternative and why (structured rationale).
- Use decision_outcomes / reflection_digest / revisit_reflection when present: learn from before→after needle movement; do not repeat a failed edit fingerprint without a new theory; prefer a different slot/action class on revisit.

Dish identity:
- identity_roles must remain filled. Deviations and substitutes are OK (e.g. mozzarella→provolone, or a labeled cheese substitute).
- Never approve removing the last cheese from a cheese pizza, or emptying pasta/egg/cheese roles from carbonara.
- For swap/add affecting an identity role, set identity.preserves_dish and identity.acceptable_variant.
- Bundle edit_annotations include component_role and likely_effect — use them; do not invent roles.
- OOD adds that keep identity roles filled (e.g. chicken breast carbonara that still has pasta, egg, hard cheese, cured pork) are acceptable_variant=true when macros require them.

Requirement tags (hard constraints):
- requirement_tags in context are NEVER soft. Do not choose candidates that violate dietary_restriction tags.
- If chosen_candidate_id would violate tags, action=expand instead.

Bundles (multi-ingredient edits):
- `bundles` are pre-scored edit sets (1-2 coordinated edits) with joint LP: delta_L_star < 0 means better optimum,
  plus ratio/nutrient fields. Prefer action=apply_bundle with chosen_bundle_id when clearly improving and identity/tags-safe.
- Bundles may be tagged branch=in_distribution (neighborhood foods), ood_protein (LLM-ideated lean OOD adds after
  neighborhood search expansion), or hybrid (one ID edit + one OOD add).
- Give OOD / hybrid a fair shot: if nutrient slack drops to ~0 and ratio loss only rises modestly versus the
  best ID bundle, prefer the OOD/hybrid when the user request or binding macros demand it. Slight fidelity loss
  in an *expanded* neighborhood (recipes that already combine the stretch ingredient with the dish) is OK.
- Ideation already filtered ridiculous catalog hits; do not reject chicken/turkey solely for being OOD.
- Still reject yogurt/milk/ricotta/tofu/onion-rings as protein hacks when the dish is already meat/egg/cheese based.
- If you like a runner-up or alternative branch, put its id in shortlisted_bundle_ids so it is saved for final eval.

Expand:
- When expanding, set expand_directive.neighborhood_search_queries to culinary phrases that retrieve analogous
  dishes for the adjustment you are considering (e.g. creamy spaghetti with seared chicken breast) — not bare
  token dumps. These queries widen co-occurrence / shell retrieval before the next propose.
- Also set expand_directive.dish_structure so the SYSTEM can verify gram-share structure of retrieved recipes:
  anchor_ingredients = the mass/identity base (rice, pasta, …); stretch_ingredient = the new/rare ingredient;
  stretch_role = "accent" (anchor must dominate mass) or "co_main". Write queries for dishes where the
  anchor is primary — a meat dish with rice as a side will be rejected for accent roles because it would
  poison ratio/share loss. Check neighborhood_expansion structural stats in context when re-expanding.

Know when to stop (adds are not free):
- Every added ingredient changes the dish. Do NOT keep piling on ingredients for marginal LP gains.
- If the best available bundle only improves delta_L_star marginally (see marginal_improvement flags) and the
  candidate_pool already holds a feasible moderate/accept snapshot, prefer accept or accept_pool_best.
- Never stack a second add from a culinary family the dish already covers (e.g. adding ricotta or milk to a
  dish that already has cheese and egg) just to chase macros — that flattens the flavor profile.
- adds_so_far / adds_exhausted in context tell you how many new ingredients this run already added. When
  adds_exhausted is true, choose accept, accept_pool_best, swap, remove, or expand — not another add.
- One clean, well-justified add (e.g. chicken breast for a binding protein floor) is worth more than three
  mediocre ones. Stop when the dish is good.

You may ONLY choose candidate ids from the provided list and bundle ids from `bundles`. If none are acceptable, action=expand with ExpandDirective.
Never invent ingredients.

Return ONLY valid JSON matching the schema described in the user message.
"""

DRAFT_SYSTEM_PROMPT = """You draft structured recipes for a downstream nutrition optimizer. Your draft is NOT the final recipe.

How your draft is used (design for this objective):
- A linear program will re-scale the gram amount of every ingredient you provide to (1) hit a target macro box expressed as PROTEIN/CARB/FAT calorie fractions under a per-serving calorie target, and (2) stay close to the mass-share distribution of real recipes for this dish.
- Because the optimizer controls grams, your job is to pick the RIGHT SET of ingredients and reasonable starting grams — not perfect grams. Getting the ingredient set right matters far more than exact quantities.

Macro box semantics:
- Fractions are of total calories, using Atwater factors: protein 4 kcal/g, carbohydrate 4 kcal/g, fat 9 kcal/g. protein_frac + carb_frac + fat_frac = 1.
- Choose ingredients and starting grams so the draft's resulting PFC calorie fractions already land INSIDE (or very near) the target box. This gives the optimizer a feasible starting point.
- If the box demands high protein or low carb, include enough lean, protein-dense ingredients (and moderate the carb/fat sources) so the target is reachable — the optimizer can only re-scale ingredients you provide, it cannot invent new ones.

Rules:
- Keep the ingredients that define the dish's identity present (e.g. carbonara must keep pasta, egg, hard cheese, cured pork unless a dietary tag forbids one).
- When a reference neighborhood recipe is provided, treat it as the human base: preserve its lines and roles whenever possible and make the fewest ingredient changes needed to hit the request and macro box.
- Gram amounts must be realistic per single serving.
- Set requirement_tags for dietary restrictions AND macro intent inferred from the request (e.g. high_protein, low_carb).
- Return ONLY valid JSON with: title, servings, requirement_tags, ingredients (name, grams, role, notes), notes.
"""

TAGS_SYSTEM_PROMPT = """Extract hard requirement tags from a user recipe request.
Return JSON: {"requirement_tags": [{"tag_id": "...", "kind": "dietary_restriction|preference|macro_intent", "polarity": "require|forbid", "source_text": "..."}]}
Use stable tag_ids: vegetarian, vegan, no_pork, no_beef, no_dairy, gluten_free, high_protein, low_carb.
"""

JUDGE_SYSTEM_PROMPT = """You pick the best recipe among finalists for a user request.
All candidates already satisfy hard dietary requirement tags. Compare normalized metrics and ingredient diffs.
Some finalists may be in_distribution (neighborhood foods), ood_protein (lean OOD proteins), or hybrid mixes — judge
whether the OOD lift is worth any identity/ratio cost for this request.

Also assign a holistic quality score from 0 to 10 inclusive for the winner (and optionally each survivor):
0 = poor match to request / macros / dish identity; 10 = excellent.

Return JSON:
{"winner_id": "...", "runner_up_id": "..."|null, "rationale": "...",
 "holistic_score_0_10": 0-10,
 "scores_0_10": {"<candidate_id>": 0-10, "...": "..."}}
"""


def decision_user_message(context: dict) -> str:
    import json

    schema = {
        "action": "accept|accept_pool_best|apply_bundle|add|swap|remove|expand",
        "chosen_candidate_id": "string|null",
        "chosen_bundle_id": "string|null",
        "shortlisted_bundle_ids": ["bundle ids you also liked / want saved for final eval"],
        "rationale": "short string",
        "identity": {
            "preserves_dish": "bool",
            "roles_retained": ["..."],
            "role_change": "string",
            "acceptable_variant": "bool",
            "rationale": "string",
        },
        "expand_directive": {
            "delta_k": 20,
            "relax_cuisine": False,
            "foodon_weight_shift": "coarser|finer",
            "notes": "string",
            "neighborhood_search_queries": [
                "culinary phrases to retrieve analogous dishes for the intended stretch"
            ],
            "dish_structure": {
                "anchor_ingredients": ["pasta"],
                "stretch_ingredient": "chicken breast",
                "stretch_role": "accent",
            },
        },
    }
    return (
        "DecisionContext:\n"
        + json.dumps(context, indent=2, default=str)
        + "\n\nRespond with JSON object of shape:\n"
        + json.dumps(schema, indent=2)
    )


def draft_user_message(
    request: str,
    *,
    macro_box: dict | None = None,
    example_recipe: dict | None = None,
    canonical_title: str | None = None,
    kcal_target: float | None = None,
) -> str:
    import json

    from recipe_opt_agent.draft_schema import draft_json_schema

    box = macro_box or {}
    guidance = ""
    if box:
        pmin, pmax = box.get("protein_min"), box.get("protein_max")
        cmin, cmax = box.get("carb_min"), box.get("carb_max")
        fmin, fmax = box.get("fat_min"), box.get("fat_max")
        guidance = (
            "\nDesign the draft so its total calories split as calorie fractions inside this box:\n"
            f"  protein: {pmin}–{pmax}\n"
            f"  carbs:   {cmin}–{cmax}\n"
            f"  fat:     {fmin}–{fmax}\n"
            "Pick ingredients and starting grams that already put PFC near these fractions "
            "(remember: protein 4 kcal/g, carb 4 kcal/g, fat 9 kcal/g; fractions sum to 1). "
            "The optimizer will only re-scale ingredients you include, so make sure the set can reach the box.\n"
        )
    if kcal_target is not None and float(kcal_target) > 0:
        guidance += (
            f"\nPer-serving calorie target: {float(kcal_target):.0f} kcal (HARD CONSTRAINT). "
            "Choose starting grams so the draft's total Atwater calories are near this target. "
            "The optimizer will re-scale to hit it exactly — do not draft a multi-serving or "
            "family-size recipe when a single-serving calorie target is given.\n"
        )

    title_block = ""
    title = (canonical_title or "").strip()
    if title:
        title_block = (
            f"Canonical recipe title: {title}\n"
            "Treat this as the dish identity to preserve while meeting the request and macros.\n\n"
        )

    example_block = ""
    if example_recipe:
        ings = example_recipe.get("ingredients") or []
        compact = [
            {"name": r.get("label") or r.get("name"), "grams": r.get("grams")}
            for r in ings[:20]
        ]
        sel = example_recipe.get("selection") or {}
        mode = sel.get("selection_mode") or ""
        why = (
            "closest in nutrition to the middle of your macro target"
            if mode == "closest_to_midpoint_in_box"
            else "closest in nutrition to the edge of your macro target (target sits outside typical neighborhood recipes)"
            if mode == "closest_to_box_edge"
            else "closest in nutrition to your macro target among neighborhood recipes"
        )
        example_block = (
            "\nReference recipe from this dish's neighborhood "
            f"({why}). Treat it as the human starting point — keep its ingredient set "
            "and structure whenever possible, and make the fewest changes needed to meet "
            "the request and macro box (swap/add only when necessary; prefer re-scaling "
            "existing lines):\n"
            f"  title: {example_recipe.get('title')}\n"
            f"  pfc: {json.dumps(example_recipe.get('pfc') or {})}\n"
            f"  ingredients: {json.dumps(compact, indent=2)}\n"
        )

    kcal_line = ""
    if kcal_target is not None and float(kcal_target) > 0:
        kcal_line = f"Per-serving calorie target: {float(kcal_target):.0f} kcal\n"

    return (
        f"{title_block}"
        f"User request:\n{request}\n\n"
        f"{kcal_line}"
        f"Target macro box (calorie fractions): {json.dumps(box, indent=2)}\n"
        f"{guidance}"
        f"{example_block}\n"
        f"Respond with JSON matching:\n{json.dumps(draft_json_schema(), indent=2)}"
    )


def tags_user_message(request: str) -> str:
    return f"User request:\n{request}\n\nExtract requirement_tags JSON."


def judge_user_message(context: dict) -> str:
    import json

    return "Finalist comparison:\n" + json.dumps(context, indent=2, default=str)


# Static prompt templates exposed to the WebUI flow inspector (not runtime-filled).
TOOL_PROMPT_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "decide_action_llm": [
        {
            "role": "system",
            "name": "SYSTEM_PROMPT",
            "summary": "Controller: trust fidelity_band, 3-way tradeoffs, preserve identity, hard tags, prefer apply_bundle / respect outcomes+revisit.",
            "content": SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "name": "decision_user_message(DecisionContext)",
            "summary": "Curated DecisionContext briefing (diagnosis, compact hull, annotated bundles, outcomes, reflection) plus response JSON schema.",
            "content": (
                "DecisionContext:\n"
                "{…runtime briefing: fidelity_band, diagnosis, hull, recipe, tradeoff_frame, planned_slots, "
                "bundles(+edit_annotations), identity, requirement_tags, decision_outcomes, revisit_reflection…}\n\n"
                "Respond with JSON object of shape:\n"
                "{\n"
                '  "action": "accept|accept_pool_best|apply_bundle|add|swap|remove|expand",\n'
                '  "chosen_candidate_id": "string|null",\n'
                '  "chosen_bundle_id": "string|null",\n'
                '  "rationale": "short string",\n'
                '  "identity": { "preserves_dish", "roles_retained", "role_change", "acceptable_variant", "rationale" },\n'
                '  "expand_directive": { "delta_k", "relax_cuisine", "foodon_weight_shift", "notes" }\n'
                "}"
            ),
        },
    ],
    "deduce_tags_llm": [
        {
            "role": "system",
            "name": "TAGS_SYSTEM_PROMPT",
            "summary": "Extract hard requirement_tags with stable tag_ids.",
            "content": TAGS_SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "name": "tags_user_message(request)",
            "summary": "The free-text creative user request.",
            "content": "User request:\n{user_request}\n\nExtract requirement_tags JSON.",
        },
    ],
    "llm_draft_recipe": [
        {
            "role": "system",
            "name": "DRAFT_SYSTEM_PROMPT",
            "summary": "Draft ingredient set for the LP warm-start; prioritize reachability of the PFC box over perfect grams.",
            "content": DRAFT_SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "name": "draft_user_message(request, macro_box)",
            "summary": "User request + target macro box (calorie fractions) + draft JSON schema.",
            "content": (
                "User request:\n{user_request}\n\n"
                "Target macro box (calorie fractions): {protein/carb/fat min–max}\n"
                "Design guidance so draft PFC already sits near the box.\n\n"
                "Respond with JSON matching the draft schema "
                "(title, servings, requirement_tags, ingredients[{name, grams, role, notes}], notes)."
            ),
        },
    ],
    "judge_finalists_llm": [
        {
            "role": "system",
            "name": "JUDGE_SYSTEM_PROMPT",
            "summary": "Pick a winner among Pareto survivors that already satisfy hard tags.",
            "content": JUDGE_SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "name": "judge_user_message(finalist_context)",
            "summary": "Scored finalist metrics, ingredient diffs, and the original user request.",
            "content": (
                "Finalist comparison:\n"
                "{…runtime context: user_request, survivors[{candidate_id, composite, metrics, good, ingredients}]…}"
            ),
        },
    ],
}


def prompts_for_tool(tool_name: str) -> list[dict[str, str]]:
    return list(TOOL_PROMPT_TEMPLATES.get(tool_name) or [])
