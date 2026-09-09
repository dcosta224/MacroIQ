"""OpenAI structured JSON helpers for the recipe opt agent."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from recipe_opt_agent.prompts import (
    DRAFT_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TAGS_SYSTEM_PROMPT,
    decision_user_message,
    draft_user_message,
    judge_user_message,
    tags_user_message,
)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))


def decide_action_llm(context: dict[str, Any], *, model: str = "gpt-4o-mini") -> dict[str, Any]:
    """Call OpenAI chat completions; return parsed decision dict with `_llm_trace`.

    If OPENAI_API_KEY is missing, returns a deterministic heuristic decision.
    """
    system = SYSTEM_PROMPT
    user = decision_user_message(context)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if not os.environ.get("OPENAI_API_KEY"):
        decision = _heuristic_decision(context)
        decision["_llm_trace"] = {
            "mode": "heuristic",
            "model": None,
            "messages": messages,
            "raw_response": json.dumps(decision, indent=2, default=str),
            "parsed": {k: v for k, v in decision.items() if k != "_llm_trace"},
            "rationale": decision.get("rationale"),
        }
        return decision

    from recipe_opt_agent.observability import get_openai_client

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=messages,
    )
    content = resp.choices[0].message.content or "{}"
    data = _extract_json(content)
    decision = _normalize_decision(data, context)
    usage = None
    if getattr(resp, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }
    decision["_llm_trace"] = {
        "mode": "openai",
        "model": model,
        "messages": messages,
        "raw_response": content,
        "parsed": {k: v for k, v in decision.items() if k != "_llm_trace"},
        "rationale": decision.get("rationale"),
        "usage": usage,
    }
    return decision


def _tags_from_context(context: dict[str, Any]):
    from recipe_opt_agent.requirement_tags import RequirementTag

    raw = context.get("requirement_tags") or []
    out: list[RequirementTag] = []
    for r in raw:
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


def _candidate_passes_tags(cand: dict[str, Any], context: dict[str, Any]) -> bool:
    from recipe_opt_agent.requirement_tags import ingredient_passes_tags

    tags = _tags_from_context(context)
    if not tags:
        return True
    label = str(cand.get("label") or "")
    return ingredient_passes_tags(label, tags)


def _bundle_passes_tags(bundle: dict[str, Any], context: dict[str, Any]) -> bool:
    from recipe_opt_agent.requirement_tags import ingredient_passes_tags

    tags = _tags_from_context(context)
    if not tags:
        return True
    for edit in bundle.get("edits") or []:
        if edit.get("action") in {"add", "swap"}:
            if not ingredient_passes_tags(str(edit.get("label") or ""), tags):
                return False
    return True


def _normalize_decision(data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    action = str(data.get("action", "expand"))
    allowed = {"accept", "accept_pool_best", "apply_bundle", "add", "swap", "remove", "expand"}
    if action not in allowed:
        action = "expand"
    cand_id = data.get("chosen_candidate_id")
    bundle_id = data.get("chosen_bundle_id")
    cands = {c["candidate_id"]: c for c in context.get("candidates", [])}
    bundles = {str(b.get("bundle_id")): b for b in (context.get("bundles") or [])}
    valid_ids = set(cands.keys())
    if cand_id is not None and str(cand_id) not in valid_ids:
        cand_id = None
        if action in {"add", "swap", "remove"}:
            action = "expand"
    identity = data.get("identity") or {}
    if action == "apply_bundle":
        bundle = bundles.get(str(bundle_id)) if bundle_id is not None else None
        if bundle is None:
            bundle_id = None
            action = "expand"
        elif not _bundle_passes_tags(bundle, context):
            bundle_id = None
            action = "expand"
        elif identity.get("preserves_dish") is False or identity.get("acceptable_variant") is False:
            bundle_id = None
            action = "expand"
    if action in {"swap", "remove", "add"} and cand_id is not None:
        cand = cands.get(str(cand_id))
        if cand and not _candidate_passes_tags(cand, context):
            cand_id = None
            action = "expand"
        if identity.get("preserves_dish") is False or identity.get("acceptable_variant") is False:
            cand_id = None
            action = "expand"
    return {
        "action": action,
        "chosen_candidate_id": cand_id,
        "chosen_bundle_id": bundle_id if action == "apply_bundle" else None,
        "shortlisted_bundle_ids": [
            str(x)
            for x in (data.get("shortlisted_bundle_ids") or [])
            if str(x) in bundles
        ][:6],
        "rationale": str(data.get("rationale", "")),
        "identity": identity,
        "expand_directive": data.get("expand_directive")
        or {"delta_k": 20, "relax_cuisine": False, "foodon_weight_shift": "coarser", "notes": ""},
    }


def _heuristic_decision(context: dict[str, Any]) -> dict[str, Any]:
    from recipe_opt_agent.telemetry import clear_favorite_bundle

    band = context.get("fidelity_band", "must_retry")
    diagnosis = (context.get("diagnosis") or {}).get("diagnosis")
    cands = [c for c in (context.get("candidates") or []) if _candidate_passes_tags(c, context)]
    if band == "accept":
        return {
            "action": "accept",
            "chosen_candidate_id": None,
            "rationale": "Heuristic: fidelity_band=accept",
            "identity": {"preserves_dish": True, "acceptable_variant": True, "roles_retained": context.get("identity_roles", []), "role_change": "", "rationale": "no change"},
            "expand_directive": None,
        }
    if band == "moderate" and not cands:
        return {
            "action": "accept_pool_best",
            "chosen_candidate_id": None,
            "rationale": "Heuristic: moderate band, no candidates",
            "identity": {"preserves_dish": True, "acceptable_variant": True, "roles_retained": context.get("identity_roles", []), "role_change": "", "rationale": "keep pool"},
            "expand_directive": None,
        }
    bundles = [
        b
        for b in (context.get("bundles") or [])
        if b.get("lp_evaluated")
        and b.get("delta_L_star") is not None
        and _bundle_passes_tags(b, context)
        and not b.get("oscillation_blocked")
    ]
    # Prefer clear favorite (mirrors graph auto-apply); else any improving bundle.
    favorite = clear_favorite_bundle(
        bundles,
        passes_tags_fn=lambda b: _bundle_passes_tags(b, context),
        ood_delta_handicap=0.015,
    )
    best_b = favorite or (min(bundles, key=lambda b: b["delta_L_star"]) if bundles else None)
    if best_b is not None and float(best_b["delta_L_star"]) < 0.0:
        return {
            "action": "apply_bundle",
            "chosen_candidate_id": None,
            "chosen_bundle_id": best_b.get("bundle_id"),
            "rationale": (
                f"Heuristic: bundle {best_b.get('bundle_id')} improves L* by "
                f"{-float(best_b['delta_L_star']):.4f}"
                + (" (clear favorite)" if favorite is not None else "")
            ),
            "identity": {
                "preserves_dish": True,
                "acceptable_variant": True,
                "roles_retained": context.get("identity_roles", []),
                "role_change": ", ".join(str(e.get("label")) for e in best_b.get("edits") or []),
                "rationale": "joint-LP-scored bundle from slot retrieval",
            },
            "expand_directive": None,
        }
    if cands:
        # Prefer add with lowest L_star
        best = min(cands, key=lambda c: (c.get("L_star") is None, c.get("L_star") or 0.0))
        return {
            "action": best.get("action", "add"),
            "chosen_candidate_id": best["candidate_id"],
            "rationale": f"Heuristic: pick best candidate for {diagnosis}",
            "identity": {
                "preserves_dish": True,
                "acceptable_variant": True,
                "roles_retained": context.get("identity_roles", []),
                "role_change": best.get("label", ""),
                "rationale": "candidate from retrieval shortlist",
            },
            "expand_directive": None,
        }
    return {
        "action": "expand",
        "chosen_candidate_id": None,
        "rationale": "Heuristic: no candidates; expand neighborhood",
        "identity": {"preserves_dish": True, "acceptable_variant": True, "roles_retained": context.get("identity_roles", []), "role_change": "", "rationale": "expand"},
        "expand_directive": {"delta_k": 20, "relax_cuisine": False, "foodon_weight_shift": "coarser", "notes": "auto"},
    }


def _call_json_llm(
    *,
    system: str,
    user: str,
    model: str,
    heuristic_fn: Callable[[], dict[str, Any]] | None = None,
    temperature: float | None = 0.2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if not os.environ.get("OPENAI_API_KEY"):
        data = heuristic_fn() if heuristic_fn else {}
        trace = {"mode": "heuristic", "model": None, "messages": messages, "raw_response": json.dumps(data)}
        return data, trace

    from recipe_opt_agent.observability import get_openai_client

    client = get_openai_client()
    create_kwargs: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    # Some models (e.g. gpt-5.5) only accept the default temperature; omit the param.
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    resp = client.chat.completions.create(**create_kwargs)
    content = resp.choices[0].message.content or "{}"
    data = _extract_json(content)
    trace = {"mode": "openai", "model": model, "messages": messages, "raw_response": content}
    return data, trace


def deduce_tags_llm(request: str, *, model: str = "gpt-4.1-nano") -> list[dict[str, Any]]:
    from recipe_opt_agent.requirement_tags import deduce_tags_from_text

    def _heuristic() -> dict[str, Any]:
        return {"requirement_tags": [t.to_dict() for t in deduce_tags_from_text(request)]}

    data, _ = _call_json_llm(
        system=TAGS_SYSTEM_PROMPT,
        user=tags_user_message(request),
        model=model,
        heuristic_fn=_heuristic,
    )
    return list(data.get("requirement_tags") or [])


def _heuristic_draft(request: str, *, macro_box: dict | None = None) -> dict[str, Any]:
    req = request.lower()
    box = macro_box or {}
    # Infer macro intent from the target box (calorie fractions), not just keywords.
    protein_mid = None
    carb_mid = None
    if box.get("protein_min") is not None and box.get("protein_max") is not None:
        protein_mid = 0.5 * (float(box["protein_min"]) + float(box["protein_max"]))
    if box.get("carb_min") is not None and box.get("carb_max") is not None:
        carb_mid = 0.5 * (float(box["carb_min"]) + float(box["carb_max"]))
    box_high_protein = protein_mid is not None and protein_mid >= 0.25
    box_low_carb = carb_mid is not None and carb_mid <= 0.35
    wants_protein = "protein" in req or "40" in req or box_high_protein
    wants_low_carb = "low carb" in req or "low-carb" in req or box_low_carb

    title = "Custom Recipe"
    tags: list[dict[str, Any]] = []
    ings: list[dict[str, Any]] = []
    if "carbonara" in req:
        title = "High-Protein Carbonara" if wants_protein else "Carbonara"
        # Lower starting pasta grams when the box is high-protein / low-carb.
        pasta_g = 60.0 if (wants_protein or wants_low_carb) else 80.0
        ings = [
            {"name": "spaghetti", "grams": pasta_g, "role": "pasta"},
            {"name": "egg", "grams": 100.0, "role": "egg"},
            {"name": "parmesan cheese", "grams": 40.0, "role": "cheese"},
        ]
        if "vegetarian" in req or "veggie" in req:
            tags.append({"tag_id": "vegetarian", "kind": "dietary_restriction", "polarity": "require"})
            if wants_protein:
                tags.append({"tag_id": "high_protein", "kind": "macro_intent", "polarity": "require"})
                ings.append({"name": "tofu", "grams": 120.0, "role": "protein"})
                ings.append({"name": "mushrooms", "grams": 80.0, "role": "umami"})
        else:
            if "no pork" not in req:
                ings.append({"name": "guanciale", "grams": 50.0, "role": "cured_pork"})
            if wants_protein:
                tags.append({"tag_id": "high_protein", "kind": "macro_intent", "polarity": "require"})
                ings.append({"name": "chicken breast", "grams": 120.0, "role": "protein"})
        if wants_low_carb:
            tags.append({"tag_id": "low_carb", "kind": "macro_intent", "polarity": "require"})
    else:
        veg = "vegetarian" in req or "veggie" in req
        protein_line = (
            {"name": "tofu", "grams": 150.0, "role": "protein"}
            if veg
            else {"name": "chicken breast", "grams": 150.0, "role": "protein"}
        )
        # Bias grams by target: more protein / less carb when the box calls for it.
        if wants_protein:
            protein_line["grams"] = 200.0
        carb_g = 70.0 if wants_low_carb else 100.0
        ings = [
            protein_line,
            {"name": "rice", "grams": carb_g, "role": "carb"},
            {"name": "olive oil", "grams": 10.0, "role": "fat"},
        ]
        if veg:
            tags.append({"tag_id": "vegetarian", "kind": "dietary_restriction", "polarity": "require"})
        if wants_protein:
            tags.append({"tag_id": "high_protein", "kind": "macro_intent", "polarity": "require"})
        if wants_low_carb:
            tags.append({"tag_id": "low_carb", "kind": "macro_intent", "polarity": "require"})
    from recipe_opt_agent.requirement_tags import deduce_tags_from_text

    for t in deduce_tags_from_text(request):
        if t.to_dict() not in tags:
            tags.append(t.to_dict())
    # Dedup tags by tag_id (branches above may add the same macro-intent tag).
    seen_tag_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for t in tags:
        tid = t.get("tag_id")
        if tid in seen_tag_ids:
            continue
        seen_tag_ids.add(tid)
        deduped.append(t)
    return {"title": title, "servings": 2, "requirement_tags": deduped, "ingredients": ings, "notes": request}


def llm_draft_recipe(
    request: str,
    *,
    macro_box: dict | None = None,
    example_recipe: dict | None = None,
    canonical_title: str | None = None,
    kcal_target: float | None = None,
    model: str = "gpt-4.1-mini",
    temperature: float | None = 0.2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data, trace = _call_json_llm(
        system=DRAFT_SYSTEM_PROMPT,
        user=draft_user_message(
            request,
            macro_box=macro_box,
            example_recipe=example_recipe,
            canonical_title=canonical_title,
            kcal_target=kcal_target,
        ),
        model=model,
        temperature=temperature,
        heuristic_fn=lambda: _heuristic_draft(request, macro_box=macro_box),
    )
    return data, trace


def judge_finalists_llm(
    context: dict[str, Any],
    *,
    model: str = "gpt-4.1-mini",
) -> tuple[dict[str, Any], dict[str, Any]]:
    survivors = context.get("survivors") or []

    def _heuristic() -> dict[str, Any]:
        if not survivors:
            return {"winner_id": None, "rationale": "no survivors", "runner_up_id": None}
        best = max(survivors, key=lambda s: s.get("composite", 0.0))
        runner = survivors[1]["candidate_id"] if len(survivors) > 1 else None
        return {
            "winner_id": best.get("candidate_id"),
            "rationale": "Heuristic: highest composite score",
            "runner_up_id": runner,
            "holistic_score_0_10": round(float(best.get("composite") or 0.0) * 10.0, 1),
            "scores_0_10": {
                str(s.get("candidate_id")): round(float(s.get("composite") or 0.0) * 10.0, 1)
                for s in survivors
                if s.get("candidate_id")
            },
        }

    data, trace = _call_json_llm(
        system=JUDGE_SYSTEM_PROMPT,
        user=judge_user_message(context),
        model=model,
        heuristic_fn=_heuristic,
    )
    return data, trace
