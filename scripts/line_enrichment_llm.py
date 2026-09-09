"""LLM line enrichment for compound, vague, ambiguous, and parenthetical cases."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ingredient_parse_llm import DEFAULT_MODEL, MODEL_PRICING, normalize_ingredient_key
from openai_fallback import AllKeysExhaustedError
from resolution_plan import ResolutionPlan, build_resolution_plan, needs_line_enrichment

PROMPT_VERSION = "line_enrichment_v1"

SYSTEM_PROMPT = (
    "Enrich one recipe ingredient line for gram-resolution planning.\n"
    "resolution_paths (ordered, first success wins) may include:\n"
    "- embedded_mass: parenthetical mass like (8 oz.)\n"
    "- explicit_mass, explicit_volume, count_portion\n"
    "- parenthetical_mass_override: parenthetical total weight overrides count\n"
    "flags (non-exclusive): vague_amount, micro_amount, ambiguous_quantity_accepted, compound_ingredient, "
    "negligible_calorie_compound\n"
    "For dash/pinch units use explicit_volume (not count_portion).\n"
    "Set authoritative_mass_is_total=true when parenthetical mass is total weight, not per-piece.\n"
    "For compounds (parsley and chives), set is_compound=true and list components.\n"
    "certainty 0.0-1.0; rationale one short sentence."
)

RESPONSE_SCHEMA = {
    "name": "line_enrichment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "resolution_paths": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "embedded_mass",
                        "explicit_mass",
                        "explicit_volume",
                        "count_portion",
                        "parenthetical_mass_override",
                    ],
                },
            },
            "flags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "vague_amount",
                        "micro_amount",
                        "ambiguous_quantity_accepted",
                        "compound_ingredient",
                        "negligible_calorie_compound",
                    ],
                },
            },
            "quantity": {"type": ["number", "null"]},
            "unit": {"type": ["string", "null"]},
            "embedded_mass_qty": {"type": ["number", "null"]},
            "embedded_mass_unit": {"type": ["string", "null"]},
            "count_size_tokens": {"type": "array", "items": {"type": "string"}},
            "count_unit_tokens": {"type": "array", "items": {"type": "string"}},
            "is_compound": {"type": "boolean"},
            "components": {"type": "array", "items": {"type": "string"}},
            "authoritative_mass_is_total": {"type": "boolean"},
            "negligible_calories": {"type": "boolean"},
            "certainty": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": [
            "resolution_paths",
            "flags",
            "quantity",
            "unit",
            "embedded_mass_qty",
            "embedded_mass_unit",
            "count_size_tokens",
            "count_unit_tokens",
            "is_compound",
            "components",
            "authoritative_mass_is_total",
            "negligible_calories",
            "certainty",
            "rationale",
        ],
        "additionalProperties": False,
    },
}


def build_user_prompt(ingredient: str, rules_plan: ResolutionPlan) -> str:
    return (
        f"INGREDIENT: {ingredient}\n\n"
        f"RULES_PLAN: {json.dumps(rules_plan.to_dict(), default=str)}\n\n"
        "Refine resolution_paths, flags, and authoritative amounts."
    )


def validate_response(parsed: dict[str, Any]) -> str | None:
    try:
        c = float(parsed["certainty"])
        if c < 0 or c > 1:
            return "certainty_out_of_range"
    except (TypeError, ValueError, KeyError):
        return "invalid_certainty"
    return None


async def enrich_one_async(
    client: Any,
    model: str,
    ingredient: str,
    rules_plan: ResolutionPlan,
) -> dict[str, Any]:
    async def _invoke(extra: str | None = None):
        user = build_user_prompt(ingredient, rules_plan)
        if extra:
            user += f"\n\n{extra}"
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
            temperature=0,
        )
        content = resp.choices[0].message.content
        return json.loads(content), content, resp.usage

    prompt_tokens = completion_tokens = 0
    error: str | None = None
    parsed: dict[str, Any] = {}
    raw_response: str | None = None

    try:
        parsed, raw_response, usage = await _invoke()
        prompt_tokens += usage.prompt_tokens
        completion_tokens += usage.completion_tokens
        validation_error = validate_response(parsed)
        if validation_error:
            parsed, raw_response, usage = await _invoke(
                f"Validation failed ({validation_error}). Fix and resubmit."
            )
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
            validation_error = validate_response(parsed)
            if validation_error:
                error = validation_error
    except AllKeysExhaustedError:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
    cost = prompt_tokens / 1e6 * pricing["input"] + completion_tokens / 1e6 * pricing["output"]

    return {
        "ingredient_norm": normalize_ingredient_key(ingredient),
        "ingredient_raw": ingredient,
        "enrichment": parsed if not error else {},
        "certainty": parsed.get("certainty"),
        "rationale": parsed.get("rationale"),
        "error": error,
        "response": raw_response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "price_estimate_usd": round(cost, 8),
    }


def run_line_enrichment_sync(
    items: list[tuple[str, dict[str, Any]]],
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 8,
    progress_writer: Any = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Enrich lines that need LLM; items are (ingredient, parse_fields)."""
    from openai_fallback import get_async_openai_client

    unique: dict[str, tuple[str, ResolutionPlan]] = {}
    for ingredient, fields in items:
        line = str(ingredient)
        key = normalize_ingredient_key(line)
        if not key or key in unique:
            continue
        rules_plan = build_resolution_plan(fields, ingredient_raw=line)
        if needs_line_enrichment(line, fields, rules_plan):
            unique[key] = (line, rules_plan)

    if progress_writer is not None and unique:
        progress_writer.add_prompt_budget(len(unique))

    async def _run():
        client = get_async_openai_client()
        sem = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = []
        total = len(unique)
        done = 0

        async def _one(key: str):
            nonlocal done
            async with sem:
                raw, plan = unique[key]
                row = await enrich_one_async(client, model, raw, plan)
                done += 1
                if progress_writer is not None:
                    progress_writer.record_enrichment_row(
                        done=done,
                        total=total,
                        ingredient_norm=key,
                        error=row.get("error"),
                    )
                return row

        tasks = [asyncio.create_task(_one(k)) for k in unique]
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)
        return results

    rows = asyncio.run(_run()) if unique else []
    cache = {r["ingredient_norm"]: r for r in rows}
    return cache, rows


def apply_enrichment_to_plan(
    ingredient: str,
    parse_fields: dict[str, Any],
    *,
    enrichment_cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[ResolutionPlan, str, dict[str, Any]]:
    """Build plan from rules; merge LLM enrichment when cached."""
    line = str(ingredient)
    rules_plan = build_resolution_plan(parse_fields, ingredient_raw=line)
    meta: dict[str, Any] = {
        "enrichment_source": "rules",
        "line_enrichment_certainty": None,
        "line_enrichment_rationale": None,
    }
    if not enrichment_cache or not needs_line_enrichment(line, parse_fields, rules_plan):
        return rules_plan, "rules", meta

    key = normalize_ingredient_key(line)
    llm_row = enrichment_cache.get(key)
    if not llm_row or llm_row.get("error") or not llm_row.get("enrichment"):
        return rules_plan, "rules", meta

    plan = build_resolution_plan(
        parse_fields,
        ingredient_raw=line,
        enrichment=llm_row["enrichment"],
    )
    meta.update({
        "enrichment_source": "llm",
        "line_enrichment_certainty": llm_row.get("certainty"),
        "line_enrichment_rationale": llm_row.get("rationale"),
        "line_enrichment": llm_row["enrichment"],
    })
    return plan, "llm", meta
