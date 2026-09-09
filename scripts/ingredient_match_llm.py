"""LLM-judge ingredient matching pilot (async + Supabase checkpoints + MLflow).

For each ingredient line in a random sample of full-dataset RecipeNLG recipes:

1. Retrieve a union candidate set (full-text lexical + global semantic) from the
   `food_4macro` USDA catalog, with lexical/semantic/staged scores.
2. Filter the recipe directions to steps that mention the ingredient.
3. Ask an OpenAI model to pick the best `fdc_id` for nutrition, with a short
   rationale and a certainty score (structured JSON output).

Every ingredient gets exactly one LLM call (no skipping). Candidate retrieval and
staged scoring are CPU/numpy work done synchronously up front; the LLM calls are
I/O-bound and run concurrently via asyncio + AsyncOpenAI. Results are checkpointed
incrementally into Supabase (`inference` schema) as calls complete, so an abort
loses at most one in-flight batch. Local CSV/JSON reports are also written.

One script execution = one Supabase experiment row and one MLflow run. Per-LLM-call
rows land in `inference.match_inferences_0`. MLflow runs are logged under a single
experiment (default `ingredient_match_llm`; override with `--mlflow-experiment`).

Run:
    uv run python scripts/ingredient_match_llm.py --n-recipes 100 --seed 42
    uv run python scripts/ingredient_match_llm.py --sample-manifest Data/recipes/mvp_sample_1000.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from db import load_dotenv
import inference_store
from budget_guard import BudgetConfig, BudgetGuard
from ingredient_match_staged import (
    LLMRetrievalConfig,
    StagedFoodIndex,
    StagedMatchConfig,
    batched_dequant_similarities,
    match_query,
    query_from_parsed_row,
    retrieve_llm_candidates,
)
from ingredient_query_cache import (
    DEFAULT_WORK_DIR as FOOD_CACHE_DIR,
    load_or_build_food_artifacts,
    load_or_build_recipe_artifacts,
)
from load_food_4macro import load_food_4macro
from openai_fallback import AllKeysExhaustedError
from progress_utils import iter_progress
from recipe_directions import parse_directions_list, relevant_direction_steps
from recipe_match_summary import summarize_recipe_matches
from sample_recipes import DEFAULT_MVP_MANIFEST, load_sampled_recipes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLM_WORK_DIR = ROOT / "scratch" / "recipe_matching_llm_100"
MLFLOW_DIR = ROOT / "mlruns"
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT / "mlartifacts"
# Default MLflow experiment; each script execution adds one run under this name.
MLFLOW_EXPERIMENT = "ingredient_match_llm"

PROMPT_VERSION = "v2"
# Supabase tables: one experiment row per execution, one inference row per LLM call.
EXPERIMENTS_TABLE = "match_experiments_0"
INFERENCES_TABLE = "match_inferences_0"

# Synthetic "non-caloric / negligible" food the judge may pick when an ingredient
# carries no meaningful calories and no real USDA entry matches. Injected into every
# candidate prompt (it is NOT in food_4macro, so it never enters retrieval). Must stay
# in sync with the row created by sql/31_add_sentinel_food.sql.
SENTINEL_FDC_ID = 999000001
SENTINEL_DATA_TYPE = "sentinel"
SENTINEL_DESCRIPTION = "NON-CALORIC OR NEGLIGIBLE INGREDIENT (ice, plain salt, garnish)"

# Separate sentinel for generic water lines (moisture tracking). sql/35_add_water_sentinel_food.sql
WATER_SENTINEL_FDC_ID = 999000002
WATER_SENTINEL_DATA_TYPE = "sentinel"
WATER_SENTINEL_DESCRIPTION = (
    "WATER (recipe moisture; track separately from negligible salt/garnish)"
)

SENTINEL_FDC_IDS = frozenset({SENTINEL_FDC_ID, WATER_SENTINEL_FDC_ID})

# OpenAI list pricing (USD per 1M tokens). Update if pricing changes.
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You match one recipe ingredient to the best USDA FoodData Central entry (fdc_id) "
    "for nutrition lookup.\n"
    "Each candidate line is formatted `fdc_id | description | L | S`:\n"
    "  L = lexical score (word/text overlap between the ingredient and the description).\n"
    "  S = semantic score (meaning similarity from embeddings).\n"
    "Both range 0-1; higher means a closer match. Treat them as evidence, not absolute "
    "truth — read the descriptions yourself.\n"
    "Selection rules:\n"
    "- Choose best identity; if none exact, closest same-type substitute (e.g. white wine vinegar→distilled vinegar).\n"
    "- Abstain (fdc_id null) only when no plausible substitute AND expected contribution ≥20 kcal for this amount.\n"
    "- negligible_calories: true only if this qty likely contributes <20 kcal. Never for flour, sugar, butter, oil.\n"
    "- 'Raw, generic, unbranded' is only a tie-breaker among same-food candidates.\n"
    f"- If no candidate refers to the same food BUT the ingredient is essentially non-caloric "
    f"or nutritionally negligible (e.g. ice, plain salt, a garnish), pick fdc_id "
    f"{SENTINEL_FDC_ID} (the '{SENTINEL_DESCRIPTION}' entry) and set certainty to reflect how "
    f"sure you are that it carries no meaningful calories. Do not report certainty 0 for "
    f"something inconsequential.\n"
    f"- Plain recipe water (water, cold/hot/boiling water): pick fdc_id "
    f"{WATER_SENTINEL_FDC_ID} ('{WATER_SENTINEL_DESCRIPTION}'). Never use the negligible "
    f"sentinel for water.\n"
    f"- Set fdc_id to null ONLY when no candidate fits AND the ingredient is NOT non-caloric "
    f"(i.e. it has real calories but none of the candidates represent it).\n"
    "Use the recipe steps only as context for how the ingredient is used. Output JSON only."
)

RESPONSE_SCHEMA = {
    "name": "ingredient_match",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "fdc_id": {
                "type": ["integer", "null"],
                "description": "Best or closest-substitute fdc_id; null only if ≥20 kcal expected and no fit.",
            },
            "certainty": {
                "type": "number",
                "description": "0-1 joint confidence in fdc match/substitute and portion-unit fit.",
            },
            "rationale": {
                "type": "string",
                "description": "<= 20 words.",
            },
            "matched_portion_id": {
                "type": ["integer", "null"],
                "description": "USDA food_portion.id when volume/count and fit>0; else null.",
            },
            "negligible_calories": {
                "type": "boolean",
                "description": "True only if this ingredient amount likely contributes <20 kcal total.",
            },
        },
        "required": ["fdc_id", "certainty", "rationale", "matched_portion_id", "negligible_calories"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Food index + prompt construction
# ---------------------------------------------------------------------------


def build_food_index(
    food_cache_dir: Path,
    config: StagedMatchConfig,
    *,
    force: bool = False,
    show_progress: bool = True,
) -> StagedFoodIndex:
    """Load cached food_4macro embeddings and assemble the staged index."""
    food_raw = load_food_4macro()
    print(f"food_4macro rows: {len(food_raw):,}", flush=True)
    food_name_emb, food_prep_emb, food_dequant_emb = load_or_build_food_artifacts(
        food_raw, food_cache_dir, force=force, show_progress=show_progress
    )[1:4]
    return StagedFoodIndex.from_catalog(
        food_raw,
        name_embeddings=food_name_emb,
        prep_embeddings=food_prep_emb,
        dequant_embeddings=food_dequant_emb,
        config=config,
        show_progress=show_progress,
    )


def _append_sentinel_candidate(prompt_candidates: pd.DataFrame) -> pd.DataFrame:
    """Append the non-caloric sentinel row so the judge can always select it.

    The sentinel is not retrieved (it has no embedding and is absent from
    food_4macro), so it carries lexical/semantic/staged scores of 0 and a rank
    just past the retrieved candidates. It is flagged ``in_llm_prompt`` so it
    flows into the prompt, valid fdc_id set, and description lookup.
    """
    next_rank = (int(prompt_candidates["rank"].max()) + 1) if not prompt_candidates.empty else 1
    sentinel = {
        "rank": next_rank,
        "fdc_id": SENTINEL_FDC_ID,
        "data_type": SENTINEL_DATA_TYPE,
        "description": SENTINEL_DESCRIPTION,
        "lexical_dequant": 0.0,
        "dequant_sem": 0.0,
        "retrieval_score": 0.0,
        "staged_final_score": 0.0,
        "staged_base_score": 0.0,
        "staged_prep_score": 0.0,
        "is_staged_top1": False,
        "in_llm_prompt": True,
    }
    return pd.concat([prompt_candidates, pd.DataFrame([sentinel])], ignore_index=True)


def _append_water_sentinel_candidate(prompt_candidates: pd.DataFrame) -> pd.DataFrame:
    """Append the water moisture sentinel (distinct from negligible sentinel)."""
    if not prompt_candidates.empty and (
        prompt_candidates["fdc_id"].astype(int) == WATER_SENTINEL_FDC_ID
    ).any():
        return prompt_candidates
    next_rank = (int(prompt_candidates["rank"].max()) + 1) if not prompt_candidates.empty else 1
    sentinel = {
        "rank": next_rank,
        "fdc_id": WATER_SENTINEL_FDC_ID,
        "data_type": WATER_SENTINEL_DATA_TYPE,
        "description": WATER_SENTINEL_DESCRIPTION,
        "lexical_dequant": 0.0,
        "dequant_sem": 0.0,
        "retrieval_score": 0.0,
        "staged_final_score": 0.0,
        "staged_base_score": 0.0,
        "staged_prep_score": 0.0,
        "is_staged_top1": False,
        "in_llm_prompt": True,
    }
    return pd.concat([prompt_candidates, pd.DataFrame([sentinel])], ignore_index=True)


def format_candidate_block(prompt_candidates: pd.DataFrame, max_chars: int) -> str:
    if prompt_candidates.empty:
        return "(none)"
    lines = []
    for row in prompt_candidates.itertuples(index=False):
        desc = str(row.description)[:max_chars]
        lines.append(
            f"{row.fdc_id} | {desc} | {row.lexical_dequant:.2f} | {row.dequant_sem:.2f}"
        )
    return "\n".join(lines)


def build_user_prompt(
    ingredient: str,
    name: str,
    preparation: str,
    unit: str,
    prompt_candidates: pd.DataFrame,
    steps: list[str],
    max_chars: int,
) -> str:
    parts = [
        f"INGREDIENT: {ingredient}",
        f"PARSED: name={name or '-'}; prep={preparation or '-'}; unit={unit or '-'}",
        "",
        "CANDIDATES (fdc_id | description | L | S):",
        format_candidate_block(prompt_candidates, max_chars),
    ]
    if steps:
        parts.append("")
        parts.append("STEPS:")
        for i, step in enumerate(steps, 1):
            parts.append(f"{i}. {step}")
    parts.append("")
    parts.append("Select the best fdc_id for this ingredient in this recipe.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Synchronous pre-compute: retrieval, staged baseline, prompt, candidates
# ---------------------------------------------------------------------------


def precompute_payloads(
    parsed: pd.DataFrame,
    name_emb,
    prep_emb,
    dequant_emb,
    food_index: StagedFoodIndex,
    directions_by_recipe: dict[int, list[str]],
    retr_config: LLMRetrievalConfig,
    *,
    limit: int | None = None,
    chunk_size: int = 256,
    progress_writer: Any = None,
) -> list[dict[str, Any]]:
    """All CPU work (retrieval + staged scoring + prompt assembly) per ingredient.

    Semantic similarities are computed in batched matrix-matrix products (one per
    `chunk_size` ingredients) rather than a matrix-vector product per ingredient.
    """
    import numpy as np

    n = len(parsed) if limit is None else min(limit, len(parsed))
    payloads: list[dict[str, Any]] = []

    n_chunks = (n + chunk_size - 1) // chunk_size
    print(f"Batched retrieval: {n} ingredients in {n_chunks} chunk(s) of {chunk_size} "
          f"(food matrix {food_index.dequant_matrix.shape if food_index.dequant_matrix is not None else 'none'})",
          flush=True)

    show_progress = progress_writer is None or progress_writer.show_secondary_progress
    progress = iter_progress(range(n), total=n, desc="Retrieval + prompts", enabled=show_progress)
    progress_iter = iter(progress)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # One BLAS matmul for the whole chunk: (n_food, chunk).
        sims_chunk = batched_dequant_similarities(food_index, dequant_emb[start:end])

        for i in range(start, end):
            next(progress_iter, None)
            row = parsed.iloc[i]
            recipe_id = int(row["recipe_id"])
            ingredient_idx = int(row["ingredient_idx"])
            ingredient = str(row["ingredient"])
            name = str(row.get("name") or "")
            preparation = str(row.get("preparation") or "")
            unit = str(row.get("unit") or "")

            query = query_from_parsed_row(row, name_emb[i], prep_emb[i], dequant_emb[i])
            staged = match_query(query, food_index)
            staged_top1 = staged.get("matched_fdc_id")

            sims_i = sims_chunk[:, i - start] if sims_chunk.size else None
            cand_df = retrieve_llm_candidates(
                query, food_index, retr_config,
                staged_top1_fdc_id=staged_top1, precomputed_sims=sims_i,
            )
            n_lexical_pool = int(cand_df.attrs.get("n_lexical_pool", 0)) if not cand_df.empty else 0
            n_semantic_pool = int(cand_df.attrs.get("n_semantic_pool", 0)) if not cand_df.empty else 0

            prompt_candidates = cand_df[cand_df["in_llm_prompt"]] if not cand_df.empty else cand_df
            n_retrieved_candidates = int(len(prompt_candidates))
            # Always offer the non-caloric sentinel as a selectable escape hatch (it is not
            # in food_4macro / retrieval, so inject it directly into the prompt candidate set).
            prompt_candidates = _append_sentinel_candidate(prompt_candidates)
            valid_fdc_ids = (
                set(int(x) for x in prompt_candidates["fdc_id"]) if not prompt_candidates.empty else set()
            )
            prompt_desc = (
                {int(r.fdc_id): str(r.description) for r in prompt_candidates.itertuples(index=False)}
                if not prompt_candidates.empty else {}
            )

            steps = relevant_direction_steps(ingredient, directions_by_recipe.get(recipe_id, []))
            user_prompt = build_user_prompt(
                ingredient, name, preparation, unit,
                prompt_candidates, steps, retr_config.description_max_chars,
            )

            top10 = cand_df.head(retr_config.top10_size) if not cand_df.empty else cand_df
            top10_rows = [
                {
                    "rank": int(c.rank), "fdc_id": int(c.fdc_id), "data_type": c.data_type,
                    "description": c.description, "lexical_dequant": float(c.lexical_dequant),
                    "dequant_sem": float(c.dequant_sem), "retrieval_score": float(c.retrieval_score),
                    "staged_final_score": float(c.staged_final_score),
                    "staged_base_score": float(c.staged_base_score),
                    "staged_prep_score": float(c.staged_prep_score),
                    "in_llm_prompt": bool(c.in_llm_prompt), "is_staged_top1": bool(c.is_staged_top1),
                }
                for c in top10.itertuples(index=False)
            ] if not top10.empty else []

            staged_in_candidates = (
                staged_top1 is not None and not cand_df.empty
                and bool((cand_df["fdc_id"] == staged_top1).any())
            )
            staged_in_top10 = (
                staged_top1 is not None and not top10.empty
                and bool((top10["fdc_id"] == staged_top1).any())
            )

            payloads.append({
                "recipe_id": recipe_id, "ingredient_idx": ingredient_idx,
                "ingredient": ingredient, "name": name, "preparation": preparation,
                "dequantified": str(row.get("dequantified") or ""), "unit": unit,
                "staged_fdc_id": staged_top1,
                "staged_description": staged.get("matched_description"),
                "staged_match_score": staged.get("match_score"),
                "staged_match_quality": staged.get("match_quality"),
                "staged_base_score": staged.get("base_score"),
                "staged_prep_score": staged.get("prep_score"),
                "n_candidates_llm": n_retrieved_candidates,
                "n_lexical_pool": n_lexical_pool, "n_semantic_pool": n_semantic_pool,
                "staged_top1_in_llm_candidates": staged_in_candidates,
                "staged_top1_in_top10": staged_in_top10,
                "n_relevant_steps": len(steps), "relevant_steps": " | ".join(steps),
                "user_prompt": user_prompt, "valid_fdc_ids": valid_fdc_ids,
                "prompt_desc": prompt_desc, "top10_rows": top10_rows,
            })

    return payloads


# ---------------------------------------------------------------------------
# Async LLM judging
# ---------------------------------------------------------------------------


async def judge_async(
    client: Any,
    model: str,
    user_prompt: str,
    valid_fdc_ids: set[int],
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """One judge call with a single retry on invalid fdc_id."""
    sys_prompt = system_prompt or SYSTEM_PROMPT

    async def _invoke(extra: str | None):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt + (f"\n\n{extra}" if extra else "")},
        ]
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
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
        parsed, raw_response, usage = await _invoke(None)
        prompt_tokens += usage.prompt_tokens
        completion_tokens += usage.completion_tokens
        fdc_id = parsed.get("fdc_id")
        if fdc_id is not None and int(fdc_id) not in valid_fdc_ids:
            hint = (
                "Your previous fdc_id was not in the candidate list. "
                "Choose only from the listed fdc_id values, or null."
            )
            parsed, raw_response, usage = await _invoke(hint)
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
            fdc_id = parsed.get("fdc_id")
            if fdc_id is not None and int(fdc_id) not in valid_fdc_ids:
                error = "invalid_fdc_id"
                parsed["fdc_id"] = None
    except AllKeysExhaustedError:
        raise
    except Exception as exc:  # network / parse / API error
        error = f"{type(exc).__name__}: {exc}"

    matched_portion_id = parsed.get("matched_portion_id")
    if matched_portion_id is not None:
        try:
            matched_portion_id = int(matched_portion_id)
        except (TypeError, ValueError):
            matched_portion_id = None

    negligible = bool(parsed.get("negligible_calories", False))

    return {
        "fdc_id": parsed.get("fdc_id"),
        "certainty": parsed.get("certainty"),
        "rationale": parsed.get("rationale"),
        "matched_portion_id": matched_portion_id,
        "negligible_calories": negligible,
        "response": raw_response,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": error,
    }


def assemble_rows(
    payload: dict[str, Any],
    judge: dict[str, Any],
    *,
    run_id: str,
    run_name: str,
    model: str,
    pricing: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the experiment row + candidate rows from a payload and judge result."""
    llm_fdc_id = judge["fdc_id"]
    llm_fdc_id = int(llm_fdc_id) if llm_fdc_id is not None else None
    llm_desc = payload["prompt_desc"].get(llm_fdc_id) if llm_fdc_id is not None else None

    cost = (
        judge["prompt_tokens"] / 1e6 * pricing["input"]
        + judge["completion_tokens"] / 1e6 * pricing["output"]
    )
    staged_top1 = payload["staged_fdc_id"]
    ts = datetime.now(timezone.utc)
    ts_parts = inference_store.split_timestamp(ts)

    exp = {
        "run_id": run_id, "run_name": run_name, "model": model,
        "recipe_id": payload["recipe_id"], "ingredient_idx": payload["ingredient_idx"],
        "ingredient": payload["ingredient"], "name": payload["name"],
        "preparation": payload["preparation"], "dequantified": payload["dequantified"],
        "unit": payload["unit"],
        "system_prompt": SYSTEM_PROMPT, "prompt": payload["user_prompt"],
        "response": judge["response"],
        "llm_fdc_id": llm_fdc_id, "llm_description": llm_desc,
        "llm_certainty": judge["certainty"], "llm_rationale": judge["rationale"],
        "llm_agrees_with_staged": llm_fdc_id is not None and llm_fdc_id == staged_top1,
        "llm_abstained": llm_fdc_id is None and judge["error"] is None,
        "llm_error": judge["error"],
        "staged_fdc_id": staged_top1, "staged_description": payload["staged_description"],
        "staged_match_score": payload["staged_match_score"],
        "staged_match_quality": payload["staged_match_quality"],
        "staged_base_score": payload["staged_base_score"],
        "staged_prep_score": payload["staged_prep_score"],
        "n_candidates_llm": payload["n_candidates_llm"],
        "n_lexical_pool": payload["n_lexical_pool"],
        "n_semantic_pool": payload["n_semantic_pool"],
        "staged_top1_in_llm_candidates": payload["staged_top1_in_llm_candidates"],
        "staged_top1_in_top10": payload["staged_top1_in_top10"],
        "n_relevant_steps": payload["n_relevant_steps"],
        "relevant_steps": payload["relevant_steps"],
        "prompt_tokens": judge["prompt_tokens"],
        "completion_tokens": judge["completion_tokens"],
        "total_tokens": judge["prompt_tokens"] + judge["completion_tokens"],
        "price_estimate_usd": round(cost, 8),
        **ts_parts,
    }

    cand_rows = []
    for c in payload["top10_rows"]:
        cand_rows.append({
            "run_id": run_id, "recipe_id": payload["recipe_id"],
            "ingredient_idx": payload["ingredient_idx"], "rank": c["rank"],
            "fdc_id": c["fdc_id"], "data_type": c["data_type"],
            "description": c["description"], "lexical_dequant": c["lexical_dequant"],
            "dequant_sem": c["dequant_sem"], "retrieval_score": c["retrieval_score"],
            "staged_final_score": c["staged_final_score"],
            "staged_base_score": c["staged_base_score"],
            "staged_prep_score": c["staged_prep_score"],
            "in_llm_prompt": c["in_llm_prompt"], "is_staged_top1": c["is_staged_top1"],
            "is_llm_pick": llm_fdc_id is not None and c["fdc_id"] == llm_fdc_id,
            "ts": ts,
        })
    return exp, cand_rows


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


async def run_judging(
    payloads: list[dict[str, Any]],
    *,
    run_id: str,
    run_name: str,
    model: str,
    pricing: dict[str, float],
    concurrency: int,
    flush_every: int,
    conn,
    use_supabase: bool,
    budget_config: BudgetConfig | None = None,
    log_every: int = 25,
    heartbeat_sec: float = 15.0,
    verbose: bool = True,
    assemble_fn=assemble_rows,
    system_prompt: str | None = None,
    disk_checkpoint_path: Any = None,
    disk_flush_every: int = 100,
    total_dataset_lines: int | None = None,
    progress_writer: Any = None,
    dequant_cache_runtime: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """Dispatch concurrent LLM calls; checkpoint in batches; stream observability.

    Observability:
    - per-call line (when `verbose`): pick, agreement mark, certainty, latency
    - aggregate line every `log_every` completions: rate, ETA, cost, tokens,
      agreement/abstain/error rates, latency p50/p95, in-flight count
    - heartbeat every `heartbeat_sec` seconds even if nothing completed (stalls)
    DB writes are batched via execute_values and flushed every `flush_every`.
    """
    from openai_fallback import AllKeysExhaustedError, get_async_openai_client, get_key_pool_status

    client = get_async_openai_client()
    sem = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    db_executor = ThreadPoolExecutor(max_workers=1) if use_supabase else None

    guard = (
        BudgetGuard(conn, run_id, budget_config)
        if (use_supabase and budget_config is not None)
        else None
    )
    breaker: dict[str, Any] = {
        "tripped": False,
        "verdict": None,
        "window": None,
        "skipped": 0,
        "reason": None,
        "key_status": None,
    }

    all_exp: list[dict[str, Any]] = []
    all_cand: list[dict[str, Any]] = []
    buf_exp: list[dict[str, Any]] = []
    buf_cand: list[dict[str, Any]] = []
    total = len(payloads)

    stats = {
        "done": 0, "active": 0, "cost": 0.0, "tokens": 0,
        "agree": 0, "abstain": 0, "errors": 0, "checkpointed": 0,
        "no_portion": 0,
    }
    latencies: list[float] = []
    t_start = time.time()
    disk_path = Path(disk_checkpoint_path) if disk_checkpoint_path else None
    stream_via_writer = progress_writer is not None

    def _inferred_at_str(exp: dict[str, Any]) -> str:
        raw = exp.get("inferred_at")
        if raw:
            return str(raw)
        ts = exp.get("ts")
        if hasattr(ts, "strftime"):
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        ts_time = exp.get("ts_time")
        if ts_time:
            return str(ts_time)
        return ""

    def agg_line(tag: str) -> str:
        done = stats["done"]
        rate = done / max(time.time() - t_start, 1e-6)
        eta = (total - done) / rate if rate > 0 else float("inf")
        p50 = _pct(latencies, 0.50)
        p95 = _pct(latencies, 0.95)
        nop = stats["no_portion"]
        nop_run = f"no_portion {nop / max(done, 1):.1%} ({nop}/{done})"
        nop_overall = ""
        if total_dataset_lines and disk_path and disk_path.is_file():
            try:
                from judge_checkpoint import load_judge_checkpoint, no_portion_rate

                on_disk, _, rate_all = no_portion_rate(load_judge_checkpoint(disk_path))
                nop_overall = f" | overall {rate_all:.1%} ({on_disk}/{total_dataset_lines})"
            except Exception:
                pass
        return (
            f"{tag} {done}/{total} ({done / total:.0%}) | {rate:.1f}/s ETA {eta:4.0f}s | "
            f"${stats['cost']:.4f} {stats['tokens']:,}tok | "
            f"agree {stats['agree'] / max(done,1):.0%} abstain {stats['abstain'] / max(done,1):.0%} "
            f"err {stats['errors']} | {nop_run}{nop_overall} | "
            f"lat p50 {p50:.2f}s p95 {p95:.2f}s | "
            f"inflight {stats['active']}/{concurrency} | ckpt {stats['checkpointed']}"
        )

    async def worker(payload):
        # Breaker check before and after acquiring the slot: do not start new
        # calls once tripped (in-flight calls already past this point finish).
        if breaker["tripped"]:
            return payload, None
        async with sem:
            if breaker["tripped"]:
                return payload, None
            stats["active"] += 1
            t0 = time.perf_counter()
            precomputed = payload.get("hardcoded_judge")
            if precomputed is None and dequant_cache_runtime is not None:
                precomputed = dequant_cache_runtime.lookup(payload)
            if precomputed is not None:
                judge = dict(precomputed)
                judge["latency_sec"] = time.perf_counter() - t0
                stats["active"] -= 1
                return payload, judge
            try:
                judge = await judge_async(
                    client,
                    model,
                    payload["user_prompt"],
                    payload["valid_fdc_ids"],
                    system_prompt=system_prompt,
                )
            except AllKeysExhaustedError:
                if not breaker["tripped"]:
                    breaker["tripped"] = True
                    breaker["reason"] = "openai_keys_exhausted"
                    breaker["key_status"] = get_key_pool_status()
                    if verbose:
                        print(
                            "  [openai] All API keys exhausted — halting new LLM calls, "
                            "draining in-flight requests…",
                            flush=True,
                        )
                return payload, None
            finally:
                stats["active"] -= 1
            judge["latency_sec"] = time.perf_counter() - t0
        return payload, judge

    def flush(exp_rows, cand_rows):
        inference_store.upsert_inferences(conn, exp_rows)
        inference_store.upsert_candidates(conn, cand_rows)

    def flush_disk(exp_rows):
        if not disk_path or not exp_rows:
            return
        from judge_checkpoint import merge_judge_checkpoint, no_portion_rate

        n_disk = merge_judge_checkpoint(disk_path, exp_rows)
        on_disk = pd.read_parquet(disk_path)
        _, _, rate_all = no_portion_rate(on_disk)
        denom = total_dataset_lines or n_disk
        nop = int(on_disk["grams_status"].eq("no_portion").sum())
        if verbose:
            print(
                f"  .. checkpointed {len(exp_rows)} rows -> disk "
                f"(total {n_disk}/{denom}, no_portion {nop / max(denom, 1):.1%})",
                flush=True,
            )

    async def heartbeat():
        try:
            while True:
                await asyncio.sleep(heartbeat_sec)
                if verbose and stats["done"] < total:
                    print(agg_line("  [hb]"), flush=True)
        except asyncio.CancelledError:
            return

    tasks = [asyncio.create_task(worker(p)) for p in payloads]
    hb_task = asyncio.create_task(heartbeat())
    if verbose:
        print(f"Dispatched {total} judge tasks @ concurrency {concurrency}; "
              f"checkpoint every {flush_every}, log every {log_every}, heartbeat {heartbeat_sec:.0f}s",
              flush=True)

    try:
        for fut in asyncio.as_completed(tasks):
            payload, judge = await fut
            if judge is None:
                # Worker skipped because the breaker was already tripped.
                breaker["skipped"] += 1
                continue
            exp, cand_rows = assemble_fn(
                payload, judge, run_id=run_id, run_name=run_name, model=model, pricing=pricing
            )
            if dequant_cache_runtime is not None:
                dequant_cache_runtime.record_completion(payload, judge, exp)
            all_exp.append(exp)
            all_cand.extend(cand_rows)
            buf_exp.append(exp)
            buf_cand.extend(cand_rows)

            stats["done"] += 1
            stats["cost"] += exp["price_estimate_usd"]
            stats["tokens"] += exp["total_tokens"]
            if exp["llm_agrees_with_staged"]:
                stats["agree"] += 1
            if exp["llm_abstained"]:
                stats["abstain"] += 1
            if exp["llm_error"] is not None:
                stats["errors"] += 1
            if exp.get("grams_status") == "no_portion":
                stats["no_portion"] += 1
            latencies.append(judge.get("latency_sec", 0.0))

            if verbose:
                pick = exp["llm_fdc_id"]
                flag = "OK" if exp["llm_error"] is None else f"ERR:{exp['llm_error']}"
                mark = "=" if exp["llm_agrees_with_staged"] else ("~" if pick is not None else "X")
                ing = exp["ingredient"][:34]
                ts_label = _inferred_at_str(exp)
                ts_prefix = f"{ts_label} " if ts_label else ""
                print(
                    f"[{stats['done']:>4}/{total}] {ts_prefix}"
                    f"r{exp['recipe_id']}#{exp['ingredient_idx']} "
                    f"{ing!r:36} -> {str(pick):>8} {mark} cert={exp['llm_certainty']} "
                    f"{judge.get('latency_sec', 0.0):.2f}s {flag}",
                    flush=True,
                )

            if progress_writer is not None:
                progress_writer.record_judge_row(exp)

            if stats["done"] % log_every == 0 and verbose:
                print(agg_line(">>"), flush=True)
                if dequant_cache_runtime is not None:
                    cs = dequant_cache_runtime.summary()
                    print(
                        f"  [dequant_cache] initial={cs['initial_hits']} growth={cs['runtime_growth_hits']} "
                        f"llm={cs['llm_calls']} terms+={cs['terms_added_during_run']} "
                        f"total_terms={cs['final_terms']}",
                        flush=True,
                    )

            if (
                disk_path
                and not stream_via_writer
                and len(buf_exp) >= disk_flush_every
            ):
                await loop.run_in_executor(None, flush_disk, list(buf_exp))
                stats["checkpointed"] += len(buf_exp)
                buf_exp.clear()
                buf_cand.clear()

            if use_supabase and len(buf_exp) >= flush_every:
                await loop.run_in_executor(db_executor, flush, list(buf_exp), list(buf_cand))
                stats["checkpointed"] += len(buf_exp)
                print(f"  .. checkpointed {len(buf_exp)} rows -> Supabase "
                      f"(total {stats['checkpointed']}/{total})", flush=True)
                buf_exp.clear()
                buf_cand.clear()

            # Budget circuit breaker: every N completed calls, check DB spend.
            if guard is not None and not breaker["tripped"] and guard.should_check(stats["done"]):
                # Flush pending rows first so the spend query reflects all calls.
                if buf_exp:
                    await loop.run_in_executor(db_executor, flush, list(buf_exp), list(buf_cand))
                    stats["checkpointed"] += len(buf_exp)
                    buf_exp.clear()
                    buf_cand.clear()
                verdict = await loop.run_in_executor(db_executor, guard.check, stats["done"])
                print(
                    f"  [budget] after {verdict['calls_completed']} calls: "
                    f"past-day ${verdict['past_day_spend_usd']:.2f} | "
                    f"recent ${verdict['spend_since_last_usd']:.4f} over "
                    f"{verdict['seconds_since_last']:.0f}s = "
                    f"${verdict['rate_usd_per_min']:.3f}/min"
                    + (f" | TRIPPED: {verdict['reason']}" if verdict["tripped"] else " | ok"),
                    flush=True,
                )
                if verdict["tripped"]:
                    breaker["tripped"] = True
                    breaker["verdict"] = verdict
                    breaker["window"] = await loop.run_in_executor(
                        db_executor, guard.spend_window, budget_config.daily_limit_usd
                    )
                    print("  [budget] BREAKER TRIPPED — halting new LLM calls, "
                          "draining in-flight requests…", flush=True)

        if disk_path and buf_exp and not stream_via_writer:
            await loop.run_in_executor(None, flush_disk, list(buf_exp))
            stats["checkpointed"] += len(buf_exp)

        if progress_writer is not None:
            progress_writer.compact_parquet()

        if use_supabase and buf_exp:
            await loop.run_in_executor(db_executor, flush, list(buf_exp), list(buf_cand))
            stats["checkpointed"] += len(buf_exp)
            print(f"  .. checkpointed final {len(buf_exp)} rows -> Supabase "
                  f"(total {stats['checkpointed']}/{total})", flush=True)
    finally:
        hb_task.cancel()
        if db_executor is not None:
            db_executor.shutdown(wait=True)

    if verbose:
        print(agg_line("== final"), flush=True)
        if breaker["skipped"]:
            if breaker.get("reason") == "openai_keys_exhausted":
                print(
                    f"== OpenAI key pool exhausted; skipped {breaker['skipped']} un-started calls",
                    flush=True,
                )
            else:
                print(f"== budget breaker skipped {breaker['skipped']} un-started calls", flush=True)
    return all_exp, all_cand, (breaker if breaker["tripped"] else None)


# ---------------------------------------------------------------------------
# Reports (local + DB + MLflow)
# ---------------------------------------------------------------------------


def compute_report(
    matches_df: pd.DataFrame,
    recipes: pd.DataFrame,
    *,
    model: str,
    pricing: dict[str, float],
    elapsed: float,
) -> dict[str, Any]:
    n_calls = len(matches_df)
    prompt_total = int(matches_df["prompt_tokens"].sum())
    completion_total = int(matches_df["completion_tokens"].sum())
    input_cost = prompt_total / 1e6 * pricing["input"]
    output_cost = completion_total / 1e6 * pricing["output"]
    certainty = pd.to_numeric(matches_df["llm_certainty"], errors="coerce").dropna()

    def q(s, x):
        return round(float(s.quantile(x)), 4) if len(s) else None

    return {
        "model": model,
        "n_recipes": int(recipes["recipe_id"].nunique()),
        "n_ingredients": n_calls,
        "n_llm_calls": n_calls,
        "n_llm_errors": int(matches_df["llm_error"].notna().sum()),
        "elapsed_sec": round(elapsed, 1),
        "prompt_tokens_total": prompt_total,
        "completion_tokens_total": completion_total,
        "total_tokens": prompt_total + completion_total,
        "cost_input_usd": round(input_cost, 4),
        "cost_output_usd": round(output_cost, 4),
        "cost_total_usd": round(input_cost + output_cost, 4),
        "cost_avg_per_ingredient_usd": round((input_cost + output_cost) / max(n_calls, 1), 6),
        "prompt_tokens_avg": round(prompt_total / max(n_calls, 1), 1),
        "completion_tokens_avg": round(completion_total / max(n_calls, 1), 1),
        "abstain_rate": round(float(matches_df["llm_abstained"].mean()), 4),
        "error_rate": round(float(matches_df["llm_error"].notna().mean()), 4),
        "agreement_rate": round(float(matches_df["llm_agrees_with_staged"].mean()), 4),
        "staged_top1_in_llm_candidates_rate": round(
            float(matches_df["staged_top1_in_llm_candidates"].mean()), 4),
        "staged_top1_in_top10_rate": round(
            float(matches_df["staged_top1_in_top10"].mean()), 4),
        "certainty_mean": round(float(certainty.mean()), 4) if len(certainty) else None,
        "certainty_median": q(certainty, 0.5),
        "certainty_std": round(float(certainty.std()), 4) if len(certainty) > 1 else None,
        "certainty_p01": q(certainty, 0.01),
        "certainty_p05": q(certainty, 0.05),
        "certainty_p10": q(certainty, 0.10),
        "certainty_p90": q(certainty, 0.90),
        "n_disagreements": int((~matches_df["llm_agrees_with_staged"]).sum()),
        "pricing_per_1m": pricing,
    }


def write_local_reports(
    work_dir: Path,
    matches_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    recipes: pd.DataFrame,
    report: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        "matches": work_dir / "ingredient_matches_llm.csv",
        "candidates": work_dir / "ingredient_candidates_top10.csv",
        "summary": work_dir / "recipe_match_summary_llm.csv",
        "cost": work_dir / "cost_report.json",
        "eval": work_dir / "llm_eval_summary.json",
        "manifest": work_dir / "llm_run_manifest.json",
    }
    # Drop bulky prompt/response columns from the flat CSV (kept in Supabase).
    csv_cols = [c for c in matches_df.columns if c not in ("system_prompt", "prompt", "response")]
    matches_df[csv_cols].to_csv(paths["matches"], index=False)
    candidates_df.to_csv(paths["candidates"], index=False)

    staged_for_summary = matches_df.rename(
        columns={"staged_match_quality": "match_quality", "staged_match_score": "match_score"}
    )
    summary = summarize_recipe_matches(staged_for_summary, recipes[["recipe_id"]])
    summary.to_csv(paths["summary"], index=False)

    paths["cost"].write_text(json.dumps(report, indent=2) + "\n")
    paths["eval"].write_text(json.dumps({
        k: report[k] for k in (
            "n_ingredients", "abstain_rate", "error_rate", "agreement_rate",
            "staged_top1_in_llm_candidates_rate", "staged_top1_in_top10_rate",
            "certainty_mean", "certainty_median", "certainty_std",
            "certainty_p01", "certainty_p05", "certainty_p10", "certainty_p90",
            "n_disagreements",
        )
    }, indent=2) + "\n")
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def log_to_mlflow(
    *,
    experiment_name: str,
    run_name: str,
    model: str,
    seed: int,
    concurrency: int,
    n_recipes: int,
    retr_config: LLMRetrievalConfig,
    pricing: dict[str, float],
    report: dict[str, Any],
    artifact_paths: dict[str, Path],
    run_id: str,
) -> str | None:
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover
        print(f"MLflow unavailable, skipping logging: {exc}", flush=True)
        return None

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "model": model,
            "model_version": model,
            "prompt_version": PROMPT_VERSION,
            "retrieval_version": "union_lex_sem_v1",
            "supabase_schema": "inference",
            "supabase_experiments_table": EXPERIMENTS_TABLE,
            "supabase_inferences_table": INFERENCES_TABLE,
            "supabase_run_id": run_id,
            "task": "ingredient_fdc_match",
        })
        params = {
            "model": model, "seed": seed, "concurrency": concurrency,
            "n_recipes": n_recipes, "prompt_version": PROMPT_VERSION,
            "price_input_per_1m": pricing["input"],
            "price_output_per_1m": pricing["output"],
        }
        params.update({f"retr_{k}": v for k, v in asdict(retr_config).items()})
        mlflow.log_params(params)

        metric_keys = [
            "n_ingredients", "n_llm_calls", "n_llm_errors", "elapsed_sec",
            "prompt_tokens_total", "completion_tokens_total", "total_tokens",
            "cost_input_usd", "cost_output_usd", "cost_total_usd",
            "cost_avg_per_ingredient_usd", "prompt_tokens_avg", "completion_tokens_avg",
            "abstain_rate", "error_rate", "agreement_rate",
            "staged_top1_in_llm_candidates_rate", "staged_top1_in_top10_rate",
            "certainty_mean", "certainty_median", "certainty_std",
            "certainty_p01", "certainty_p05", "certainty_p10", "certainty_p90",
            "n_disagreements",
        ]
        mlflow.log_metrics({k: report[k] for k in metric_keys if report.get(k) is not None})

        for path in artifact_paths.values():
            if path.is_file():
                mlflow.log_artifact(str(path))

        return run.info.run_id


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pilot(
    *,
    n_recipes: int,
    seed: int,
    model: str,
    work_dir: Path,
    food_cache_dir: Path,
    retr_config: LLMRetrievalConfig,
    match_config: StagedMatchConfig,
    concurrency: int,
    flush_every: int,
    limit: int | None,
    use_supabase: bool,
    use_mlflow: bool,
    mlflow_experiment: str,
    sample_manifest: Path | None = None,
    run_id: str | None = None,
    chunk_size: int = 256,
    log_every: int = 25,
    heartbeat_sec: float = 15.0,
    verbose: bool = True,
    budget_config: BudgetConfig | None = None,
) -> None:
    load_dotenv()
    work_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or uuid.uuid4().hex
    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{model}__seed{seed}__{stamp}"
    pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
    print(f"=== LLM ingredient judge ===\nMLflow experiment: {mlflow_experiment}\n"
          f"MLflow run: {run_name}\nrun_id={run_id}", flush=True)

    def phase(label: str) -> float:
        print(f"\n--- {label} ---", flush=True)
        return time.perf_counter()

    t = phase(
        f"Loading {n_recipes} recipes from manifest {sample_manifest}"
        if sample_manifest is not None
        else f"Sampling {n_recipes} recipes (seed={seed}) from full RecipeNLG"
    )
    ids_path = None if sample_manifest is not None else work_dir / "sampled_recipe_ids.json"
    recipes, recipe_ingredients, sampled_ids = load_sampled_recipes(
        n=n_recipes,
        seed=seed,
        ids_path=ids_path,
        sample_manifest=sample_manifest,
    )
    print(f"Loaded {len(recipes)} recipes -> {len(recipe_ingredients):,} ingredient lines "
          f"({time.perf_counter() - t:.1f}s)", flush=True)

    directions_by_recipe = {
        int(r.recipe_id): parse_directions_list(r.directions)
        for r in recipes.itertuples(index=False)
    }

    t = phase("Parsing + embedding sampled ingredient lines (batched encoder)")
    parsed, name_emb, prep_emb, dequant_emb, _meta = load_or_build_recipe_artifacts(
        recipe_ingredients, work_dir
    )
    print(f"Embeddings ready ({time.perf_counter() - t:.1f}s)", flush=True)

    t = phase("Building food index from cached embeddings")
    food_index = build_food_index(food_cache_dir, match_config)
    print(f"Food index ready ({time.perf_counter() - t:.1f}s)", flush=True)

    t = phase("Batched retrieval + staged scoring + prompt assembly")
    payloads = precompute_payloads(
        parsed, name_emb, prep_emb, dequant_emb, food_index,
        directions_by_recipe, retr_config, limit=limit, chunk_size=chunk_size,
    )
    print(f"Prepared {len(payloads)} ingredient prompts ({time.perf_counter() - t:.1f}s); "
          f"starting LLM judging (concurrency={concurrency})…", flush=True)

    conn = None
    if use_supabase:
        print("Connecting to Supabase and ensuring inference schema…", flush=True)
        conn = inference_store.open_connection()
        run_start = {
            "run_id": run_id, "run_name": run_name, "model": model,
            "mlflow_experiment": mlflow_experiment, "prompt_version": PROMPT_VERSION,
            "seed": seed, "n_recipes": int(recipes["recipe_id"].nunique()),
            "n_ingredients": len(payloads), "concurrency": concurrency,
            "retrieval_config": asdict(retr_config), "pricing": pricing,
            "sampled_recipe_ids": sampled_ids, "status": "running",
            "started_at": started_at,
        }
        inference_store.upsert_experiment(conn, run_start)

    if use_supabase and budget_config is not None:
        print(f"Budget guard active: daily cap ${budget_config.daily_limit_usd:.2f}, "
              f"rate cap ${budget_config.rate_limit_usd_per_min:.2f}/min, "
              f"check every {budget_config.check_every} calls", flush=True)
    elif budget_config is not None and not use_supabase:
        print("WARNING: budget guard requires Supabase; disabled (--no-supabase).", flush=True)

    started = time.time()
    all_exp, all_cand, breaker = asyncio.run(run_judging(
        payloads, run_id=run_id, run_name=run_name, model=model, pricing=pricing,
        concurrency=concurrency, flush_every=flush_every, conn=conn,
        use_supabase=use_supabase, budget_config=budget_config,
        log_every=log_every, heartbeat_sec=heartbeat_sec, verbose=verbose,
    ))
    elapsed = time.time() - started

    matches_df = pd.DataFrame(all_exp)
    candidates_df = pd.DataFrame(all_cand)
    report = compute_report(matches_df, recipes, model=model, pricing=pricing, elapsed=elapsed)

    manifest = {
        "run_id": run_id, "run_name": run_name,
        "timestamp_utc": started_at.isoformat(), "model": model, "seed": seed,
        "concurrency": concurrency, "n_recipes_requested": n_recipes,
        "n_recipes_loaded": int(recipes["recipe_id"].nunique()),
        "retrieval_config": asdict(retr_config), "pricing": pricing,
        "sampled_recipe_ids": sampled_ids,
    }
    artifact_paths = write_local_reports(
        work_dir, matches_df, candidates_df, recipes, report, manifest
    )
    print(f"\nWrote local reports -> {work_dir}", flush=True)
    print(f"Cost: ${report['cost_total_usd']:.4f} over {report['n_llm_calls']} calls "
          f"({report['total_tokens']:,} tokens); agreement "
          f"{report['agreement_rate']:.1%}, abstain {report['abstain_rate']:.1%}", flush=True)

    mlflow_run_id = None
    if use_mlflow:
        mlflow_run_id = log_to_mlflow(
            experiment_name=mlflow_experiment, run_name=run_name, model=model,
            seed=seed, concurrency=concurrency, n_recipes=n_recipes,
            retr_config=retr_config, pricing=pricing,
            report=report, artifact_paths=artifact_paths, run_id=run_id,
        )
        if mlflow_run_id:
            print(f"Logged MLflow run {mlflow_run_id} (experiment '{mlflow_experiment}')", flush=True)

    if use_supabase and conn is not None:
        finished_at = datetime.now(timezone.utc)
        run_finish = {
            "run_id": run_id, "run_name": run_name, "model": model,
            "mlflow_run_id": mlflow_run_id, "mlflow_experiment": mlflow_experiment,
            "prompt_version": PROMPT_VERSION, "seed": seed,
            "n_recipes": report["n_recipes"], "n_ingredients": report["n_ingredients"],
            "n_llm_calls": report["n_llm_calls"], "n_llm_errors": report["n_llm_errors"],
            "prompt_tokens_total": report["prompt_tokens_total"],
            "completion_tokens_total": report["completion_tokens_total"],
            "total_tokens": report["total_tokens"],
            "cost_input_usd": report["cost_input_usd"],
            "cost_output_usd": report["cost_output_usd"],
            "cost_total_usd": report["cost_total_usd"],
            "abstain_rate": report["abstain_rate"], "error_rate": report["error_rate"],
            "agreement_rate": report["agreement_rate"],
            "staged_top1_in_llm_candidates_rate": report["staged_top1_in_llm_candidates_rate"],
            "staged_top1_in_top10_rate": report["staged_top1_in_top10_rate"],
            "certainty_mean": report["certainty_mean"],
            "certainty_median": report["certainty_median"],
            "certainty_std": report["certainty_std"],
            "certainty_p01": report["certainty_p01"], "certainty_p05": report["certainty_p05"],
            "certainty_p10": report["certainty_p10"], "certainty_p90": report["certainty_p90"],
            "elapsed_sec": report["elapsed_sec"], "concurrency": concurrency,
            "retrieval_config": asdict(retr_config), "pricing": pricing,
            "sampled_recipe_ids": sampled_ids,
            "status": "aborted_budget" if breaker else "completed",
            "started_at": started_at, "finished_at": finished_at,
        }
        inference_store.upsert_experiment(conn, run_finish)
        conn.close()
        print(f"Supabase: wrote experiment row + {len(all_exp)} inferences + "
              f"{len(all_cand)} candidates "
              f"(run_id={run_id}, status={run_finish['status']})", flush=True)

    if breaker:
        _print_budget_abort(breaker, report, run_id, run_name)
        raise SystemExit(2)


def _print_budget_abort(
    breaker: dict[str, Any], report: dict[str, Any], run_id: str, run_name: str
) -> None:
    v = breaker["verdict"]
    w = breaker["window"] or {}
    bar = "=" * 72
    print("\n" + bar, flush=True)
    print("BUDGET LIMIT REACHED — run aborted after writing completed results", flush=True)
    print(bar, flush=True)
    print(f"  reason            : {v['reason']}", flush=True)
    print(f"  run_id            : {run_id}", flush=True)
    print(f"  run_name          : {run_name}", flush=True)
    print(f"  calls completed   : {report['n_llm_calls']} "
          f"(skipped {breaker['skipped']} un-started)", flush=True)
    print(f"  this run spend    : ${report['cost_total_usd']:.4f} "
          f"({report['total_tokens']:,} tokens)", flush=True)
    print(f"  past-day spend    : ${v['past_day_spend_usd']:.2f} (global, rolling 24h)", flush=True)
    print(f"  recent rate       : ${v['rate_usd_per_min']:.3f}/min "
          f"(${v['spend_since_last_usd']:.4f} over {v['seconds_since_last']:.0f}s)", flush=True)
    if w:
        amt = w.get("amount_requested_usd")
        if w.get("reached_amount"):
            print(f"  last ${amt:.0f} window : {w['window_start']}  ->  {w['window_end']}", flush=True)
            print(f"                      spent over {w['duration_min']} min "
                  f"(covered ${w['covered_usd']:.2f})", flush=True)
        else:
            print(f"  last ${amt:.0f} window : only ${w['lookback_total_usd']:.2f} spent in "
                  f"lookback window {w['window_start']} -> {w['window_end']}", flush=True)
    print("  Spend-check audit : inference.spend_checks_0 (run_id above)", flush=True)
    print(bar, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-judge ingredient matching pilot.")
    parser.add_argument("--n-recipes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_LLM_WORK_DIR)
    parser.add_argument("--food-cache-dir", type=Path, default=FOOD_CACHE_DIR)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent OpenAI calls (I/O-bound).")
    parser.add_argument("--flush-every", type=int, default=20,
                        help="Checkpoint to Supabase every N completed calls.")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Ingredients per batched semantic-similarity matmul.")
    parser.add_argument("--log-every", type=int, default=25,
                        help="Print an aggregate progress line every N completions.")
    parser.add_argument("--heartbeat-sec", type=float, default=15.0,
                        help="Seconds between heartbeat status lines (stall detection).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-call lines (keep aggregate + heartbeat).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap ingredient lines judged (smoke test).")
    parser.add_argument("--run-id", default=None, help="Reuse a run_id (upsert/resume).")
    parser.add_argument("--no-supabase", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--mlflow-experiment",
        default=MLFLOW_EXPERIMENT,
        help=f"MLflow experiment name (default: {MLFLOW_EXPERIMENT}). Each execution logs one run.",
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=None,
        help=(
            f"JSON manifest with recipe_ids (e.g. diversity MVP sample). "
            f"Default MVP path when set without value: {DEFAULT_MVP_MANIFEST.name} under Data/recipes/."
        ),
    )
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--semantic-floor", type=float, default=None)
    parser.add_argument("--lexical-floor", type=float, default=None)
    # Budget circuit breaker
    parser.add_argument("--daily-budget-usd", type=float, default=10.0,
                        help="Abort if global spend in the last 24h exceeds this.")
    parser.add_argument("--rate-limit-usd-min", type=float, default=0.50,
                        help="Abort if recent spend rate exceeds this (USD/minute).")
    parser.add_argument("--budget-check-every", type=int, default=100,
                        help="Run a DB spend check every N completed calls.")
    parser.add_argument("--no-budget-guard", action="store_true",
                        help="Disable the spend circuit breaker.")
    args = parser.parse_args()

    retr_config = LLMRetrievalConfig()
    if args.max_candidates is not None:
        retr_config.max_candidates = args.max_candidates
    if args.semantic_floor is not None:
        retr_config.semantic_score_floor = args.semantic_floor
    if args.lexical_floor is not None:
        retr_config.lexical_score_floor = args.lexical_floor

    budget_config = None if args.no_budget_guard else BudgetConfig(
        daily_limit_usd=args.daily_budget_usd,
        rate_limit_usd_per_min=args.rate_limit_usd_min,
        check_every=args.budget_check_every,
    )

    run_pilot(
        n_recipes=args.n_recipes, seed=args.seed, model=args.model,
        work_dir=args.work_dir, food_cache_dir=args.food_cache_dir,
        retr_config=retr_config, match_config=StagedMatchConfig(),
        concurrency=args.concurrency, flush_every=args.flush_every, limit=args.limit,
        use_supabase=not args.no_supabase, use_mlflow=not args.no_mlflow,
        mlflow_experiment=args.mlflow_experiment,
        sample_manifest=args.sample_manifest,
        run_id=args.run_id, chunk_size=args.chunk_size, log_every=args.log_every,
        heartbeat_sec=args.heartbeat_sec, verbose=not args.quiet,
        budget_config=budget_config,
    )


if __name__ == "__main__":
    main()
