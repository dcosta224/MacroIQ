"""Portion-aware LLM-judge ingredient matching (v3).

Same flow as ingredient_match_llm.py but with tiered retrieval that prefers
fdc_ids with volume/count portions for non-mass ingredients, and post-match
gram resolution metadata.

Run:
    uv run python scripts/ingredient_match_llm_portion.py --n-recipes 100 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from amount_kind import classify_from_parsed_row
from db import connect, load_dotenv
import inference_store
from budget_guard import BudgetConfig, BudgetGuard
from ingredient_match_llm import (
    DEFAULT_MODEL,
    MODEL_PRICING,
    SENTINEL_DATA_TYPE,
    SENTINEL_DESCRIPTION,
    SENTINEL_FDC_ID,
    WATER_SENTINEL_DATA_TYPE,
    WATER_SENTINEL_DESCRIPTION,
    WATER_SENTINEL_FDC_ID,
    build_food_index,
    compute_report,
    judge_async,
    log_to_mlflow,
    run_judging,
    write_local_reports,
    _append_sentinel_candidate,
    _append_water_sentinel_candidate,
    _print_budget_abort,
)
from generic_ingredient_defaults import (
    build_hardcoded_judge,
    inject_default_candidate,
    lookup_generic_default,
)
from ingredient_match_staged import (
    LLMRetrievalConfig,
    StagedMatchConfig,
    batched_dequant_similarities,
    match_query,
    query_from_parsed_row,
)
from ingredient_query_cache import (
    DEFAULT_WORK_DIR as FOOD_CACHE_DIR,
    load_or_build_recipe_artifacts,
)
from portion_aware_match import (
    build_pipeline_path,
    build_user_prompt_portion,
    retrieve_llm_candidates_portion_aware,
)
from portion_candidate_index import load_or_build_portion_summary_index
from portion_gram import (
    build_count_portion_index,
    build_portion_capability_sets,
    build_portion_index,
    resolve_grams_from_parsed_row,
)
from resolution_plan import plan_from_parsed_row
from progress_utils import iter_progress
from recipe_directions import parse_directions_list, relevant_direction_steps
from sample_recipes import load_sampled_recipes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLM_WORK_DIR = ROOT / "scratch" / "recipe_matching_llm_100_portion"
MLFLOW_EXPERIMENT = "ingredient_match_llm_portion"
PROMPT_VERSION = "v7_generic_defaults_prep_semantic_portion_capable"

SYSTEM_PROMPT = (
    "Match recipe ingredient → USDA fdc_id.\n"
    "Candidates: fdc_id | description | L | S | P | portions | fit "
    "(P=V/C/Cm/VC/-; fit=unit match 0-1).\n"
    "Best identity first; else closest same-type substitute with usable portions "
    "(e.g. white wine vinegar→distilled vinegar). Never substitute different forms "
    "(e.g. garlic powder→garlic raw, dried→fresh). Abstain only if ≥20 kcal expected "
    "and no substitute.\n"
    "Volume/count: prefer P!=- and fit>0; set matched_portion_id when fit>0. "
    "Never pick P=- for volume/count unless negligible_calories.\n"
    "When SEMANTIC_FALLBACK is present and all primary candidates have fit=0, pick "
    "best identity from SEMANTIC_FALLBACK; set negligible_calories=true if this qty "
    "likely <20 kcal and matched_portion_id=null.\n"
    "certainty 0-1 = joint confidence in fdc identity/substitute AND portion-unit fit (lower if either weak).\n"
    "negligible_calories=true only if THIS qty likely <20 kcal. Never flour, sugar, butter, oil.\n"
    f"Plain recipe water (water, cold/hot/boiling water): {WATER_SENTINEL_FDC_ID}. "
    f"Ice/plain salt/garnish: {SENTINEL_FDC_ID}. JSON only."
)


def precompute_payloads_portion(
    parsed: pd.DataFrame,
    name_emb,
    prep_emb,
    dequant_emb,
    food_index,
    directions_by_recipe: dict[int, list[str]],
    retr_config: LLMRetrievalConfig,
    capabilities,
    volume_index: dict,
    count_index: dict,
    portion_summary_index: dict | None = None,
    *,
    limit: int | None = None,
    chunk_size: int = 256,
    progress_writer: Any = None,
) -> list[dict[str, Any]]:
    n = len(parsed) if limit is None else min(limit, len(parsed))
    payloads: list[dict[str, Any]] = []
    volume_fdc_ids = set(capabilities.volume_fdc_ids)
    count_fdc_ids = set(capabilities.count_fdc_ids)

    n_chunks = (n + chunk_size - 1) // chunk_size
    print(
        f"Batched portion-aware retrieval: {n} ingredients in {n_chunks} chunk(s) of {chunk_size}",
        flush=True,
    )
    if n:
        preview_n = min(5, n)
        print(f"  first {preview_n} ingredient(s):", flush=True)
        for pi in range(preview_n):
            row = parsed.iloc[pi]
            print(
                f"    [{pi + 1}] recipe={int(row['recipe_id'])} "
                f"idx={int(row['ingredient_idx'])} "
                f"{str(row.get('ingredient', ''))[:100]}",
                flush=True,
            )

    show_progress = progress_writer is None or progress_writer.show_secondary_progress
    progress = iter_progress(range(n), total=n, desc="Retrieval + prompts", enabled=show_progress)
    progress_iter = iter(progress)

    for chunk_idx, start in enumerate(range(0, n, chunk_size), start=1):
        end = min(start + chunk_size, n)
        chunk_n = end - start
        t_chunk = time.perf_counter()
        print(
            f"  chunk {chunk_idx}/{n_chunks}: rows {start}-{end - 1} ({chunk_n} ingredients) "
            f"— batched dequant similarities…",
            flush=True,
        )
        t_sims = time.perf_counter()
        sims_chunk = batched_dequant_similarities(food_index, dequant_emb[start:end])
        sims_sec = time.perf_counter() - t_sims
        if sims_chunk.size:
            print(
                f"  chunk {chunk_idx}/{n_chunks}: dequant sims {sims_sec:.1f}s "
                f"({sims_chunk.shape[0]:,} foods × {sims_chunk.shape[1]} queries)",
                flush=True,
            )
        else:
            print(
                f"  chunk {chunk_idx}/{n_chunks}: dequant sims {sims_sec:.1f}s (empty)",
                flush=True,
            )

        for i in range(start, end):
            next(progress_iter, None)
            pos_in_chunk = i - start + 1
            if pos_in_chunk == 1 or pos_in_chunk % 16 == 0 or i == end - 1:
                row_peek = parsed.iloc[i]
                print(
                    f"  chunk {chunk_idx}/{n_chunks}: processing {pos_in_chunk}/{chunk_n} "
                    f"(global {i + 1}/{n}) "
                    f"{str(row_peek.get('ingredient', ''))[:80]!r}",
                    flush=True,
                )
                if progress_writer is not None:
                    progress_writer.record_chunk_progress("payloads", i + 1, n)
            row = parsed.iloc[i]
            row_dict = row.to_dict()
            recipe_id = int(row["recipe_id"])
            ingredient_idx = int(row["ingredient_idx"])
            ingredient = str(row["ingredient"])
            name = str(row.get("name") or "")
            preparation = str(row.get("preparation") or "")
            unit = str(row.get("unit") or "")

            plan = plan_from_parsed_row(row_dict)
            amount_kind = (
                row_dict.get("amount_kind_final")
                or plan.primary_amount_kind
                or classify_from_parsed_row(row_dict)
            )
            query = query_from_parsed_row(row, name_emb[i], prep_emb[i], dequant_emb[i])
            staged = match_query(query, food_index)
            staged_top1 = staged.get("matched_fdc_id")

            sims_i = sims_chunk[:, i - start] if sims_chunk.size else None
            retr = retrieve_llm_candidates_portion_aware(
                query,
                food_index,
                capabilities,
                retr_config,
                amount_kind=amount_kind,
                staged_top1_fdc_id=staged_top1,
                precomputed_sims=sims_i,
                parsed_row=row_dict,
                resolution_plan=plan,
                portion_summary_index=portion_summary_index,
            )
            cand_df = retr.candidates
            n_lexical_pool = int(cand_df.attrs.get("n_lexical_pool", 0)) if not cand_df.empty else 0
            n_semantic_pool = int(cand_df.attrs.get("n_semantic_pool", 0)) if not cand_df.empty else 0

            prompt_candidates = cand_df[cand_df["in_llm_prompt"]] if not cand_df.empty else cand_df
            semantic_fb = retr.semantic_fallback
            if semantic_fb is not None and not semantic_fb.empty:
                prompt_for_judge = pd.concat(
                    [prompt_candidates, semantic_fb], ignore_index=True
                ).drop_duplicates(subset=["fdc_id"], keep="first")
            else:
                prompt_for_judge = prompt_candidates

            generic_match = lookup_generic_default(name)
            hardcoded_judge = None
            if generic_match is not None:
                prompt_for_judge = inject_default_candidate(prompt_for_judge, generic_match)
                hardcoded_judge = build_hardcoded_judge(generic_match)

            n_retrieved_candidates = int(len(prompt_for_judge))
            skip_sentinel = (
                retr.portion_filter_kind in ("volume", "count") and not retr.mass_in_text
            )
            if not skip_sentinel:
                prompt_for_judge = _append_sentinel_candidate(prompt_for_judge)
                if generic_match is None or not generic_match.is_water_sentinel:
                    prompt_for_judge = _append_water_sentinel_candidate(prompt_for_judge)
            valid_fdc_ids = (
                set(int(x) for x in prompt_for_judge["fdc_id"])
                if not prompt_for_judge.empty
                else set()
            )
            prompt_desc = (
                {
                    int(r.fdc_id): str(r.description)
                    for r in prompt_for_judge.itertuples(index=False)
                }
                if not prompt_for_judge.empty
                else {}
            )

            steps = relevant_direction_steps(ingredient, directions_by_recipe.get(recipe_id, []))
            user_prompt = build_user_prompt_portion(
                ingredient,
                name,
                preparation,
                unit,
                amount_kind,
                prompt_candidates,
                steps,
                retr_config.description_max_chars,
                mass_in_text=retr.mass_in_text,
                query_tokens=list(retr.query_tokens),
                quantity=row.get("quantity"),
                semantic_fallback=semantic_fb,
            )

            top10 = cand_df.head(retr_config.top10_size) if not cand_df.empty else cand_df
            top10_rows = [
                {
                    "rank": int(c.rank),
                    "fdc_id": int(c.fdc_id),
                    "data_type": c.data_type,
                    "description": c.description,
                    "lexical_dequant": float(c.lexical_dequant),
                    "dequant_sem": float(c.dequant_sem),
                    "retrieval_score": float(c.retrieval_score),
                    "staged_final_score": float(c.staged_final_score),
                    "staged_base_score": float(c.staged_base_score),
                    "staged_prep_score": float(c.staged_prep_score),
                    "in_llm_prompt": bool(c.in_llm_prompt),
                    "is_staged_top1": bool(c.is_staged_top1),
                    "portion_flag": getattr(c, "portion_flag", "-"),
                    "has_volume_portion": bool(getattr(c, "has_volume_portion", False)),
                    "has_count_portion": bool(getattr(c, "has_count_portion", False)),
                    "portion_match_score": float(getattr(c, "portion_match_score", 0.0) or 0.0),
                    "portion_summary": getattr(c, "portion_summary", "-"),
                    "blended_score": float(getattr(c, "blended_score", c.retrieval_score)),
                }
                for c in top10.itertuples(index=False)
            ] if not top10.empty else []

            staged_in_candidates = (
                staged_top1 is not None
                and not cand_df.empty
                and bool((cand_df["fdc_id"] == staged_top1).any())
            )
            staged_in_top10 = (
                staged_top1 is not None
                and not top10.empty
                and bool((top10["fdc_id"] == staged_top1).any())
            )

            payloads.append(
                {
                    "recipe_id": recipe_id,
                    "ingredient_idx": ingredient_idx,
                    "ingredient": ingredient,
                    "name": name,
                    "preparation": preparation,
                    "dequantified": str(row.get("dequantified") or ""),
                    "unit": unit,
                    "quantity": row.get("quantity"),
                    "amount_kind": amount_kind,
                    "resolution_plan": plan.to_dict(),
                    "retrieval_tier": retr.retrieval_tier,
                    "mass_in_text": retr.mass_in_text,
                    "portion_query_tokens": list(retr.query_tokens),
                    "portion_filter_kind": retr.portion_filter_kind,
                    "has_semantic_fallback": semantic_fb is not None and not semantic_fb.empty,
                    "n_tier1_union": retr.n_tier1_union,
                    "tier1_max_score": retr.tier1_max_score,
                    "staged_fdc_id": staged_top1,
                    "staged_description": staged.get("matched_description"),
                    "staged_match_score": staged.get("match_score"),
                    "staged_match_quality": staged.get("match_quality"),
                    "staged_base_score": staged.get("base_score"),
                    "staged_prep_score": staged.get("prep_score"),
                    "n_candidates_llm": n_retrieved_candidates,
                    "n_lexical_pool": n_lexical_pool,
                    "n_semantic_pool": n_semantic_pool,
                    "staged_top1_in_llm_candidates": staged_in_candidates,
                    "staged_top1_in_top10": staged_in_top10,
                    "n_relevant_steps": len(steps),
                    "relevant_steps": " | ".join(steps),
                    "user_prompt": user_prompt,
                    "valid_fdc_ids": valid_fdc_ids,
                    "prompt_desc": prompt_desc,
                    "top10_rows": top10_rows,
                    "parsed_row": row_dict,
                    "volume_fdc_ids": volume_fdc_ids,
                    "count_fdc_ids": count_fdc_ids,
                    "volume_index": volume_index,
                    "count_index": count_index,
                    "hardcoded_judge": hardcoded_judge,
                    "generic_default_key": generic_match.match_key if generic_match else None,
                    "llm_water_sentinel": bool(
                        generic_match.is_water_sentinel if generic_match else False
                    ),
                }
            )

        chunk_sec = time.perf_counter() - t_chunk
        print(
            f"  chunk {chunk_idx}/{n_chunks}: done in {chunk_sec:.1f}s "
            f"({len(payloads)} payloads total)",
            flush=True,
        )

    return payloads


def _fdc_has_portion_for_kind(
    fdc_id: int,
    amount_kind: str,
    *,
    volume_fdc_ids: set[int],
    count_fdc_ids: set[int],
) -> bool:
    if amount_kind == "volume":
        return int(fdc_id) in volume_fdc_ids
    if amount_kind == "count":
        return int(fdc_id) in count_fdc_ids
    return True


def _rescue_portion_capable_pick(
    payload: dict[str, Any],
    llm_fdc_id: int | None,
    llm_desc: str | None,
    *,
    negligible: bool,
) -> tuple[int | None, str | None]:
    """Prefer a portion-capable top-10 candidate when the judge pick cannot resolve grams."""
    amount_kind = str(payload.get("amount_kind") or "")
    if amount_kind not in ("volume", "count") or payload.get("mass_in_text") or negligible:
        return llm_fdc_id, llm_desc

    volume_fdc_ids = {int(x) for x in payload.get("volume_fdc_ids") or []}
    count_fdc_ids = {int(x) for x in payload.get("count_fdc_ids") or []}
    if llm_fdc_id is not None and _fdc_has_portion_for_kind(
        llm_fdc_id,
        amount_kind,
        volume_fdc_ids=volume_fdc_ids,
        count_fdc_ids=count_fdc_ids,
    ):
        return llm_fdc_id, llm_desc

    viable = [
        row
        for row in payload.get("top10_rows") or []
        if row.get("in_llm_prompt")
        and _fdc_has_portion_for_kind(
            int(row["fdc_id"]),
            amount_kind,
            volume_fdc_ids=volume_fdc_ids,
            count_fdc_ids=count_fdc_ids,
        )
    ]
    if not viable:
        return llm_fdc_id, llm_desc

    best = max(
        viable,
        key=lambda row: (
            float(row.get("portion_match_score") or 0.0),
            float(row.get("blended_score") or row.get("retrieval_score") or 0.0),
            float(row.get("retrieval_score") or 0.0),
        ),
    )
    rescued_id = int(best["fdc_id"])
    rescued_desc = str(best.get("description") or payload.get("prompt_desc", {}).get(rescued_id) or "")
    return rescued_id, rescued_desc or llm_desc


def assemble_rows_portion(
    payload: dict[str, Any],
    judge: dict[str, Any],
    *,
    run_id: str,
    run_name: str,
    model: str,
    pricing: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    llm_fdc_id = judge["fdc_id"]
    llm_fdc_id = int(llm_fdc_id) if llm_fdc_id is not None else None
    llm_desc = payload["prompt_desc"].get(llm_fdc_id) if llm_fdc_id is not None else None
    negligible = bool(judge.get("negligible_calories", False))
    if not judge.get("dequant_cache"):
        llm_fdc_id, llm_desc = _rescue_portion_capable_pick(
            payload,
            llm_fdc_id,
            llm_desc,
            negligible=negligible,
        )

    parsed_for_resolve = dict(payload["parsed_row"])
    if payload.get("resolution_plan"):
        parsed_for_resolve["resolution_plan"] = payload["resolution_plan"]

    gram_result = resolve_grams_from_parsed_row(
        parsed_for_resolve,
        llm_fdc_id,
        portion_index=payload["volume_index"],
        count_portion_index=payload["count_index"],
        matched_portion_id=judge.get("matched_portion_id"),
        llm_negligible_calories=negligible,
    )

    cost = (
        judge["prompt_tokens"] / 1e6 * pricing["input"]
        + judge["completion_tokens"] / 1e6 * pricing["output"]
    )
    staged_top1 = payload["staged_fdc_id"]
    ts = datetime.now(timezone.utc)
    ts_parts = inference_store.split_timestamp(ts)

    pipeline_path = build_pipeline_path(
        amount_kind=str(payload["amount_kind"]),
        retrieval_tier=str(payload["retrieval_tier"]),
        grams_status=gram_result.status,
        llm_fdc_id=llm_fdc_id,
    )

    exp = {
        "run_id": run_id,
        "run_name": run_name,
        "model": model,
        "recipe_id": payload["recipe_id"],
        "ingredient_idx": payload["ingredient_idx"],
        "ingredient": payload["ingredient"],
        "name": payload["name"],
        "preparation": payload["preparation"],
        "dequantified": payload["dequantified"],
        "unit": payload["unit"],
        "quantity": payload.get("quantity"),
        "amount_kind": payload["amount_kind"],
        "retrieval_tier": payload["retrieval_tier"],
        "portion_filter_kind": payload["portion_filter_kind"],
        "n_tier1_union": payload["n_tier1_union"],
        "tier1_max_score": payload["tier1_max_score"],
        "system_prompt": SYSTEM_PROMPT,
        "prompt": payload["user_prompt"],
        "response": judge["response"],
        "llm_fdc_id": llm_fdc_id,
        "llm_description": llm_desc,
        "llm_certainty": judge["certainty"],
        "llm_rationale": judge["rationale"],
        "matched_portion_id": judge.get("matched_portion_id"),
        "llm_negligible_calories": negligible,
        "llm_water_sentinel": bool(
            judge.get("is_water_sentinel")
            or payload.get("llm_water_sentinel")
            or (
                llm_fdc_id is not None
                and int(llm_fdc_id) == WATER_SENTINEL_FDC_ID
            )
        ),
        "llm_hardcoded": bool(judge.get("hardcoded")),
        "generic_default_key": judge.get("generic_default_key") or payload.get("generic_default_key"),
        "llm_pick_has_volume_portion": (
            llm_fdc_id is not None and int(llm_fdc_id) in payload["volume_fdc_ids"]
        ),
        "llm_pick_has_count_portion": (
            llm_fdc_id is not None and int(llm_fdc_id) in payload["count_fdc_ids"]
        ),
        "grams": gram_result.grams,
        "grams_status": gram_result.status,
        "grams_method": gram_result.method,
        "pipeline_path": pipeline_path,
        "llm_agrees_with_staged": llm_fdc_id is not None and llm_fdc_id == staged_top1,
        "llm_abstained": llm_fdc_id is None and judge["error"] is None,
        "llm_error": judge["error"],
        "staged_fdc_id": staged_top1,
        "staged_description": payload["staged_description"],
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
        "prompt_version": PROMPT_VERSION,
        **ts_parts,
    }

    cand_rows = []
    for c in payload["top10_rows"]:
        cand_rows.append(
            {
                "run_id": run_id,
                "recipe_id": payload["recipe_id"],
                "ingredient_idx": payload["ingredient_idx"],
                "rank": c["rank"],
                "fdc_id": c["fdc_id"],
                "data_type": c["data_type"],
                "description": c["description"],
                "lexical_dequant": c["lexical_dequant"],
                "dequant_sem": c["dequant_sem"],
                "retrieval_score": c["retrieval_score"],
                "staged_final_score": c["staged_final_score"],
                "staged_base_score": c["staged_base_score"],
                "staged_prep_score": c["staged_prep_score"],
                "in_llm_prompt": c["in_llm_prompt"],
                "is_staged_top1": c["is_staged_top1"],
                "portion_flag": c.get("portion_flag"),
                "has_volume_portion": c.get("has_volume_portion"),
                "has_count_portion": c.get("has_count_portion"),
                "is_llm_pick": llm_fdc_id is not None and c["fdc_id"] == llm_fdc_id,
                "ts": ts,
            }
        )
    return exp, cand_rows


def compute_portion_report(matches_df: pd.DataFrame, base_report: dict[str, Any]) -> dict[str, Any]:
    report = dict(base_report)
    resolved = matches_df["grams"].notna()
    report["gram_resolvable_rate"] = round(float(resolved.mean()), 4) if len(matches_df) else None
    report["grams_status_counts"] = (
        matches_df["grams_status"].value_counts().to_dict() if "grams_status" in matches_df else {}
    )
    report["amount_kind_counts"] = (
        matches_df["amount_kind"].value_counts().to_dict() if "amount_kind" in matches_df else {}
    )
    report["retrieval_tier_counts"] = (
        matches_df["retrieval_tier"].value_counts().to_dict()
        if "retrieval_tier" in matches_df
        else {}
    )
    if "amount_kind" in matches_df.columns:
        for kind in ("volume", "count", "mass"):
            sub = matches_df[matches_df["amount_kind"] == kind]
            if len(sub):
                report[f"gram_resolvable_rate_{kind}"] = round(
                    float(sub["grams"].notna().mean()), 4
                )
    return report


def run_pilot_portion(
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
    print(
        f"=== LLM ingredient judge (portion-aware v3) ===\n"
        f"MLflow experiment: {mlflow_experiment}\nMLflow run: {run_name}\nrun_id={run_id}",
        flush=True,
    )

    def phase(label: str) -> float:
        print(f"\n--- {label} ---", flush=True)
        return time.perf_counter()

    t = phase(
        f"Loading {n_recipes} recipes (seed={seed})"
        if sample_manifest is None
        else f"Loading recipes from manifest {sample_manifest}"
    )
    ids_path = None if sample_manifest is not None else work_dir / "sampled_recipe_ids.json"
    recipes, recipe_ingredients, sampled_ids = load_sampled_recipes(
        n=n_recipes,
        seed=seed,
        ids_path=ids_path,
        sample_manifest=sample_manifest,
    )
    print(
        f"Loaded {len(recipes)} recipes -> {len(recipe_ingredients):,} ingredient lines "
        f"({time.perf_counter() - t:.1f}s)",
        flush=True,
    )

    directions_by_recipe = {
        int(r.recipe_id): parse_directions_list(r.directions)
        for r in recipes.itertuples(index=False)
    }

    t = phase("Parsing + embedding sampled ingredient lines")
    parsed, name_emb, prep_emb, dequant_emb, _meta = load_or_build_recipe_artifacts(
        recipe_ingredients, work_dir
    )
    print(f"Embeddings ready ({time.perf_counter() - t:.1f}s)", flush=True)

    t = phase("Building food index + portion capability sets")
    food_index = build_food_index(food_cache_dir, match_config)
    with connect() as conn:
        capabilities = build_portion_capability_sets(conn)
        volume_index = build_portion_index(conn)
        count_index = build_count_portion_index(conn)
        portion_summary_index = load_or_build_portion_summary_index(conn)
    print(
        f"Portion sets: {len(capabilities.volume_fdc_ids):,} volume, "
        f"{len(capabilities.count_fdc_ids):,} count fdc_ids",
        flush=True,
    )

    t = phase("Batched portion-aware retrieval + prompt assembly")
    payloads = precompute_payloads_portion(
        parsed,
        name_emb,
        prep_emb,
        dequant_emb,
        food_index,
        directions_by_recipe,
        retr_config,
        capabilities,
        volume_index,
        count_index,
        portion_summary_index,
        limit=limit,
        chunk_size=chunk_size,
    )
    print(
        f"Prepared {len(payloads)} ingredient prompts ({time.perf_counter() - t:.1f}s); "
        f"starting LLM judging (concurrency={concurrency})…",
        flush=True,
    )

    conn = None
    if use_supabase:
        conn = inference_store.open_connection()
        inference_store.upsert_experiment(
            conn,
            {
                "run_id": run_id,
                "run_name": run_name,
                "model": model,
                "mlflow_experiment": mlflow_experiment,
                "prompt_version": PROMPT_VERSION,
                "seed": seed,
                "n_recipes": int(recipes["recipe_id"].nunique()),
                "n_ingredients": len(payloads),
                "concurrency": concurrency,
                "retrieval_config": asdict(retr_config),
                "pricing": pricing,
                "sampled_recipe_ids": sampled_ids,
                "status": "running",
                "started_at": started_at,
            },
        )

    started = time.time()
    all_exp, all_cand, breaker = asyncio.run(
        run_judging(
            payloads,
            run_id=run_id,
            run_name=run_name,
            model=model,
            pricing=pricing,
            concurrency=concurrency,
            flush_every=flush_every,
            conn=conn,
            use_supabase=use_supabase,
            budget_config=budget_config,
            log_every=log_every,
            heartbeat_sec=heartbeat_sec,
            verbose=verbose,
            assemble_fn=assemble_rows_portion,
            system_prompt=SYSTEM_PROMPT,
        )
    )
    elapsed = time.time() - started

    matches_df = pd.DataFrame(all_exp)
    candidates_df = pd.DataFrame(all_cand)
    base_report = compute_report(
        matches_df, recipes, model=model, pricing=pricing, elapsed=elapsed
    )
    report = compute_portion_report(matches_df, base_report)

    manifest = {
        "run_id": run_id,
        "run_name": run_name,
        "timestamp_utc": started_at.isoformat(),
        "model": model,
        "seed": seed,
        "prompt_version": PROMPT_VERSION,
        "concurrency": concurrency,
        "n_recipes_requested": n_recipes,
        "n_recipes_loaded": int(recipes["recipe_id"].nunique()),
        "retrieval_config": asdict(retr_config),
        "pricing": pricing,
        "sampled_recipe_ids": sampled_ids,
        "portion_volume_fdc_ids": len(capabilities.volume_fdc_ids),
        "portion_count_fdc_ids": len(capabilities.count_fdc_ids),
    }
    artifact_paths = write_local_reports(
        work_dir, matches_df, candidates_df, recipes, report, manifest
    )
    (work_dir / "llm_eval_summary_portion.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"\nWrote local reports -> {work_dir}", flush=True)
    print(
        f"Cost: ${report['cost_total_usd']:.4f}; gram resolvable "
        f"{report.get('gram_resolvable_rate', 0):.1%}",
        flush=True,
    )

    if use_mlflow:
        mlflow_run_id = log_to_mlflow(
            experiment_name=mlflow_experiment,
            run_name=run_name,
            model=model,
            seed=seed,
            concurrency=concurrency,
            n_recipes=n_recipes,
            retr_config=retr_config,
            pricing=pricing,
            report=report,
            artifact_paths=artifact_paths,
            run_id=run_id,
        )
        if mlflow_run_id:
            print(f"Logged MLflow run {mlflow_run_id}", flush=True)

    if use_supabase and conn is not None:
        inference_store.upsert_experiment(
            conn,
            {
                "run_id": run_id,
                "run_name": run_name,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "status": "aborted_budget" if breaker else "completed",
                "finished_at": datetime.now(timezone.utc),
                **{k: report[k] for k in report if k != "pricing"},
            },
        )
        conn.close()

    if breaker:
        _print_budget_abort(breaker, report, run_id, run_name)
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Portion-aware LLM ingredient matching.")
    parser.add_argument("--n-recipes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_LLM_WORK_DIR)
    parser.add_argument("--food-cache-dir", type=Path, default=FOOD_CACHE_DIR)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--heartbeat-sec", type=float, default=15.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-supabase", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", default=MLFLOW_EXPERIMENT)
    parser.add_argument("--sample-manifest", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--semantic-floor", type=float, default=None)
    parser.add_argument("--lexical-floor", type=float, default=None)
    parser.add_argument("--daily-budget-usd", type=float, default=10.0)
    parser.add_argument("--rate-limit-usd-min", type=float, default=0.50)
    parser.add_argument("--budget-check-every", type=int, default=100)
    parser.add_argument("--no-budget-guard", action="store_true")
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

    run_pilot_portion(
        n_recipes=args.n_recipes,
        seed=args.seed,
        model=args.model,
        work_dir=args.work_dir,
        food_cache_dir=args.food_cache_dir,
        retr_config=retr_config,
        match_config=StagedMatchConfig(),
        concurrency=args.concurrency,
        flush_every=args.flush_every,
        limit=args.limit,
        use_supabase=not args.no_supabase,
        use_mlflow=not args.no_mlflow,
        mlflow_experiment=args.mlflow_experiment,
        sample_manifest=args.sample_manifest,
        run_id=args.run_id,
        chunk_size=args.chunk_size,
        log_every=args.log_every,
        heartbeat_sec=args.heartbeat_sec,
        verbose=not args.quiet,
        budget_config=budget_config,
    )


if __name__ == "__main__":
    main()
