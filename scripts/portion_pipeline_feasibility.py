#!/usr/bin/env python3
"""Feasibility report: portion-needing ingredients → fdc_id match → gram resolution.

Phases (each cached under --out-dir when run_manifest.json matches):
  1. amount_classification.parquet — rules + line enrichment LLM
  2. judge_matches_raw.parquet — portion-aware LLM fdc judging
  3. pipeline_matches.parquet — rules grams + optional LLM portion pick
  4. feasibility_report.json

Each run logs to MLflow (experiment ``portion_pipeline_feasibility``) with an auto-incrementing
``feasibility_version`` (max existing version in the experiment + 1, starting at 1). Pass
``--no-mlflow`` to skip.

Usage:
  uv run python scripts/portion_pipeline_feasibility.py --n-recipes 1000 --seed 42
  uv run python scripts/portion_pipeline_feasibility.py --force-all      # full fresh run
  uv run python scripts/portion_pipeline_feasibility.py --finalize-only  # report from cache
  uv run python scripts/portion_pipeline_feasibility.py --force-judging    # redo LLM match
  uv run python scripts/portion_pipeline_feasibility.py --only-no-portion --force-payloads  # retry failures
  uv run python scripts/portion_pipeline_feasibility.py --only-unresolved --force-payloads  # retry missing fdc/grams

Targeted no_portion re-run (after v4; writes to separate dir, baseline unchanged):
  cd Capstone && uv run python scripts/portion_pipeline_feasibility.py \\
    --n-recipes 1000 --seed 42 --only-no-portion --force-payloads \\
    --baseline-dir scratch/EDA/portion_feasibility_1000 \\
    2>&1 | tee -a scratch/EDA/portion_feasibility_1000_v4_no_portion/run.log

Targeted unresolved re-run (v4 baseline; missing llm_fdc_id and/or grams):
  cd Capstone && uv run python scripts/portion_pipeline_feasibility.py \\
    --n-recipes 1000 --seed 42 --only-unresolved --force-payloads --force-judging \\
    --baseline-dir scratch/EDA/portion_feasibility_1000_v4_no_portion \\
    --out-dir scratch/EDA/portion_feasibility_1000_v5_unresolved \\
    2>&1 | tee scratch/EDA/portion_feasibility_1000_v5_unresolved/run.log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import random
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from amount_kind import missing_quantity
from line_enrichment_llm import apply_enrichment_to_plan, run_line_enrichment_sync
from resolution_plan import build_resolution_plan, needs_line_enrichment
from db import connect, load_dotenv
from feasibility_mlflow import (
    DEFAULT_EXPERIMENT,
    log_feasibility_run,
    next_feasibility_version,
)
from feasibility_progress import FeasibilityProgressWriter
from ingredient_match_llm import (
    DEFAULT_MODEL,
    MODEL_PRICING,
    build_food_index,
    run_judging,
)
from ingredient_match_llm_portion import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    assemble_rows_portion,
    precompute_payloads_portion,
)
from judge_checkpoint import completed_keys, load_judge_checkpoint
from build_dequant_norm_cache import (
    apply_dequant_cache_to_payloads,
    create_dequant_cache_runtime,
    load_dequant_norm_cache,
    resolve_dequant_cache_path,
)
from ingredient_match_staged import LLMRetrievalConfig, StagedMatchConfig
from ingredient_query_cache import load_or_build_recipe_artifacts
from portion_candidate_index import load_or_build_portion_summary_index
from progress_utils import iter_progress
from portion_gram import (
    build_count_portion_index,
    build_portion_capability_sets,
    build_portion_index,
    classify_food_portion_row,
    load_portion_rows_cache,
    resolve_grams_from_parsed_row,
    resolve_quantity_fields,
    _load_portion_rows_for_fdc,
)
from portion_resolve_llm import apply_portion_pick, pick_portion_sync
from recipe_directions import parse_directions_list
from sample_recipes import load_sampled_recipes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "scratch" / "EDA" / "portion_feasibility_1000"
DEFAULT_BASELINE = DEFAULT_OUT
DEFAULT_V4_OUT = ROOT / "scratch" / "EDA" / "portion_feasibility_1000_v4_no_portion"
DEFAULT_V5_OUT = ROOT / "scratch" / "EDA" / "portion_feasibility_1000_v5_unresolved"
PartialRetryMode = str  # "no_portion" | "unresolved"
DEFAULT_FOOD_CACHE = ROOT / "scratch" / "recipe_matching_llm_100_portion"
MANIFEST_VERSION = 3

AMOUNT_PATH = "amount_classification.parquet"
AMOUNT_LLM_PATH = "line_enrichment_llm_calls.parquet"
JUDGE_RAW_PATH = "judge_matches_raw.parquet"
MATCHES_PATH = "pipeline_matches.parquet"
PAYLOADS_PATH = "payloads.pkl"
MANIFEST_PATH = "run_manifest.json"
REPORT_PATH = "feasibility_report.json"


def _paths(out_dir: Path) -> dict[str, Path]:
    return {
        "amount": out_dir / AMOUNT_PATH,
        "amount_llm": out_dir / AMOUNT_LLM_PATH,
        "judge_raw": out_dir / JUDGE_RAW_PATH,
        "matches": out_dir / MATCHES_PATH,
        "payloads": out_dir / PAYLOADS_PATH,
        "manifest": out_dir / MANIFEST_PATH,
        "report": out_dir / REPORT_PATH,
    }


def build_manifest(
    *,
    n_recipes: int,
    seed: int,
    model: str,
    limit: int | None,
    n_lines: int,
) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "n_recipes": n_recipes,
        "seed": seed,
        "model": model,
        "limit": limit,
        "n_lines": n_lines,
        "prompt_version": PROMPT_VERSION,
    }


def load_manifest(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / MANIFEST_PATH
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def manifest_matches(
    saved: dict[str, Any] | None,
    current: dict[str, Any],
    *,
    ignore_version: bool = False,
) -> bool:
    if saved is None:
        return False
    if ignore_version:
        keys = [k for k in current if k != "version"]
        return all(saved.get(k) == current.get(k) for k in keys)
    return saved == current


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    (out_dir / MANIFEST_PATH).write_text(json.dumps(manifest, indent=2) + "\n")


def mark_openai_partial_manifest(
    manifest: dict[str, Any],
    *,
    phase: str,
    completed: int,
    total: int,
    key_status: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Annotate manifest so the same command can resume after key exhaustion."""
    from openai_fallback import OPENAI_PARTIAL_STATUS

    manifest["status"] = OPENAI_PARTIAL_STATUS
    manifest["openai_key_status"] = key_status or {}
    manifest["resume"] = {
        "phase": phase,
        "completed": completed,
        "total": total,
        "pending": max(total - completed, 0),
        "artifact_files": [
            AMOUNT_PATH,
            AMOUNT_LLM_PATH,
            JUDGE_RAW_PATH,
            MATCHES_PATH,
            PAYLOADS_PATH,
        ],
        "next_command_flags": _resume_flags_for_phase(phase),
    }
    if extra:
        manifest["resume"].update(extra)
    return manifest


def _resume_flags_for_phase(phase: str) -> list[str]:
    if phase == "judging":
        return []
    if phase == "amount_classify":
        return ["--force-amount"]
    if phase == "portion_llm":
        return ["--force-judging"]
    return []


def print_openai_resume_hint(out_dir: Path, manifest: dict[str, Any]) -> None:
    resume = manifest.get("resume") or {}
    phase = resume.get("phase", "unknown")
    completed = resume.get("completed", "?")
    total = resume.get("total", "?")
    flags = " ".join(resume.get("next_command_flags") or [])
    print(
        f"\nOpenAI API keys exhausted during {phase} ({completed}/{total} done). "
        f"Partial artifacts saved under {out_dir}. "
        f"Re-run the same command{' ' + flags if flags else ''} to resume.\n",
        flush=True,
    )


def is_openai_partial_manifest(manifest: dict[str, Any] | None) -> bool:
    from openai_fallback import OPENAI_PARTIAL_STATUS

    return bool(manifest and manifest.get("status") == OPENAI_PARTIAL_STATUS)


def partial_retry_mode(
    *,
    only_no_portion: bool = False,
    only_unresolved: bool = False,
) -> PartialRetryMode | None:
    if only_no_portion and only_unresolved:
        raise ValueError("Use only one of --only-no-portion or --only-unresolved")
    if only_unresolved:
        return "unresolved"
    if only_no_portion:
        return "no_portion"
    return None


def resolve_run_dirs(
    *,
    out_dir: Path,
    baseline_dir: Path | None,
    retry_mode: PartialRetryMode | None,
) -> tuple[Path, Path]:
    """Return (baseline_dir, write_dir). Partial retries never write into baseline."""
    if retry_mode is None:
        return out_dir, out_dir

    baseline = (baseline_dir or out_dir).resolve()
    write = out_dir.resolve()
    if write == baseline:
        write = DEFAULT_V5_OUT if retry_mode == "unresolved" else DEFAULT_V4_OUT
    return baseline, write


def _seed_baseline_artifacts(
    baseline_paths: dict[str, Path],
    write_paths: dict[str, Path],
) -> None:
    """Copy read-only baseline inputs into write dir when missing (never overwrites baseline)."""
    import shutil

    for key in ("amount", "amount_llm"):
        src, dst = baseline_paths[key], write_paths[key]
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
            print(f"Copied baseline {src.name} -> {dst}", flush=True)

    src_cache = baseline_paths["amount"].parent / "recipe_cache"
    dst_cache = write_paths["amount"].parent / "recipe_cache"
    if src_cache.is_dir() and not dst_cache.exists():
        shutil.copytree(src_cache, dst_cache)
        print(f"Copied baseline recipe_cache -> {dst_cache}", flush=True)


def _load_baseline_judge_df(baseline_paths: dict[str, Path]) -> pd.DataFrame:
    """Full prior run frame used to preserve non-retried rows (prefer final matches)."""
    for cache in (baseline_paths["matches"], baseline_paths["judge_raw"]):
        if cache.is_file():
            return pd.read_parquet(cache)
    return pd.DataFrame()


def classify_ingredient_lines(
    recipe_ingredients: pd.DataFrame,
    *,
    model: str,
    progress_writer: Any = None,
    enrichment_concurrency: int = 8,
    only_enrich_keys: set[tuple[int, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Rules parse + resolution plan; selective LLM line enrichment."""
    rows: list[dict[str, Any]] = []
    enrich_items: list[tuple[str, dict[str, Any]]] = []

    for row in recipe_ingredients.itertuples(index=False):
        ingredient = str(row.ingredient)
        key = _ingredient_key(int(row.recipe_id), int(row.ingredient_idx))
        rules = resolve_quantity_fields(ingredient, method="rules")
        parse_fields = {
            "quantity": rules.get("quantity"),
            "unit": rules.get("unit"),
            "name": rules.get("name"),
            "size": rules.get("size"),
            "parse_status": rules.get("parse_status"),
            "amount_kind": rules.get("amount_kind"),
        }
        rules_plan = build_resolution_plan(parse_fields, ingredient_raw=ingredient)
        if needs_line_enrichment(ingredient, parse_fields, rules_plan):
            if only_enrich_keys is None or key in only_enrich_keys:
                enrich_items.append((ingredient, parse_fields))
        rows.append({
            "recipe_id": int(row.recipe_id),
            "ingredient_idx": int(row.ingredient_idx),
            "ingredient": ingredient,
            **rules,
        })

    llm_cache: dict[str, dict[str, Any]] = {}
    llm_log: list[dict[str, Any]] = []
    if enrich_items:
        llm_cache, llm_log = run_line_enrichment_sync(
            enrich_items,
            model=model,
            concurrency=enrichment_concurrency,
            progress_writer=progress_writer,
        )

    final_rows: list[dict[str, Any]] = []
    for r in rows:
        parse_fields = {
            "quantity": r.get("quantity"),
            "unit": r.get("unit"),
            "name": r.get("name"),
            "size": r.get("size"),
            "parse_status": r.get("parse_status"),
            "amount_kind": r.get("amount_kind"),
        }
        plan, source, meta = apply_enrichment_to_plan(
            r["ingredient"], parse_fields, enrichment_cache=llm_cache
        )
        kind = plan.primary_amount_kind
        needs_portion = (
            "count_portion" in plan.resolution_paths
            or "explicit_volume" in plan.resolution_paths
        )
        final_rows.append({
            **r,
            "amount_kind_final": kind,
            "amount_kind_source": source,
            "needs_portion": needs_portion,
            "resolution_plan": plan.to_dict(),
            "resolution_paths": plan.resolution_paths,
            "plan_flags": plan.flags,
            **meta,
        })

    df = pd.DataFrame(final_rows)
    summary = {
        "n_lines": len(df),
        "amount_kind_final_counts": df["amount_kind_final"].value_counts().to_dict(),
        "amount_kind_source_counts": df["amount_kind_source"].value_counts().to_dict(),
        "n_needs_portion": int(df["needs_portion"].sum()),
        "n_unknown_rules": int((df["amount_kind"] == "unknown").sum()),
        "n_llm_line_enrichment_calls": len(llm_log),
        "resolution_path_counts": df["resolution_paths"].explode().value_counts().to_dict()
        if "resolution_paths" in df.columns
        else {},
        "plan_flag_counts": df["plan_flags"].explode().value_counts().to_dict()
        if "plan_flags" in df.columns
        else {},
    }
    return df, pd.DataFrame(llm_log), summary


def load_or_classify_amounts(
    recipe_ingredients: pd.DataFrame,
    *,
    model: str,
    paths: dict[str, Path],
    manifest: dict[str, Any],
    force: bool,
    retry_mode: PartialRetryMode | None = None,
    baseline_paths: dict[str, Path] | None = None,
    retry_limit: int | None = None,
    progress_writer: Any = None,
    enrichment_concurrency: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    saved = load_manifest(paths["amount"].parent)
    if (
        not force
        and paths["amount"].is_file()
        and (retry_mode is not None or manifest_matches(saved, manifest))
    ):
        amount_df = pd.read_parquet(paths["amount"])
        amount_llm_df = (
            pd.read_parquet(paths["amount_llm"])
            if paths["amount_llm"].is_file()
            else pd.DataFrame()
        )
        summary = {
            "n_lines": len(amount_df),
            "amount_kind_final_counts": amount_df["amount_kind_final"].value_counts().to_dict(),
            "amount_kind_source_counts": amount_df["amount_kind_source"].value_counts().to_dict(),
            "n_needs_portion": int(amount_df["needs_portion"].sum()),
            "n_unknown_rules": int((amount_df["amount_kind"] == "unknown").sum()),
            "n_llm_line_enrichment_calls": len(amount_llm_df),
            "from_cache": True,
        }
        print(f"Loaded cached amount classification ({len(amount_df)} lines)", flush=True)
        return amount_df, amount_llm_df, summary

    only_enrich_keys: set[tuple[int, int]] | None = None
    if retry_mode is not None and baseline_paths is not None:
        only_enrich_keys = _partial_retry_keys(
            baseline_paths, retry_mode=retry_mode, retry_limit=retry_limit
        )
        if only_enrich_keys:
            print(
                f"Line enrichment LLM scoped to {len(only_enrich_keys)} partial-retry line(s)",
                flush=True,
            )

    amount_df, amount_llm_df, summary = classify_ingredient_lines(
        recipe_ingredients,
        model=model,
        progress_writer=progress_writer,
        enrichment_concurrency=enrichment_concurrency,
        only_enrich_keys=only_enrich_keys,
    )
    amount_df.to_parquet(paths["amount"], index=False)
    if not amount_llm_df.empty:
        amount_llm_df.to_parquet(paths["amount_llm"], index=False)
    write_manifest(paths["amount"].parent, manifest)
    print(f"Amount classification: {summary}", flush=True)
    return amount_df, amount_llm_df, summary


def _ingredient_key(recipe_id: int, ingredient_idx: int) -> tuple[int, int]:
    return (int(recipe_id), int(ingredient_idx))


def _row_in_retry_keys(
    row: Any,
    only_keys: set[tuple[int, int]] | None,
) -> bool:
    if only_keys is None:
        return True
    rid = int(row.recipe_id if hasattr(row, "recipe_id") else row["recipe_id"])
    iidx = int(row.ingredient_idx if hasattr(row, "ingredient_idx") else row["ingredient_idx"])
    return _ingredient_key(rid, iidx) in only_keys


def _fdc_has_classifiable_portion(
    fdc_id: int,
    amount_kind: str,
    *,
    volume_fdc_ids: set[int],
    count_fdc_ids: set[int],
) -> bool:
    fid = int(fdc_id)
    if amount_kind == "volume":
        return fid in volume_fdc_ids
    if amount_kind == "count":
        return fid in count_fdc_ids
    return False


def _portion_llm_row_eligible(row) -> bool:
    """Rescue only when rules still have no_portion after judge portion + heuristics."""
    if bool(row.get("llm_negligible_calories")):
        return False
    if not bool(row.get("needs_portion")):
        return False
    if pd.isna(row.get("llm_fdc_id")) or pd.isna(row.get("quantity")):
        return False
    if pd.notna(row.get("grams")):
        return False
    if row.get("rules_grams_status") != "no_portion":
        return False
    amount_kind = str(row.get("amount_kind_final") or row.get("amount_kind"))
    return amount_kind in ("volume", "count")


def _count_portion_llm_calls(
    matches_df: pd.DataFrame,
    *,
    conn,
    portion_rows_cache: dict[int, list[dict[str, Any]]] | None,
    only_keys: set[tuple[int, int]] | None = None,
) -> int:
    n = 0
    for _, row in matches_df.iterrows():
        if not _row_in_retry_keys(row, only_keys):
            continue
        if not _portion_llm_row_eligible(row):
            continue
        fdc_id = int(row["llm_fdc_id"])
        if portion_rows_cache is not None:
            raw_rows = portion_rows_cache.get(fdc_id, [])
        else:
            raw_rows = _load_portion_rows_for_fdc(conn, fdc_id)
        if raw_rows:
            n += 1
    return n


def apply_portion_llm_pass(
    matches_df: pd.DataFrame,
    parsed_lookup: dict[tuple[int, int], dict[str, Any]],
    *,
    conn,
    model: str,
    portion_rows_cache: dict[int, list[dict[str, Any]]] | None = None,
    progress_writer: Any = None,
    only_keys: set[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """LLM portion pick when judge omitted matched_portion_id or rules conversion failed."""
    from datetime import datetime, timezone

    out = matches_df.copy()
    n_picks = 0
    n_rescued = 0
    if progress_writer is not None:
        n_portion_llm = _count_portion_llm_calls(
            matches_df,
            conn=conn,
            portion_rows_cache=portion_rows_cache,
            only_keys=only_keys,
        )
        progress_writer.set_phase("portion_llm", total=n_portion_llm)

    if only_keys is not None:
        print(
            f"Portion LLM pass: scoped to {len(only_keys)} partial-retry line(s)",
            flush=True,
        )

    for i, row in out.iterrows():
        if not _row_in_retry_keys(row, only_keys):
            continue
        if not _portion_llm_row_eligible(row):
            continue

        fdc_id = int(row["llm_fdc_id"])
        if portion_rows_cache is not None:
            raw_rows = portion_rows_cache.get(fdc_id, [])
        else:
            raw_rows = _load_portion_rows_for_fdc(conn, fdc_id)
        if not raw_rows:
            continue

        amount_kind = str(row.get("amount_kind_final") or row.get("amount_kind"))
        parsed = parsed_lookup.get((int(row["recipe_id"]), int(row["ingredient_idx"])), {})
        llm_meta = pick_portion_sync(
            model,
            ingredient=str(row["ingredient"]),
            quantity=float(row["quantity"]),
            unit=parsed.get("unit"),
            name=parsed.get("name"),
            amount_kind=amount_kind,
            fdc_id=fdc_id,
            raw_rows=raw_rows,
        )
        n_picks += 1
        if progress_writer is not None:
            progress_writer.record_portion_llm_pick()
        else:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            print(
                f"[portion_llm {n_picks}] {ts} r{int(row['recipe_id'])}#{int(row['ingredient_idx'])} "
                f"{str(row['ingredient'])[:40]!r}",
                flush=True,
            )
        out.at[i, "portion_llm_certainty"] = llm_meta.get("certainty")
        out.at[i, "portion_llm_rationale"] = llm_meta.get("rationale")

        pid = llm_meta.get("portion_id")
        if pid is None:
            continue

        gram_result = apply_portion_pick(
            int(pid),
            raw_rows,
            quantity=float(row["quantity"]),
            unit=parsed.get("unit"),
            name=parsed.get("name"),
            amount_kind=amount_kind,
        )
        if gram_result.grams is not None:
            n_rescued += 1
            out.at[i, "grams"] = gram_result.grams
            out.at[i, "grams_status"] = gram_result.status
            out.at[i, "grams_method"] = gram_result.method
            out.at[i, "portion_id"] = gram_result.portion_id
            out.at[i, "portion_resolved_by_llm"] = True

    if progress_writer is None or not progress_writer.quiet:
        print(f"Portion LLM pass: {n_picks} picks, {n_rescued} rescued", flush=True)
    return out


def enrich_matches_with_rules_grams(
    matches_df: pd.DataFrame,
    parsed_lookup: dict[tuple[int, int], dict[str, Any]],
    *,
    conn,
    volume_index: dict,
    count_index: dict,
    portion_rows_cache: dict[int, list[dict[str, Any]]] | None = None,
    progress_writer: FeasibilityProgressWriter | None = None,
    only_keys: set[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    out = matches_df.copy()
    rules_grams: list[float | None] = []
    rules_status: list[str | None] = []
    rules_method: list[str | None] = []
    usda_avail: list[bool] = []
    portion_llm_cert: list[float | None] = []
    portion_llm_rat: list[str | None] = []
    portion_resolved_by_llm: list[bool] = []

    volume_fdc_ids = set(volume_index.keys())
    count_fdc_ids = set(count_index.keys())
    n = len(out)
    if only_keys is not None:
        print(
            f"Rules gram enrichment: {len(only_keys)} partial-retry line(s) "
            f"(preserving baseline for {n - len(only_keys)} others)",
            flush=True,
        )
    else:
        print(f"Rules gram enrichment: {n} rows", flush=True)

    show_progress = progress_writer is None or progress_writer.show_secondary_progress
    for row in iter_progress(
        out.itertuples(index=False), total=n, desc="Rules grams", enabled=show_progress
    ):
        if not _row_in_retry_keys(row, only_keys):
            rules_grams.append(getattr(row, "rules_grams", None))
            rules_status.append(getattr(row, "rules_grams_status", None))
            rules_method.append(getattr(row, "rules_grams_method", None))
            usda_avail.append(bool(getattr(row, "usda_portion_available", False)))
            portion_llm_cert.append(getattr(row, "portion_llm_certainty", None))
            portion_llm_rat.append(getattr(row, "portion_llm_rationale", None))
            portion_resolved_by_llm.append(bool(getattr(row, "portion_resolved_by_llm", False)))
            continue

        parsed = parsed_lookup.get((int(row.recipe_id), int(row.ingredient_idx)), {})
        amount_kind = str(
            getattr(row, "amount_kind_final", None)
            or getattr(row, "amount_kind", None)
            or parsed.get("amount_kind_final", "")
        )
        parsed_for_resolve = {**parsed, "amount_kind": amount_kind, "amount_kind_final": amount_kind}

        matched_pid = getattr(row, "matched_portion_id", None)
        if pd.isna(matched_pid):
            matched_pid = None
        neglig = bool(getattr(row, "llm_negligible_calories", False))
        rules_result = resolve_grams_from_parsed_row(
            parsed_for_resolve,
            int(row.llm_fdc_id) if pd.notna(row.llm_fdc_id) else None,
            portion_index=volume_index,
            count_portion_index=count_index,
            matched_portion_id=int(matched_pid) if matched_pid is not None else None,
            portion_rows_cache=portion_rows_cache,
            llm_negligible_calories=neglig,
        )
        rules_grams.append(rules_result.grams)
        rules_status.append(rules_result.status)
        rules_method.append(rules_result.method)

        avail = False
        if pd.notna(row.llm_fdc_id) and amount_kind in ("volume", "count"):
            avail = _fdc_has_classifiable_portion(
                int(row.llm_fdc_id),
                amount_kind,
                volume_fdc_ids=volume_fdc_ids,
                count_fdc_ids=count_fdc_ids,
            )
        usda_avail.append(avail)
        portion_llm_cert.append(getattr(row, "portion_llm_certainty", None))
        portion_llm_rat.append(getattr(row, "portion_llm_rationale", None))
        portion_resolved_by_llm.append(bool(getattr(row, "portion_resolved_by_llm", False)))

    out["rules_grams"] = rules_grams
    out["rules_grams_status"] = rules_status
    out["rules_grams_method"] = rules_method
    out["usda_portion_available"] = usda_avail
    out["portion_llm_certainty"] = portion_llm_cert
    out["portion_llm_rationale"] = portion_llm_rat
    out["portion_resolved_by_llm"] = portion_resolved_by_llm
    return out


def attach_amount_fields(matches_df: pd.DataFrame, amount_df: pd.DataFrame) -> pd.DataFrame:
    kind_lookup = amount_df.set_index(["recipe_id", "ingredient_idx"])
    out = matches_df.copy()
    for col, src in (
        ("amount_kind_final", "amount_kind_final"),
        ("amount_kind_rules", "amount_kind"),
        ("amount_kind_source", "amount_kind_source"),
        ("needs_portion", "needs_portion"),
    ):
        out[col] = [
            kind_lookup.loc[(int(r.recipe_id), int(r.ingredient_idx)), src]
            for r in out.itertuples(index=False)
        ]
    return out


def _rate(mask: pd.Series, denom: int) -> float:
    return round(float(mask.sum()) / max(denom, 1), 4)


def build_feasibility_report(
    amount_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    *,
    amount_summary: dict[str, Any],
) -> dict[str, Any]:
    # Drop amount-phase columns from matches to avoid _x/_y suffix collision on merge.
    merge_keys = {"recipe_id", "ingredient_idx", "ingredient"}
    overlap = (set(amount_df.columns) & set(matches_df.columns)) - merge_keys
    matches_slim = matches_df.drop(columns=list(overlap), errors="ignore")

    merged = amount_df.merge(
        matches_slim,
        on=["recipe_id", "ingredient_idx", "ingredient"],
        how="left",
    )

    has_fdc = merged["llm_fdc_id"].notna()
    has_grams = merged["grams"].notna()
    both = has_fdc & has_grams
    has_qty = ~merged["quantity"].map(missing_quantity) if "quantity" in merged.columns else pd.Series(True, index=merged.index)

    needs = merged[merged["needs_portion"].fillna(False)]
    needs_fdc = needs["llm_fdc_id"].notna()
    needs_grams = needs["grams"].notna()
    needs_both = needs_fdc & needs_grams

    needs_with_fdc = needs[needs["llm_fdc_id"].notna()]
    usda_avail = needs_with_fdc["usda_portion_available"].fillna(False)
    rules_ok = needs_with_fdc["rules_grams"].notna()
    llm_rescued = needs_with_fdc["portion_resolved_by_llm"].fillna(False)

    report: dict[str, Any] = {
        **amount_summary,
        "n_match_rows": len(matches_df),
        "fdc_match_rate_all": _rate(has_fdc, len(merged)),
        "gram_resolve_rate_all": _rate(has_grams, len(merged)),
        "fdc_and_gram_rate_all": _rate(both, len(merged)),
        "n_no_quantity_lines": int((~has_qty).sum()),
        "gram_resolve_rate_measurable": _rate(merged.loc[has_qty, "grams"].notna(), int(has_qty.sum())),
        "fdc_and_gram_rate_measurable": _rate(
            merged.loc[has_qty, "llm_fdc_id"].notna() & merged.loc[has_qty, "grams"].notna(),
            int(has_qty.sum()),
        ),
        "n_needs_portion_lines": int(needs.shape[0]),
        "fdc_match_rate_needs_portion": _rate(needs_fdc, len(needs)),
        "gram_resolve_rate_needs_portion": _rate(needs_grams, len(needs)),
        "fdc_and_gram_rate_needs_portion": _rate(needs_both, len(needs)),
        "usda_classifiable_portion_rate_given_fdc": _rate(usda_avail, len(needs_with_fdc)),
        "rules_gram_rate_needs_portion_given_fdc": _rate(rules_ok, len(needs_with_fdc)),
        "llm_portion_rescue_rate_needs_portion": _rate(llm_rescued, len(needs_with_fdc)),
        "grams_status_counts": matches_df["grams_status"].value_counts().to_dict()
        if "grams_status" in matches_df.columns
        else {},
        "rules_grams_status_counts": matches_df["rules_grams_status"].value_counts().to_dict()
        if "rules_grams_status" in matches_df.columns
        else {},
        "amount_kind_final_counts": amount_df["amount_kind_final"].value_counts().to_dict(),
        "judge_error_count": int(matches_df["llm_error"].notna().sum())
        if "llm_error" in matches_df.columns
        else 0,
    }

    for kind in ("volume", "count", "mass"):
        sub = merged[merged["amount_kind_final"] == kind]
        if len(sub):
            report[f"n_{kind}"] = len(sub)
            report[f"fdc_and_gram_rate_{kind}"] = _rate(
                sub["llm_fdc_id"].notna() & sub["grams"].notna(), len(sub)
            )
            if kind in ("volume", "count"):
                sub_fdc = sub[sub["llm_fdc_id"].notna()]
                if len(sub_fdc):
                    report[f"rules_gram_rate_{kind}_given_fdc"] = _rate(
                        sub_fdc["rules_grams"].notna(), len(sub_fdc)
                    )
                    report[f"usda_portion_available_rate_{kind}"] = _rate(
                        sub_fdc["usda_portion_available"].fillna(False), len(sub_fdc)
                    )

    return report


def _enrich_report_with_judge_metrics(report: dict[str, Any], matches_df: pd.DataFrame) -> None:
    if "price_estimate_usd" in matches_df.columns:
        report["judge_cost_usd"] = round(float(matches_df["price_estimate_usd"].sum()), 4)
    if "grams_status" in matches_df.columns:
        no_portion = matches_df["grams_status"] == "no_portion"
        report["no_portion_count"] = int(no_portion.sum())
        report["no_portion_rate"] = _rate(no_portion, len(matches_df))


def _finish_openai_partial_feasibility(
    *,
    amount_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    amount_summary: dict[str, Any],
    manifest: dict[str, Any],
    paths: dict[str, Path],
    write_dir: Path,
    t0: float,
    n_recipes: int,
    seed: int,
    model: str,
    sampled_ids: list[int] | None,
    skip_portion_llm: bool,
    progress_writer: Any = None,
) -> dict[str, Any]:
    """Save partial pipeline artifacts and a feasibility report after key exhaustion."""
    from openai_fallback import OPENAI_PARTIAL_STATUS

    matches_df.to_parquet(paths["matches"], index=False)
    write_manifest(write_dir, manifest)
    report = build_feasibility_report(amount_df, matches_df, amount_summary=amount_summary)
    report.update(
        {
            "status": OPENAI_PARTIAL_STATUS,
            "elapsed_sec": round(time.perf_counter() - t0, 1),
            "n_recipes": n_recipes,
            "seed": seed,
            "model": model,
            "sampled_recipe_ids": sampled_ids,
            "skip_portion_llm": skip_portion_llm,
            "resume": manifest.get("resume"),
        }
    )
    _enrich_report_with_judge_metrics(report, matches_df)
    paths["report"].write_text(json.dumps(report, indent=2) + "\n")
    if progress_writer is not None:
        progress_writer.finalize()
    print_openai_resume_hint(write_dir, manifest)
    print(json.dumps(report, indent=2), flush=True)
    return report


def _finish_feasibility_run(
    *,
    report: dict[str, Any],
    manifest: dict[str, Any],
    matches_df: pd.DataFrame,
    paths: dict[str, Path],
    write_dir: Path,
    feasibility_version: int | None,
    mlflow_experiment: str,
    use_mlflow: bool,
    extra_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if extra_report:
        report.update(extra_report)
    _enrich_report_with_judge_metrics(report, matches_df)

    if feasibility_version is not None:
        report["feasibility_version"] = feasibility_version
        manifest["feasibility_version"] = feasibility_version

    paths["report"].write_text(json.dumps(report, indent=2) + "\n")
    write_manifest(write_dir, manifest)
    print(json.dumps(report, indent=2), flush=True)

    if not use_mlflow or feasibility_version is None:
        return report

    run_name = (
        f"feasibility_v{feasibility_version}_n{report.get('n_recipes')}_seed{report.get('seed')}"
    )
    mlflow_run_id = log_feasibility_run(
        experiment_name=mlflow_experiment,
        run_name=run_name,
        feasibility_version=feasibility_version,
        report=report,
        params={
            "n_recipes": report.get("n_recipes"),
            "seed": report.get("seed"),
            "model": report.get("model"),
            "prompt_version": manifest.get("prompt_version"),
            "only_no_portion": manifest.get("only_no_portion", False),
            "baseline_dir": manifest.get("baseline_dir"),
            "write_dir": str(write_dir),
            "finalize_only": report.get("finalize_only", False),
            "skip_portion_llm": report.get("skip_portion_llm", False),
        },
        artifact_paths={
            "feasibility_report": paths["report"],
            "pipeline_matches": paths["matches"],
            "judge_matches_raw": paths["judge_raw"],
            "run_manifest": write_dir / MANIFEST_PATH,
        },
        tags={"prompt_version": str(manifest.get("prompt_version", ""))},
    )
    if mlflow_run_id:
        report["mlflow_run_id"] = mlflow_run_id
        report["mlflow_experiment"] = mlflow_experiment
        paths["report"].write_text(json.dumps(report, indent=2) + "\n")
        manifest["mlflow_run_id"] = mlflow_run_id
        write_manifest(write_dir, manifest)
        print(
            f"Logged MLflow run {mlflow_run_id} "
            f"(experiment '{mlflow_experiment}', feasibility_version={feasibility_version})",
            flush=True,
        )
    return report


def _load_payloads(paths: dict[str, Path], manifest: dict[str, Any], force: bool) -> list[dict] | None:
    saved = load_manifest(paths["payloads"].parent)
    if force or not paths["payloads"].is_file() or not manifest_matches(saved, manifest):
        return None
    print(f"Loaded cached payloads ({paths['payloads'].stat().st_size // 1024} KB)", flush=True)
    with paths["payloads"].open("rb") as f:
        return pickle.load(f)


def _save_payloads(paths: dict[str, Path], payloads: list[dict]) -> None:
    with paths["payloads"].open("wb") as f:
        pickle.dump(payloads, f, protocol=pickle.HIGHEST_PROTOCOL)


def _subset_parsed_and_embeddings(
    parsed: pd.DataFrame,
    name_emb,
    prep_emb,
    dequant_emb,
    keys: set[tuple[int, int]],
) -> tuple[pd.DataFrame, Any, Any, Any]:
    """Keep parsed rows (and aligned embedding rows) whose keys are in *keys*."""
    if not keys:
        empty = parsed.iloc[0:0].reset_index(drop=True)
        return empty, name_emb[:0], prep_emb[:0], dequant_emb[:0]
    idx = np.array(
        [
            i
            for i, r in enumerate(parsed.itertuples(index=False))
            if (int(r.recipe_id), int(r.ingredient_idx)) in keys
        ],
        dtype=int,
    )
    return (
        parsed.iloc[idx].reset_index(drop=True),
        name_emb[idx],
        prep_emb[idx],
        dequant_emb[idx],
    )


def _print_partial_retry_subset_preview(
    parsed: pd.DataFrame,
    *,
    retry_keys: set[tuple[int, int]],
    total_lines: int,
    retry_mode: PartialRetryMode,
) -> None:
    label = "unresolved (missing fdc and/or grams)" if retry_mode == "unresolved" else "no_portion"
    print(
        f"partial retry ({label}): {len(parsed)} / {total_lines} lines "
        f"({len(retry_keys)} keys from baseline)",
        flush=True,
    )
    for i, ing in enumerate(parsed["ingredient"].head(8), start=1):
        print(f"  [{i}] {str(ing)[:120]}", flush=True)


def _load_baseline_matches_df(paths: dict[str, Path]) -> pd.DataFrame:
    for cache in (paths["matches"], paths["judge_raw"]):
        if cache.is_file():
            return pd.read_parquet(cache)
    raise FileNotFoundError(
        "Partial retry requires pipeline_matches.parquet or judge_matches_raw.parquet "
        "from a prior run"
    )


def _partial_retry_keys(
    paths: dict[str, Path],
    *,
    retry_mode: PartialRetryMode,
    retry_limit: int | None = None,
) -> set[tuple[int, int]]:
    df = _load_baseline_matches_df(paths)
    if retry_mode == "no_portion":
        if "grams_status" not in df.columns:
            raise ValueError("--only-no-portion requires grams_status on baseline artifacts")
        sub = df[df["grams_status"] == "no_portion"]
    else:
        has_fdc = df["llm_fdc_id"].notna() if "llm_fdc_id" in df.columns else pd.Series(False, index=df.index)
        has_grams = df["grams"].notna() if "grams" in df.columns else pd.Series(False, index=df.index)
        sub = df[~has_fdc | ~has_grams]
    keys = {
        (int(r.recipe_id), int(r.ingredient_idx))
        for r in sub.itertuples(index=False)
    }
    if retry_limit is not None and retry_limit >= 0:
        keys = set(sorted(keys)[:retry_limit])
    return keys


def _no_portion_keys(paths: dict[str, Path]) -> set[tuple[int, int]]:
    return _partial_retry_keys(paths, retry_mode="no_portion")


def _filter_payloads(
    payloads: list[dict],
    *,
    keys: set[tuple[int, int]] | None = None,
    skip_keys: set[tuple[int, int]] | None = None,
) -> list[dict]:
    out: list[dict] = []
    for p in payloads:
        key = (int(p["recipe_id"]), int(p["ingredient_idx"]))
        if keys is not None and key not in keys:
            continue
        if skip_keys and key in skip_keys:
            continue
        out.append(p)
    return out


def _v4_completed_keys(df: pd.DataFrame) -> set[tuple[int, int]]:
    if df.empty or "prompt_version" not in df.columns:
        return set()
    sub = df[df["prompt_version"] == PROMPT_VERSION]
    return {(int(r.recipe_id), int(r.ingredient_idx)) for r in sub.itertuples(index=False)}


def run_judging_cached(
    payloads: list[dict],
    *,
    model: str,
    concurrency: int,
    paths: dict[str, Path],
    manifest: dict[str, Any],
    force: bool,
    retry_mode: PartialRetryMode | None = None,
    baseline_paths: dict[str, Path] | None = None,
    retry_limit: int | None = None,
    total_dataset_lines: int | None = None,
    disk_flush_every: int = 100,
    judge_log_every: int = 25,
    heartbeat_sec: float = 15.0,
    progress_writer: Any = None,
    dequant_cache_path: Path | None = None,
    no_dequant_cache: bool = False,
) -> pd.DataFrame:
    saved = load_manifest(paths["judge_raw"].parent)
    cache_path = paths["judge_raw"]
    baseline_paths = baseline_paths or paths

    retry_keys: set[tuple[int, int]] | None = None
    base_df = load_judge_checkpoint(cache_path)

    if retry_mode is not None:
        retry_keys = _partial_retry_keys(
            baseline_paths, retry_mode=retry_mode, retry_limit=retry_limit
        )
        payloads = _filter_payloads(payloads, keys=retry_keys)
        write_ckpt = load_judge_checkpoint(cache_path)
        if not force and not write_ckpt.empty:
            skip = _v4_completed_keys(write_ckpt) & retry_keys
            payloads = _filter_payloads(payloads, skip_keys=skip)
        base_df = _load_baseline_judge_df(baseline_paths)
        if not base_df.empty and retry_keys:
            base_df = base_df[
                ~base_df.apply(
                    lambda r: (int(r["recipe_id"]), int(r["ingredient_idx"])) in retry_keys,
                    axis=1,
                )
            ]
        print(
            f"partial retry ({retry_mode}): {len(retry_keys)} failure keys, {len(payloads)} to judge "
            f"(baseline preserved: {len(base_df)} rows)",
            flush=True,
        )
    elif not force and not cache_path.is_file() and paths["matches"].is_file() and manifest_matches(saved, manifest):
        cache_path = paths["matches"]
    elif not force and cache_path.is_file() and manifest_matches(saved, manifest) and retry_mode is None:
        if not is_openai_partial_manifest(saved):
            df = pd.read_parquet(cache_path)
            if cache_path == paths["matches"] and not paths["judge_raw"].is_file():
                judge_cols = [c for c in df.columns if c not in (
                    "rules_grams", "rules_grams_status", "rules_grams_method",
                    "usda_portion_available", "portion_resolved_by_llm",
                )]
                df[judge_cols].to_parquet(paths["judge_raw"], index=False)
                print("Migrated judge cache → judge_matches_raw.parquet", flush=True)
            n_ok = df["llm_fdc_id"].notna().sum() if "llm_fdc_id" in df.columns else 0
            print(f"Loaded cached judge results ({len(df)} rows, {n_ok} fdc matches)", flush=True)
            return df
        pending = len(payloads)
        if cache_path.is_file():
            skip = completed_keys(load_judge_checkpoint(cache_path))
            payloads = _filter_payloads(payloads, skip_keys=skip)
            print(
                f"Resuming judge after OpenAI key exhaustion: "
                f"{len(skip)} done, {len(payloads)} pending (of {pending} total payloads)",
                flush=True,
            )

    if not payloads:
        if not base_df.empty:
            print("No pending judge payloads; using cached judge results", flush=True)
            return base_df
        raise ValueError("No payloads to judge")

    resolved_cache = resolve_dequant_cache_path(dequant_cache_path)
    dequant_cache_runtime = None
    if not no_dequant_cache:
        cache_write_path = paths["judge_raw"].parent / "dequant_norm_llm_cache.json"
        dequant_cache_runtime = create_dequant_cache_runtime(
            load_path=resolved_cache,
            write_path=cache_write_path,
        )
        n_cache_hits, payloads = apply_dequant_cache_to_payloads(
            payloads,
            dequant_cache_runtime.entries,
            runtime=dequant_cache_runtime,
        )
        manifest["dequant_cache_path"] = str(resolved_cache or cache_write_path)
        manifest["dequant_cache_write_path"] = str(cache_write_path)
        manifest["dequant_cache_terms_initial"] = dequant_cache_runtime.stats["initial_terms"]
        manifest["dequant_cache_hits_initial"] = n_cache_hits
        print(
            f"Dequant cache: {n_cache_hits:,}/{len(payloads):,} payloads hit "
            f"({dequant_cache_runtime.stats['initial_terms']} initial terms from "
            f"{resolved_cache.name if resolved_cache else 'empty'})",
            flush=True,
        )
    elif dequant_cache_path is not None:
        print(f"Warning: dequant cache disabled via --no-dequant-cache", flush=True)

    if cache_path.is_file() and retry_mode is None and not force:
        skip = completed_keys(load_judge_checkpoint(cache_path))
        if skip:
            before = len(payloads)
            payloads = _filter_payloads(payloads, skip_keys=skip)
            if before != len(payloads):
                print(
                    f"Skipping {len(skip)} already-judged lines ({len(payloads)} pending)",
                    flush=True,
                )

    run_id = uuid.uuid4().hex
    run_name = manifest.get("run_name", "feasibility")
    pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])
    quiet = bool(progress_writer is not None and getattr(progress_writer, "quiet", False))

    all_exp, _, breaker = asyncio.run(
        run_judging(
            payloads,
            run_id=run_id,
            run_name=run_name,
            model=model,
            pricing=pricing,
            concurrency=concurrency,
            flush_every=100,
            conn=None,
            use_supabase=False,
            assemble_fn=assemble_rows_portion,
            system_prompt=SYSTEM_PROMPT,
            disk_checkpoint_path=cache_path,
            disk_flush_every=disk_flush_every,
            log_every=judge_log_every,
            heartbeat_sec=heartbeat_sec,
            total_dataset_lines=total_dataset_lines or manifest.get("n_lines"),
            progress_writer=progress_writer,
            verbose=not quiet,
            dequant_cache_runtime=dequant_cache_runtime,
        )
    )

    if dequant_cache_runtime is not None:
        saved_cache = dequant_cache_runtime.save()
        cache_summary = dequant_cache_runtime.summary()
        manifest["dequant_cache_stats"] = cache_summary
        manifest["dequant_cache_terms_final"] = cache_summary["final_terms"]
        manifest["dequant_cache_hits_total"] = cache_summary["total_cache_hits"]
        print(
            f"Dequant cache saved: {saved_cache} | "
            f"hits initial={cache_summary['initial_hits']} growth={cache_summary['runtime_growth_hits']} "
            f"llm={cache_summary['llm_calls']} terms+={cache_summary['terms_added_during_run']} "
            f"final_terms={cache_summary['final_terms']}",
            flush=True,
        )

    keys_exhausted = bool(breaker and breaker.get("reason") == "openai_keys_exhausted")

    if progress_writer is not None:
        new_df = pd.DataFrame(all_exp)
        if retry_mode is not None and not base_df.empty:
            df = pd.concat([base_df, new_df], ignore_index=True)
        elif new_df.empty:
            df = load_judge_checkpoint(cache_path)
        else:
            df = new_df
        df.to_parquet(cache_path, index=False)
    else:
        new_df = pd.DataFrame(all_exp)
        if retry_mode is not None and not base_df.empty:
            df = pd.concat([base_df, new_df], ignore_index=True)
        else:
            df = new_df
        df.to_parquet(cache_path, index=False)
    if retry_mode is not None:
        manifest["partial_retry_mode"] = retry_mode
        manifest["only_no_portion"] = retry_mode == "no_portion"
        manifest["only_unresolved"] = retry_mode == "unresolved"
    total_lines = total_dataset_lines or manifest.get("n_lines") or len(df)
    if keys_exhausted:
        mark_openai_partial_manifest(
            manifest,
            phase="judging",
            completed=len(completed_keys(df)),
            total=int(total_lines),
            key_status=breaker.get("key_status") if breaker else None,
        )
        print_openai_resume_hint(paths["judge_raw"].parent, manifest)
    else:
        manifest.pop("status", None)
        manifest.pop("resume", None)
        manifest.pop("openai_key_status", None)
    write_manifest(paths["judge_raw"].parent, manifest)
    n_ok = df["llm_fdc_id"].notna().sum()
    print(f"Saved judge results ({len(df)} rows, {n_ok} fdc matches)", flush=True)
    return df


def _clear_phase_caches(paths: dict[str, Path]) -> None:
    for key in ("amount", "amount_llm", "judge_raw", "matches", "payloads", "report"):
        p = paths[key]
        if p.is_file():
            p.unlink()
            print(f"Removed cached {p.name}", flush=True)


def run_feasibility(
    *,
    n_recipes: int,
    seed: int,
    model: str,
    out_dir: Path,
    baseline_dir: Path | None,
    food_cache_dir: Path,
    limit: int | None,
    concurrency: int,
    skip_portion_llm: bool,
    force_amount: bool,
    force_judging: bool,
    force_payloads: bool,
    force_all: bool,
    finalize_only: bool,
    only_no_portion: bool = False,
    only_unresolved: bool = False,
    retry_limit: int | None = None,
    use_mlflow: bool = True,
    mlflow_experiment: str = DEFAULT_EXPERIMENT,
    progress_mode: str = "default",
    disk_flush_every: int = 100,
    judge_log_every: int = 25,
    parquet_compact_every: int = 50,
    enrichment_concurrency: int = 8,
    sample_manifest: Path | None = None,
    recipe_csv: Path | None = None,
    sample_lines: int | None = None,
    sample_seed: int = 42,
    recipe_cache_dir: Path | None = None,
    dequant_cache_path: Path | None = None,
    no_dequant_cache: bool = False,
) -> dict[str, Any]:
    load_dotenv()
    retry_mode = partial_retry_mode(
        only_no_portion=only_no_portion,
        only_unresolved=only_unresolved,
    )
    baseline_dir_resolved, write_dir = resolve_run_dirs(
        out_dir=out_dir,
        baseline_dir=baseline_dir,
        retry_mode=retry_mode,
    )
    write_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(write_dir)
    baseline_paths = _paths(baseline_dir_resolved)
    t0 = time.perf_counter()

    progress_writer: FeasibilityProgressWriter | None = None
    if progress_mode == "colab":
        progress_writer = FeasibilityProgressWriter(
            write_dir,
            run_id=write_dir.name,
            quiet=True,
            flush_every=10,
            parquet_compact_every=parquet_compact_every,
            s3_sync_every=parquet_compact_every,
        )
        disk_flush_every = 10
        judge_log_every = max(judge_log_every, 10_000)
        heartbeat_sec = 10.0
    else:
        heartbeat_sec = 15.0

    if retry_mode is not None:
        if progress_writer is None or not progress_writer.quiet:
            print(
                f"partial retry ({retry_mode}): baseline (read-only) {baseline_dir_resolved}\n"
                f"              writes -> {write_dir}",
                flush=True,
            )
        _seed_baseline_artifacts(baseline_paths, paths)

    if force_all:
        force_amount = force_judging = force_payloads = True

    if force_all and not finalize_only and retry_mode is None:
        _clear_phase_caches(paths)

    quiet_ctx = (
        progress_writer.quiet_context()
        if progress_writer is not None
        else nullcontext()
    )

    with quiet_ctx:
        return _run_feasibility_pipeline(
            n_recipes=n_recipes,
            seed=seed,
            model=model,
            limit=limit,
            concurrency=concurrency,
            skip_portion_llm=skip_portion_llm,
            force_amount=force_amount,
            force_judging=force_judging,
            force_payloads=force_payloads,
            force_all=force_all,
            finalize_only=finalize_only,
            retry_mode=retry_mode,
            retry_limit=retry_limit,
            use_mlflow=use_mlflow,
            mlflow_experiment=mlflow_experiment,
            enrichment_concurrency=enrichment_concurrency,
            sample_manifest=sample_manifest,
            recipe_csv=recipe_csv,
            sample_lines=sample_lines,
            sample_seed=sample_seed,
            recipe_cache_dir=recipe_cache_dir,
            baseline_dir_resolved=baseline_dir_resolved,
            write_dir=write_dir,
            paths=paths,
            baseline_paths=baseline_paths,
            progress_writer=progress_writer,
            disk_flush_every=disk_flush_every,
            judge_log_every=judge_log_every,
            heartbeat_sec=heartbeat_sec,
            food_cache_dir=food_cache_dir,
            t0=t0,
            dequant_cache_path=dequant_cache_path,
            no_dequant_cache=no_dequant_cache,
        )


def _run_feasibility_pipeline(
    *,
    n_recipes: int,
    seed: int,
    model: str,
    limit: int | None,
    concurrency: int,
    skip_portion_llm: bool,
    force_amount: bool,
    force_judging: bool,
    force_payloads: bool,
    force_all: bool,
    finalize_only: bool,
    retry_mode: PartialRetryMode | None,
    retry_limit: int | None,
    use_mlflow: bool,
    mlflow_experiment: str,
    enrichment_concurrency: int,
    sample_manifest: Path | None,
    recipe_csv: Path | None,
    recipe_cache_dir: Path | None,
    sample_lines: int | None,
    sample_seed: int,
    baseline_dir_resolved: Path,
    write_dir: Path,
    paths: dict[str, Path],
    baseline_paths: dict[str, Path],
    progress_writer: FeasibilityProgressWriter | None,
    disk_flush_every: int,
    judge_log_every: int,
    heartbeat_sec: float,
    food_cache_dir: Path,
    t0: float,
    dequant_cache_path: Path | None = None,
    no_dequant_cache: bool = False,
) -> dict[str, Any]:
    from sample_recipes import DEFAULT_RECIPE_CSV, load_sampled_recipes

    recipes, recipe_ingredients, sampled_ids = load_sampled_recipes(
        n=n_recipes,
        seed=seed,
        sample_manifest=sample_manifest,
        recipe_csv=recipe_csv or DEFAULT_RECIPE_CSV,
    )
    if limit is not None:
        recipe_ingredients = recipe_ingredients.head(limit)

    if sample_lines is not None:
        n_sample = min(int(sample_lines), len(recipe_ingredients))
        rng = random.Random(int(sample_seed))
        pick = sorted(rng.sample(range(len(recipe_ingredients)), n_sample))
        recipe_ingredients = recipe_ingredients.iloc[pick].reset_index(drop=True)

    manifest = build_manifest(
        n_recipes=n_recipes,
        seed=seed,
        model=model,
        limit=limit,
        n_lines=len(recipe_ingredients),
    )
    if sample_lines is not None:
        manifest["sample_lines"] = int(sample_lines)
        manifest["sample_seed"] = int(sample_seed)
        manifest["sampled_line_indices"] = pick
        manifest["sampled_line_keys"] = [
            [int(r.recipe_id), int(r.ingredient_idx)]
            for r in recipe_ingredients.itertuples(index=False)
        ]
    manifest["run_name"] = f"feasibility_{n_recipes}_seed{seed}"
    if retry_mode is not None:
        manifest["partial_retry_mode"] = retry_mode
        manifest["only_no_portion"] = retry_mode == "no_portion"
        manifest["only_unresolved"] = retry_mode == "unresolved"
        manifest["baseline_dir"] = str(baseline_dir_resolved)
    if retry_limit is not None:
        manifest["retry_limit"] = retry_limit

    feasibility_version: int | None = None
    if use_mlflow:
        feasibility_version = next_feasibility_version(mlflow_experiment)
        print(
            f"MLflow experiment '{mlflow_experiment}', feasibility_version={feasibility_version}",
            flush=True,
        )

    if progress_writer is not None:
        progress_writer.set_phase("sample_load", total=len(recipe_ingredients))
    print(f"Loaded {len(recipes)} recipes, {len(recipe_ingredients)} ingredient lines", flush=True)

    if finalize_only:
        if not paths["matches"].is_file():
            raise FileNotFoundError(f"--finalize-only requires {paths['matches']}")
        amount_df = pd.read_parquet(paths["amount"])
        matches_df = pd.read_parquet(paths["matches"])
        amount_summary = {
            "n_lines": len(amount_df),
            "n_needs_portion": int(amount_df["needs_portion"].sum()),
            "from_cache": True,
        }
        report = build_feasibility_report(amount_df, matches_df, amount_summary=amount_summary)
        return _finish_feasibility_run(
            report=report,
            manifest=manifest,
            matches_df=matches_df,
            paths=paths,
            write_dir=write_dir,
            feasibility_version=feasibility_version,
            mlflow_experiment=mlflow_experiment,
            use_mlflow=use_mlflow,
            extra_report={
                "elapsed_sec": round(time.perf_counter() - t0, 1),
                "n_recipes": n_recipes,
                "seed": seed,
                "model": model,
                "finalize_only": True,
            },
        )

    if progress_writer is not None:
        progress_writer.set_phase("amount_classify", total=len(recipe_ingredients))
    amount_df, amount_llm_df, amount_summary = load_or_classify_amounts(
        recipe_ingredients,
        model=model,
        paths=paths,
        manifest=manifest,
        force=force_amount,
        retry_mode=retry_mode,
        baseline_paths=baseline_paths if retry_mode is not None else None,
        retry_limit=retry_limit,
        progress_writer=progress_writer,
        enrichment_concurrency=enrichment_concurrency,
    )

    if progress_writer is not None:
        progress_writer.set_phase("recipe_artifacts")
    directions_by_recipe = {
        int(r.recipe_id): parse_directions_list(r.directions)
        for r in recipes.itertuples(index=False)
    }

    show_progress = progress_writer is None or progress_writer.show_secondary_progress
    parsed, name_emb, prep_emb, dequant_emb, _ = load_or_build_recipe_artifacts(
        recipe_ingredients,
        recipe_cache_dir or paths["amount"].parent / "recipe_cache",
        force=force_all and retry_mode is None,
        show_progress=show_progress,
    )

    kind_lookup = amount_df.set_index(["recipe_id", "ingredient_idx"])
    parsed = parsed.copy()
    for col, src in (
        ("amount_kind_final", "amount_kind_final"),
        ("amount_kind_rules", "amount_kind"),
        ("amount_kind_source", "amount_kind_source"),
        ("needs_portion", "needs_portion"),
        ("resolution_plan", "resolution_plan"),
        ("line_enrichment", "line_enrichment"),
    ):
        if src in kind_lookup.columns:
            parsed[col] = [
                kind_lookup.loc[(int(r.recipe_id), int(r.ingredient_idx)), src]
                for r in parsed.itertuples(index=False)
            ]

    parsed_lookup = {
        (int(r["recipe_id"]), int(r["ingredient_idx"])): r
        for r in parsed.to_dict(orient="records")
    }

    match_config = StagedMatchConfig()
    retr_config = LLMRetrievalConfig()
    food_index = build_food_index(
        food_cache_dir, match_config, force=force_all, show_progress=show_progress
    )

    with connect() as conn:
        capabilities = build_portion_capability_sets(conn)
        volume_index = build_portion_index(conn)
        count_index = build_count_portion_index(conn)
        portion_summary_index = load_or_build_portion_summary_index(
            conn, refresh=force_all
        )

        retry_keys_for_retrieval: set[tuple[int, int]] | None = None
        parsed_for_payloads = parsed
        name_emb_p, prep_emb_p, dequant_emb_p = name_emb, prep_emb, dequant_emb
        if retry_mode is not None:
            retry_keys_for_retrieval = _partial_retry_keys(
                baseline_paths, retry_mode=retry_mode, retry_limit=retry_limit
            )
            parsed_for_payloads, name_emb_p, prep_emb_p, dequant_emb_p = (
                _subset_parsed_and_embeddings(
                    parsed,
                    name_emb,
                    prep_emb,
                    dequant_emb,
                    retry_keys_for_retrieval,
                )
            )
            _print_partial_retry_subset_preview(
                parsed_for_payloads,
                retry_keys=retry_keys_for_retrieval,
                total_lines=len(recipe_ingredients),
                retry_mode=retry_mode,
            )
            manifest[f"n_{retry_mode}_retry"] = len(retry_keys_for_retrieval)

        payloads = _load_payloads(paths, manifest, force_payloads)
        if payloads is not None and retry_mode is not None and retry_keys_for_retrieval:
            before = len(payloads)
            payloads = _filter_payloads(payloads, keys=retry_keys_for_retrieval)
            print(
                f"Filtered cached payloads: {before} -> {len(payloads)} {retry_mode} rows",
                flush=True,
            )
        if payloads is None:
            if progress_writer is not None:
                progress_writer.set_phase(
                    "payloads",
                    total=len(parsed_for_payloads) if limit is None else min(limit, len(parsed_for_payloads)),
                )
            payloads = precompute_payloads_portion(
                parsed_for_payloads,
                name_emb_p,
                prep_emb_p,
                dequant_emb_p,
                food_index,
                directions_by_recipe,
                retr_config,
                capabilities,
                volume_index,
                count_index,
                portion_summary_index,
                limit=limit,
                progress_writer=progress_writer,
            )
            _save_payloads(paths, payloads)
            write_manifest(write_dir, manifest)

        if progress_writer is not None:
            progress_writer.set_phase("judging", total=len(payloads))
        matches_df = run_judging_cached(
            payloads,
            model=model,
            concurrency=concurrency,
            paths=paths,
            manifest=manifest,
            force=force_judging,
            retry_mode=retry_mode,
            baseline_paths=baseline_paths if retry_mode is not None else None,
            retry_limit=retry_limit,
            total_dataset_lines=len(recipe_ingredients),
            disk_flush_every=disk_flush_every,
            judge_log_every=judge_log_every,
            heartbeat_sec=heartbeat_sec,
            progress_writer=progress_writer,
            dequant_cache_path=None if no_dequant_cache else dequant_cache_path,
            no_dequant_cache=no_dequant_cache,
        )
        matches_df = attach_amount_fields(matches_df, amount_df)

        if is_openai_partial_manifest(load_manifest(write_dir)):
            return _finish_openai_partial_feasibility(
                amount_df=amount_df,
                matches_df=matches_df,
                amount_summary=amount_summary,
                manifest=manifest,
                paths=paths,
                write_dir=write_dir,
                t0=t0,
                n_recipes=n_recipes,
                seed=seed,
                model=model,
                sampled_ids=sampled_ids,
                skip_portion_llm=skip_portion_llm,
                progress_writer=progress_writer,
            )

        portion_rows_cache: dict[int, list[dict[str, Any]]] | None = None
        llm_scope_keys = retry_keys_for_retrieval if retry_mode is not None else None
        needs_enrich = "rules_grams" not in matches_df.columns
        if needs_enrich or force_judging or force_amount or force_all:
            if progress_writer is not None:
                progress_writer.set_phase("rules_grams")
            if llm_scope_keys is not None:
                scoped = matches_df[
                    matches_df.apply(
                        lambda r: _ingredient_key(int(r["recipe_id"]), int(r["ingredient_idx"]))
                        in llm_scope_keys,
                        axis=1,
                    )
                ]
                matched_fdc_ids = {
                    int(x) for x in scoped["llm_fdc_id"].dropna().unique()
                }
            else:
                matched_fdc_ids = {
                    int(x) for x in matches_df["llm_fdc_id"].dropna().unique()
                }
            print(
                f"Batch-loading portion rows for {len(matched_fdc_ids):,} matched fdc_ids…",
                flush=True,
            )
            portion_rows_cache = load_portion_rows_cache(conn, matched_fdc_ids)
            matches_df = enrich_matches_with_rules_grams(
                matches_df,
                parsed_lookup,
                conn=conn,
                volume_index=volume_index,
                count_index=count_index,
                portion_rows_cache=portion_rows_cache,
                progress_writer=progress_writer,
                only_keys=llm_scope_keys,
            )
            if llm_scope_keys is not None:
                retry_mask = matches_df.apply(
                    lambda r: _ingredient_key(int(r["recipe_id"]), int(r["ingredient_idx"]))
                    in llm_scope_keys,
                    axis=1,
                )
                matches_df.loc[retry_mask, "grams"] = matches_df.loc[retry_mask, "rules_grams"]
                matches_df.loc[retry_mask, "grams_status"] = matches_df.loc[
                    retry_mask, "rules_grams_status"
                ]
                matches_df.loc[retry_mask, "grams_method"] = matches_df.loc[
                    retry_mask, "rules_grams_method"
                ]
            else:
                matches_df["grams"] = matches_df["rules_grams"]
                matches_df["grams_status"] = matches_df["rules_grams_status"]
                matches_df["grams_method"] = matches_df["rules_grams_method"]

        if not skip_portion_llm:
            already_done = (
                "portion_resolved_by_llm" in matches_df.columns
                and matches_df["portion_resolved_by_llm"].fillna(False).any()
            )
            should_run_portion_llm = (
                retry_mode is not None
                or force_judging
                or force_all
                or not already_done
            )
            if should_run_portion_llm:
                from openai_fallback import AllKeysExhaustedError, get_key_pool_status

                try:
                    matches_df = apply_portion_llm_pass(
                        matches_df,
                        parsed_lookup,
                        conn=conn,
                        model=model,
                        portion_rows_cache=portion_rows_cache,
                        progress_writer=progress_writer,
                        only_keys=llm_scope_keys,
                    )
                except AllKeysExhaustedError:
                    mark_openai_partial_manifest(
                        manifest,
                        phase="portion_llm",
                        completed=int(matches_df["portion_resolved_by_llm"].fillna(False).sum())
                        if "portion_resolved_by_llm" in matches_df.columns
                        else 0,
                        total=_count_portion_llm_calls(
                            matches_df,
                            conn=conn,
                            portion_rows_cache=portion_rows_cache,
                            only_keys=llm_scope_keys,
                        ),
                        key_status=get_key_pool_status(),
                    )
                    write_manifest(write_dir, manifest)
                    matches_df.to_parquet(paths["matches"], index=False)
                    return _finish_openai_partial_feasibility(
                        amount_df=amount_df,
                        matches_df=matches_df,
                        amount_summary=amount_summary,
                        manifest=manifest,
                        paths=paths,
                        write_dir=write_dir,
                        t0=t0,
                        n_recipes=n_recipes,
                        seed=seed,
                        model=model,
                        sampled_ids=sampled_ids,
                        skip_portion_llm=skip_portion_llm,
                        progress_writer=progress_writer,
                    )

    if progress_writer is not None:
        progress_writer.set_phase("report")
    matches_df.to_parquet(paths["matches"], index=False)
    write_manifest(write_dir, manifest)

    report = build_feasibility_report(amount_df, matches_df, amount_summary=amount_summary)
    if progress_writer is not None:
        progress_writer.finalize()
    return _finish_feasibility_run(
        report=report,
        manifest=manifest,
        matches_df=matches_df,
        paths=paths,
        write_dir=write_dir,
        feasibility_version=feasibility_version,
        mlflow_experiment=mlflow_experiment,
        use_mlflow=use_mlflow,
        extra_report={
            "elapsed_sec": round(time.perf_counter() - t0, 1),
            "n_recipes": n_recipes,
            "seed": seed,
            "model": model,
            "sampled_recipe_ids": sampled_ids,
            "skip_portion_llm": skip_portion_llm,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Portion pipeline feasibility report")
    parser.add_argument("--n-recipes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--food-cache-dir", type=Path, default=DEFAULT_FOOD_CACHE)
    parser.add_argument("--limit", type=int, default=None, help="Limit ingredient lines for dry run")
    parser.add_argument(
        "--sample-lines",
        type=int,
        default=None,
        help="Random sample N ingredient lines (after --limit) using --sample-seed",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="RNG seed for --sample-lines (default: 42)",
    )
    parser.add_argument(
        "--retry-limit",
        type=int,
        default=None,
        help=(
            "With --only-unresolved/--only-no-portion: max failure lines to re-process; "
            "all LLM calls (judge, portion pick, line enrichment) are limited to those keys"
        ),
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--skip-portion-llm", action="store_true")
    parser.add_argument("--finalize-only", action="store_true", help="Rebuild report from cached parquets")
    parser.add_argument("--force-amount", action="store_true", help="Recompute amount classification")
    parser.add_argument("--force-judging", action="store_true", help="Re-run LLM fdc matching")
    parser.add_argument("--force-payloads", action="store_true", help="Recompute retrieval payloads")
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Full fresh run: clear phase caches, re-embed, re-classify, re-judge, re-enrich",
    )
    parser.add_argument(
        "--only-no-portion",
        action="store_true",
        help="Re-judge no_portion rows; read --baseline-dir, write separate v4 output dir",
    )
    parser.add_argument(
        "--only-unresolved",
        action="store_true",
        help="Re-judge rows missing llm_fdc_id and/or grams; merge with baseline resolved rows",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Prior run artifacts to read (default depends on retry mode). Never modified.",
    )
    parser.add_argument(
        "--dequant-cache",
        type=Path,
        default=None,
        help="dequant_norm LLM skip cache JSON (default: data/dequant_norm_llm_cache.json if present)",
    )
    parser.add_argument(
        "--no-dequant-cache",
        action="store_true",
        help="Disable dequant_norm cache even if a default file exists",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument(
        "--mlflow-experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT})",
    )
    args = parser.parse_args()

    out = args.out_dir
    if args.n_recipes != 1000:
        out = out.parent / f"portion_feasibility_n{args.n_recipes}_seed{args.seed}"

    baseline = args.baseline_dir
    if args.only_no_portion and baseline is None:
        baseline = DEFAULT_BASELINE
    if args.only_unresolved and baseline is None:
        baseline = DEFAULT_V4_OUT

    run_feasibility(
        n_recipes=args.n_recipes,
        seed=args.seed,
        model=args.model,
        out_dir=out,
        baseline_dir=baseline,
        food_cache_dir=args.food_cache_dir,
        limit=args.limit,
        concurrency=args.concurrency,
        skip_portion_llm=args.skip_portion_llm,
        force_amount=args.force_amount,
        force_judging=args.force_judging,
        force_payloads=args.force_payloads,
        force_all=args.force_all,
        finalize_only=args.finalize_only,
        only_no_portion=args.only_no_portion,
        only_unresolved=args.only_unresolved,
        retry_limit=args.retry_limit,
        use_mlflow=not args.no_mlflow,
        mlflow_experiment=args.mlflow_experiment,
        sample_lines=args.sample_lines,
        sample_seed=args.sample_seed,
        dequant_cache_path=args.dequant_cache,
        no_dequant_cache=args.no_dequant_cache,
    )


if __name__ == "__main__":
    main()
