"""Tests for feasibility progress streaming."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from feasibility_progress import FeasibilityProgressWriter
from judge_checkpoint import append_judge_jsonl, compact_jsonl_to_parquet, load_judge_checkpoint


def _row(recipe_id: int, ingredient_idx: int) -> dict:
    ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    return {
        "recipe_id": recipe_id,
        "ingredient_idx": ingredient_idx,
        "ingredient": "flour",
        "llm_fdc_id": 100 + ingredient_idx,
        "grams_status": "resolved",
        "ts": ts,
        "inferred_at": "2025-06-15T12:00:00Z",
    }


def test_append_jsonl_and_compact(tmp_path: Path) -> None:
    jsonl = tmp_path / "judge_stream.jsonl"
    parquet = tmp_path / "judge_matches_raw.parquet"
    for i in range(3):
        append_judge_jsonl(jsonl, _row(1, i))
    n = compact_jsonl_to_parquet(jsonl, parquet)
    assert n == 3
    df = load_judge_checkpoint(parquet)
    assert len(df) == 3
    assert set(df["ingredient_idx"].astype(int)) == {0, 1, 2}


def test_compact_upserts_by_key(tmp_path: Path) -> None:
    jsonl = tmp_path / "stream.jsonl"
    parquet = tmp_path / "out.parquet"
    append_judge_jsonl(jsonl, _row(1, 0))
    r2 = _row(1, 0)
    r2["llm_fdc_id"] = 999
    append_judge_jsonl(jsonl, r2)
    n = compact_jsonl_to_parquet(jsonl, parquet)
    assert n == 1
    assert int(load_judge_checkpoint(parquet).iloc[0]["llm_fdc_id"]) == 999


def test_progress_writer_judge_rows(tmp_path: Path) -> None:
    writer = FeasibilityProgressWriter(tmp_path, run_id="testrun", parquet_compact_every=2)
    writer.set_phase("judging", total=5)
    for i in range(3):
        writer.record_judge_row(_row(10, i))
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["phase"] == "judging"
    assert progress["judging"]["done"] == 3
    assert progress["judging"]["total"] == 5
    assert (tmp_path / "judge_stream.jsonl").is_file()
    assert (tmp_path / "judge_matches_raw.parquet").is_file()
    assert len(load_judge_checkpoint(tmp_path / "judge_matches_raw.parquet")) == 2


def test_quiet_flush_writes_stats(tmp_path: Path) -> None:
    writer = FeasibilityProgressWriter(
        tmp_path, run_id="quiet", quiet=True, flush_every=2, parquet_compact_every=2
    )
    writer.set_phase("judging", total=4)
    for i in range(4):
        writer.record_judge_row(_row(1, i))
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["prompts"]["done"] == 4
    assert "stats" in progress
    assert progress["stats"]["judge_fdc_matched"] == 4
    writer.finalize()


def test_per_phase_bars_enrich_then_judge(tmp_path: Path) -> None:
    writer = FeasibilityProgressWriter(tmp_path, quiet=True, flush_every=1, parquet_compact_every=10)
    writer.set_phase("amount_classify", total=1000)
    writer.add_prompt_budget(3)
    for i in range(3):
        writer.record_enrichment_row(
            done=i + 1, total=3, ingredient_norm=f"ing{i}", error=None if i else "bad"
        )
    writer.set_phase("judging", total=2)
    for i in range(2):
        writer.record_judge_row(_row(2, i))
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["prompts"]["done"] == 5
    assert progress["prompts"]["total"] == 5
    assert progress["stats"]["enrichment_errors"] == 1
    assert progress["judging"]["done"] == 2
    writer.finalize()


def test_run_judging_with_progress_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ingredient_match_llm import assemble_rows, run_judging

    async def fake_judge(client, model, user_prompt, valid_fdc_ids, *, system_prompt=None):
        return {
            "fdc_id": 42,
            "certainty": 0.9,
            "rationale": "ok",
            "response": "{}",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "error": None,
        }

    monkeypatch.setattr("ingredient_match_llm.judge_async", fake_judge)

    writer = FeasibilityProgressWriter(tmp_path, parquet_compact_every=1)
    writer.set_phase("judging", total=2)

    def _payload(idx: int) -> dict:
        return {
            "recipe_id": 1,
            "ingredient_idx": idx,
            "ingredient": "flour",
            "user_prompt": "pick",
            "valid_fdc_ids": {42},
            "name": "flour",
            "preparation": "",
            "dequantified": "flour",
            "unit": "cup",
            "prompt_desc": {42: "flour"},
            "staged_fdc_id": 42,
            "staged_description": "flour",
            "staged_match_score": 0.9,
            "staged_match_quality": "high",
            "staged_base_score": 0.8,
            "staged_prep_score": 0.7,
            "n_candidates_llm": 1,
            "n_lexical_pool": 1,
            "n_semantic_pool": 1,
            "staged_top1_in_llm_candidates": True,
            "staged_top1_in_top10": True,
            "n_relevant_steps": 0,
            "relevant_steps": [],
            "top10_rows": [],
        }

    disk_path = tmp_path / "judge_matches_raw.parquet"
    asyncio.run(
        run_judging(
            [_payload(0), _payload(1)],
            run_id="rid",
            run_name="test",
            model="test-model",
            pricing={"input": 0.0, "output": 0.0},
            concurrency=2,
            flush_every=100,
            conn=None,
            use_supabase=False,
            disk_checkpoint_path=disk_path,
            disk_flush_every=1,
            log_every=1,
            progress_writer=writer,
            assemble_fn=assemble_rows,
        )
    )
    writer.finalize()
    assert len(load_judge_checkpoint(disk_path)) == 2
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["judging"]["done"] == 2
