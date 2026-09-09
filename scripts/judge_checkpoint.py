"""Disk checkpoint merge for LLM judge rows (feasibility runs)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

MERGE_KEYS = ("recipe_id", "ingredient_idx")


def load_judge_checkpoint(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def completed_keys(df: pd.DataFrame) -> set[tuple[int, int]]:
    if df.empty:
        return set()
    return {
        (int(r.recipe_id), int(r.ingredient_idx))
        for r in df.itertuples(index=False)
    }


def merge_judge_checkpoint(path: Path, new_rows: list[dict[str, Any]]) -> int:
    """Upsert judge rows by (recipe_id, ingredient_idx). Returns total rows on disk."""
    if not new_rows:
        return len(load_judge_checkpoint(path))
    incoming = pd.DataFrame(new_rows)
    existing = load_judge_checkpoint(path)
    if existing.empty:
        merged = incoming
    else:
        keys = set(
            zip(
                incoming["recipe_id"].astype(int),
                incoming["ingredient_idx"].astype(int),
                strict=True,
            )
        )
        keep = existing[
            ~existing.apply(
                lambda r: (int(r["recipe_id"]), int(r["ingredient_idx"])) in keys,
                axis=1,
            )
        ]
        merged = pd.concat([keep, incoming], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return len(merged)


def no_portion_rate(df: pd.DataFrame) -> tuple[int, int, float]:
    """Return (no_portion_count, total, rate)."""
    if df.empty or "grams_status" not in df.columns:
        return 0, 0, 0.0
    total = len(df)
    nop = int((df["grams_status"] == "no_portion").sum())
    return nop, total, nop / max(total, 1)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_judge_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one judge row as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")


def _read_judge_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def compact_jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> int:
    """Merge all JSONL rows into parquet by (recipe_id, ingredient_idx); last row wins."""
    rows = _read_judge_jsonl(jsonl_path)
    if not rows:
        return len(load_judge_checkpoint(parquet_path))
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["recipe_id"]), int(row["ingredient_idx"]))
        by_key[key] = row
    return merge_judge_checkpoint(parquet_path, list(by_key.values()))
