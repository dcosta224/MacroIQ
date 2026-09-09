"""Helpers for MVP resolved-recipe loading from feasibility pipeline artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

SENTINEL_FDC_ID = 999_000_001
WATER_SENTINEL_FDC_ID = 999_000_002
PORTION_ID_RE = re.compile(r"portion#(\d+)", re.IGNORECASE)

DEFAULT_V4_DIR = (
    Path(__file__).resolve().parents[1]
    / "scratch"
    / "EDA"
    / "portion_feasibility_1000_v4_no_portion"
)


def _row_field(row: Any, col: str) -> Any:
    if hasattr(row, col):
        return getattr(row, col)
    try:
        return row[col]
    except (KeyError, TypeError, IndexError):
        return None


def extract_portion_id(row: Any) -> int | None:
    """Resolve USDA food_portion.id from structured fields or grams_method text."""
    for col in ("portion_id", "matched_portion_id"):
        val = _row_field(row, col)
        if val is not None and pd.notna(val):
            return int(val)
    for col in ("grams_method", "rules_grams_method"):
        val = _row_field(row, col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        m = PORTION_ID_RE.search(str(val))
        if m:
            return int(m.group(1))
    return None


def normalize_fdc_id(val: Any) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    fid = int(val)
    if fid == SENTINEL_FDC_ID:
        return None
    return fid


def fully_resolved_recipe_ids(matches: pd.DataFrame) -> set[int]:
    """Recipe ids where every ingredient line has non-null grams."""
    if "grams" not in matches.columns:
        raise ValueError("matches DataFrame missing 'grams' column")
    grouped = matches.groupby("recipe_id")["grams"].apply(lambda s: s.notna().all())
    return {int(rid) for rid, ok in grouped.items() if ok}


def build_portion_label(
    portion_description: str | None,
    modifier: str | None,
    measure_unit_name: str | None,
) -> str | None:
    parts = [
        p.strip()
        for p in (portion_description, modifier, measure_unit_name)
        if p and str(p).strip()
    ]
    return " | ".join(parts) if parts else None


def load_v4_matches(artifacts_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    root = artifacts_dir or DEFAULT_V4_DIR
    matches_path = root / "pipeline_matches.parquet"
    manifest_path = root / "run_manifest.json"
    if not matches_path.is_file():
        raise FileNotFoundError(f"Missing {matches_path}")
    matches = pd.read_parquet(matches_path)
    manifest: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    return matches, manifest
