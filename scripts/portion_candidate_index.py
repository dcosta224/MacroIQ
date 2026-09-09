"""Per-fdc portion summaries and portion-match scoring for retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from portion_gram import (
    CONTAINER_UNIT_TOKENS,
    PORTION_INDEX_SQL,
    _extract_volume_unit_from_text,
    classify_food_portion_row,
    normalize_count_portion_row,
    normalize_portion_row,
)
from unit_convert import UnitConversionError, normalize_volume_unit
from usda_volume_units import text_has_volume

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "scratch" / "portion_summaries_by_fdc.json"


@dataclass(frozen=True)
class PortionSummaryLine:
    portion_id: int
    rules_class: str
    modifier: str
    measure_unit: str
    portion_description: str
    gram_weight: float
    count_label: str | None

    def display_token(self) -> str:
        if self.count_label:
            base = self.count_label
        elif self.modifier:
            base = self.modifier
        elif self.measure_unit:
            base = self.measure_unit
        else:
            base = self.portion_description or "unit"
        return f"{base}={self.gram_weight:.0f}g"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _line_from_row(row: dict[str, Any]) -> PortionSummaryLine | None:
    gw = float(row.get("gram_weight") or 0)
    if gw <= 0:
        return None
    tag = classify_food_portion_row(row)
    count_c = normalize_count_portion_row(row)
    vol_c = normalize_portion_row(row)
    count_label = count_c.count_label if count_c else None
    return PortionSummaryLine(
        portion_id=int(row["id"]),
        rules_class=tag,
        modifier=str(row.get("modifier") or "").strip(),
        measure_unit=str(row.get("measure_unit_name") or "").strip(),
        portion_description=str(row.get("portion_description") or "").strip(),
        gram_weight=gw,
        count_label=count_label,
    )


def build_portion_summary_index(conn) -> dict[int, list[PortionSummaryLine]]:
    index: dict[int, list[PortionSummaryLine]] = {}
    with conn.cursor() as cur:
        cur.execute(PORTION_INDEX_SQL)
        columns = [desc[0] for desc in cur.description]
        for db_row in cur.fetchall():
            row = dict(zip(columns, db_row, strict=True))
            line = _line_from_row(row)
            if line is None:
                continue
            index.setdefault(int(row["fdc_id"]), []).append(line)
    return index


def load_or_build_portion_summary_index(
    conn,
    *,
    cache_path: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> dict[int, list[PortionSummaryLine]]:
    if cache_path.is_file() and not refresh:
        raw = json.loads(cache_path.read_text())
        return {
            int(fid): [PortionSummaryLine(**d) for d in lines]
            for fid, lines in raw.items()
        }
    index = build_portion_summary_index(conn)
    serializable = {
        str(fid): [line.to_dict() for line in lines]
        for fid, lines in index.items()
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(serializable) + "\n")
    return index


def format_portion_summary_compact(lines: list[PortionSummaryLine], *, max_lines: int = 3) -> str:
    if not lines:
        return "-"
    return "; ".join(line.display_token() for line in lines[:max_lines])


def _volume_units_from_text(*parts: str) -> set[str]:
    units: set[str] = set()
    for part in parts:
        if not part:
            continue
        try:
            units.add(normalize_volume_unit(part))
        except UnitConversionError:
            pass
        if text_has_volume(part):
            unit = _extract_volume_unit_from_text(part)
            if unit:
                units.add(unit)
    return units


def _volume_query_units(query_tokens: list[str]) -> set[str]:
    return _volume_units_from_text(*query_tokens)


def _line_volume_units(line: PortionSummaryLine) -> set[str]:
    parts = (line.modifier, line.measure_unit, line.portion_description)
    if line.rules_class != "volume" and not text_has_volume(*parts):
        return set()
    return _volume_units_from_text(*parts)


def _line_is_volume(line: PortionSummaryLine) -> bool:
    return line.rules_class == "volume" or text_has_volume(
        line.modifier, line.measure_unit, line.portion_description
    )


def portion_match_score(
    lines: list[PortionSummaryLine],
    query_tokens: list[str],
    *,
    amount_kind: str | None = None,
) -> float:
    """Score how well an fdc's portion rows match recipe count/volume tokens."""
    if not lines or not query_tokens:
        return 0.0
    best = 0.0
    size_tokens = {"small", "medium", "large", "extra-large", "jumbo"}
    for line in lines:
        text = " ".join(
            filter(
                None,
                [
                    line.count_label,
                    line.modifier,
                    line.measure_unit,
                    line.portion_description,
                ],
            )
        ).lower()
        score = 0.0
        for tok in query_tokens:
            t = tok.lower().replace("extra large", "extra-large")
            if t in size_tokens:
                if t in text or t.replace("-", " ") in text:
                    score += 0.35
                continue
            if re.search(rf"\b{re.escape(t)}\b", text):
                score += 0.5
            elif t in text:
                score += 0.25
        if amount_kind == "volume":
            query_vol = _volume_query_units(query_tokens)
            line_vol = _line_volume_units(line)
            if query_vol and line_vol:
                score = max(score, 0.45)
            if _line_is_volume(line):
                score += 0.1
        if amount_kind == "count" and line.rules_class == "count":
            score += 0.1
        best = max(best, min(score, 1.0))
    return round(best, 4)


def has_container_mass_portion(lines: list[PortionSummaryLine]) -> bool:
    for line in lines:
        mod = line.modifier.lower()
        if line.rules_class == "mass" and any(c in mod for c in CONTAINER_UNIT_TOKENS):
            return True
    return False


def food_aware_portion_score(
    raw_score: float,
    retrieval_score: float,
) -> float:
    """Down-rank high portion fit when semantic/retrieval identity is weak."""
    if raw_score <= 0:
        return 0.0
    factor = min(1.0, float(retrieval_score) * 1.5)
    return round(raw_score * factor, 4)


def summarize_fdc_portions(
    summary_index: dict[int, list[PortionSummaryLine]],
    fdc_id: int,
    query_tokens: list[str],
    *,
    amount_kind: str | None = None,
    retrieval_score: float = 1.0,
) -> tuple[float, str, int | None]:
    """Return (match_score, compact_display, best_portion_id)."""
    lines = summary_index.get(int(fdc_id), [])
    raw = portion_match_score(lines, query_tokens, amount_kind=amount_kind)
    score = food_aware_portion_score(raw, retrieval_score)
    if not lines:
        return 0.0, "-", None
    best_line = max(
        lines,
        key=lambda ln: portion_match_score([ln], query_tokens, amount_kind=amount_kind),
    )
    return score, format_portion_summary_compact(lines, max_lines=4), best_line.portion_id
