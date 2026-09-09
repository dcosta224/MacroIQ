"""Golden tests for portion resolution (rules only; no LLM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from portion_gram import CountPortionCandidate, PortionCandidate, resolve_grams_from_plan  # noqa: E402
from resolution_plan import build_resolution_plan  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "portion_resolution_cases.json"


def _load_cases() -> list[dict]:
    data = json.loads(FIXTURES.read_text())
    return data["cases"]


def _build_count_index(raw: dict) -> dict[int, list[CountPortionCandidate]]:
    out: dict[int, list[CountPortionCandidate]] = {}
    for fdc_id, rows in raw.items():
        out[int(fdc_id)] = [CountPortionCandidate(**row) for row in rows]
    return out


def _build_volume_index(raw: dict) -> dict[int, list[PortionCandidate]]:
    out: dict[int, list[PortionCandidate]] = {}
    for fdc_id, rows in raw.items():
        out[int(fdc_id)] = [PortionCandidate(**row) for row in rows]
    return out


def _assert_close(a: float, b: float, *, rel: float = 1e-3) -> None:
    if abs(a - b) > rel * max(abs(a), abs(b), 1.0):
        raise AssertionError(f"{a} != {b} (rel tol {rel})")


def test_resolution_case(case: dict) -> None:
    parse = dict(case.get("parse") or {})
    if parse.get("quantity") is None and "quantity" not in case.get("parse", {}):
        pass
    ingredient = case["ingredient"]
    plan = build_resolution_plan(parse, ingredient_raw=ingredient)
    if case.get("enrichment"):
        from resolution_plan import _apply_enrichment

        _apply_enrichment(plan, case["enrichment"])

    count_index = _build_count_index(case.get("count_portion_index") or {})
    volume_index = _build_volume_index(case.get("portion_index") or {})
    matched_portion_id = case.get("matched_portion_id")
    conn = None
    if case.get("raw_portion_rows"):
        class _FakeConn:
            def __init__(self, rows):
                self._rows = rows

        conn = _FakeConn(case["raw_portion_rows"])
        import portion_gram as pg

        orig = pg._load_portion_rows_for_fdc
        pg._load_portion_rows_for_fdc = lambda _conn, _fdc: case["raw_portion_rows"]
        try:
            result = resolve_grams_from_plan(
                plan,
                case.get("fdc_id"),
                ingredient_raw=ingredient,
                name=parse.get("name"),
                count_portion_index=count_index,
                portion_index=volume_index,
                matched_portion_id=matched_portion_id,
                conn=conn,
                llm_negligible_calories=bool(case.get("llm_negligible_calories", False)),
            )
        finally:
            pg._load_portion_rows_for_fdc = orig
    else:
        result = resolve_grams_from_plan(
            plan,
            case.get("fdc_id"),
            ingredient_raw=ingredient,
            name=parse.get("name"),
            count_portion_index=count_index,
            portion_index=volume_index,
            matched_portion_id=matched_portion_id,
            llm_negligible_calories=bool(case.get("llm_negligible_calories", False)),
        )

    assert result.status == case["expected_status"]
    if "expected_grams" in case:
        assert result.grams is not None
        _assert_close(result.grams, case["expected_grams"])
    if "expected_grams_min" in case:
        assert result.grams is not None
        assert case["expected_grams_min"] <= result.grams <= case["expected_grams_max"]


def main() -> None:
    cases = _load_cases()
    for case in cases:
        test_resolution_case(case)
        print(f"OK: {case['id']}")
    print(f"All {len(cases)} cases passed.")


if __name__ == "__main__":
    main()
