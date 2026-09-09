"""Persist MacroIQ UI runs to mvp_logs.macroiq_runs in Supabase."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _connect():
    from db import connect

    return connect()


def _ensure_schema(cur) -> None:
    path = ROOT / "sql" / "43_create_macroiq_runs.sql"
    if not path.is_file():
        return
    lines = [
        line
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]
    sql = "\n".join(lines)
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)


def _compact_result(final: dict[str, Any] | None) -> dict[str, Any]:
    """Keep a useful but bounded snapshot (avoid multi-MB neighborhood matrices)."""
    final = final or {}
    browse = list(final.get("browse_candidates") or [])
    compact_browse = []
    for card in browse[:8]:
        if not isinstance(card, dict):
            continue
        scores = card.get("display_scores") or {}
        summary = card.get("score_summary") or {}
        compact_browse.append(
            {
                "candidate_id": card.get("candidate_id"),
                "title": card.get("title"),
                "branch": card.get("branch"),
                "is_recommended": card.get("is_recommended"),
                "source": card.get("source"),
                "score_summary": summary,
                "macros": (scores.get("macros") or summary.get("macros")),
                "ratio_loss": (scores.get("ratio_loss") or {}).get("value")
                if isinstance(scores.get("ratio_loss"), dict)
                else summary.get("ratio_loss"),
                "iqr_in_band_frac": summary.get("iqr_in_band_frac"),
                "n_ingredients": len(scores.get("ingredients") or []),
            }
        )
    chosen = final.get("chosen") or {}
    display = final.get("display_scores") or {}
    macros = display.get("macros") or {}
    return {
        "agent_mode": final.get("agent_mode"),
        "title": final.get("title"),
        "canonical_id": final.get("canonical_id"),
        "judge_rationale": final.get("judge_rationale"),
        "run_telemetry": final.get("run_telemetry"),
        "winner_candidate_id": (
            (chosen.get("entry") or {}).get("candidate_id")
            or chosen.get("candidate_id")
            or chosen.get("arbiter_winner_id")
        ),
        "macros": macros,
        "ratio_loss": (display.get("ratio_loss") or {}).get("value"),
        "nutrient_loss": (display.get("nutrient_loss") or {}).get("value"),
        "holistic_0_10": (display.get("holistic_0_10") or {}).get("value"),
        "browse_candidates": compact_browse,
        "n_browse_candidates": len(browse),
        "iteration": final.get("iteration"),
        "fidelity_band": final.get("fidelity_band"),
    }


def start_macroiq_run(
    *,
    mode: str,
    request: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str | None:
    """Insert a running row; returns run_id or None if logging fails."""
    run_id = str(uuid.uuid4())
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                _ensure_schema(cur)
                cur.execute(
                    """
                    INSERT INTO mvp_logs.macroiq_runs (
                        run_id, status, mode, agent_mode, canonical_id, title,
                        taste_text, user_request, kcal_target, use_macro_targets,
                        request_json, config_json
                    )
                    VALUES (
                        %s, 'running', %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s::jsonb, %s::jsonb
                    )
                    """,
                    (
                        run_id,
                        mode,
                        mode,
                        request.get("canonical_id"),
                        request.get("title"),
                        request.get("taste_text"),
                        request.get("user_request"),
                        request.get("kcal_target"),
                        request.get("use_macro_targets"),
                        json.dumps(request, default=str),
                        json.dumps(config or {}, default=str),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        return run_id
    except Exception as exc:
        import sys

        print(f"[macroiq_runs] start failed: {exc}", file=sys.stderr)
        return None


def finish_macroiq_run(
    run_id: str | None,
    *,
    status: str = "done",
    final: dict[str, Any] | None = None,
    error_message: str | None = None,
    mode: str | None = None,
) -> None:
    if not run_id:
        return
    compact = _compact_result(final)
    browse = compact.get("browse_candidates") or []
    winner_id = compact.get("winner_candidate_id")
    winner_ratio = compact.get("ratio_loss")
    winner_cal = (compact.get("macros") or {}).get("calories")
    if browse:
        top = browse[0] if isinstance(browse[0], dict) else {}
        winner_id = top.get("candidate_id") or winner_id
        winner_ratio = top.get("ratio_loss") if top.get("ratio_loss") is not None else winner_ratio
        macros = top.get("macros") or {}
        if isinstance(macros, dict) and macros.get("calories") is not None:
            winner_cal = macros.get("calories")
    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE mvp_logs.macroiq_runs
                    SET finished_at = now(),
                        status = %s,
                        mode = COALESCE(%s, mode),
                        agent_mode = COALESCE(%s, agent_mode),
                        result_json = %s::jsonb,
                        browse_candidates = %s::jsonb,
                        error_message = %s,
                        winner_candidate_id = %s,
                        winner_ratio_loss = %s,
                        winner_calories = %s
                    WHERE run_id = %s
                    """,
                    (
                        status,
                        mode,
                        mode,
                        json.dumps(compact, default=str),
                        json.dumps(browse, default=str),
                        error_message,
                        winner_id,
                        winner_ratio,
                        winner_cal,
                        run_id,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        import sys

        print(f"[macroiq_runs] finish failed: {exc}", file=sys.stderr)
