#!/usr/bin/env python3
"""Re-run selected high-protein cases and compare to the baseline suite.

Usage:
  PYTHONPATH=scripts:. uv run python tests/rerun_grounding_compare.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

BASELINE_SUITE = (
    ROOT
    / "scratch"
    / "recipe_opt_runs"
    / "eval_suites"
    / "high_protein_20_20260719T200634Z"
)

# Previously bad + two strong controls from the baseline suite.
CASES: list[tuple[int, str, str]] = [
    (35, "creative", "BBQ Ribs"),
    (67, "creative", "Bobotie"),
    (449, "creative", "Stuffed Grape Leaves"),
    (187, "neighborhood", "Focaccia"),
    (193, "creative_example", "Fried Rice"),
    (109, "creative", "Cheeseburger"),
]


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def _baseline_index() -> dict[tuple[int, str], dict[str, Any]]:
    summary = json.loads((BASELINE_SUITE / "high_protein_eval_summary.json").read_text())
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for r in summary.get("rows") or []:
        cid = r.get("canonical_id")
        mode = r.get("mode")
        if cid is None or mode is None:
            continue
        out[(int(cid), str(mode))] = r
    return out


def _chosen_labels(result: dict[str, Any]) -> list[str]:
    cr = result.get("chosen_recipe") or {}
    return [str(i.get("label") or i.get("name") or "") for i in (cr.get("ingredients") or [])]


def _grounding_summary(result: dict[str, Any]) -> dict[str, Any]:
    gr = (result.get("problem") or {}).get("grounding_report") or result.get("grounding_report") or {}
    return {
        "n_matched": len(gr.get("matched") or []),
        "n_substituted": len(gr.get("substituted") or []),
        "n_unresolved": len(gr.get("unresolved") or []),
        "unresolved_names": [u.get("name") for u in (gr.get("unresolved") or [])][:8],
        "matched_pairs": [
            f"{m.get('name')} → {m.get('label')}" for m in (gr.get("matched") or [])[:8]
        ],
    }


def main() -> None:
    _load_dotenv()
    from recipe_opt_agent.config import AgentConfig
    from recipe_opt_agent.eval_artifacts import EvalSuiteRecorder, run_agent_with_artifacts
    from recipe_opt_agent.macro_target_suggestions import suggest_high_protein_targets_for_canonical
    from tests.run_high_protein_eval import (
        _extract_metrics,
        _prepare_case,
        _slug,
        _user_request,
    )

    baseline = _baseline_index()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite = EvalSuiteRecorder(name=f"grounding_compare_{stamp}")
    print(f"Suite dir: {suite.suite_dir}", flush=True)
    print(f"Baseline:  {BASELINE_SUITE}", flush=True)

    rows: list[dict[str, Any]] = []
    for cid, mode, title in CASES:
        print(f"\n=== {title} (id={cid}) · {mode} ===", flush=True)
        hp = suggest_high_protein_targets_for_canonical(cid, pad_pct=2)
        if hp.get("error"):
            print(f"  SKIP: {hp['error']}", flush=True)
            continue
        box = hp["box"]
        mid = hp["midpoint"]
        case_name = f"{_slug(title)}__{mode}"
        t0 = time.perf_counter()
        try:
            problem, agent_mode, req = _prepare_case(
                mode=mode,
                canonical_id=cid,
                title=title,
                box=box,
                target_mid=mid,
            )
            cfg = AgentConfig(
                protein_min=box["protein_min"],
                protein_max=box["protein_max"],
                carb_min=box["carb_min"],
                carb_max=box["carb_max"],
                fat_min=box["fat_min"],
                fat_max=box["fat_max"],
                max_iterations=3,
                F_accept=1.0,
                F_max=1.5,
                agent_mode=agent_mode,
            )
            result, rec = run_agent_with_artifacts(
                problem=problem,
                case_name=case_name,
                agent_mode=agent_mode,
                suite=suite,
                taste_text=title if mode == "neighborhood" else req,
                title=title,
                user_request=req,
                canonical_id=cid,
                config=cfg,
                extra_tags=["grounding_compare", mode],
                metadata={
                    "eval_mode": mode,
                    "canonical_id": cid,
                    "title": title,
                    "macro_targets": box,
                    "baseline_suite": str(BASELINE_SUITE),
                },
            )
            metrics = _extract_metrics(result)
            elapsed = time.perf_counter() - t0
            fe = result.get("final_evaluation") or {}
            base = baseline.get((cid, mode)) or {}
            row = {
                "canonical_id": cid,
                "title": title,
                "mode": mode,
                "elapsed_s": round(elapsed, 2),
                "run_dir": str(rec.run_dir),
                "after": {
                    **metrics,
                    "pfc_after": (result.get("opt") or {}).get("pfc_after"),
                    "status": result.get("status"),
                    "fidelity_band": (result.get("diagnosis") or {}).get("fidelity_band")
                    if isinstance(result.get("diagnosis"), dict)
                    else result.get("fidelity_band"),
                    "identity_roles": result.get("identity_roles"),
                    "labels": _chosen_labels(result),
                    "grounding": _grounding_summary(result),
                    "judge_summary": (fe.get("summary_markdown") or "")[:400],
                    "odd_ingredients": fe.get("odd_ingredients"),
                    "concerns": fe.get("concerns"),
                    "strengths": fe.get("strengths"),
                },
                "before": {
                    "ratio_loss": base.get("ratio_loss"),
                    "nutrient_loss": base.get("nutrient_loss"),
                    "holistic_0_10": base.get("holistic_0_10"),
                    "error": base.get("error"),
                    "run_dir": base.get("run_dir"),
                },
            }
            rows.append(row)
            a, b = row["after"], row["before"]
            print(
                f"  before hol={b.get('holistic_0_10')} ratio={b.get('ratio_loss')} nut={b.get('nutrient_loss')}",
                flush=True,
            )
            print(
                f"  after  hol={a.get('holistic_0_10')} ratio={a.get('ratio_loss')} nut={a.get('nutrient_loss')} "
                f"band={a.get('fidelity_band')} ({elapsed:.0f}s)",
                flush=True,
            )
            print(f"  labels: {a.get('labels')}", flush=True)
            if a.get("grounding", {}).get("matched_pairs"):
                print(f"  grounded: {a['grounding']['matched_pairs']}", flush=True)
            if a.get("grounding", {}).get("unresolved_names"):
                print(f"  unresolved: {a['grounding']['unresolved_names']}", flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  FAILED ({elapsed:.0f}s): {exc}", flush=True)
            base = baseline.get((cid, mode)) or {}
            rows.append(
                {
                    "canonical_id": cid,
                    "title": title,
                    "mode": mode,
                    "elapsed_s": round(elapsed, 2),
                    "error": str(exc),
                    "before": {
                        "ratio_loss": base.get("ratio_loss"),
                        "nutrient_loss": base.get("nutrient_loss"),
                        "holistic_0_10": base.get("holistic_0_10"),
                        "error": base.get("error"),
                    },
                }
            )

    out_path = suite.suite_dir / "grounding_compare_report.json"
    out_path.write_text(json.dumps({"baseline": str(BASELINE_SUITE), "rows": rows}, indent=2, default=str))
    print(f"\nWrote {out_path}", flush=True)

    # Compact table
    print("\n=== BEFORE → AFTER ===", flush=True)
    for r in rows:
        b = r.get("before") or {}
        a = r.get("after") or {}
        if r.get("error"):
            print(f"{r['title']:28s} {r['mode']:18s} FAILED: {r['error'][:80]}", flush=True)
            continue
        print(
            f"{r['title']:28s} {r['mode']:18s} "
            f"hol {b.get('holistic_0_10')}→{a.get('holistic_0_10')}  "
            f"ratio {b.get('ratio_loss')}→{a.get('ratio_loss')}  "
            f"nut {b.get('nutrient_loss')}→{a.get('nutrient_loss')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
