#!/usr/bin/env python3
"""High-protein agent eval: 20 canonical dishes × 3 modes.

Modes:
  1. neighborhood — canonical start + high-protein macro box
  2. creative — creative/OOD graph with neighborhood catalog context
  3. creative_example — creative, but LLM draft is shown a neighborhood
     example recipe (strong semantic match + closest PFC to the target)

Targets: neighborhood mean PFC with protein +10pp, carbs −5pp, fat −5pp,
then a ±2% box on each macro (rounded to nearest percent).

Usage:
  PYTHONPATH=scripts:. uv run python tests/run_high_protein_eval.py
  PYTHONPATH=scripts:. uv run python tests/run_high_protein_eval.py --n-dishes 5 --max-iterations 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

MODES = ("neighborhood", "creative", "creative_example")


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


def _slug(title: str) -> str:
    import re

    s = re.sub(r"[^a-z0-9]+", "_", (title or "").lower()).strip("_")
    return (s[:48] or "dish")


def _extract_metrics(result: dict[str, Any]) -> dict[str, Any]:
    display = result.get("display_scores") or {}
    feval = result.get("final_evaluation") or {}
    judge = result.get("judge_result") or {}

    def _val(card: Any) -> float | None:
        if isinstance(card, dict):
            v = card.get("value")
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        return None

    ratio = _val(display.get("ratio_loss"))
    nutrient = _val(display.get("nutrient_loss"))
    holistic = _val(display.get("holistic_0_10"))
    if holistic is None and feval.get("overall_score_0_10") is not None:
        try:
            holistic = float(feval["overall_score_0_10"])
        except (TypeError, ValueError):
            holistic = None
    if holistic is None and judge.get("holistic_score_0_10") is not None:
        try:
            holistic = float(judge["holistic_score_0_10"])
        except (TypeError, ValueError):
            holistic = None

    return {
        "ratio_loss": ratio,
        "nutrient_loss": nutrient,
        "holistic_0_10": holistic,
        "status": result.get("status"),
        "dietary_violation_flag": feval.get("dietary_violation_flag"),
        "final_evaluation_overall": feval.get("overall_score_0_10"),
        "ratio_source": (display.get("ratio_loss") or {}).get("source")
        if isinstance(display.get("ratio_loss"), dict)
        else None,
        "nutrient_source": (display.get("nutrient_loss") or {}).get("source")
        if isinstance(display.get("nutrient_loss"), dict)
        else None,
        "holistic_source": (display.get("holistic_0_10") or {}).get("source")
        if isinstance(display.get("holistic_0_10"), dict)
        else None,
    }


def _user_request(title: str, box: dict[str, float]) -> str:
    p = 100.0 * 0.5 * (box["protein_min"] + box["protein_max"])
    c = 100.0 * 0.5 * (box["carb_min"] + box["carb_max"])
    f = 100.0 * 0.5 * (box["fat_min"] + box["fat_max"])
    return (
        f"Higher-protein {title}: about {p:.0f}% protein, {c:.0f}% carbs, {f:.0f}% fat "
        f"(calorie shares). Keep the dish recognizable but boost protein relative to a "
        f"typical neighborhood version."
    )


def _build_example(
    canonical_id: int,
    title: str,
    target_mid: dict[str, float],
    target_box: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    from canonical_optimization import CanonicalNeighborhood
    from recipe_opt_agent.example_recipe import pick_example_recipe_near_targets

    nb = CanonicalNeighborhood.build(int(canonical_id), fast=True, use_cache=True)
    return pick_example_recipe_near_targets(
        lines_df=nb.lines_df,
        recipe_ids=list(nb.recipe_ids),
        query=title,
        target_mid=target_mid,
        target_box=target_box,
    )


def _prepare_case(
    *,
    mode: str,
    canonical_id: int,
    title: str,
    box: dict[str, float],
    target_mid: dict[str, float],
) -> tuple[dict[str, Any], str, str]:
    """Return (problem, agent_mode, user_request)."""
    from recipe_opt_agent.creative_loader import load_creative_problem
    from recipe_opt_agent.example_recipe import attach_example_recipe_to_problem
    from recipe_opt_agent.problem_loader import load_canonical_problem

    req = _user_request(title, box)
    kwargs = dict(
        protein_min=box["protein_min"],
        protein_max=box["protein_max"],
        carb_min=box["carb_min"],
        carb_max=box["carb_max"],
        fat_min=box["fat_min"],
        fat_max=box["fat_max"],
    )

    if mode == "neighborhood":
        problem = load_canonical_problem(
            int(canonical_id),
            prefer_nutrition_start=True,
            start_metric="l1_pfc",
            fast_neighborhood=True,
            **kwargs,
        )
        return problem, "neighborhood", req

    # creative + creative_example share creative graph
    problem = load_creative_problem(
        user_request=req,
        canonical_id=int(canonical_id),
        offline=False,
        **kwargs,
    )
    if mode == "creative_example":
        example = _build_example(canonical_id, title, target_mid, target_box=box)
        problem = attach_example_recipe_to_problem(problem, example)
    return problem, "creative", req


def run_eval(
    *,
    n_dishes: int = 20,
    max_iterations: int = 3,
    min_neighborhood: int = 10,
    name: str = "high_protein_20",
    modes: tuple[str, ...] = MODES,
    live: bool = True,
) -> Path:
    from recipe_opt_agent.config import AgentConfig
    from recipe_opt_agent.eval_artifacts import EvalSuiteRecorder, run_agent_with_artifacts
    from recipe_opt_agent.macro_target_suggestions import suggest_high_protein_targets_for_canonical
    from recipe_opt_agent.observability import ensure_tracing_env
    from canonical_optimization import fetch_top_canonical_dishes

    if not live:
        os.environ.pop("OPENAI_API_KEY", None)

    ensure_tracing_env(project=os.environ.get("LANGSMITH_PROJECT") or "recipe-opt-agent-eval")
    suite = EvalSuiteRecorder(name=name)
    suite_dir = suite.start()
    print(f"Suite dir: {suite_dir}", flush=True)

    dishes = fetch_top_canonical_dishes(limit=n_dishes, min_neighborhood=min_neighborhood)
    if dishes is None or dishes.empty:
        raise RuntimeError("No canonical dishes returned — check DB / local store.")
    dish_rows = dishes.head(n_dishes).to_dict(orient="records")
    print(f"Dishes: {len(dish_rows)} · modes={list(modes)}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    targets_by_dish: dict[str, Any] = {}

    for di, row in enumerate(dish_rows, start=1):
        cid = int(row["canonical_recipe_id"])
        title = str(row.get("title") or f"canonical_{cid}")
        print(f"\n=== [{di}/{len(dish_rows)}] {title} (id={cid}) ===", flush=True)

        try:
            hp = suggest_high_protein_targets_for_canonical(cid, pad_pct=2)
        except Exception as exc:
            print(f"  SKIP targets failed: {exc}", flush=True)
            summary_rows.append(
                {
                    "canonical_id": cid,
                    "title": title,
                    "error": f"targets: {exc}",
                }
            )
            continue
        if hp.get("error"):
            print(f"  SKIP: {hp['error']}", flush=True)
            summary_rows.append({"canonical_id": cid, "title": title, "error": hp["error"]})
            continue

        box = hp["box"]
        mid = hp["midpoint"]
        targets_by_dish[str(cid)] = hp
        print(
            f"  mean PFC → high-P mid "
            f"{100*mid['protein']:.0f}/{100*mid['carbs']:.0f}/{100*mid['fat']:.0f} "
            f"±{hp['pad_pct']}%",
            flush=True,
        )

        for mode in modes:
            case_name = f"{_slug(title)}__{mode}"
            t0 = time.perf_counter()
            print(f"  → {mode} …", flush=True)
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
                    max_iterations=max_iterations,
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
                    extra_tags=["high_protein_eval", mode],
                    metadata={
                        "eval_mode": mode,
                        "canonical_id": cid,
                        "title": title,
                        "macro_targets": box,
                        "target_midpoint": mid,
                        "neighborhood_mean_pfc": hp.get("neighborhood_mean_pfc"),
                        "example_recipe": (problem.get("retrieval_context") or {}).get(
                            "example_recipe"
                        ),
                    },
                )
                metrics = _extract_metrics(result)
                elapsed = time.perf_counter() - t0
                row_out = {
                    "canonical_id": cid,
                    "title": title,
                    "mode": mode,
                    "agent_mode": agent_mode,
                    "case_name": case_name,
                    "run_id": rec.run_id,
                    "run_dir": str(rec.run_dir),
                    "elapsed_s": round(elapsed, 2),
                    "macro_box": box,
                    "target_midpoint": mid,
                    "neighborhood_mean_pfc": hp.get("neighborhood_mean_pfc"),
                    "example_recipe": (problem.get("retrieval_context") or {}).get(
                        "example_recipe"
                    ),
                    **metrics,
                    "error": None,
                }
                summary_rows.append(row_out)
                print(
                    f"     done {elapsed:.0f}s · ratio={metrics.get('ratio_loss')} "
                    f"nutrient={metrics.get('nutrient_loss')} "
                    f"holistic={metrics.get('holistic_0_10')}",
                    flush=True,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                tb = traceback.format_exc()
                print(f"     FAILED ({elapsed:.0f}s): {exc}", flush=True)
                summary_rows.append(
                    {
                        "canonical_id": cid,
                        "title": title,
                        "mode": mode,
                        "case_name": case_name,
                        "elapsed_s": round(elapsed, 2),
                        "error": str(exc),
                        "traceback": tb,
                        "ratio_loss": None,
                        "nutrient_loss": None,
                        "holistic_0_10": None,
                    }
                )

    # Aggregate means by mode
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        rows = [r for r in summary_rows if r.get("mode") == mode and not r.get("error")]
        def _mean(key: str) -> float | None:
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            return float(sum(vals) / len(vals)) if vals else None

        by_mode[mode] = {
            "n_ok": len(rows),
            "n_total": sum(1 for r in summary_rows if r.get("mode") == mode),
            "mean_ratio_loss": _mean("ratio_loss"),
            "mean_nutrient_loss": _mean("nutrient_loss"),
            "mean_holistic_0_10": _mean("holistic_0_10"),
        }

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir),
        "n_dishes": len(dish_rows),
        "modes": list(modes),
        "max_iterations": max_iterations,
        "target_spec": {
            "protein_delta": 0.10,
            "carb_delta": -0.05,
            "fat_delta": -0.05,
            "pad_pct": 2,
        },
        "by_mode": by_mode,
        "targets_by_dish": targets_by_dish,
        "rows": summary_rows,
    }
    out_path = suite_dir / "high_protein_eval_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Flat metrics CSV-friendly JSONL
    metrics_path = suite_dir / "high_protein_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        for r in summary_rows:
            f.write(json.dumps(r, default=str) + "\n")

    print("\n=== SUMMARY BY MODE ===", flush=True)
    print(json.dumps(by_mode, indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)
    return suite_dir


def main() -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-dishes", type=int, default=20)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--min-neighborhood", type=int, default=10)
    p.add_argument("--name", type=str, default="high_protein_20")
    p.add_argument(
        "--modes",
        type=str,
        default=",".join(MODES),
        help="Comma-separated: neighborhood,creative,creative_example",
    )
    p.add_argument("--offline", action="store_true", help="Force heuristic LLMs (no OpenAI)")
    args = p.parse_args()
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"Unknown mode {m!r}; choose from {MODES}")
    run_eval(
        n_dishes=args.n_dishes,
        max_iterations=args.max_iterations,
        min_neighborhood=args.min_neighborhood,
        name=args.name,
        modes=modes,
        live=not args.offline,
    )


if __name__ == "__main__":
    main()
