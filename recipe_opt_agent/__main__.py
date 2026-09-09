"""CLI entry: python -m recipe_opt_agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recipe optimization agent (LangGraph)")
    parser.add_argument("--canonical-id", type=int, default=443)
    parser.add_argument("--taste", type=str, default="classic carbonara")
    parser.add_argument("--protein-min", type=float, default=0.19)
    parser.add_argument("--protein-max", type=float, default=0.23)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--fixture", type=str, default="", help="Path to JSON problem fixture (offline)")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args(argv)

    from recipe_opt_agent.config import AgentConfig
    from recipe_opt_agent.runner import run_recipe_opt_agent

    cfg = AgentConfig(
        protein_min=args.protein_min,
        protein_max=args.protein_max,
        max_iterations=args.max_iters,
    )

    from recipe_opt_agent.problem_loader import load_canonical_problem, load_fixture_problem

    if args.fixture:
        problem = load_fixture_problem(args.fixture)
        title = problem.get("title", f"fixture {args.fixture}")
    else:
        problem = load_canonical_problem(args.canonical_id)
        title = problem.get("title") or f"canonical {args.canonical_id}"

    result = run_recipe_opt_agent(
        problem=problem,
        taste_text=args.taste,
        title=title,
        canonical_id=args.canonical_id,
        config=cfg,
    )
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
