#!/usr/bin/env python3
"""Upload the Colab input bundle for the 1000-recipe OSS feasibility run.

Run locally once (Mac) after baseline/v4 artifacts exist:

  uv run python scripts/export_colab_s3_bundle.py
  uv run python scripts/export_colab_s3_bundle.py --dry-run

Requires deploy/aws.env (or S3_BUCKET_ARTIFACTS env) and AWS credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "scratch" / "EDA" / "portion_feasibility_1000_v4_no_portion" / "feasibility_report.json"
DEFAULT_BASELINE_REPORT = ROOT / "scratch" / "EDA" / "portion_feasibility_1000" / "feasibility_report.json"
DEFAULT_FOOD_CSV = ROOT / "scratch" / "food_4macro.csv"
DEFAULT_FOOD_CACHE = ROOT / "scratch" / "recipe_matching_llm_100_portion"
DEFAULT_RECIPE_CACHE = ROOT / "scratch" / "EDA" / "portion_feasibility_1000" / "recipe_cache"
BUNDLE_PREFIX = "colab/feasibility_1000_seed42"

BASELINE_SUMMARY_KEYS = [
    "n_lines",
    "n_recipes",
    "seed",
    "model",
    "fdc_match_rate_all",
    "gram_resolve_rate_all",
    "fdc_and_gram_rate_all",
    "fdc_match_rate_needs_portion",
    "gram_resolve_rate_needs_portion",
    "fdc_and_gram_rate_needs_portion",
    "n_llm_line_enrichment_calls",
    "judge_error_count",
    "grams_status_counts",
    "rules_grams_status_counts",
    "llm_portion_rescue_rate_needs_portion",
    "elapsed_sec",
]


def _load_aws_env() -> dict[str, str]:
    env_path = ROOT / "deploy" / "aws.env"
    out: dict[str, str] = {}
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            out[key.strip()] = val.strip()
    return out


def _resolve_bucket(aws_env: dict[str, str]) -> str:
    bucket = os.environ.get("S3_BUCKET_ARTIFACTS") or aws_env.get("S3_BUCKET_ARTIFACTS")
    if not bucket:
        raise SystemExit(
            "Set S3_BUCKET_ARTIFACTS in deploy/aws.env or the environment."
        )
    return bucket


def _extract_recipe_ids(report_path: Path) -> list[int]:
    report = json.loads(report_path.read_text())
    ids = report.get("sampled_recipe_ids")
    if not ids:
        raise SystemExit(f"No sampled_recipe_ids in {report_path}")
    return [int(x) for x in ids]


def _baseline_summary(report_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    summary = {k: report[k] for k in BASELINE_SUMMARY_KEYS if k in report}
    summary["source_report"] = str(report_path.relative_to(ROOT))
    return summary


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def build_bundle(
    *,
    staging_dir: Path,
    report_path: Path,
    food_csv: Path,
    food_cache: Path,
    recipe_cache: Path,
) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)

    ids = _extract_recipe_ids(report_path)
    manifest = {"n": len(ids), "seed": 42, "recipe_ids": ids}
    (staging_dir / "sampled_recipe_ids.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    if not food_csv.is_file():
        raise SystemExit(f"Missing food_4macro cache: {food_csv}")
    shutil.copy2(food_csv, staging_dir / "food_4macro.csv")

    if not food_cache.is_dir():
        raise SystemExit(f"Missing food embedding cache: {food_cache}")
    _copy_tree(food_cache, staging_dir / "food_cache")

    if not recipe_cache.is_dir():
        raise SystemExit(f"Missing recipe embedding cache: {recipe_cache}")
    _copy_tree(recipe_cache, staging_dir / "recipe_cache")

    summary = _baseline_summary(report_path)
    (staging_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(f"Bundle staged at {staging_dir}")
    print(f"  recipes: {len(ids)}")
    print(f"  food_cache: {food_cache}")
    print(f"  recipe_cache: {recipe_cache}")


def upload_bundle(staging_dir: Path, bucket: str, *, dry_run: bool) -> None:
    dest = f"s3://{bucket}/{BUNDLE_PREFIX}/"
    cmd = ["aws", "s3", "sync", str(staging_dir), dest, "--delete"]
    print(" ".join(cmd))
    if dry_run:
        cmd.insert(1, "--dryrun")
    subprocess.run(cmd, check=True)
    print(f"Uploaded bundle to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Colab S3 input bundle (local Mac only)")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="feasibility_report.json with sampled_recipe_ids",
    )
    parser.add_argument("--food-csv", type=Path, default=DEFAULT_FOOD_CSV)
    parser.add_argument("--food-cache-dir", type=Path, default=DEFAULT_FOOD_CACHE)
    parser.add_argument("--recipe-cache-dir", type=Path, default=DEFAULT_RECIPE_CACHE)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=ROOT / "scratch" / "colab_bundle_staging",
    )
    parser.add_argument("--dry-run", action="store_true", help="aws s3 sync --dryrun only")
    parser.add_argument("--skip-upload", action="store_true", help="Stage files only")
    args = parser.parse_args()

    report = args.report
    if not report.is_file() and DEFAULT_BASELINE_REPORT.is_file():
        print(f"Report not found at {report}; falling back to {DEFAULT_BASELINE_REPORT}")
        report = DEFAULT_BASELINE_REPORT

    build_bundle(
        staging_dir=args.staging_dir,
        report_path=report,
        food_csv=args.food_csv,
        food_cache=args.food_cache_dir,
        recipe_cache=args.recipe_cache_dir,
    )

    if args.skip_upload:
        return

    aws_env = _load_aws_env()
    bucket = _resolve_bucket(aws_env)
    upload_bundle(args.staging_dir, bucket, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
