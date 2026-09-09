# Colab OSS ingredient resolution

Run the full portion/FDC resolution pipeline on Google Colab using an open-source Hugging Face model instead of OpenAI. The notebook uses the **same 1,000 recipes** (seed 42) as the GPT baseline and writes results to S3 for comparison.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Colab GPU** | A100 40GB+ recommended (~2–4 h full run with vLLM). T4 works for smoke tests only. |
| **Colab high RAM** | Helps hold embedding caches (~1.5 GB) |
| **Supabase** | Read-only pooler credentials (`PG_POOL_*`) for `usda.food_portion` |
| **AWS** | Credentials with read/write on the artifacts bucket |
| **Hugging Face** | `HF_TOKEN` for model download (recommended) |
| **S3 input bundle** | Upload once from your Mac (see below) |

## One-time: export input bundle (local Mac)

From the Capstone repo root, after baseline artifacts exist:

```bash
uv run python scripts/export_colab_s3_bundle.py
```

This stages and uploads:

```
s3://{S3_BUCKET_ARTIFACTS}/colab/feasibility_1000_seed42/
  sampled_recipe_ids.json    # canonical 1000 recipe IDs
  food_4macro.csv
  food_cache/                # pre-built MiniLM embeddings
  recipe_cache/              # pre-built recipe embeddings
  baseline_summary.json      # GPT v4 metrics for comparison
```

Dry-run upload:

```bash
uv run python scripts/export_colab_s3_bundle.py --dry-run
```

## Colab secrets

Set these in **Secrets** (or export as env vars). The notebook reads them via `google.colab.userdata`.

| Variable | Purpose |
|----------|---------|
| `PG_POOL_USER` | Supabase pooler user |
| `PG_PASSWORD` | Supabase password |
| `PG_POOL_HOST` | e.g. `aws-1-us-east-1.pooler.supabase.com` |
| `PG_POOL_SESSION_PORT` | `5432` |
| `PG_DATABASE` | `postgres` |
| `PG_SSL_MODE` | `require` |
| `HF_TOKEN` | Hugging Face model download |
| `AWS_ACCESS_KEY_ID` | S3 pull/push |
| `AWS_SECRET_ACCESS_KEY` | S3 pull/push |
| `AWS_DEFAULT_REGION` | e.g. `us-east-1` |
| `S3_BUCKET_ARTIFACTS` | Artifacts bucket (from `deploy/aws.env`) |
| `S3_BUCKET_RAW` | Raw bucket (for `RecipeNLG.csv` if not cached locally) |

Optional:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OSS_MODEL_ID` | `Qwen/Qwen2.5-14B-Instruct` | HF model for all LLM calls |
| `COLAB_LIMIT` | (none) | Smoke test: limit ingredient lines (e.g. `100`) |
| `COLAB_MIN_BATCH` | `8` | Minimum parallel judge batch size |

**Not needed:** `OPENAI_API_KEY`, MLflow.

## Run the notebook

1. Open [`notebooks/colab_ingredient_resolution_oss.ipynb`](../notebooks/colab_ingredient_resolution_oss.ipynb) in Colab.
2. **Runtime → Change runtime type → GPU** (A100 if available).
3. Set secrets (table above).
4. **Runtime → Run all**.

The notebook:

1. Clones this repo (`scripts/` pipeline code).
2. Syncs the S3 input bundle to `/content/capstone_cache/`.
3. Downloads `RecipeNLG.csv` from the raw bucket if needed.
4. Probes GPU/RAM and loads **Qwen2.5-14B-Instruct** (4-bit; falls back to 7B on OOM).
5. Monkey-patches LLM hooks (`judge_async`, line enrichment, portion pick).
6. Calls `run_feasibility(n_recipes=1000, seed=42, use_mlflow=False, ...)`.
7. Uploads outputs to S3.

## Monitoring progress during a run

The pipeline cell shows **one tqdm bar per LLM pass** (enrichment lines, judge calls, portion picks). Stdout and warnings are suppressed during the run.

Artifacts under `/content/capstone_runs/<RUN_ID>/`:

| File | Updates |
|------|---------|
| `progress.json` | Every **10** prompts with live stats (errors, fdc match rate, ETA) |
| `judge_stream.jsonl` | Full judge row appended after each inference |
| `judge_matches_raw.parquet` | Compacted every 10 judge rows (+ at end) |
| `phase_log.jsonl` | Phase start/end events |

Poll from your Mac after periodic S3 sync:

```bash
aws s3 cp s3://{S3_BUCKET_ARTIFACTS}/colab/runs/{run_id}/progress.json -
```

**Not judge progress:** `recipe_cache/recipe_ingredients_parsed.parquet` is embedding prep (always ~8,754 lines for the full sample).

Optional env vars:

| Variable | Default | Purpose |
|----------|---------|---------|
| `COLAB_MIN_BATCH` | `8` | Minimum `judge_concurrency` and vLLM `max_num_seqs` on A100 |
| `COLAB_PROGRESS_S3_PREFIX` | (set by notebook) | boto3 upload of progress files every 10 prompts |

## S3 output layout

```
s3://{S3_BUCKET_ARTIFACTS}/colab/runs/{run_id}/
  progress.json
  judge_stream.jsonl
  phase_log.jsonl
  run_manifest.json
  feasibility_report.json
  amount_classification.parquet
  judge_matches_raw.parquet
  pipeline_matches.parquet
  oss_model_meta.json
```

Compare `feasibility_report.json` to `baseline_summary.json` from the input bundle.

## Model and concurrency

| Backend | When | `judge_concurrency` |
|---------|------|---------------------|
| **vLLM** (preferred) | A100 40GB+ | **≥ 8** (up to 12 on 80GB; `COLAB_MIN_BATCH`) |
| **transformers + 4-bit** | vLLM install fails | **≥ 8** on 14GB+ VRAM; 2 on T4 |

Default model: **Qwen/Qwen2.5-14B-Instruct** (NF4 4-bit). OOM → **Qwen/Qwen2.5-7B-Instruct**.

All OSS LLM calls (enrichment, judge, portion pick) use **strict JSON schema** enforcement:
vLLM `StructuredOutputsParams(json=...)` on A100; `lm-format-enforcer` on the transformers fallback.
Token budgets: enrichment/judge 1024 (auto-doubles up to 2048 if truncated).

## Runtime expectations

| Setup | Full run (~8,754 lines) |
|-------|-------------------------|
| A100 + 14B + vLLM | ~2–4 hours |
| A100 + transformers fallback | ~4–6 hours |
| T4 + 7B | ~6–10 hours (smoke test recommended) |
| `COLAB_LIMIT=100` | ~20–30 min on A100 |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Supabase SSL / connection errors | Check `PG_SSL_MODE=require`, pooler host/port, password |
| HF model download 401 | Set `HF_TOKEN` |
| CUDA OOM | Notebook auto-falls back to 7B; or set `COLAB_LIMIT=100` |
| vLLM install fails | Notebook uses transformers automatically |
| Missing bundle | Run `export_colab_s3_bundle.py` locally |
| Recipe CSV missing | Set `S3_BUCKET_RAW` or upload `RecipeNLG.csv` into the bundle |

## Partner workflow

1. Run export script once after a new baseline.
2. Open notebook, run all cells.
3. Pull results: `aws s3 sync s3://{artifacts}/colab/runs/{run_id}/ ./scratch/colab_oss_run/`
4. Compare metrics in `feasibility_report.json` vs `baseline_summary.json`.
