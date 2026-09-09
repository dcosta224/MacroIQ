# Recipe opt agent — LangSmith observability & eval navigation

This guide covers: (1) turning on LangSmith, (2) what we log for agent decisions, (3) how to navigate the UI quickly, and (4) where full local artifacts live.

## 1. One-time LangSmith setup

1. Open [smith.langchain.com](https://smith.langchain.com) and sign in (free developer plan is enough).
2. **Settings → API Keys → Create API Key**. Copy it once.
3. Add to your local `.env` (never commit the key):

```bash
# Prefer modern names (both work; we dual-set them in code)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=recipe-opt-agent-eval

# Legacy aliases (optional if you already have them)
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=lsv2_...
# LANGCHAIN_PROJECT=recipe-opt-agent-eval
```

4. Confirm OpenAI is set (`OPENAI_API_KEY=...`) for live runs.
5. Smoke-test:

```bash
PYTHONPATH=scripts:. uv run python tests/run_eval_suite.py --live
```

You should see `LangSmith: tracing=on key=set project=recipe-opt-agent-eval`. Within ~30s, traces appear under that project in the UI.

**EU region:** also set `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`.

## 2. What gets saved on every eval suite run

### Local (always)

`tests/run_eval_suite.py` writes:

```
scratch/recipe_opt_runs/eval_suites/<name>_<timestamp>/
  suite_summary.json          # aggregates + per-run index
  suite_metrics.jsonl         # one metrics row per case
  runs/<case>__<id>/
    manifest.json             # config + pointers
    final.json                # agent final payload + telemetry
    steps.json                # every graph node update
    tools.json                # every tool call (hull, LP, retrieve, score, decide, …)
    retrieval.json            # plan_slots / retrieve_slots / score_bundles only
    llm_calls.json            # full prompts + raw responses + usage
    transcripts.jsonl         # UI-style transcript entries
    events.jsonl              # raw SSE-equivalent event stream
    flow.json                 # node/edge graph meta
    metrics.json              # actionable + observability scores (see below)
    run_metadata.json         # mode, query, macros, F_accept/max_iter, models, graph
    flow.json                 # node/edge graph meta
```

Huge matrices (`M`, full `next_problem`) are summarized so artifacts stay readable; presence/shape is kept.

### LangSmith (when tracing is on)

| Signal | How it appears | Why it matters |
|--------|----------------|----------------|
| Graph nodes | Run tree spans (`diagnose`, `propose`, `decide`, …) | See which path fired |
| OpenAI calls | Nested LLM spans (via `wrap_openai`) | Prompt / latency / tokens |
| Tags | `recipe-opt-agent`, `creative`\|`neighborhood`, `case:…`, `suite:…` | Filter a suite or mode |
| Metadata | `case_name`, `suite_id`, `complex_model`, … | Group A/B arms |
| Feedback scores | Numeric keys listed below | Sort / dashboard / alerts |

## 3. Metrics: actionable vs observability

Feedback is posted in two classes (comment `class=actionable` / `class=observability`). Both land on the LangSmith root run and in local `metrics.json` (flat keys + nested `actionable` / `observability` groups).

### Actionable (decision knobs)

| Metric | If bad… | Likely lever |
|--------|---------|--------------|
| `final_status_ok` | Many non-accept | `F_accept` / `max_iterations` / retrieval |
| `final_L_max_norm` / `final_n_red` | High fidelity loss | Neighborhood / bundle scoring / box |
| `final_nutrient_slack` | Macros missed | Draft model / macro_gap slots / box |
| `final_holistic` | Low taste overlap | Draft prompt / intent scoring |
| `tag_violations_final` | >0 | Tag filters in retrieve/apply |
| `n_auto_applies` vs `n_llm_calls` | Auto≈0 but `lp_agreement_rate` high | Widen `auto_apply_margin` |
| `lp_agreement_rate` | Low | Context/prompt or auto-gate too aggressive |
| `escalate_rate` | Always escalate | Tighten escalate rules / context size |
| `expand_count` | High | Candidate quality / tags too strict |
| `oscillation_hits` | High | Fingerprint logic / slot diversity |
| `elapsed_s` / tokens | Cost spike | Model tier / auto-apply / context curation |

### Observability (auxiliary / signal quality)

| Metric | What it tells you |
|--------|-------------------|
| `foodon_mean/max_aggregation_levels`, `foodon_n_aggregated` | How deep leaves roll up to basis (neighborhood coarseness) |
| `foodon_min/mean_hits_in_recipe`, `foodon_n_low_hit_basis` | Whether share/ratio terms have enough neighborhood mass |
| `foodon_n_unmapped`, `foodon_frac_aggregated` | Mapping coverage |
| `neighborhood_n_recipes`, `n_ingredients_final` | Problem size |
| `final_ratio_term` | Ratio-loss needle at end (alongside nutrient/holistic) |
| `pool_size_final`, `n_applies`, `n_decision_outcomes` | Path / churn shape |
| `hull_outside_final`, `grounding_resolve_rate` | Geometry / creative grounding health |

Full per-ingredient FoodOn detail remains on `foodon_basis_report` (local artifacts + diagnose tool), not as a feedback score.

### Run metadata (LangSmith + `run_metadata.json`)

Every eval / web / LangSmith run attaches:

- `agent_mode` (`neighborhood` \| `creative`), `user_request` / `taste_text`, `title`, `canonical_id`
- `macro_targets`, `F_accept`, `F_max`, `max_iterations`, auto-apply margins
- `models` by stage (`decide_routine`, `decide_escalate`, `draft`, `tags`, `judge`, `identity_extract`)
- `graph.nodes` / `graph.edges` for the active mode
- neighborhood size / cache flag when available

Filter LangSmith by metadata fields or tags (`creative` / `neighborhood`, `suite:…`, `case:…`).

## 4. How to navigate LangSmith so you iterate fast

### Daily loop (5 minutes)

1. Open **Tracing** → project `recipe-opt-agent-eval`.
2. Filter **Tag** = `suite:<id>` from the suite stdout line, or **Metadata** `suite_id`.
3. Sort by **Latency** or feedback `elapsed_s` — open the slowest run first.
4. In the run tree, click in this order:
   - **`decide`** — auto vs LLM? escalate? agreed with LP?
   - Nested **ChatOpenAI / OpenAI** under draft/decide — read the curated DecisionContext and JSON reply.
   - **`propose` → retrieve_slots / score_bundles`** — empty shortlists? best `delta_L_star` ≈ 0?
   - **`diagnose`** — band + `L_max_norm` / reds.
5. Cross-check local `runs/.../llm_calls.json` and `retrieval.json` if you need the exact tool payload (LangSmith may truncate large JSON).

### Decision playbook

| What you see | Do this next |
|--------------|--------------|
| Many LLM decides that pick the LP-best bundle | Raise `auto_apply_margin` / trust auto more |
| Auto-apply then fidelity worse | Tighten `auto_apply_delta_eps` or require identity-safe annotation |
| Draft box far from target (`nutrient_slack`) | Change `creative_model` or draft prompt / box midpoints |
| Tag violations | Inspect `retrieve_slots` drops; harden filters before decide |
| High `escalate_rate` on first iteration | Check dietary tags + near-tie margin in `model_policy` |
| Oscillation / expand loops | Look at `recent_edit_fingerprints` in DecisionContext |

### A/B complex models

```bash
PYTHONPATH=scripts:. uv run python tests/run_eval_suite.py --ab-complex --live
```

Filter metadata `complex_model=gpt-4.1-mini` vs `gpt-4o`. Compare feedback `final_status_ok`, `tag_violations_final`, `final_holistic`, and draft spans side-by-side. Prefer 4.1-mini unless 4o wins clearly on ≥2 axes.

## 5. Offline vs live

| Mode | Command | Artifacts | LangSmith |
|------|---------|-----------|-----------|
| Offline heuristic | `python tests/run_eval_suite.py` | Yes | No (unless you leave tracing on; spans will be graph-only) |
| Live | `python tests/run_eval_suite.py --live` | Yes | Yes if `LANGSMITH_TRACING=true` |
| Complex A/B | `python tests/run_eval_suite.py --ab-complex` | Yes | Yes |

## 6. Code map

| Module | Role |
|--------|------|
| [`recipe_opt_agent/observability.py`](../recipe_opt_agent/observability.py) | Env normalize, OpenAI wrap, run config, feedback scores |
| [`recipe_opt_agent/eval_artifacts.py`](../recipe_opt_agent/eval_artifacts.py) | Suite/run recorders + `run_agent_with_artifacts` |
| [`tests/run_eval_suite.py`](../tests/run_eval_suite.py) | CLI entrypoint |

Web UI streaming (`on_event`) is the same event shape as eval artifacts — one flight recorder for both.
