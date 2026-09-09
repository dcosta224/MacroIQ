# MacroIQ

**Keeping favorite dishes recognizable under nutrition goals.**

[Live demo](http://macroiq.org) · [Product UI](recipe_opt_web/) · [Agent](recipe_opt_agent/) · [Design notes](docs/recipe_opt_agent.md)

MacroIQ is a Berkeley MIDS Capstone project. Ask for a high-protein carbonara, a no-pork BBQ plate, or a vegetarian bobotie with a real macro box—and get a recipe that is still that dish after every ingredient is linked to USDA FoodData Central and checked against how people actually cook it.

This README is written for recruiters and hiring managers browsing the portfolio. It highlights **what the product does**, **what is actually novel**, and **where in the repo** to look for systems engineering and production work. Setup detail for collaborators lives further down and in linked docs.

---

## The problem in one breath

Large language models draft recipes that *read* well and then fall apart in the kitchen: wrong USDA matches, missed protein targets, or a “carbonara” that no longer looks like carbonara. Calorie trackers measure after the fact; they do not redesign a named dish. The useful problem sits in between—**nutrition goals without sacrificing what makes the meal worth cooking**.

---

## What we built

Users name a dish (or type a free-text request), set protein / carbohydrate / fat calorie shares and dietary rules, and watch an agent redesign the recipe in measurable steps.

Under the hood:

1. **Ground** free-form ingredient lines to USDA foods and gram amounts  
2. **Locate** the dish in a FoodOn neighborhood of real human recipes (RecipeNLG / CuisineNLG)  
3. **Optimize** amounts with a convex program so macros and typical proportions stay honest  
4. **Edit** with a LangGraph loop—propose a small swap/add/remove, re-score with the optimizer, keep going only while the suggestion improves  

Language models brainstorm edible changes. Structured data and math decide whether the plate still works.

```text
  User request + macro box
            │
            ▼
  FoodOn neighborhood of real recipes
            │
            ▼
  USDA FDC grounding (foods + grams)
            │
            ▼
  Hull check + CVXPY LP  ←── amounts owned by the optimizer
            │
            ▼
  LangGraph agent loop (diagnose → propose → decide → apply)
            │
            ▼
  Auditable recipe: macros, ratio fidelity, dietary survival
```

**Try it:** [macroiq.org](http://macroiq.org) · local product UI at `/` via [`recipe_opt_web/`](recipe_opt_web/) · personal pantry optimizer at `/personal`.

---

## Main innovation

> **LLM proposes the edit. The neighborhood + linear program own the grams.**

Most “AI recipe” demos ask a model for a finished ingredient list and hope the numbers work out. MacroIQ treats recipe design as **constrained optimization with an agent in the loop**:

| Idea | Why it matters | Where to look |
|------|----------------|---------------|
| **Weighted empirical LP** | Minimizes how far mass shares drift from real dishes for that family, subject to Atwater protein/carb/fat calorie fractions (and optional fiber as a gram constraint—not a fake calorie share) | [`scripts/weighted_empirical_opt.py`](scripts/weighted_empirical_opt.py) |
| **FoodOn neighborhood priors** | “Still carbonara?” is learned from related resolved recipes, not from vibes in the prompt | [`scripts/canonical_optimization.py`](scripts/canonical_optimization.py), FoodOn caches under [`foodon_web/cache/`](foodon_web/cache/) |
| **Conical hull / feasibility** | Before chasing edits, ask whether the current ingredient set can reach the macro box at all | [`scripts/hull_geometry.py`](scripts/hull_geometry.py), [`scripts/opt_diagnosis.py`](scripts/opt_diagnosis.py) |
| **LangGraph control loop** | Diagnose → propose bundles → LP-score → auto-apply clear favorites or ask the LLM → apply → stop on fidelity bands | [`recipe_opt_agent/graph.py`](recipe_opt_agent/graph.py), [`docs/recipe_opt_agent.md`](docs/recipe_opt_agent.md) |
| **Structured OOD protein** | When the neighborhood cannot stretch protein, propose a checked lean-protein add (e.g. turkey on ribs) instead of free-associating beans and tenderloin | [`recipe_opt_agent/ood_branch.py`](recipe_opt_agent/ood_branch.py), win stories in [`docs/agent_vs_gpt55_presentation_wins.md`](docs/agent_vs_gpt55_presentation_wins.md) |
| **LLM seed → LP refine** | For the personal pantry path: one model call drafts amounts toward macros + fiber; a trust-region LP enforces the box without reinventing the dish | [`recipe_opt_agent/personal_pipeline.py`](recipe_opt_agent/personal_pipeline.py) |

The non-technical version: we do not trust the chat model with the calculator. We trust it with *ideas*, then verify those ideas against USDA numbers and the way similar recipes are actually built.

---

## Systems thinking (where to look)

This repo is not a single notebook glued to an API. End-to-end food modeling required pipelines, caches, eval harnesses, and honest failure modes.

| Theme | What we did | Paths |
|-------|-------------|-------|
| **Data foundation** | Loaded USDA FoodData Central, RecipeNLG (~2M recipes), and FAO density conversions into Supabase Postgres (`usda` / `recipe` / `conversions`) | [`sql/`](sql/), [`scripts/load_recipes.py`](scripts/load_recipes.py), setup notes below |
| **Ingredient resolution** | Parse → retrieve → judge → grams → macros. A polished draft is useless until lines resolve to real `fdc_id`s and gram weights | [`scripts/portion_pipeline_feasibility.py`](scripts/portion_pipeline_feasibility.py), [`docs/portion_resolution_roadmap.md`](docs/portion_resolution_roadmap.md) |
| **Human-in-the-loop caches** | Curator UIs for FoodOn labels, dequant/FDC portion caches, and grounding eval corrections—because missingness and bad links cap everything downstream | [`foodon_cache_ui/`](foodon_cache_ui/), [`dequant_cache_ui/`](dequant_cache_ui/), [`eval_fdc_grounding_ui/`](eval_fdc_grounding_ui/) |
| **GPU batch resolution** | On-demand EC2 + Qwen for large corpus grounding, S3 artifacts, gate-then-scale runs | [`ec2/`](ec2/) |
| **Soft vs hard constraints** | Soft nutrient slack can trade a small miss for typicality; dietary tags and dish identity stay hard | [`docs/recipe_opt_agent.md`](docs/recipe_opt_agent.md), [`recipe_opt_agent/requirement_tags.py`](recipe_opt_agent/requirement_tags.py) |
| **Eval as product surface** | Shared FDC scoring for agent vs one-shot frontier models; suites for macros, dietary bans, and identity under stretch targets | [`tests/run_eval_suite.py`](tests/run_eval_suite.py), [`docs/ischool_project_gallery_submission.md`](docs/ischool_project_gallery_submission.md) |

**Takeaway for hiring managers:** the interesting work is the *system*—how grounding quality, neighborhood geometry, and optimizer incentives interact—not a single clever prompt.

---

## Production deployment (where to look)

| Surface | Detail | Paths |
|---------|--------|-------|
| **Live staging** | [macroiq.org](http://macroiq.org) — Dockerized FastAPI app on EC2 (host `:80` → container `:8000`) | [`docs/ECR_EC2_DEPLOY.md`](docs/ECR_EC2_DEPLOY.md) |
| **CI → registry** | Push to `deployment` builds `linux/amd64` and pushes to ECR (`macroiq:<sha>` and `:deployment`) | [`.github/workflows/push-ecr.yml`](.github/workflows/push-ecr.yml), [`Dockerfile`](Dockerfile) |
| **Deploy scripts** | Pull image, systemd unit, start/stop staging without leaving GPU/demo spend running | [`scripts/deploy/`](scripts/deploy/), [`infra/aws/`](infra/aws/) |
| **Product surfaces** | MacroIQ UI, developer playground with live LangGraph transcript, loop demo, personal pantry lab | [`recipe_opt_web/`](recipe_opt_web/) |
| **Data plane** | Supabase for nutrition + neighborhood caches; S3 for batch artifacts; secrets at runtime, not baked into the image | [`docs/AWS_WORKFLOW.md`](docs/AWS_WORKFLOW.md) |

Staging is **on-demand**—we stop EC2 when idle. That is intentional cost control, not an unfinished deploy story.

---

## Results (honest)

We compared MacroIQ to one-shot **GPT-5.5** on the same requests and macro boxes, with **shared USDA grounding** so both sides are scored the same way. Metrics: holistic quality (LLM judge), ingredient-ratio fidelity to real dishes (lower is better), and nutrient-box loss (lower is better).

- MacroIQ wins **ratio** and **nutrient** fit in every reported suite.  
- On **preserve identity**, holistic **5.80** sits next to a human-recipe reference mean of **5.85** (GPT-5.5: **3.92**).  
- Under **dietary restrictions**, MacroIQ leads on all three metrics.  
- On **general quality**, GPT-5.5 can edge holistic slightly; MacroIQ still leads on the measurable fit metrics that keep a meal honest after measurement.

Concrete failure modes for one-shot drafts show up in head-to-heads: missing a protein box after resolution, collapsing a no-pork request into a spice pile, or drifting far from typical proportions. The agent’s edit-and-remeasure loop is built to catch those before a suggestion is shown.

Full table and framing: [`docs/ischool_project_gallery_submission.md`](docs/ischool_project_gallery_submission.md) · walkthrough cases: [`docs/agent_vs_gpt55_presentation_wins.md`](docs/agent_vs_gpt55_presentation_wins.md).

We also document where we do **not** win every suite or metric—see [`docs/canonical_unconstrained_eval.md`](docs/canonical_unconstrained_eval.md). Grounding quality (~71% FDC+grams on a gated sample) remains the binding constraint on corpus coverage.

---

## Stack

**Python 3.11** · **FastAPI / Uvicorn** · **LangGraph** · **CVXPY** · **OpenAI API** · **NumPy / SciPy** · **sentence-transformers** · **Supabase Postgres / pgvector** · **Docker** · **AWS (ECR, EC2, S3)** · **GitHub Actions** · **uv** · HTML/JS product UIs  

GPU batch path: vLLM / Transformers (Qwen3) on EC2. Dependency groups and extras: [`pyproject.toml`](pyproject.toml).

---

## Explore the repo

| If you care about… | Start here |
|--------------------|------------|
| Product experience | [`recipe_opt_web/`](recipe_opt_web/), [macroiq.org](http://macroiq.org) |
| Agent architecture | [`recipe_opt_agent/`](recipe_opt_agent/), [`docs/recipe_opt_agent.md`](docs/recipe_opt_agent.md) |
| Optimizer math | [`scripts/weighted_empirical_opt.py`](scripts/weighted_empirical_opt.py), [`scripts/hull_geometry.py`](scripts/hull_geometry.py) |
| Grounding pipeline | [`scripts/portion_pipeline_feasibility.py`](scripts/portion_pipeline_feasibility.py) |
| Deploy path | [`docs/ECR_EC2_DEPLOY.md`](docs/ECR_EC2_DEPLOY.md), [`Dockerfile`](Dockerfile) |
| Eval story | [`docs/ischool_project_gallery_submission.md`](docs/ischool_project_gallery_submission.md) |
| Demo video (local asset) | [`docs/animated_demo_stage_v8.mp4`](docs/animated_demo_stage_v8.mp4) |

---

## Quick start (collaborators)

```bash
# Environment
cp .env.example .env          # OPENAI_API_KEY; PG_* if using Supabase
uv sync --extra notebook --extra pipeline --extra dev

# Optional local recipe store (preferred over DB for demos)
PYTHONPATH=scripts:. uv run python scripts/download_cap40_recipe_store.py
export RECIPE_DATA_SOURCE=local

# Product UI
PYTHONPATH=scripts:. uv run python -m recipe_opt_web --reload
# → http://127.0.0.1:8010            MacroIQ
# → http://127.0.0.1:8010/playground developer playground
# → http://127.0.0.1:8010/personal   pantry → plate lab
```

Offline agent smoke:

```bash
PYTHONPATH=scripts:. uv run python -m recipe_opt_agent \
  --fixture tests/fixtures/recipe_opt/synthetic_problem.json \
  --out scratch/recipe_opt_runs/demo.json

PYTHONPATH=scripts:. uv run pytest tests/test_recipe_opt_agent_graph.py -q
```

Deeper partner setup (data loaders, USDA `\copy`, embeddings, AWS cost notes): historically covered in this README’s data sections—see [`docs/AWS_WORKFLOW.md`](docs/AWS_WORKFLOW.md), [`sql/`](sql/), and [`recipe_opt_web/README.md`](recipe_opt_web/README.md).

### Data sources (attribution)

- **USDA FoodData Central** — U.S. Department of Agriculture  
- **RecipeNLG / Open Recipes** — see original dataset terms  
- **FoodOn** — food ontology  
- **FAO/INFOODS Density Database v2.0** — volume ↔ mass factors  

Raw dumps under `Data/` are gitignored (multi‑GB).

---

## Acknowledgements

Built for the UC Berkeley I School / MIDS Capstone program. Thanks to instructors, partners, and classmates who cooked, tasted, and argued with the agent’s suggestions—and to the public nutrition and recipe corpora that make grounded cooking AI possible.
