# MacroIQ

**An agentic recipe design system that keeps favorite dishes recognizable under nutrition goals.**

[Project write-up](https://www.ischool.berkeley.edu/programs/mids/capstone/2026b-summer/macroiq-0) · [MVP Live Demo](http://macroiq.org) · [Product UI](recipe_opt_web/) · [Agent](recipe_opt_agent/)

UC Berkeley MIDS Capstone (Summer 2026) · Daniel Costa, Kadin Wilkins

---

## Problem

LLMs draft recipes that read well and fail in the kitchen—missed macros, lost dish identity, or amounts that don’t map to real foods. Trackers measure after the fact; they don’t redesign a named meal. People want nutrition goals without sacrificing quality.

## Solution

MacroIQ grounds recipe redesign in RecipeNLG, USDA FoodData Central, and FoodOn. An agent proposes small ingredient edits; a convex optimizer chooses grams; structured checks decide whether the plate still works. Users set a dish, macro box, and dietary rules, then audit each step.

```text
request + macros → FoodOn neighborhood → USDA grounding
                 → hull + CVXPY LP → LangGraph edit loop → recipe
```

## Innovation

The LLM proposes edits. The neighborhood and linear program own the grams.

| Piece | Code |
|-------|------|
| Weighted empirical LP (Atwater PFC, optional fiber) | [`scripts/weighted_empirical_opt.py`](scripts/weighted_empirical_opt.py) |
| FoodOn neighborhood priors | [`scripts/canonical_optimization.py`](scripts/canonical_optimization.py) |
| Feasibility / conical hull | [`scripts/hull_geometry.py`](scripts/hull_geometry.py) |
| LangGraph agent loop | [`recipe_opt_agent/graph.py`](recipe_opt_agent/graph.py) |
| Ingredient → FDC + grams | [`scripts/portion_pipeline_feasibility.py`](scripts/portion_pipeline_feasibility.py) |

Design notes: [`docs/recipe_opt_agent.md`](docs/recipe_opt_agent.md)

## Systems

| Area | Paths |
|------|-------|
| Supabase loaders / SQL | [`sql/`](sql/), [`scripts/load_recipes.py`](scripts/load_recipes.py) |
| Grounding pipeline | [`scripts/portion_pipeline_feasibility.py`](scripts/portion_pipeline_feasibility.py) |
| Curator UIs | [`foodon_cache_ui/`](foodon_cache_ui/), [`dequant_cache_ui/`](dequant_cache_ui/), [`eval_fdc_grounding_ui/`](eval_fdc_grounding_ui/) |
| GPU batch resolution | [`ec2/`](ec2/) |
| Eval vs GPT-5.5 | [`tests/run_eval_suite.py`](tests/run_eval_suite.py), [write-up](https://www.ischool.berkeley.edu/programs/mids/capstone/2026b-summer/macroiq-0) |

## Deployment

| Piece | Paths |
|-------|-------|
| MVP Live Demo | [macroiq.org](http://macroiq.org) |
| Docker + ECR CI | [`Dockerfile`](Dockerfile), [`.github/workflows/push-ecr.yml`](.github/workflows/push-ecr.yml) |
| EC2 deploy | [`docs/ECR_EC2_DEPLOY.md`](docs/ECR_EC2_DEPLOY.md), [`scripts/deploy/`](scripts/deploy/) |
| Product surfaces | [`recipe_opt_web/`](recipe_opt_web/) (`/`, `/playground`, `/personal`) |

## Results

Same requests and boxes as one-shot GPT-5.5; shared USDA scoring. MacroIQ wins **ratio** and **nutrient** in every suite; on **preserve identity**, holistic **5.80** ≈ human recipes **5.85** (GPT-5.5: **3.92**). Full table: [project write-up](https://www.ischool.berkeley.edu/programs/mids/capstone/2026b-summer/macroiq-0).

## Stack

Python 3.11 · FastAPI · LangGraph · CVXPY · OpenAI · sentence-transformers · Supabase / pgvector · Docker · AWS (ECR, EC2, S3) · GitHub Actions · uv

## Quick start

```bash
cp .env.example .env
uv sync --extra notebook --extra pipeline --extra dev
export RECIPE_DATA_SOURCE=local   # after download_cap40_recipe_store.py if needed

PYTHONPATH=scripts:. uv run python -m recipe_opt_web --reload
# http://127.0.0.1:8010
```

## Acknowledgements

UC Berkeley I School / MIDS Capstone and Data Science 210. Data: USDA FoodData Central, RecipeNLG, FoodOn, FAO/INFOODS density database.
