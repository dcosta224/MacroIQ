# MacroIQ

UC Berkeley MIDS Capstone (Summer 2026) · Daniel Costa, Kadin Wilkins

**[Project Write-Up](https://www.ischool.berkeley.edu/programs/mids/capstone/2026b-summer/macroiq-0)** · **[MVP Live Demo](http://macroiq.org)** · **[Product UI](recipe_opt_web/)** · **[Agent](recipe_opt_agent/)**

## Problem & Motivation

Ask a large language model to create a recipe for "high-protein carbonara" and you usually get something that reads like a recipe, but falls apart in the kitchen. Traditional nutrition tools can tell you whether a meal meets your goals, but not how to redesign it without compromising the dish. The hard problem sits in between:

**People want meals that satisfy their nutrition goals without sacrificing quality.**

In practice, AI-generated recipes often fall flat once they leave the chat window. They miss the user's nutrition targets, drift so far from the original dish that the recipe no longer resembles what was requested, or recommend ingredient combinations that don't hold up in the kitchen. Underneath these failures is a lack of grounded reasoning: vague ingredient descriptions, inconsistent quantities, and nutrition estimates that don't reliably map to real values. Health-oriented home cooks need recommendations they can trust, not recipes that merely sound plausible.

MacroIQ treats this as a data science problem: start with real human-tested recipes and nutrition markers, propose data-driven ingredient edits, and leverage an LLM to check every candidate against the user’s goals before producing a recommendation.

## Our Solution

MacroIQ is an agentic recipe design system that bridges the gap between language models and structured food science. Rather than asking an LLM to invent a recipe in one pass, MacroIQ grounds recipe generation in millions of human-created recipes, authoritative nutrition databases, and mathematical optimization. The system treats recipe design as an iterative decision-making problem: propose ingredient changes, measure their nutritional and culinary consequences, and continue refining until the recipe satisfies user constraints.

This approach allows MacroIQ to generate recipes that simultaneously satisfy macronutrient targets, accommodate dietary restrictions, and preserve dish identity while remaining explainable to the user. Every recommendation is accompanied by evidence showing how ingredient choices compare with similar real recipes and whether nutrition goals have actually been achieved.

Users interact with MacroIQ by naming a dish, specifying nutritional targets and dietary preferences, and observing the agent progressively redesign the recipe. Rather than producing a single opaque answer, the system exposes each optimization step, allowing users to both understand and trust why a recommendation was made.

---

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
