# syntax=docker/dockerfile:1
# Recipe optimization agent image (CPU-only PyTorch). See pyproject.toml [tool.uv.sources].
# Installs main deps only: `uv sync --frozen --no-dev` (no notebook/pipeline/mvp extras).
#
# UI: recipe_opt_web serves MacroIQ at `/` (static/macroiq.html + .css/.js).
# Developer playground remains at `/playground` (static/index.html).
#
# Runtime caches baked into the image:
#   - foodon_web/cache/{foodon_index,foodon_hierarchy,fdc_foodon_map}.json
#   - data/dequant_norm_llm_cache.json  (FDC grounding dequant lookups)
# Neighborhood Jaccard cache is NOT in the image — it lives in Supabase
# (recipe.canonical_neighborhood_cache, cache_version matching the agent).

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/.cache/huggingface \
    PYTHONPATH=/app/scripts:/app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY recipe_opt_agent/ recipe_opt_agent/
# Includes MacroIQ product UI (static/macroiq.*) and playground (static/index.html).
COPY recipe_opt_web/ recipe_opt_web/
# Dequant-cache lookup package used by grounding (DraftDequantCache).
COPY eval_fdc_grounding_ui/__init__.py eval_fdc_grounding_ui/__init__.py
COPY eval_fdc_grounding_ui/draft_cache.py eval_fdc_grounding_ui/draft_cache.py
# Schema bootstrap for neighborhood + MacroIQ run logging.
RUN mkdir -p sql
COPY sql/42_create_canonical_neighborhood_cache.sql sql/42_create_canonical_neighborhood_cache.sql
COPY sql/43_create_macroiq_runs.sql sql/43_create_macroiq_runs.sql
# Agent runtime modules under scripts/ (PYTHONPATH). Avoid shipping the whole
# scripts tree so pipeline/notebook helpers stay out of the image.
COPY scripts/db.py \
     scripts/mvp_data.py \
     scripts/mvp_nutrient_fit.py \
     scripts/mvp_recipe_ranker.py \
     scripts/recipe_macro_optimizer.py \
     scripts/recipe_data_access.py \
     scripts/recipe_similarity.py \
     scripts/hull_geometry.py \
     scripts/loss_field.py \
     scripts/opt_diagnosis.py \
     scripts/weighted_empirical_opt.py \
     scripts/canonical_optimization.py \
     scripts/neighborhood_expansion.py \
     scripts/augmentation_retrieve.py \
     scripts/foodon_index.py \
     scripts/foodon_hierarchy_cache.py \
     scripts/portion_gram.py \
     scripts/amount_kind.py \
     scripts/unit_convert.py \
     scripts/unit_aliases.py \
     scripts/usda_volume_units.py \
     scripts/recipe_parse_rules.py \
     scripts/resolution_plan.py \
     scripts/parse_recipe_ingredient.py \
     scripts/build_dequant_norm_cache.py \
     scripts/dequant_volume_anchor.py \
     scripts/ingredient_query_cache.py \
     scripts/ingredient_match_staged.py \
     scripts/resolved_recipe_portion.py \
     scripts/load_food_4macro.py \
     scripts/progress_utils.py \
     scripts/
RUN mkdir -p foodon_web/cache data
COPY foodon_web/cache/foodon_index.json foodon_web/cache/foodon_index.json
COPY foodon_web/cache/foodon_hierarchy.json foodon_web/cache/foodon_hierarchy.json
COPY foodon_web/cache/fdc_foodon_map.json foodon_web/cache/fdc_foodon_map.json
COPY data/dequant_norm_llm_cache.json data/dequant_norm_llm_cache.json

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Warm MiniLM weights used by neighborhood / creative embedding paths.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/scripts:/app \
    RECIPE_DATA_SOURCE=db \
    DEQUANT_CACHE_PATH=/app/data/dequant_norm_llm_cache.json \
    MACROIQ_FAST_DEMO=1 \
    MACROIQ_SAVE_NEIGHBORHOOD_CACHE=1

COPY --from=builder /app /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "recipe_opt_web.server:app", "--host", "0.0.0.0", "--port", "8000"]
