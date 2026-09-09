-- Cached Jaccard / FoodOn rollup neighborhood for a canonical dish.
-- Written by precompute_recipe_loss_fields.py (full Jaccard build);
-- read by CanonicalNeighborhood.build(use_cache=True) to skip recomputation.
-- Idempotent: safe to re-run.

CREATE SCHEMA IF NOT EXISTS recipe;

CREATE TABLE IF NOT EXISTS recipe.canonical_neighborhood_cache (
    canonical_recipe_id   bigint PRIMARY KEY,
    title                 text,
    n_recipes             integer NOT NULL,
    starting_recipe_id    text NOT NULL,
    cut_nodes             text[] NOT NULL,
    best_nodes            text[] NOT NULL,
    basis_nodes           text[] NOT NULL,
    recipe_sets           jsonb NOT NULL,
    rollup_chains         jsonb NOT NULL,
    basis_shares          jsonb NOT NULL,
    build_params          jsonb NOT NULL DEFAULT '{}'::jsonb,
    cache_version         integer NOT NULL DEFAULT 1,
    computed_at           timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_canonical_neighborhood_cache_computed
    ON recipe.canonical_neighborhood_cache (computed_at DESC);
