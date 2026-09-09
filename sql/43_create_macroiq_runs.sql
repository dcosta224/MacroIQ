-- MacroIQ UI run history (logged from recipe_opt_web /api/run).

CREATE SCHEMA IF NOT EXISTS mvp_logs;

CREATE TABLE IF NOT EXISTS mvp_logs.macroiq_runs (
    run_id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          timestamptz NOT NULL DEFAULT now(),
    finished_at         timestamptz,
    status              text NOT NULL DEFAULT 'running',
    mode                text,
    agent_mode          text,
    canonical_id        bigint,
    title               text,
    taste_text          text,
    user_request        text,
    kcal_target         double precision,
    use_macro_targets   boolean,
    request_json        jsonb NOT NULL DEFAULT '{}'::jsonb,
    config_json         jsonb NOT NULL DEFAULT '{}'::jsonb,
    result_json         jsonb,
    browse_candidates   jsonb,
    error_message       text,
    winner_candidate_id text,
    winner_ratio_loss   double precision,
    winner_calories     double precision
);

CREATE INDEX IF NOT EXISTS idx_macroiq_runs_created_at
    ON mvp_logs.macroiq_runs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_macroiq_runs_canonical_id
    ON mvp_logs.macroiq_runs (canonical_id);

ALTER TABLE mvp_logs.macroiq_runs ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON mvp_logs.macroiq_runs FROM anon, authenticated;
GRANT ALL ON mvp_logs.macroiq_runs TO service_role;
