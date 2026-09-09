-- Sentinel non-caloric / negligible food entry.
--
-- A synthetic stand-in the LLM judge can select when an ingredient carries no
-- meaningful calories (water, ice, plain salt, garnish, etc.) and no real USDA
-- entry matches. All four core macros (protein 1003, fat 1004, carb 1005,
-- energy 1008) are 0, so downstream nutrition resolution contributes nothing for
-- this pick. This lets the judge report an honest, non-zero certainty for
-- inconsequential ingredients instead of abstaining with certainty 0.
--
-- The sentinel lives in usda.food + usda.food_nutrient only (NOT in the
-- food_4macro materialized view), so it never enters lexical/semantic retrieval
-- and never needs an embedding. The judge sees it because the matching script
-- injects it into every candidate prompt. Keep fdc_id in sync with
-- SENTINEL_FDC_ID in scripts/ingredient_match_llm.py.
--
-- Idempotent: safe to re-run.

SET search_path TO usda, public;

INSERT INTO usda.food (fdc_id, data_type, description, food_category_id, publication_date)
VALUES (
    999000001,
    'sentinel',
    'NON-CALORIC OR NEGLIGIBLE INGREDIENT (ice, plain salt, garnish)',
    NULL,
    NULL
)
ON CONFLICT (fdc_id) DO UPDATE
    SET data_type   = EXCLUDED.data_type,
        description = EXCLUDED.description;

-- Replace any prior sentinel nutrient rows (food_nutrient has no natural unique
-- key on (fdc_id, nutrient_id), so delete-then-insert keeps this idempotent).
DELETE FROM usda.food_nutrient WHERE fdc_id = 999000001;

INSERT INTO usda.food_nutrient (id, fdc_id, nutrient_id, amount)
VALUES
    (9990000003, 999000001, 1003, 0),  -- protein (g)
    (9990000004, 999000001, 1004, 0),  -- total fat (g)
    (9990000005, 999000001, 1005, 0),  -- carbohydrate (g)
    (9990000008, 999000001, 1008, 0);  -- energy (kcal)
