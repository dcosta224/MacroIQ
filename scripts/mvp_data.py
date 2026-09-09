"""Database loaders for MVP pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from db import connect
from recipe_macro_optimizer import NUTRIENT_IDS, build_macro_matrix

try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None


def _parse_embedding(val: Any) -> np.ndarray:
    if isinstance(val, (list, tuple, np.ndarray)):
        return np.asarray(val, dtype=np.float32)
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        return np.array([float(x) for x in s.strip("[]").split(",")], dtype=np.float32)
    raise ValueError(f"Cannot parse embedding: {type(val)}")


def fetch_mvp_recipe_ids(cur) -> list[int]:
    cur.execute(
        """
        SELECT DISTINCT rn.recipe_id
        FROM recipe.recipe_nutrients rn
        INNER JOIN recipe.resolved_recipes rr ON rr.recipe_id = rn.recipe_id
        ORDER BY rn.recipe_id
        """
    )
    return [int(r[0]) for r in cur.fetchall()]


def fetch_recipe_nutrients(cur, recipe_ids: list[int]) -> pd.DataFrame:
    if not recipe_ids:
        return pd.DataFrame()
    cur.execute(
        """
        SELECT recipe_id, recipe_name, protein_g, total_lipid_fat_g,
               carbohydrate_by_difference_g, energy_kcal
        FROM recipe.recipe_nutrients
        WHERE recipe_id = ANY(%s)
        ORDER BY recipe_id
        """,
        (recipe_ids,),
    )
    cols = [
        "recipe_id",
        "recipe_name",
        "protein_g",
        "total_lipid_fat_g",
        "carbohydrate_by_difference_g",
        "energy_kcal",
    ]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def fetch_recipe_embeddings(cur, recipe_ids: list[int]) -> tuple[np.ndarray, list[int]]:
    conn = cur.connection
    if register_vector is not None:
        register_vector(conn)
    cur.execute(
        """
        SELECT recipe_id, embedding
        FROM recipe.recipe_nlg_embedding
        WHERE recipe_id = ANY(%s)
        ORDER BY recipe_id
        """,
        (recipe_ids,),
    )
    rows = cur.fetchall()
    if not rows:
        return np.zeros((0, 384), dtype=np.float32), []
    ids = [int(r[0]) for r in rows]
    embs = np.vstack([_parse_embedding(r[1]) for r in rows])
    return embs, ids


def parse_nlg_ingredients(semantic_text: str) -> str:
    """Ingredient list from recipe.recipe_nlg_features.semantic_text (title | ing1 ing2 …)."""
    text = (semantic_text or "").strip()
    if not text:
        return ""
    if "|" in text:
        return text.split("|", 1)[1].strip()
    return text


def fetch_recipe_features(cur, recipe_ids: list[int]) -> dict[int, dict[str, Any]]:
    cur.execute(
        """
        SELECT recipe_id, title_clean, semantic_text, ingredient_count
        FROM recipe.recipe_nlg_features
        WHERE recipe_id = ANY(%s)
        """,
        (recipe_ids,),
    )
    return {
        int(r[0]): {
            "title_clean": r[1],
            "semantic_text": r[2],
            "ingredient_count": int(r[3]),
            "nlg_ingredients": parse_nlg_ingredients(r[2]),
        }
        for r in cur.fetchall()
    }


def fetch_resolved_ingredients(cur, recipe_id: int) -> pd.DataFrame:
    cur.execute(
        """
        SELECT recipe_id, ingredient_idx, ingredient, fdc_id, fdc_description,
               portion_id, portion_label, quantity, unit, gram_weight
        FROM recipe.resolved_recipes
        WHERE recipe_id = %s
        ORDER BY ingredient_idx
        """,
        (recipe_id,),
    )
    cols = [
        "recipe_id",
        "ingredient_idx",
        "ingredient",
        "fdc_id",
        "fdc_description",
        "portion_id",
        "portion_label",
        "quantity",
        "unit",
        "gram_weight",
    ]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def fetch_food_nutrients_for_recipe(cur, fdc_ids: list[int]) -> pd.DataFrame:
    """Fetch macro nutrients; prefers local store when RECIPE_DATA_SOURCE=local.

    ``cur`` may be None when using the local store.
    """
    try:
        from recipe_data_access import data_source, get_store

        if data_source() == "local":
            return get_store().food_nutrients(list(map(int, fdc_ids or [])))
    except Exception:
        pass
    if not fdc_ids:
        return pd.DataFrame(columns=["fdc_id", "nutrient_id", "amount"])
    if cur is None:
        from db import connect

        conn = connect()
        try:
            with conn.cursor() as c:
                return fetch_food_nutrients_for_recipe(c, fdc_ids)
        finally:
            conn.close()
    cur.execute(
        """
        SELECT fdc_id, nutrient_id, amount
        FROM usda.food_nutrient
        WHERE fdc_id = ANY(%s) AND nutrient_id = ANY(%s) AND amount IS NOT NULL
        """,
        (fdc_ids, list(NUTRIENT_IDS)),
    )
    return pd.DataFrame(cur.fetchall(), columns=["fdc_id", "nutrient_id", "amount"])


def build_recipe_macro_inputs(
    ingredients: pd.DataFrame,
    food_nutrients: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return x0 (gram weights) and M (4 x n) per-gram macro matrix aligned to ingredients."""
    n = len(ingredients)
    x0 = ingredients["gram_weight"].astype(float).to_numpy()
    fdc_ids = ingredients["fdc_id"].astype(int).tolist()

    fn = food_nutrients.copy()
    if fn.empty:
        return x0, np.zeros((4, n))

    pivot = fn.pivot_table(index="fdc_id", columns="nutrient_id", values="amount", aggfunc="first")
    nutrient_order = list(NUTRIENT_IDS)
    matrix_rows = []
    for fid in fdc_ids:
        if fid not in pivot.index:
            matrix_rows.append([0.0] * 4)
            continue
        row = pivot.loc[fid]
        matrix_rows.append([float(row.get(nid, 0.0) or 0.0) for nid in nutrient_order])
    per_100g = np.array(matrix_rows, dtype=float)  # (n, 4) USDA per-100g
    _, M = build_macro_matrix(x0, per_100g)
    return x0, M


def load_mvp_corpus() -> dict[str, Any]:
    conn = connect()
    try:
        with conn.cursor() as cur:
            recipe_ids = fetch_mvp_recipe_ids(cur)
            nutrients = fetch_recipe_nutrients(cur, recipe_ids)
            embs, emb_ids = fetch_recipe_embeddings(cur, recipe_ids)
            features = fetch_recipe_features(cur, recipe_ids)
    finally:
        conn.close()

    # Align embeddings to nutrient rows
    id_to_idx = {rid: i for i, rid in enumerate(emb_ids)}
    aligned_embs = []
    aligned_ids = []
    aligned_names = []
    aligned_nutrients = []
    for _, row in nutrients.iterrows():
        rid = int(row["recipe_id"])
        if rid not in id_to_idx:
            continue
        aligned_ids.append(rid)
        aligned_embs.append(embs[id_to_idx[rid]])
        aligned_names.append(str(row["recipe_name"]))
        aligned_nutrients.append(row.to_dict())

    return {
        "recipe_ids": aligned_ids,
        "recipe_names": aligned_names,
        "embeddings": np.vstack(aligned_embs) if aligned_embs else np.zeros((0, 384)),
        "nutrient_rows": aligned_nutrients,
        "features": features,
    }
