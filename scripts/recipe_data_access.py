"""Local-first access to cap40 recipe schema (+ filtered USDA nutrients).

Toggle with env ``RECIPE_DATA_SOURCE``:
  - ``local`` (default): read parquet store under Data/recipe_local_store/cap40
  - ``db``: query Supabase/Postgres via ``db.connect``

Example:
  from recipe_data_access import get_store
  store = get_store()  # local by default
  lines = store.load_canonical_lines(443)
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE_DIR = ROOT / "Data" / "recipe_local_store" / "cap40"
OPTIONAL_FDC_MAP_PATH = ROOT / "foodon_web" / "data" / "carbonara_fdc_foodon_map.json"

Backend = Literal["local", "db"]


def _as_list(val: Any) -> list[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [val]
    try:
        return list(val)
    except TypeError:
        return [val]


def _as_dict(val: Any) -> dict[str, Any]:
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_str_keyed_lists(val: Any) -> dict[str, list[Any]]:
    raw = _as_dict(val)
    return {str(k): _as_list(v) for k, v in raw.items()}


def data_source() -> Backend:
    raw = (os.environ.get("RECIPE_DATA_SOURCE") or "local").strip().lower()
    if raw in {"db", "database", "postgres", "supabase"}:
        return "db"
    return "local"


def store_dir() -> Path:
    override = os.environ.get("RECIPE_LOCAL_STORE")
    return Path(override) if override else DEFAULT_STORE_DIR


class RecipeDataStore:
    """Unified reader for recipe.* (+ filtered usda.*) used by the agent."""

    def __init__(self, backend: Backend | None = None, root: Path | None = None):
        self.backend: Backend = backend or data_source()
        self.root = Path(root) if root is not None else store_dir()
        self._cache: dict[str, pd.DataFrame] = {}
        self._emb_ids: np.ndarray | None = None
        self._emb_mat: np.ndarray | None = None

    # ------------------------------------------------------------------ loaders
    def _local_path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def _require_local(self) -> None:
        if not (self.root / "manifest.json").is_file():
            raise FileNotFoundError(
                f"Local recipe store missing at {self.root}. "
                "Run: PYTHONPATH=scripts python scripts/download_cap40_recipe_store.py"
            )

    def _load_parquet(self, rel: str) -> pd.DataFrame:
        if rel in self._cache:
            return self._cache[rel]
        self._require_local()
        path = self.root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        self._cache[rel] = df
        return df

    def _db_read(self, sql: str, params: Any = None) -> pd.DataFrame:
        from db import connect

        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = '0'")
            return pd.read_sql(sql, conn, params=params)
        finally:
            conn.close()

    # ------------------------------------------------------------------ tables
    def resolved_recipes(self, recipe_ids: list[int] | None = None) -> pd.DataFrame:
        if self.backend == "local":
            df = self._load_parquet("recipe/resolved_recipes.parquet")
        else:
            if recipe_ids is None:
                df = self._db_read("SELECT * FROM recipe.resolved_recipes")
            else:
                df = self._db_read(
                    "SELECT * FROM recipe.resolved_recipes WHERE recipe_id = ANY(%s)",
                    (list(map(int, recipe_ids)),),
                )
        if recipe_ids is not None and self.backend == "local":
            keep = set(map(int, recipe_ids))
            df = df[df["recipe_id"].astype(int).isin(keep)].copy()
        return df

    def canonical_matches(self, *, canonical_id: int | None = None) -> pd.DataFrame:
        if self.backend == "local":
            df = self._load_parquet("recipe/canonical_matches.parquet")
        else:
            if canonical_id is None:
                df = self._db_read("SELECT * FROM recipe.canonical_matches")
            else:
                df = self._db_read(
                    "SELECT * FROM recipe.canonical_matches WHERE canonical_recipe_id = %s",
                    (int(canonical_id),),
                )
        if canonical_id is not None and self.backend == "local":
            df = df[df["canonical_recipe_id"].astype(int) == int(canonical_id)].copy()
        return df

    def canonical_recipes(self) -> pd.DataFrame:
        if self.backend == "local":
            return self._load_parquet("recipe/canonical_recipes.parquet")
        return self._db_read("SELECT * FROM recipe.canonical_recipes")

    def neighborhood_cache_row(self, canonical_id: int) -> dict[str, Any] | None:
        if self.backend == "local":
            path = self._local_path("recipe", "canonical_neighborhood_cache.parquet")
            if not path.is_file():
                return None
            df = self._load_parquet("recipe/canonical_neighborhood_cache.parquet")
            hit = df[df["canonical_recipe_id"].astype(int) == int(canonical_id)]
            if hit.empty:
                return None
            row = hit.iloc[0].to_dict()
            for key in ("cut_nodes", "best_nodes", "basis_nodes"):
                row[key] = _as_list(row.get(key))
            row["recipe_sets"] = _normalize_str_keyed_lists(row.get("recipe_sets"))
            row["rollup_chains"] = _normalize_str_keyed_lists(row.get("rollup_chains"))
            row["build_params"] = _as_dict(row.get("build_params"))
            shares = row.get("basis_shares")
            if isinstance(shares, str):
                row["basis_shares"] = json.loads(shares)
            elif isinstance(shares, np.ndarray):
                row["basis_shares"] = shares.tolist()
            elif shares is None or (isinstance(shares, float) and pd.isna(shares)):
                row["basis_shares"] = []
            elif not isinstance(shares, list):
                row["basis_shares"] = _as_list(shares)
            return row
        df = self._db_read(
            """
            SELECT canonical_recipe_id, title, n_recipes, starting_recipe_id,
                   cut_nodes, best_nodes, basis_nodes,
                   recipe_sets, rollup_chains, basis_shares,
                   build_params, cache_version, computed_at
            FROM recipe.canonical_neighborhood_cache
            WHERE canonical_recipe_id = %s
            """,
            (int(canonical_id),),
        )
        if df.empty:
            return None
        row = df.iloc[0].to_dict()
        for key in ("cut_nodes", "best_nodes", "basis_nodes"):
            row[key] = _as_list(row.get(key))
        row["recipe_sets"] = _normalize_str_keyed_lists(row.get("recipe_sets"))
        row["rollup_chains"] = _normalize_str_keyed_lists(row.get("rollup_chains"))
        row["build_params"] = _as_dict(row.get("build_params"))
        shares = row.get("basis_shares")
        if isinstance(shares, str):
            row["basis_shares"] = json.loads(shares)
        elif shares is None:
            row["basis_shares"] = []
        elif not isinstance(shares, list):
            row["basis_shares"] = _as_list(shares)
        return row

    def recipe_loss_fields(self, *, canonical_id: int | None = None) -> pd.DataFrame:
        if self.backend == "local":
            path = self._local_path("recipe", "recipe_loss_fields.parquet")
            if not path.is_file():
                return pd.DataFrame()
            df = self._load_parquet("recipe/recipe_loss_fields.parquet")
        else:
            if canonical_id is None:
                df = self._db_read("SELECT * FROM recipe.recipe_loss_fields")
            else:
                df = self._db_read(
                    """
                    SELECT recipe_nlg_id, points_p, points_c, points_f, hard_loss, hard_feasible,
                           protein_frac, carb_frac, fat_frac, grid_n, canonical_recipe_id
                    FROM recipe.recipe_loss_fields
                    WHERE canonical_recipe_id = %s
                    """,
                    (int(canonical_id),),
                )
        if canonical_id is not None and self.backend == "local":
            df = df[df["canonical_recipe_id"].astype(int) == int(canonical_id)].copy()
        return df

    def recipe_nlg(self, recipe_ids: list[int] | None = None) -> pd.DataFrame:
        if self.backend == "local":
            df = self._load_parquet("recipe/recipe_nlg.parquet")
        else:
            if recipe_ids is None:
                df = self._db_read("SELECT * FROM recipe.recipe_nlg")
            else:
                df = self._db_read(
                    "SELECT * FROM recipe.recipe_nlg WHERE id = ANY(%s)",
                    (list(map(int, recipe_ids)),),
                )
        if recipe_ids is not None and self.backend == "local":
            keep = set(map(int, recipe_ids))
            df = df[df["id"].astype(int).isin(keep)].copy()
        return df

    def recipe_nlg_features(self, recipe_ids: list[int] | None = None) -> pd.DataFrame:
        if self.backend == "local":
            df = self._load_parquet("recipe/recipe_nlg_features.parquet")
        else:
            if recipe_ids is None:
                df = self._db_read("SELECT * FROM recipe.recipe_nlg_features")
            else:
                df = self._db_read(
                    "SELECT * FROM recipe.recipe_nlg_features WHERE recipe_id = ANY(%s)",
                    (list(map(int, recipe_ids)),),
                )
        if recipe_ids is not None and self.backend == "local":
            keep = set(map(int, recipe_ids))
            df = df[df["recipe_id"].astype(int).isin(keep)].copy()
        return df

    def recipe_embeddings(self, recipe_ids: list[int]) -> tuple[np.ndarray, list[int]]:
        ids = [int(x) for x in recipe_ids]
        if not ids:
            return np.zeros((0, 384), dtype=np.float32), []
        if self.backend == "local":
            self._require_local()
            if self._emb_ids is None or self._emb_mat is None:
                id_df = pd.read_parquet(self.root / "recipe" / "recipe_nlg_embedding_ids.parquet")
                self._emb_ids = id_df["recipe_id"].astype(int).to_numpy()
                self._emb_mat = np.load(self.root / "recipe" / "recipe_nlg_embedding.f32.npy")
            index = {int(r): i for i, r in enumerate(self._emb_ids.tolist())}
            keep_idx = [index[i] for i in ids if i in index]
            keep_ids = [i for i in ids if i in index]
            if not keep_idx:
                return np.zeros((0, int(self._emb_mat.shape[1])), dtype=np.float32), []
            return self._emb_mat[keep_idx], keep_ids

        from mvp_data import fetch_recipe_embeddings
        from db import connect

        conn = connect()
        try:
            with conn.cursor() as cur:
                return fetch_recipe_embeddings(cur, ids)
        finally:
            conn.close()

    def food_nutrients(self, fdc_ids: list[int]) -> pd.DataFrame:
        ids = sorted({int(x) for x in fdc_ids})
        if not ids:
            return pd.DataFrame(columns=["fdc_id", "nutrient_id", "amount"])
        if self.backend == "local":
            df = self._load_parquet("usda/food_nutrient.parquet")
            return df[df["fdc_id"].astype(int).isin(ids)].copy()
        from mvp_data import fetch_food_nutrients_for_recipe
        from db import connect

        conn = connect()
        try:
            with conn.cursor() as cur:
                return fetch_food_nutrients_for_recipe(cur, ids)
        finally:
            conn.close()

    def food_4macro_foodon(self, fdc_ids: list[int] | None = None) -> pd.DataFrame:
        if self.backend == "local":
            df = self._load_parquet("usda/food_4macro_foodon.parquet")
        else:
            df = self._db_read("SELECT * FROM usda.food_4macro_foodon")
        if fdc_ids is not None:
            keep = set(map(int, fdc_ids))
            df = df[df["fdc_id"].astype(int).isin(keep)].copy()
        return df

    # ----------------------------------------------------------- agent helpers
    def canonical_title(self, canonical_id: int) -> str:
        df = self.canonical_recipes()
        hit = df[df["id"].astype(int) == int(canonical_id)]
        if hit.empty:
            return f"canonical_{canonical_id}"
        return str(hit.iloc[0]["title"])

    def load_canonical_lines(self, canonical_id: int) -> pd.DataFrame:
        """Resolved ingredient lines for a canonical dish (FoodOn attached)."""
        matches = self.canonical_matches(canonical_id=int(canonical_id))
        if matches.empty:
            return pd.DataFrame()
        # Drop multi-canonical recipes (same rule as SQL exclusion).
        all_matches = self.canonical_matches()
        multi = (
            all_matches.groupby("recipe_nlg_id")["canonical_recipe_id"]
            .nunique()
            .loc[lambda s: s > 1]
            .index.astype(str)
        )
        recipe_ids = [
            int(x)
            for x in matches["recipe_nlg_id"].tolist()
            if str(x) not in set(map(str, multi))
        ]
        if not recipe_ids:
            return pd.DataFrame()
        rr = self.resolved_recipes(recipe_ids)
        rr = rr[rr["fdc_id"].notna() & rr["gram_weight"].notna()].copy()
        if rr.empty:
            return pd.DataFrame()

        foodon = self.food_4macro_foodon(rr["fdc_id"].astype(int).unique().tolist())
        foodon = foodon.rename(columns={"foodon_id": "foodon_id", "foodon_label": "foodon_label"})
        # Keep only columns we need from foodon map
        keep_cols = [c for c in ("fdc_id", "foodon_id", "foodon_label") if c in foodon.columns]
        foodon = foodon[keep_cols].drop_duplicates(subset=["fdc_id"])

        merged = rr.merge(foodon, on="fdc_id", how="left")
        merged["canonical_recipe_id"] = int(canonical_id)
        # Optional offline FDC→FoodOn map fallback
        if OPTIONAL_FDC_MAP_PATH.is_file():
            fdc_map = json.loads(OPTIONAL_FDC_MAP_PATH.read_text(encoding="utf-8"))

            def resolve(row: pd.Series) -> tuple[str | None, str | None]:
                if pd.notna(row.get("foodon_id")):
                    return str(row["foodon_id"]), str(row.get("foodon_label") or "")
                hit = fdc_map.get(str(row.get("fdc_description") or ""))
                if hit:
                    return str(hit["id"]), str(hit.get("label") or "")
                return None, None

            mapped = merged.apply(resolve, axis=1, result_type="expand")
            merged["foodon_id"] = mapped[0]
            merged["foodon_label"] = mapped[1]

        merged = merged[merged["foodon_id"].notna()].copy()
        merged["recipe_nlg_id"] = merged["recipe_id"].astype(str)
        merged["fdc_id"] = merged["fdc_id"].astype(int)
        return merged

    def list_canonical_dishes(
        self,
        *,
        min_neighborhood: int = 5,
        limit: int | None = None,
        q: str | None = None,
    ) -> list[dict[str, Any]]:
        matches = self.canonical_matches()
        rr = self.resolved_recipes()
        rr = rr[rr["fdc_id"].notna() & rr["gram_weight"].notna()]
        # multi-canonical exclusion
        multi = (
            matches.groupby("recipe_nlg_id")["canonical_recipe_id"]
            .nunique()
            .loc[lambda s: s > 1]
            .index
        )
        matches = matches[~matches["recipe_nlg_id"].astype(str).isin(set(map(str, multi)))]
        resolved_ids = set(rr["recipe_id"].astype(int))
        matches = matches[matches["recipe_nlg_id"].astype(int).isin(resolved_ids)]
        counts = (
            matches.groupby("canonical_recipe_id")["recipe_nlg_id"]
            .nunique()
            .rename("n_neighborhood")
            .reset_index()
        )
        counts = counts[counts["n_neighborhood"] >= int(min_neighborhood)]
        titles = self.canonical_recipes()[["id", "title"]].rename(columns={"id": "canonical_recipe_id"})
        out_df = counts.merge(titles, on="canonical_recipe_id", how="left")
        if q and q.strip():
            pat = re.escape(q.strip())
            out_df = out_df[out_df["title"].fillna("").str.contains(pat, case=False, regex=True)]
        out_df = out_df.sort_values(
            ["n_neighborhood", "title", "canonical_recipe_id"],
            ascending=[False, True, True],
        )
        if limit is not None:
            out_df = out_df.head(int(limit))
        return [
            {
                "canonical_id": int(r.canonical_recipe_id),
                "title": str(r.title),
                "n_matches": int(r.n_neighborhood),
            }
            for r in out_df.itertuples(index=False)
        ]

    def count_canonical_dishes(self, *, min_neighborhood: int = 5, q: str | None = None) -> int:
        return len(
            self.list_canonical_dishes(min_neighborhood=min_neighborhood, limit=None, q=q)
        )


@lru_cache(maxsize=4)
def get_store(backend: str | None = None, root: str | None = None) -> RecipeDataStore:
    b: Backend | None
    if backend is None:
        b = None
    elif backend.lower() in {"db", "database", "postgres", "supabase"}:
        b = "db"
    else:
        b = "local"
    return RecipeDataStore(backend=b, root=Path(root) if root else None)


def reset_store_cache() -> None:
    get_store.cache_clear()
