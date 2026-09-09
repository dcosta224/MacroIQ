"""Canonical recipe neighborhood optimization (Jaccard + Atwater LP + Wasserstein loss).

Ported from notebooks/agent_optimization_sandbox.ipynb for deterministic benchmarking.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from db import connect
from foodon_hierarchy_cache import DEFAULT_HIERARCHY_CACHE, build_cache
from foodon_index import FoodOnIndex
from mvp_data import build_recipe_macro_inputs, fetch_food_nutrients_for_recipe, fetch_resolved_ingredients
from mvp_nutrient_fit import pfc_calorie_fractions
from recipe_macro_optimizer import compute_macros

ROOT = Path(__file__).resolve().parents[1]
FOODON_INDEX_CACHE = ROOT / "foodon_web" / "cache" / "foodon_index.json"
OPTIONAL_FDC_MAP_PATH: Path | None = ROOT / "foodon_web" / "data" / "carbonara_fdc_foodon_map.json"

NEIGHBORHOOD_CUT_PCT = 0.05
MIN_ROLLUP_NODES = 4
EXACT_SUBSET_SEARCH_LIMIT = 18

# --- Basis-node inclusion cutoff -------------------------------------------------
# A FoodOn node joins the mass-share basis only if enough neighborhood recipes hit
# it. A fixed floor is wrong: thin neighborhoods (e.g. carbonara ~24 recipes) can
# never reach a large absolute count, while big ones (fried rice ~69) over-filter
# rare-but-real roles. Use an adaptive cutoff: a fraction of neighborhood size with
# a small absolute floor and a cap so large neighborhoods do not become too strict.
BASIS_HIT_FRACTION = 0.20
MIN_BASIS_NODE_HITS = 3  # absolute floor for tiny neighborhoods
MAX_BASIS_NODE_HITS = 15  # cap so big neighborhoods don't over-filter


def adaptive_min_basis_hits(n_recipes: int) -> int:
    """Min neighborhood recipes that must hit a FoodOn node for it to enter the basis.

    ``max(MIN_BASIS_NODE_HITS, min(MAX_BASIS_NODE_HITS, ceil(BASIS_HIT_FRACTION * N)))``.
    """
    import math

    if n_recipes <= 0:
        return MIN_BASIS_NODE_HITS
    frac_hits = math.ceil(BASIS_HIT_FRACTION * n_recipes)
    return int(max(MIN_BASIS_NODE_HITS, min(MAX_BASIS_NODE_HITS, frac_hits)))


MIN_INGREDIENT_GRAMS = 1e-3
MAX_GRAM_SCALE = 100.0

# --- Neighborhood expansion (thin-neighborhood enrichment) -----------------------
# When a canonical neighborhood has too few recipes for a stable mass-share
# distribution, pull a similarity-ranked shell of extra recipes (FoodOn + embedding)
# and add their shares at a reduced weight. Core recipes always keep weight 1.0.
EXPANSION_TARGET_N = 40  # try to reach at least this many recipes for share stats
EXPANSION_MAX_SHELL = 60  # never add more than this many shell recipes
EXPANSION_MIN_SIMILARITY = 0.35  # minimum combined similarity for a shell recipe
EXPANSION_SHELL_WEIGHT = 0.5  # base down-weight applied to shell samples

# Bump when cache payload schema or Jaccard params change (invalidates old rows on read).
# v2: adaptive basis-hit cutoff + neighborhood expansion (shell recipes + sample weights).
NEIGHBORHOOD_CACHE_VERSION = 2
NEIGHBORHOOD_CACHE_DDL = (ROOT / "sql" / "42_create_canonical_neighborhood_cache.sql").read_text(
    encoding="utf-8"
)

# Global feasible envelope for point-target generation (Atwater calorie fractions).
GLOBAL_PROTEIN_FRAC = (0.15, 0.25)
GLOBAL_CARB_FRAC = (0.35, 0.55)
GLOBAL_FAT_FRAC = (0.25, 0.45)

MULTI_CANONICAL_EXCLUSION = """
  AND cm.recipe_nlg_id NOT IN (
      SELECT recipe_nlg_id
      FROM recipe.canonical_matches
      GROUP BY recipe_nlg_id
      HAVING COUNT(DISTINCT canonical_recipe_id) > 1
  )
"""

_INDEX: FoodOnIndex | None = None
_HIERARCHY: Any = None


def ensure_neighborhood_cache_table(conn=None) -> None:
    """Create recipe.canonical_neighborhood_cache if missing."""
    own = conn is None
    if own:
        conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(NEIGHBORHOOD_CACHE_DDL)
        conn.commit()
    finally:
        if own:
            conn.close()


def neighborhood_build_params(*, fast: bool) -> dict[str, Any]:
    params = {
        "cut_pct": NEIGHBORHOOD_CUT_PCT,
        "min_rollup_nodes": MIN_ROLLUP_NODES,
        "basis_hit_fraction": BASIS_HIT_FRACTION,
        "min_basis_node_hits": MIN_BASIS_NODE_HITS,
        "max_basis_node_hits": MAX_BASIS_NODE_HITS,
        "exact_subset_search_limit": EXACT_SUBSET_SEARCH_LIMIT,
        "expansion_target_n": EXPANSION_TARGET_N,
        "expansion_max_shell": EXPANSION_MAX_SHELL,
        "expansion_min_similarity": EXPANSION_MIN_SIMILARITY,
        "expansion_shell_weight": EXPANSION_SHELL_WEIGHT,
        "fast": bool(fast),
        "cache_version": NEIGHBORHOOD_CACHE_VERSION,
    }
    try:
        from recipe_opt_agent.foodon_depth import default_max_foodon_aggregation_levels

        params["max_foodon_aggregation_levels_default"] = default_max_foodon_aggregation_levels()
    except Exception:
        pass
    return params


def neighborhood_to_cache_payload(nb: "CanonicalNeighborhood", *, fast: bool = False) -> dict[str, Any]:
    """Serialize Jaccard / rollup / basis artifacts for Postgres JSONB storage."""
    shares: list[dict[str, Any]] = []
    has_weight = (
        nb.basis_share_df is not None
        and not nb.basis_share_df.empty
        and "weight" in nb.basis_share_df.columns
    )
    if nb.basis_share_df is not None and not nb.basis_share_df.empty:
        for row in nb.basis_share_df.itertuples(index=False):
            shares.append(
                {
                    "recipe_nlg_id": str(row.recipe_nlg_id),
                    "basis_node_id": str(row.basis_node_id),
                    "share": float(row.share),
                    "weight": float(getattr(row, "weight", 1.0)) if has_weight else 1.0,
                }
            )
    build_params = neighborhood_build_params(fast=fast)
    # Shell metadata rides inside build_params JSONB (no schema change needed).
    build_params["shell_recipe_ids"] = [str(x) for x in (nb.shell_recipe_ids or [])]
    build_params["expansion_meta"] = nb.expansion_meta or {}
    return {
        "canonical_recipe_id": int(nb.canonical_recipe_id),
        "title": str(nb.title),
        "n_recipes": int(nb.n_recipes),
        "starting_recipe_id": str(nb.starting_recipe_id),
        "cut_nodes": sorted(str(x) for x in nb.cut_nodes),
        "best_nodes": sorted(str(x) for x in nb.best_nodes),
        "basis_nodes": sorted(str(x) for x in nb.basis_nodes),
        "recipe_sets": {str(k): sorted(str(x) for x in v) for k, v in nb.recipe_sets.items()},
        "rollup_chains": {str(k): list(v) for k, v in nb.rollup_chains.items()},
        "basis_shares": shares,
        "build_params": build_params,
        "cache_version": NEIGHBORHOOD_CACHE_VERSION,
    }


def save_neighborhood_cache(nb: "CanonicalNeighborhood", *, fast: bool = False, conn=None) -> None:
    """Upsert Jaccard neighborhood cache for one canonical dish."""
    payload = neighborhood_to_cache_payload(nb, fast=fast)
    own = conn is None
    if own:
        conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '0'")
        ensure_neighborhood_cache_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
INSERT INTO recipe.canonical_neighborhood_cache (
    canonical_recipe_id, title, n_recipes, starting_recipe_id,
    cut_nodes, best_nodes, basis_nodes,
    recipe_sets, rollup_chains, basis_shares,
    build_params, cache_version, computed_at
) VALUES (
    %(canonical_recipe_id)s, %(title)s, %(n_recipes)s, %(starting_recipe_id)s,
    %(cut_nodes)s, %(best_nodes)s, %(basis_nodes)s,
    %(recipe_sets)s::jsonb, %(rollup_chains)s::jsonb, %(basis_shares)s::jsonb,
    %(build_params)s::jsonb, %(cache_version)s, now()
)
ON CONFLICT (canonical_recipe_id) DO UPDATE SET
    title = EXCLUDED.title,
    n_recipes = EXCLUDED.n_recipes,
    starting_recipe_id = EXCLUDED.starting_recipe_id,
    cut_nodes = EXCLUDED.cut_nodes,
    best_nodes = EXCLUDED.best_nodes,
    basis_nodes = EXCLUDED.basis_nodes,
    recipe_sets = EXCLUDED.recipe_sets,
    rollup_chains = EXCLUDED.rollup_chains,
    basis_shares = EXCLUDED.basis_shares,
    build_params = EXCLUDED.build_params,
    cache_version = EXCLUDED.cache_version,
    computed_at = now()
""",
                {
                    "canonical_recipe_id": payload["canonical_recipe_id"],
                    "title": payload["title"],
                    "n_recipes": payload["n_recipes"],
                    "starting_recipe_id": payload["starting_recipe_id"],
                    "cut_nodes": payload["cut_nodes"],
                    "best_nodes": payload["best_nodes"],
                    "basis_nodes": payload["basis_nodes"],
                    "recipe_sets": json.dumps(payload["recipe_sets"]),
                    "rollup_chains": json.dumps(payload["rollup_chains"]),
                    "basis_shares": json.dumps(payload["basis_shares"]),
                    "build_params": json.dumps(payload["build_params"]),
                    "cache_version": payload["cache_version"],
                },
            )
        if not getattr(conn, "autocommit", False):
            conn.commit()
    finally:
        if own:
            conn.close()


def load_neighborhood_cache_row(canonical_recipe_id: int, *, conn=None) -> dict[str, Any] | None:
    """Return cache row dict if present and version matches; else None."""
    from recipe_data_access import data_source, get_store

    if conn is None:
        try:
            data = get_store().neighborhood_cache_row(int(canonical_recipe_id))
            if data is None:
                return None
            if int(data.get("cache_version") or 0) != NEIGHBORHOOD_CACHE_VERSION:
                return None
            for key in ("recipe_sets", "rollup_chains", "basis_shares", "build_params"):
                val = data.get(key)
                if isinstance(val, str):
                    data[key] = json.loads(val)
            return data
        except FileNotFoundError:
            if data_source() == "local":
                raise
        except Exception:
            if data_source() == "local":
                raise

    own = conn is None
    if own:
        conn = connect()
    data: dict[str, Any] | None = None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
SELECT canonical_recipe_id, title, n_recipes, starting_recipe_id,
       cut_nodes, best_nodes, basis_nodes,
       recipe_sets, rollup_chains, basis_shares,
       build_params, cache_version, computed_at
FROM recipe.canonical_neighborhood_cache
WHERE canonical_recipe_id = %s
""",
                (int(canonical_recipe_id),),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            data = dict(zip(cols, row))
    except Exception:
        return None
    finally:
        if own:
            conn.close()

    if data is None:
        return None
    if int(data.get("cache_version") or 0) != NEIGHBORHOOD_CACHE_VERSION:
        return None
    for key in ("recipe_sets", "rollup_chains", "basis_shares", "build_params"):
        val = data.get(key)
        if isinstance(val, str):
            data[key] = json.loads(val)
    return data


def list_cached_neighborhood_ids(*, conn=None) -> set[int]:
    """Return canonical_recipe_id values with a valid neighborhood cache row."""
    from recipe_data_access import data_source, get_store

    if conn is None:
        try:
            store = get_store()
            if store.backend == "local":
                path = store.root / "recipe" / "canonical_neighborhood_cache.parquet"
                if not path.is_file():
                    return set()
                df = store._load_parquet("recipe/canonical_neighborhood_cache.parquet")
                if "cache_version" in df.columns:
                    df = df[df["cache_version"].astype(int) == NEIGHBORHOOD_CACHE_VERSION]
                return {int(x) for x in df["canonical_recipe_id"].tolist()}
        except FileNotFoundError:
            if data_source() == "local":
                raise
        except Exception:
            if data_source() == "local":
                raise

    own = conn is None
    if own:
        conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
SELECT canonical_recipe_id
FROM recipe.canonical_neighborhood_cache
WHERE cache_version = %s
""",
                (NEIGHBORHOOD_CACHE_VERSION,),
            )
            return {int(row[0]) for row in cur.fetchall()}
    except Exception:
        return set()
    finally:
        if own:
            conn.close()


def _get_index() -> FoodOnIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = FoodOnIndex.from_owl(cache_path=FOODON_INDEX_CACHE)
    return _INDEX


def _get_hierarchy() -> Any:
    global _HIERARCHY
    if _HIERARCHY is None:
        _HIERARCHY = build_cache(index_cache=FOODON_INDEX_CACHE, hierarchy_cache=DEFAULT_HIERARCHY_CACHE)
    return _HIERARCHY


def clean_display_label(text: str | None) -> str | None:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return s
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"[-–—]", " ", s)
    return re.sub(r"\s+", " ", s).strip() or None


def foodon_display_label(node_id: str | None) -> str | None:
    if node_id is None:
        return None
    return clean_display_label(_get_index().labels.get(node_id, node_id))


def load_canonical_title(canonical_recipe_id: int) -> str:
    from recipe_data_access import data_source, get_store

    try:
        return get_store().canonical_title(int(canonical_recipe_id))
    except FileNotFoundError:
        if data_source() == "local":
            raise
        sql = "SELECT title FROM recipe.canonical_recipes WHERE id = %s"
        conn = connect()
        try:
            row = pd.read_sql(sql, conn, params=(canonical_recipe_id,))
        finally:
            conn.close()
        if row.empty:
            return f"canonical #{canonical_recipe_id}"
        return str(row.iloc[0]["title"])


def load_canonical_lines(canonical_recipe_id: int) -> pd.DataFrame:
    from recipe_data_access import data_source, get_store

    try:
        return get_store().load_canonical_lines(int(canonical_recipe_id))
    except FileNotFoundError:
        if data_source() == "local":
            raise
    # DB fallback when store local missing but source=db, or unexpected store errors on db.
    sql = """
WITH multi AS (
    SELECT recipe_nlg_id
    FROM recipe.canonical_matches
    GROUP BY recipe_nlg_id
    HAVING COUNT(DISTINCT canonical_recipe_id) > 1
)
SELECT cm.canonical_recipe_id,
       cm.recipe_nlg_id,
       rr.ingredient_idx,
       rr.ingredient,
       rr.fdc_description,
       rr.fdc_id,
       rr.gram_weight,
       ff.foodon_id,
       ff.foodon_label
FROM recipe.canonical_matches cm
JOIN recipe.resolved_recipes rr ON rr.recipe_id = cm.recipe_nlg_id
LEFT JOIN usda.food_4macro_foodon ff ON ff.fdc_id = rr.fdc_id
LEFT JOIN multi m ON m.recipe_nlg_id = cm.recipe_nlg_id
WHERE rr.fdc_id IS NOT NULL
  AND rr.gram_weight IS NOT NULL
  AND m.recipe_nlg_id IS NULL
  AND cm.canonical_recipe_id = %s
"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '0'")
        df = pd.read_sql(sql, conn, params=(canonical_recipe_id,))
    finally:
        conn.close()

    fdc_map: dict[str, dict] = {}
    if OPTIONAL_FDC_MAP_PATH is not None and OPTIONAL_FDC_MAP_PATH.is_file():
        fdc_map = json.loads(OPTIONAL_FDC_MAP_PATH.read_text(encoding="utf-8"))

    def resolve_foodon(row: pd.Series) -> tuple[str | None, str | None]:
        if pd.notna(row.get("foodon_id")):
            return str(row["foodon_id"]), str(row.get("foodon_label") or "")
        hit = fdc_map.get(str(row["fdc_description"]))
        if hit:
            return str(hit["id"]), str(hit.get("label") or "")
        return None, None

    mapped = df.apply(resolve_foodon, axis=1, result_type="expand")
    df["foodon_id"] = mapped[0]
    df["foodon_label"] = mapped[1]
    df = df[df["foodon_id"].notna()].copy()
    df["recipe_nlg_id"] = df["recipe_nlg_id"].astype(str)
    df["fdc_id"] = df["fdc_id"].astype(int)
    return df


def fetch_top_canonical_dishes(*, limit: int = 30, min_neighborhood: int = 10) -> pd.DataFrame:
    from recipe_data_access import data_source, get_store

    try:
        rows = get_store().list_canonical_dishes(
            min_neighborhood=min_neighborhood, limit=limit
        )
        return pd.DataFrame(
            [
                {
                    "canonical_recipe_id": r["canonical_id"],
                    "title": r["title"],
                    "n_neighborhood": r["n_matches"],
                }
                for r in rows
            ]
        )
    except FileNotFoundError:
        if data_source() == "local":
            raise
    sql = f"""
SELECT cr.id AS canonical_recipe_id,
       cr.title,
       COUNT(DISTINCT cm.recipe_nlg_id) AS n_neighborhood
FROM recipe.canonical_recipes cr
JOIN recipe.canonical_matches cm ON cm.canonical_recipe_id = cr.id
JOIN recipe.resolved_recipes rr ON rr.recipe_id = cm.recipe_nlg_id
WHERE rr.fdc_id IS NOT NULL
  AND rr.gram_weight IS NOT NULL
  {MULTI_CANONICAL_EXCLUSION}
GROUP BY cr.id, cr.title
HAVING COUNT(DISTINCT cm.recipe_nlg_id) >= %s
ORDER BY n_neighborhood DESC, cr.id ASC
LIMIT %s
"""
    conn = connect()
    try:
        return pd.read_sql(sql, conn, params=(min_neighborhood, limit))
    finally:
        conn.close()


def deepest_antichain(nodes: set[str]) -> set[str]:
    hierarchy = _get_hierarchy()
    out: set[str] = set()
    for node_id in nodes:
        desc = set(hierarchy.descendants.get(node_id, []))
        if not (desc & nodes - {node_id}):
            out.add(node_id)
    return out


def is_antichain(nodes: set[str]) -> bool:
    index = _get_index()
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            if a in index.ancestry_path(b):
                return False
    return True


def atwater_kcal_from_grams(protein_g: float, fat_g: float, carbs_g: float) -> float:
    return 4.0 * protein_g + 9.0 * fat_g + 4.0 * carbs_g


def atwater_kcal(x: np.ndarray, M: np.ndarray) -> float:
    protein_g, fat_g, carbs_g, _ = compute_macros(np.asarray(x, dtype=float), M)
    return atwater_kcal_from_grams(float(protein_g), float(fat_g), float(carbs_g))


def macro_calorie_fractions_from_grams(protein_g: float, fat_g: float, carbs_g: float) -> tuple[float, float, float]:
    pk, ck, fk = protein_g * 4.0, carbs_g * 4.0, fat_g * 9.0
    total = pk + ck + fk
    if total <= 0:
        return 0.0, 0.0, 0.0
    return pk / total, ck / total, fk / total


def macro_point_distance(
    protein_g: float,
    fat_g: float,
    carbs_g: float,
    *,
    target_protein_frac: float,
    target_carb_frac: float,
    target_fat_frac: float,
) -> float:
    kcal = atwater_kcal_from_grams(protein_g, fat_g, carbs_g)
    if kcal <= 0:
        return float("inf")
    fat_f, carb_f, protein_f = pfc_calorie_fractions(fat_g, carbs_g, protein_g, kcal)
    return float(
        abs(protein_f - target_protein_frac)
        + abs(carb_f - target_carb_frac)
        + abs(fat_f - target_fat_frac)
    )


def atwater_scaled_baseline(x0: np.ndarray, M: np.ndarray, kcal_target: float) -> np.ndarray:
    x0 = np.asarray(x0, dtype=float)
    k0 = atwater_kcal(x0, M)
    if k0 <= 1e-9:
        return x0.copy()
    return x0 * (kcal_target / k0)


def atwater_fraction_constraints(
    x_var,
    M: np.ndarray,
    *,
    kcal_target: float,
    protein_frac_min: float,
    protein_frac_max: float,
    carb_frac_min: float,
    carb_frac_max: float,
    fat_frac_min: float,
    fat_frac_max: float,
) -> list:
    import cvxpy as cp

    M = np.asarray(M, dtype=float)
    p_kcal = 4.0 * (M[0] @ x_var)
    f_kcal = 9.0 * (M[1] @ x_var)
    c_kcal = 4.0 * (M[2] @ x_var)
    kcal = p_kcal + f_kcal + c_kcal

    cons: list = [kcal == kcal_target]
    for macro_kcal, frac_min, frac_max in (
        (p_kcal, protein_frac_min, protein_frac_max),
        (c_kcal, carb_frac_min, carb_frac_max),
        (f_kcal, fat_frac_min, fat_frac_max),
    ):
        cons.append((1.0 - frac_min) * macro_kcal - frac_min * (kcal - macro_kcal) >= 0)
        cons.append((1.0 - frac_max) * macro_kcal - frac_max * (kcal - macro_kcal) <= 0)
    return cons


def check_atwater_region_feasible(
    M: np.ndarray,
    *,
    kcal_target: float,
    protein_frac_min: float,
    protein_frac_max: float,
    carb_frac_min: float,
    carb_frac_max: float,
    fat_frac_min: float,
    fat_frac_max: float,
) -> tuple[bool, str]:
    import cvxpy as cp

    n = M.shape[1]
    if n == 0:
        return False, "Recipe has no ingredients."

    x = cp.Variable(n, nonneg=True)
    cons = atwater_fraction_constraints(
        x,
        M,
        kcal_target=kcal_target,
        protein_frac_min=protein_frac_min,
        protein_frac_max=protein_frac_max,
        carb_frac_min=carb_frac_min,
        carb_frac_max=carb_frac_max,
        fat_frac_min=fat_frac_min,
        fat_frac_max=fat_frac_max,
    )
    prob = cp.Problem(cp.Minimize(0), cons)
    for solver in (cp.OSQP, cp.SCS):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
                return True, "feasible"
        except cp.SolverError:
            continue
    return False, "infeasible"


def ingredient_pfc_fractions(M: np.ndarray) -> np.ndarray:
    """Per-ingredient Atwater calorie fractions (protein, carb, fat). Shape (n, 3)."""
    M = np.asarray(M, dtype=float)
    rows: list[tuple[float, float, float]] = []
    for i in range(M.shape[1]):
        rows.append(macro_calorie_fractions_from_grams(float(M[0, i]), float(M[1, i]), float(M[2, i])))
    return np.asarray(rows, dtype=float)


def pfc_fractions_from_portions(x: np.ndarray, M: np.ndarray) -> tuple[float, float, float]:
    macros = compute_macros(np.asarray(x, dtype=float), M)
    return macro_calorie_fractions_from_grams(float(macros[0]), float(macros[1]), float(macros[2]))


def check_optimizer_point_feasible(
    x0: np.ndarray,
    M: np.ndarray,
    *,
    kcal_target: float,
    target_protein_frac: float,
    target_carb_frac: float,
    target_fat_frac: float,
    total_mass: float | None = None,
    min_grams: float = MIN_INGREDIENT_GRAMS,
    max_scale: float = MAX_GRAM_SCALE,
) -> tuple[bool, str]:
    """Whether a point PFC target is reachable under optimizer constraints."""
    import cvxpy as cp

    x0 = np.asarray(x0, dtype=float)
    M = np.asarray(M, dtype=float)
    n = len(x0)
    if n == 0:
        return False, "Recipe has no ingredients."

    mass = float(total_mass if total_mass is not None else x0.sum())
    x = cp.Variable(n, nonneg=True)
    cons: list = [
        cp.sum(x) == mass,
        *atwater_fraction_constraints(
            x,
            M,
            kcal_target=kcal_target,
            protein_frac_min=target_protein_frac,
            protein_frac_max=target_protein_frac,
            carb_frac_min=target_carb_frac,
            carb_frac_max=target_carb_frac,
            fat_frac_min=target_fat_frac,
            fat_frac_max=target_fat_frac,
        ),
    ]
    for i in range(n):
        cons.append(x[i] >= min_grams)
        cons.append(x[i] <= max_scale * x0[i])

    prob = cp.Problem(cp.Minimize(0), cons)
    for solver in (cp.OSQP, cp.SCS):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
                return True, "feasible"
        except cp.SolverError:
            continue
    return False, "infeasible"


def generate_hull_perturbed_target(
    x0: np.ndarray,
    M: np.ndarray,
    *,
    seed: int,
    alpha_min: float = 0.18,
    alpha_max: float = 0.38,
    min_vertex_distance: float = 0.025,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Perturb baseline PFC moderately toward a hull vertex (ingredient macro profile).

    The conical hull of ingredient PFC vectors, projected to calorie-fraction space
    with fixed total mass, is the convex hull of per-ingredient macro ratios. Moving
    baseline toward a distant vertex by alpha in (0, 1) stays inside that hull.
    """
    rng = np.random.default_rng(seed)
    baseline = np.array(pfc_fractions_from_portions(x0, M), dtype=float)
    verts = ingredient_pfc_fractions(M)

    kcal = atwater_kcal(x0, M)
    x_ref = atwater_scaled_baseline(x0, M, kcal)
    total_mass = float(x_ref.sum())

    candidates: list[tuple[int, float]] = []
    for i, vert in enumerate(verts):
        dist = float(np.linalg.norm(vert - baseline, ord=1))
        if dist >= min_vertex_distance:
            candidates.append((i, dist))
    candidates.sort(key=lambda t: (-t[1], t[0]))

    if not candidates:
        # All ingredients share nearly the same macro profile; nudge along simplex.
        axis = int(rng.integers(0, 3))
        sign = float(rng.choice([-1.0, 1.0]))
        delta = np.zeros(3, dtype=float)
        delta[axis] = sign * rng.uniform(0.04, 0.08)
        delta -= delta.mean()
        target = baseline + delta
        target = np.clip(target, 1e-4, 1.0 - 1e-4)
        target = target / target.sum()
        meta = {"vertex_idx": None, "alpha": None, "hull_distance_l1": 0.0, "fallback": "simplex_nudge"}
    else:
        top_k = min(3, len(candidates))
        pick = candidates[int(rng.integers(0, top_k))][0]
        vert = verts[pick]
        alpha = float(rng.uniform(alpha_min, alpha_max))
        target = baseline + alpha * (vert - baseline)
        meta = {
            "vertex_idx": int(pick),
            "alpha": alpha,
            "hull_distance_l1": float(np.linalg.norm(vert - baseline, ord=1)),
            "fallback": None,
        }

    target_tuple = (float(target[0]), float(target[1]), float(target[2]))

    def _feasible_at(alpha: float, vert_idx: int | None) -> bool:
        if vert_idx is None:
            t = target_tuple
        else:
            t_arr = baseline + alpha * (verts[vert_idx] - baseline)
            t = (float(t_arr[0]), float(t_arr[1]), float(t_arr[2]))
        ok, _ = check_optimizer_point_feasible(
            x0,
            M,
            kcal_target=kcal,
            target_protein_frac=t[0],
            target_carb_frac=t[1],
            target_fat_frac=t[2],
            total_mass=total_mass,
        )
        return ok

    vert_idx = meta.get("vertex_idx")
    alpha = meta.get("alpha")
    if vert_idx is not None and alpha is not None:
        if not _feasible_at(alpha, vert_idx):
            lo, hi = 0.0, alpha
            best_alpha = None
            for _ in range(24):
                mid = (lo + hi) / 2.0
                if _feasible_at(mid, vert_idx):
                    best_alpha = mid
                    lo = mid
                else:
                    hi = mid
            if best_alpha is not None and best_alpha >= 0.03:
                alpha = best_alpha
                target = baseline + alpha * (verts[vert_idx] - baseline)
                target_tuple = (float(target[0]), float(target[1]), float(target[2]))
                meta["alpha"] = alpha
            else:
                for alt_idx, _ in candidates:
                    if alt_idx == vert_idx:
                        continue
                    trial_alpha = float(rng.uniform(alpha_min, alpha_max))
                    lo, hi = 0.0, trial_alpha
                    best_alpha = None
                    for _ in range(24):
                        mid = (lo + hi) / 2.0
                        if _feasible_at(mid, alt_idx):
                            best_alpha = mid
                            lo = mid
                        else:
                            hi = mid
                    if best_alpha is not None and best_alpha >= 0.03:
                        vert_idx = alt_idx
                        alpha = best_alpha
                        target = baseline + alpha * (verts[vert_idx] - baseline)
                        target_tuple = (float(target[0]), float(target[1]), float(target[2]))
                        meta["vertex_idx"] = vert_idx
                        meta["alpha"] = alpha
                        break

    ok, msg = check_optimizer_point_feasible(
        x0,
        M,
        kcal_target=kcal,
        target_protein_frac=target_tuple[0],
        target_carb_frac=target_tuple[1],
        target_fat_frac=target_tuple[2],
        total_mass=total_mass,
    )

    meta["optimizer_feasible"] = ok
    meta["feasibility_msg"] = msg
    meta["baseline_protein_frac"] = float(baseline[0])
    meta["baseline_carb_frac"] = float(baseline[1])
    meta["baseline_fat_frac"] = float(baseline[2])
    return target_tuple, meta


TARGET_MODES = ("close", "far", "outside_hull")
DEFAULT_SLACK_WEIGHT = 1.0
DEFAULT_TARGET_BOX_HALF_WIDTH = 0.02


def pfc_box_around_point(
    protein_frac: float,
    carb_frac: float,
    fat_frac: float,
    *,
    half_width: float = DEFAULT_TARGET_BOX_HALF_WIDTH,
) -> dict[str, float]:
    """Axis-aligned PFC box around a simplex point."""
    width = max(0.0, float(half_width))
    return {
        "protein_min": max(0.0, float(protein_frac) - width),
        "protein_max": min(1.0, float(protein_frac) + width),
        "carb_min": max(0.0, float(carb_frac) - width),
        "carb_max": min(1.0, float(carb_frac) + width),
        "fat_min": max(0.0, float(fat_frac) - width),
        "fat_max": min(1.0, float(fat_frac) + width),
    }


def pfc_box_distance(
    protein_frac: float,
    carb_frac: float,
    fat_frac: float,
    *,
    protein_min: float,
    protein_max: float,
    carb_min: float,
    carb_max: float,
    fat_min: float,
    fat_max: float,
) -> float:
    """L1 calorie-fraction distance outside a PFC box; zero inside."""
    return float(
        max(float(protein_min) - float(protein_frac), 0.0)
        + max(float(protein_frac) - float(protein_max), 0.0)
        + max(float(carb_min) - float(carb_frac), 0.0)
        + max(float(carb_frac) - float(carb_max), 0.0)
        + max(float(fat_min) - float(fat_frac), 0.0)
        + max(float(fat_frac) - float(fat_max), 0.0)
    )


def _normalize_pfc_simplex(target: np.ndarray) -> tuple[float, float, float]:
    target = np.clip(np.asarray(target, dtype=float), 1e-4, 1.0 - 1e-4)
    target = target / target.sum()
    return float(target[0]), float(target[1]), float(target[2])


def neighborhood_median_pfc(lines_df: pd.DataFrame) -> tuple[float, float, float]:
    """Median P/C/F calorie fractions across neighborhood resolved recipes."""
    if lines_df.empty:
        raise ValueError("empty neighborhood lines")
    all_fdc = lines_df["fdc_id"].astype(int).unique().tolist()
    nutrients = fetch_food_nutrients_for_recipe(None, all_fdc)

    fracs: list[tuple[float, float, float]] = []
    for _, grp in lines_df.groupby("recipe_nlg_id"):
        ing = grp[["ingredient_idx", "ingredient", "fdc_id", "gram_weight"]].copy()
        fdc_set = set(ing["fdc_id"].astype(int).tolist())
        sub_nutrients = nutrients[nutrients["fdc_id"].isin(fdc_set)]
        x0, m = build_recipe_macro_inputs(ing, sub_nutrients)
        fracs.append(pfc_fractions_from_portions(x0, m))
    med = np.median(np.asarray(fracs, dtype=float), axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def _max_feasible_alpha(
    x0: np.ndarray,
    M: np.ndarray,
    *,
    baseline: np.ndarray,
    vert: np.ndarray,
    kcal: float,
    total_mass: float,
    alpha_lo: float,
    alpha_hi: float,
) -> float:
    """Largest alpha in [alpha_lo, alpha_hi] with optimizer-feasible PFC target on segment to vert."""

    def feasible(alpha: float) -> bool:
        t = baseline + alpha * (vert - baseline)
        ok, _ = check_optimizer_point_feasible(
            x0,
            M,
            kcal_target=kcal,
            target_protein_frac=float(t[0]),
            target_carb_frac=float(t[1]),
            target_fat_frac=float(t[2]),
            total_mass=total_mass,
        )
        return ok

    if not feasible(alpha_lo):
        return alpha_lo
    if feasible(alpha_hi):
        return alpha_hi
    lo, hi = alpha_lo, alpha_hi
    best = alpha_lo
    for _ in range(24):
        mid = (lo + hi) / 2.0
        if feasible(mid):
            best = mid
            lo = mid
        else:
            hi = mid
    return best


def generate_close_target(
    x0: np.ndarray,
    M: np.ndarray,
    lines_df: pd.DataFrame,
    *,
    seed: int,
    max_l1_from_median: float = 0.06,
    median_pfc: tuple[float, float, float] | None = None,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Target within a few percentage points of neighborhood median P/C/F."""
    rng = np.random.default_rng(seed)
    median = np.array(
        median_pfc if median_pfc is not None else neighborhood_median_pfc(lines_df),
        dtype=float,
    )
    baseline = np.array(pfc_fractions_from_portions(x0, M), dtype=float)
    scale = float(rng.uniform(0.01, max_l1_from_median / 2.0))
    delta = rng.normal(size=3)
    delta -= delta.mean()
    delta = delta / (np.abs(delta).sum() + 1e-9) * scale
    target = _normalize_pfc_simplex(median + delta)
    meta = {
        "target_mode": "close",
        "neighborhood_median_protein_frac": float(median[0]),
        "neighborhood_median_carb_frac": float(median[1]),
        "neighborhood_median_fat_frac": float(median[2]),
        "l1_from_median": float(
            abs(target[0] - median[0]) + abs(target[1] - median[1]) + abs(target[2] - median[2])
        ),
        "baseline_protein_frac": float(baseline[0]),
        "baseline_carb_frac": float(baseline[1]),
        "baseline_fat_frac": float(baseline[2]),
        "optimizer_feasible": True,
        "feasibility_msg": "close_to_median",
    }
    return target, meta


def generate_far_target(
    x0: np.ndarray,
    M: np.ndarray,
    *,
    seed: int,
    alpha_lo: float = 0.55,
    alpha_hi: float = 0.98,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Target near hull edge: far along ray to most equidistant distant ingredient PFC vertex."""
    rng = np.random.default_rng(seed)
    baseline = np.array(pfc_fractions_from_portions(x0, M), dtype=float)
    verts = ingredient_pfc_fractions(M)
    kcal = atwater_kcal(x0, M)
    total_mass = float(atwater_scaled_baseline(x0, M, kcal).sum())

    ranked = sorted(
        ((i, float(np.linalg.norm(verts[i] - baseline, ord=1))) for i in range(len(verts))),
        key=lambda t: (-t[1], t[0]),
    )
    top = [i for i, d in ranked[: min(5, len(ranked))] if d >= 0.02]
    if not top:
        target_tuple, meta = generate_hull_perturbed_target(
            x0, M, seed=seed, alpha_min=0.55, alpha_max=0.95
        )
        meta["target_mode"] = "far"
        meta["fallback"] = "hull_perturbed"
        return target_tuple, meta

    # Among distant vertices, pick most equidistant shift across P/C/F components.
    diffs = [np.abs(verts[i] - baseline) for i in top]
    stds = [float(np.std(d)) for d in diffs]
    pick_pool = [top[i] for i, s in sorted(enumerate(stds), key=lambda t: (t[1], top[t[0]]))[:3]]
    vert_idx = int(pick_pool[int(rng.integers(0, len(pick_pool)))])
    vert = verts[vert_idx]
    alpha = _max_feasible_alpha(
        x0,
        M,
        baseline=baseline,
        vert=vert,
        kcal=kcal,
        total_mass=total_mass,
        alpha_lo=alpha_lo,
        alpha_hi=alpha_hi,
    )
    target = _normalize_pfc_simplex(baseline + alpha * (vert - baseline))
    ok, msg = check_optimizer_point_feasible(
        x0,
        M,
        kcal_target=kcal,
        target_protein_frac=target[0],
        target_carb_frac=target[1],
        target_fat_frac=target[2],
        total_mass=total_mass,
    )
    meta = {
        "target_mode": "far",
        "vertex_idx": vert_idx,
        "alpha": alpha,
        "hull_distance_l1": float(np.linalg.norm(vert - baseline, ord=1)),
        "vertex_equidistant_std": float(np.std(np.abs(vert - baseline))),
        "baseline_protein_frac": float(baseline[0]),
        "baseline_carb_frac": float(baseline[1]),
        "baseline_fat_frac": float(baseline[2]),
        "optimizer_feasible": ok,
        "feasibility_msg": msg,
    }
    return target, meta


def generate_outside_hull_target(
    x0: np.ndarray,
    M: np.ndarray,
    *,
    seed: int,
    outside_eps: float = 0.05,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Target ~5% beyond the conical hull along the farthest ingredient PFC ray from baseline."""
    baseline = np.array(pfc_fractions_from_portions(x0, M), dtype=float)
    verts = ingredient_pfc_fractions(M)
    kcal = atwater_kcal(x0, M)
    total_mass = float(atwater_scaled_baseline(x0, M, kcal).sum())

    vert_idx = max(
        range(len(verts)),
        key=lambda i: (float(np.linalg.norm(verts[i] - baseline, ord=1)), -i),
    )
    vert = verts[vert_idx]
    alpha = 1.0 + float(outside_eps)
    raw = baseline + alpha * (vert - baseline)
    target = _normalize_pfc_simplex(raw)
    ok, msg = check_optimizer_point_feasible(
        x0,
        M,
        kcal_target=kcal,
        target_protein_frac=target[0],
        target_carb_frac=target[1],
        target_fat_frac=target[2],
        total_mass=total_mass,
    )
    meta = {
        "target_mode": "outside_hull",
        "vertex_idx": int(vert_idx),
        "extrapolation_alpha": alpha,
        "outside_eps": float(outside_eps),
        "hull_distance_l1": float(np.linalg.norm(vert - baseline, ord=1)),
        "baseline_protein_frac": float(baseline[0]),
        "baseline_carb_frac": float(baseline[1]),
        "baseline_fat_frac": float(baseline[2]),
        "optimizer_feasible": ok,
        "feasibility_msg": msg,
        "hard_macro_feasible": ok,
    }
    return target, meta


def generate_target_for_mode(
    mode: str,
    x0: np.ndarray,
    M: np.ndarray,
    lines_df: pd.DataFrame,
    *,
    seed: int,
    median_pfc: tuple[float, float, float] | None = None,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    if mode == "close":
        return generate_close_target(x0, M, lines_df, seed=seed, median_pfc=median_pfc)
    if mode == "far":
        return generate_far_target(x0, M, seed=seed)
    if mode == "outside_hull":
        return generate_outside_hull_target(x0, M, seed=seed)
    raise ValueError(f"unknown target_mode: {mode!r}")


def empirical_cdf_l1_loss(z: float, samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return 0.0
    return float(np.mean(np.abs(samples - z)))


def ingredient_wasserstein_losses(
    x: np.ndarray,
    *,
    basis_samples: dict[str, np.ndarray],
    total_mass: float,
    ingredient_basis: list[str | None],
) -> list[float]:
    """Per-ingredient 1-Wasserstein distance to neighborhood gram-share samples."""
    x = np.asarray(x, dtype=float)
    if total_mass <= 0:
        return []
    losses: list[float] = []
    for grams, node_id in zip(x, ingredient_basis, strict=False):
        if node_id is None:
            continue
        samples = basis_samples.get(node_id, np.array([], dtype=float))
        if samples.size == 0:
            continue
        z = float(grams) / total_mass
        losses.append(empirical_cdf_l1_loss(z, samples))
    return losses


def simple_empirical_obj_value(
    x: np.ndarray,
    *,
    basis_index: list[str],
    basis_samples: dict[str, np.ndarray],
    total_mass: float,
    ingredient_basis: list[str | None],
) -> float:
    del basis_index  # kept for call-site compatibility
    losses = ingredient_wasserstein_losses(
        x,
        basis_samples=basis_samples,
        total_mass=total_mass,
        ingredient_basis=ingredient_basis,
    )
    if not losses:
        return 0.0
    return float(np.mean(losses))


def ingredient_ratio_loss_lp_terms(
    x_var,
    *,
    inv_mass: float,
    ingredient_basis: list[str | None],
    basis_samples: dict[str, np.ndarray],
    cons: list,
) -> list:
    """Convex per-ingredient Wasserstein terms for LP ratio objective (mean over ingredients)."""
    import cvxpy as cp

    per_ingredient: list = []
    for i, node_id in enumerate(ingredient_basis):
        if node_id is None:
            continue
        samples = basis_samples.get(node_id, np.array([], dtype=float))
        if samples.size == 0:
            continue
        z_i = inv_mass * x_var[i]
        u_terms = []
        for sample in samples:
            u = cp.Variable(nonneg=True)
            cons.append(u >= sample - z_i)
            cons.append(u >= z_i - sample)
            u_terms.append(u)
        per_ingredient.append(cp.sum(u_terms) / samples.size)
    return per_ingredient


def optimize_simple_empirical_obj(
    x0: np.ndarray,
    M: np.ndarray,
    *,
    basis_index: list[str],
    basis_samples: dict[str, np.ndarray],
    ingredient_basis: list[str | None],
    kcal_target: float,
    protein_frac_min: float,
    protein_frac_max: float,
    carb_frac_min: float,
    carb_frac_max: float,
    fat_frac_min: float,
    fat_frac_max: float,
    total_mass: float,
    min_grams: float = MIN_INGREDIENT_GRAMS,
    max_scale: float = MAX_GRAM_SCALE,
) -> dict[str, object]:
    import cvxpy as cp

    x0 = np.asarray(x0, dtype=float)
    M = np.asarray(M, dtype=float)
    n = len(x0)
    inv_mass = 1.0 / total_mass

    x = cp.Variable(n, nonneg=True)
    cons: list = [
        cp.sum(x) == total_mass,
        *atwater_fraction_constraints(
            x,
            M,
            kcal_target=kcal_target,
            protein_frac_min=protein_frac_min,
            protein_frac_max=protein_frac_max,
            carb_frac_min=carb_frac_min,
            carb_frac_max=carb_frac_max,
            fat_frac_min=fat_frac_min,
            fat_frac_max=fat_frac_max,
        ),
    ]
    for i in range(n):
        cons.append(x[i] >= min_grams)
        cons.append(x[i] <= max_scale * x0[i])

    per_ingredient = ingredient_ratio_loss_lp_terms(
        x,
        inv_mass=inv_mass,
        ingredient_basis=ingredient_basis,
        basis_samples=basis_samples,
        cons=cons,
    )
    ratio_obj = cp.sum(per_ingredient) / len(per_ingredient) if per_ingredient else 0

    objective = cp.Minimize(ratio_obj)
    prob = cp.Problem(objective, cons)

    status = "solver_failed"
    for solver in (cp.OSQP, cp.SCS):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                status = prob.status
                break
        except cp.SolverError:
            continue

    if x.value is None:
        return {
            "status": status,
            "x_opt": x0.copy(),
            "objective": simple_empirical_obj_value(
                x0,
                basis_index=basis_index,
                basis_samples=basis_samples,
                total_mass=total_mass,
                ingredient_basis=ingredient_basis,
            ),
            "feasible": False,
        }

    x_opt = np.asarray(x.value, dtype=float).ravel()
    obj_val = simple_empirical_obj_value(
        x_opt,
        basis_index=basis_index,
        basis_samples=basis_samples,
        total_mass=total_mass,
        ingredient_basis=ingredient_basis,
    )
    return {
        "status": status,
        "x_opt": x_opt,
        "objective": obj_val,
        "feasible": True,
        "macros_before": compute_macros(x0, M),
        "macros_after": compute_macros(x_opt, M),
    }


def optimize_simple_empirical_obj_box_slack(
    x0: np.ndarray,
    M: np.ndarray,
    *,
    basis_index: list[str],
    basis_samples: dict[str, np.ndarray],
    ingredient_basis: list[str | None],
    kcal_target: float,
    protein_frac_min: float,
    protein_frac_max: float,
    carb_frac_min: float,
    carb_frac_max: float,
    fat_frac_min: float,
    fat_frac_max: float,
    total_mass: float,
    slack_weight: float = DEFAULT_SLACK_WEIGHT,
    min_grams: float = MIN_INGREDIENT_GRAMS,
    max_scale: float = MAX_GRAM_SCALE,
) -> dict[str, object]:
    """Minimize ratio loss plus normalized L1 distance outside a PFC box.

    Nutrition bounds are soft. Six nonnegative fraction-valued slacks encode
    lower/upper violations. The objective is dimensionless:

        ratio_loss + slack_weight * sum(box_violation_fractions)

    This avoids the old kcal-valued slack penalty, which was hundreds of times
    larger than the ratio objective and therefore behaved almost like a hard
    nutrition equality.
    """
    import cvxpy as cp

    x0 = np.asarray(x0, dtype=float)
    M = np.asarray(M, dtype=float)
    n = len(x0)
    inv_mass = 1.0 / total_mass

    x = cp.Variable(n, nonneg=True)
    p_kcal = 4.0 * (M[0] @ x)
    f_kcal = 9.0 * (M[1] @ x)
    c_kcal = 4.0 * (M[2] @ x)
    kcal = p_kcal + f_kcal + c_kcal

    p_lo, p_hi = cp.Variable(nonneg=True), cp.Variable(nonneg=True)
    c_lo, c_hi = cp.Variable(nonneg=True), cp.Variable(nonneg=True)
    f_lo, f_hi = cp.Variable(nonneg=True), cp.Variable(nonneg=True)

    cons: list = [
        cp.sum(x) == total_mass,
        kcal == kcal_target,
        p_kcal >= (protein_frac_min - p_lo) * kcal_target,
        p_kcal <= (protein_frac_max + p_hi) * kcal_target,
        c_kcal >= (carb_frac_min - c_lo) * kcal_target,
        c_kcal <= (carb_frac_max + c_hi) * kcal_target,
        f_kcal >= (fat_frac_min - f_lo) * kcal_target,
        f_kcal <= (fat_frac_max + f_hi) * kcal_target,
    ]
    for i in range(n):
        cons.append(x[i] >= min_grams)
        cons.append(x[i] <= max_scale * x0[i])

    per_ingredient = ingredient_ratio_loss_lp_terms(
        x,
        inv_mass=inv_mass,
        ingredient_basis=ingredient_basis,
        basis_samples=basis_samples,
        cons=cons,
    )
    macro_slack_expr = p_lo + p_hi + c_lo + c_hi + f_lo + f_hi
    ratio_obj = cp.sum(per_ingredient) / len(per_ingredient) if per_ingredient else 0
    objective = cp.Minimize(ratio_obj + slack_weight * macro_slack_expr)
    prob = cp.Problem(objective, cons)

    status = "solver_failed"
    solvers = [
        solver
        for name in ("HIGHS", "CLARABEL", "OSQP", "SCS")
        if (solver := getattr(cp, name, None)) is not None
    ]
    for solver in solvers:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                status = prob.status
                break
        except cp.SolverError:
            continue

    ratio_only = simple_empirical_obj_value(
        x0,
        basis_index=basis_index,
        basis_samples=basis_samples,
        total_mass=total_mass,
        ingredient_basis=ingredient_basis,
    )

    if x.value is None:
        return {
            "status": status,
            "x_opt": x0.copy(),
            "objective": ratio_only,
            "ratio_objective": ratio_only,
            "macro_slack_frac": float("nan"),
            "feasible": False,
        }

    x_opt = np.asarray(x.value, dtype=float).ravel()
    slack_frac = float(
        p_lo.value + p_hi.value + c_lo.value + c_hi.value + f_lo.value + f_hi.value
    )
    ratio_val = simple_empirical_obj_value(
        x_opt,
        basis_index=basis_index,
        basis_samples=basis_samples,
        total_mass=total_mass,
        ingredient_basis=ingredient_basis,
    )
    return {
        "status": status,
        "x_opt": x_opt,
        "objective": float(prob.value) if prob.value is not None else float("nan"),
        "ratio_objective": ratio_val,
        "macro_slack_frac": slack_frac,
        "feasible": True,
        "macros_before": compute_macros(x0, M),
        "macros_after": compute_macros(x_opt, M),
    }


def optimize_simple_empirical_obj_slack(
    x0: np.ndarray,
    M: np.ndarray,
    *,
    basis_index: list[str],
    basis_samples: dict[str, np.ndarray],
    ingredient_basis: list[str | None],
    kcal_target: float,
    target_protein_frac: float,
    target_carb_frac: float,
    target_fat_frac: float,
    total_mass: float,
    slack_weight: float = DEFAULT_SLACK_WEIGHT,
    min_grams: float = MIN_INGREDIENT_GRAMS,
    max_scale: float = MAX_GRAM_SCALE,
) -> dict[str, object]:
    """Backward-compatible soft point target (a zero-width PFC box)."""
    return optimize_simple_empirical_obj_box_slack(
        x0,
        M,
        basis_index=basis_index,
        basis_samples=basis_samples,
        ingredient_basis=ingredient_basis,
        kcal_target=kcal_target,
        protein_frac_min=target_protein_frac,
        protein_frac_max=target_protein_frac,
        carb_frac_min=target_carb_frac,
        carb_frac_max=target_carb_frac,
        fat_frac_min=target_fat_frac,
        fat_frac_max=target_fat_frac,
        total_mass=total_mass,
        slack_weight=slack_weight,
        min_grams=min_grams,
        max_scale=max_scale,
    )


def load_recipe_macro_problem(recipe_id: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    from recipe_data_access import data_source, get_store

    rid = int(recipe_id)
    if data_source() == "local":
        store = get_store()
        ingredients = store.resolved_recipes([rid])
        ingredients = ingredients[
            ingredients["fdc_id"].notna() & ingredients["gram_weight"].notna()
        ].copy()
        food_nutrients = store.food_nutrients(ingredients["fdc_id"].astype(int).tolist())
        x0, M = build_recipe_macro_inputs(ingredients, food_nutrients)
        return ingredients, x0, M

    conn = connect()
    try:
        with conn.cursor() as cur:
            ingredients = fetch_resolved_ingredients(cur, rid)
            ingredients = ingredients[
                ingredients["fdc_id"].notna() & ingredients["gram_weight"].notna()
            ].copy()
            food_nutrients = fetch_food_nutrients_for_recipe(
                cur, ingredients["fdc_id"].astype(int).tolist()
            )
    finally:
        conn.close()
    x0, M = build_recipe_macro_inputs(ingredients, food_nutrients)
    return ingredients, x0, M


def macros_from_portions(
    ingredients: pd.DataFrame,
    gram_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build macro matrix from ingredients + gram vector (for LLM eval)."""
    food_nutrients = fetch_food_nutrients_for_recipe(
        None, ingredients["fdc_id"].astype(int).tolist()
    )
    ing = ingredients.copy()
    ing["gram_weight"] = gram_weights
    x0, M = build_recipe_macro_inputs(ing, food_nutrients)
    return compute_macros(x0, M), M


@dataclass
class CanonicalNeighborhood:
    canonical_recipe_id: int
    title: str
    lines_df: pd.DataFrame
    recipe_ids: list[str]
    n_recipes: int
    subtree_df: pd.DataFrame
    cut_nodes: set[str]
    best_nodes: set[str]
    recipe_sets: dict[str, set[str]]
    starting_recipe_id: str
    rollup_chains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    basis_nodes: set[str] = field(default_factory=set)
    basis_share_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    basis_index: list[str] = field(default_factory=list)
    basis_samples: dict[str, np.ndarray] = field(default_factory=dict)
    basis_sample_weights: dict[str, np.ndarray] = field(default_factory=dict)
    ingredient_basis: list[str | None] = field(default_factory=list)
    starting_ingredients: pd.DataFrame = field(default_factory=pd.DataFrame)
    x0: np.ndarray = field(default_factory=lambda: np.array([]))
    M: np.ndarray = field(default_factory=lambda: np.array([]))
    fdc_catalog: pd.DataFrame = field(default_factory=pd.DataFrame)
    shell_recipe_ids: list[str] = field(default_factory=list)
    expansion_meta: dict[str, Any] = field(default_factory=dict)
    from_cache: bool = False

    @classmethod
    def from_cache(cls, canonical_recipe_id: int) -> CanonicalNeighborhood | None:
        """Rehydrate neighborhood from recipe.canonical_neighborhood_cache + live lines SQL.

        Returns None on miss / version mismatch / empty lines. Skips Jaccard entirely.
        """
        cached = load_neighborhood_cache_row(canonical_recipe_id)
        if cached is None:
            return None

        title = str(cached.get("title") or load_canonical_title(canonical_recipe_id))
        lines_df = load_canonical_lines(canonical_recipe_id)
        if lines_df.empty:
            return None

        recipe_ids = sorted(lines_df["recipe_nlg_id"].unique(), key=lambda x: (len(x), x))
        cut_nodes = set(map(str, cached.get("cut_nodes") or []))
        best_nodes = set(map(str, cached.get("best_nodes") or []))
        basis_nodes = set(map(str, cached.get("basis_nodes") or []))
        recipe_sets_raw = cached.get("recipe_sets") or {}
        if not isinstance(recipe_sets_raw, dict):
            recipe_sets_raw = {}
        recipe_sets_raw = cached.get("recipe_sets") or {}
        if not isinstance(recipe_sets_raw, dict):
            recipe_sets_raw = {}
        recipe_sets = {
            str(k): set(map(str, v if isinstance(v, (list, tuple, set)) else (list(v) if v is not None else [])))
            for k, v in recipe_sets_raw.items()
        }
        rollup_raw = cached.get("rollup_chains") or {}
        if not isinstance(rollup_raw, dict):
            rollup_raw = {}
        rollup_chains = {
            str(k): tuple(
                map(
                    str,
                    v if isinstance(v, (list, tuple)) else (list(v) if v is not None else []),
                )
            )
            for k, v in rollup_raw.items()
        }
        shares = cached.get("basis_shares") or []
        if isinstance(shares, np.ndarray):
            shares = shares.tolist()
        if not isinstance(shares, list):
            shares = []
        basis_share_df = pd.DataFrame(shares) if shares else pd.DataFrame(
            columns=["recipe_nlg_id", "basis_node_id", "share"]
        )

        starting_recipe_id = str(cached.get("starting_recipe_id") or recipe_ids[0])
        if starting_recipe_id not in set(map(str, recipe_ids)) and recipe_ids:
            starting_recipe_id = str(recipe_ids[0])

        def rollup_to_active(leaf_id: str | None, active_nodes: set[str]) -> str | None:
            if not leaf_id:
                return None
            for node_id in rollup_chains.get(str(leaf_id), (str(leaf_id),)):
                if node_id in active_nodes:
                    return node_id
            return None

        starting_ingredients, x0, M = load_recipe_macro_problem(starting_recipe_id)
        leaf_by_fdc_idx: dict[tuple[int, int], str] = {}
        for _, row in lines_df.loc[lines_df["recipe_nlg_id"] == starting_recipe_id].iterrows():
            leaf_by_fdc_idx[(int(row["ingredient_idx"]), int(row["fdc_id"]))] = str(row["foodon_id"])

        ingredient_basis: list[str | None] = []
        for row in starting_ingredients.itertuples(index=False):
            leaf = leaf_by_fdc_idx.get((int(row.ingredient_idx), int(row.fdc_id)))
            ingredient_basis.append(rollup_to_active(leaf, basis_nodes) if leaf else None)

        index = _get_index()
        recipe_basis_nodes = {nid for nid in ingredient_basis if nid is not None}
        basis_index = sorted(recipe_basis_nodes, key=lambda nid: index.labels.get(nid, nid).lower())
        has_nodes = not basis_share_df.empty and "basis_node_id" in basis_share_df.columns
        has_weight_col = has_nodes and "weight" in basis_share_df.columns
        basis_samples = {
            nid: basis_share_df.loc[basis_share_df["basis_node_id"] == nid, "share"].to_numpy(
                dtype=float
            )
            if has_nodes
            else np.array([], dtype=float)
            for nid in basis_index
        }
        basis_sample_weights = {
            nid: (
                basis_share_df.loc[basis_share_df["basis_node_id"] == nid, "weight"].to_numpy(dtype=float)
                if has_weight_col
                else np.ones(len(basis_samples[nid]), dtype=float)
            )
            for nid in basis_index
        }
        build_params = cached.get("build_params") or {}
        if isinstance(build_params, str):
            try:
                build_params = json.loads(build_params)
            except Exception:
                build_params = {}
        shell_ids = [str(x) for x in (build_params.get("shell_recipe_ids") or [])]
        expansion_meta = build_params.get("expansion_meta") or {}
        if isinstance(expansion_meta, str):
            try:
                expansion_meta = json.loads(expansion_meta)
            except Exception:
                expansion_meta = {}

        fdc_catalog = (
            lines_df[["fdc_id", "fdc_description"]]
            .drop_duplicates(subset=["fdc_id"])
            .sort_values("fdc_id")
            .reset_index(drop=True)
        )

        return cls(
            canonical_recipe_id=int(canonical_recipe_id),
            title=title,
            lines_df=lines_df,
            recipe_ids=[str(x) for x in recipe_ids],
            n_recipes=int(cached.get("n_recipes") or len(recipe_ids)),
            subtree_df=pd.DataFrame(),
            cut_nodes=cut_nodes,
            best_nodes=best_nodes,
            recipe_sets=recipe_sets,
            starting_recipe_id=starting_recipe_id,
            rollup_chains=rollup_chains,
            basis_nodes=basis_nodes,
            basis_share_df=basis_share_df,
            basis_index=basis_index,
            basis_samples=basis_samples,
            basis_sample_weights=basis_sample_weights,
            ingredient_basis=ingredient_basis,
            starting_ingredients=starting_ingredients,
            x0=x0,
            M=M,
            fdc_catalog=fdc_catalog,
            shell_recipe_ids=shell_ids,
            expansion_meta=expansion_meta if isinstance(expansion_meta, dict) else {},
            from_cache=True,
        )

    @classmethod
    def build(
        cls,
        canonical_recipe_id: int,
        *,
        fast: bool = False,
        use_cache: bool = True,
        save_cache: bool = False,
        require_cache: bool = False,
    ) -> CanonicalNeighborhood:
        """Build neighborhood for a canonical dish.

        ``use_cache=True`` (default): try ``recipe.canonical_neighborhood_cache`` first
        (full Jaccard artifacts from precompute). On hit, skips Jaccard entirely.

        ``fast=True`` only applies on cache miss: skips combinatorial Jaccard search and
        uses the cut antichain instead.

        ``save_cache=True``: upsert cache after a fresh build (used by precompute).

        ``require_cache=True``: if the Jaccard cache row is missing/invalid, raise instead
        of falling back to a live build (used by MacroIQ / web so demos stay on cache).
        """
        if use_cache:
            cached_nb = cls.from_cache(canonical_recipe_id)
            if cached_nb is not None:
                return cached_nb
            if require_cache:
                raise ValueError(
                    f"Neighborhood cache miss for canonical_recipe_id={canonical_recipe_id}. "
                    "Precompute recipe.canonical_neighborhood_cache before running MacroIQ "
                    "(scripts/populate_neighborhood_cache.py / precompute_canonical_neighborhoods.py)."
                )

        nb = cls._build_fresh(canonical_recipe_id, fast=fast)
        if save_cache:
            try:
                save_neighborhood_cache(nb, fast=fast)
            except Exception:
                pass
        return nb

    @classmethod
    def _build_fresh(cls, canonical_recipe_id: int, *, fast: bool = False) -> CanonicalNeighborhood:
        """Full (or fast) in-memory Jaccard / rollup build without reading cache."""
        index = _get_index()
        hierarchy = _get_hierarchy()

        title = load_canonical_title(canonical_recipe_id)
        lines_df = load_canonical_lines(canonical_recipe_id)
        if lines_df.empty:
            raise ValueError(f"No resolved lines for canonical_recipe_id={canonical_recipe_id}")

        recipe_ids = sorted(lines_df["recipe_nlg_id"].unique(), key=lambda x: (len(x), x))
        n_recipes = len(recipe_ids)
        matched_leaves = set(lines_df["foodon_id"].unique())

        relevant_nodes: set[str] = set(matched_leaves)
        for leaf_id in matched_leaves:
            relevant_nodes.update(index.ancestry_path(leaf_id))

        leaf_line_counts = lines_df.groupby("foodon_id").size().to_dict()
        subtree_rows: list[dict] = []
        for node_id in sorted(relevant_nodes, key=lambda nid: index.labels.get(nid, nid).lower()):
            if node_id not in index.labels:
                continue
            desc = set(hierarchy.descendants.get(node_id, [node_id]))
            desc_leaves = desc & matched_leaves
            line_hits = int(sum(leaf_line_counts.get(lid, 0) for lid in desc_leaves))
            recipes_with_hit = lines_df.loc[lines_df["foodon_id"].isin(desc_leaves), "recipe_nlg_id"].nunique()
            subtree_rows.append(
                {
                    "foodon_id": node_id,
                    "label": index.labels.get(node_id, node_id),
                    "is_leaf": hierarchy.is_leaf(node_id),
                    "line_hits": line_hits,
                    "leaf_count": len(desc_leaves),
                    "recipe_hits": int(recipes_with_hit),
                    "recipe_pct": round(100.0 * recipes_with_hit / n_recipes, 2) if n_recipes else 0.0,
                }
            )
        subtree_df = pd.DataFrame(subtree_rows).sort_values(
            ["recipe_pct", "line_hits"], ascending=[False, False]
        )

        qualifying_nodes = set(
            subtree_df.loc[subtree_df["recipe_pct"] >= 100 * NEIGHBORHOOD_CUT_PCT, "foodon_id"]
        )
        cut_nodes = deepest_antichain(qualifying_nodes)

        def ancestry_chain(leaf_id: str) -> list[str]:
            return [leaf_id, *index.ancestry_path(leaf_id)]

        rollup_chains = {leaf_id: tuple(ancestry_chain(leaf_id)) for leaf_id in matched_leaves}
        recipe_leaf_sets = (
            lines_df.groupby("recipe_nlg_id")["foodon_id"].apply(lambda s: set(map(str, s))).to_dict()
        )

        def rollup_leaf(leaf_id: str, active_nodes: set[str]) -> str | None:
            for node_id in rollup_chains.get(leaf_id, (leaf_id,)):
                if node_id in active_nodes:
                    return node_id
            return None

        def recipe_presence_sets(active_nodes: set[str]) -> dict[str, set[str]]:
            out: dict[str, set[str]] = {}
            for recipe_id, leaves in recipe_leaf_sets.items():
                rolled = {rollup_leaf(leaf_id, active_nodes) for leaf_id in leaves}
                rolled.discard(None)
                if rolled:
                    out[recipe_id] = rolled
            return out

        if fast:
            # Agent path on cache miss: skip O(2^n) / hill-climb Jaccard search.
            best_nodes = set(cut_nodes) if cut_nodes else deepest_antichain(qualifying_nodes)
            if len(best_nodes) < MIN_ROLLUP_NODES and qualifying_nodes:
                top = sorted(
                    qualifying_nodes,
                    key=lambda n: float(
                        subtree_df.loc[subtree_df["foodon_id"] == n, "recipe_pct"].iloc[0]
                    ),
                    reverse=True,
                )
                best_nodes = deepest_antichain(set(top[: max(MIN_ROLLUP_NODES, len(top))]))
        else:
            def mean_pairwise_jaccard(sets: dict[str, set[str]]) -> float:
                ids = sorted(sets)
                if len(ids) < 2:
                    return 1.0
                scores: list[float] = []
                for a_id, b_id in itertools.combinations(ids, 2):
                    a, b = sets[a_id], sets[b_id]
                    union = a | b
                    scores.append(len(a & b) / len(union) if union else 1.0)
                return float(np.mean(scores))

            def score_subset(active_nodes: set[str]) -> float:
                if len(active_nodes) < MIN_ROLLUP_NODES:
                    return -1.0
                return mean_pairwise_jaccard(recipe_presence_sets(active_nodes))

            def is_better_subset(
                candidate: set[str], candidate_score: float, best: set[str], best_score: float
            ) -> bool:
                if candidate_score < best_score - 1e-9:
                    return False
                if candidate_score > best_score + 1e-9:
                    return True
                if len(candidate) != len(best):
                    return len(candidate) > len(best)
                return sorted(candidate) < sorted(best)

            def search_best_subset(candidates: set[str]) -> tuple[set[str], float]:
                ordered = sorted(candidates)
                best_nodes_local: set[str] = set()
                best_score = -1.0

                if len(ordered) <= EXACT_SUBSET_SEARCH_LIMIT:
                    for mask in range(1, 1 << len(ordered)):
                        active = {ordered[i] for i in range(len(ordered)) if mask & (1 << i)}
                        if not is_antichain(active):
                            continue
                        score = score_subset(active)
                        if is_better_subset(active, score, best_nodes_local, best_score):
                            best_nodes_local, best_score = active, score
                    return best_nodes_local, best_score

                current = deepest_antichain(candidates)
                if len(current) < MIN_ROLLUP_NODES:
                    top = sorted(
                        candidates,
                        key=lambda n: float(
                            subtree_df.loc[subtree_df["foodon_id"] == n, "recipe_pct"].iloc[0]
                        ),
                        reverse=True,
                    )
                    current = set(top[:MIN_ROLLUP_NODES])
                current_score = score_subset(current)
                best_nodes_local, best_score = set(current), current_score

                improved = True
                while improved:
                    improved = False
                    for node_id in sorted(candidates | best_nodes_local):
                        trial = (
                            set(best_nodes_local) - {node_id}
                            if node_id in best_nodes_local
                            else set(best_nodes_local) | {node_id}
                        )
                        if not trial or not is_antichain(trial):
                            continue
                        trial_score = score_subset(trial)
                        if is_better_subset(trial, trial_score, best_nodes_local, best_score):
                            best_nodes_local, best_score = trial, trial_score
                            improved = True
                return best_nodes_local, best_score

            best_nodes, _ = search_best_subset(cut_nodes)

        recipe_sets = recipe_presence_sets(best_nodes)

        starting_rows: list[dict] = []
        for recipe_id in recipe_ids:
            rolled = recipe_sets.get(recipe_id, set())
            hit = rolled & best_nodes
            grams = float(lines_df.loc[lines_df["recipe_nlg_id"] == recipe_id, "gram_weight"].sum())
            starting_rows.append(
                {
                    "recipe_nlg_id": recipe_id,
                    "subtrees_hit": len(hit),
                    "total_grams": grams,
                }
            )
        starting_df = pd.DataFrame(starting_rows).sort_values(
            ["subtrees_hit", "total_grams", "recipe_nlg_id"],
            ascending=[False, False, True],
        )
        starting_recipe_id = str(starting_df.iloc[0]["recipe_nlg_id"])

        def rollup_to_active(leaf_id: str, active_nodes: set[str]) -> str | None:
            for node_id in rollup_chains.get(str(leaf_id), (str(leaf_id),)):
                if node_id in active_nodes:
                    return node_id
            return None

        min_basis_hits = adaptive_min_basis_hits(n_recipes)
        hits_ok = set(subtree_df.loc[subtree_df["recipe_hits"] >= min_basis_hits, "foodon_id"])
        basis_nodes = deepest_antichain(hits_ok)

        # --- Thin-neighborhood expansion: add a down-weighted similarity shell -------
        recipe_weight: dict[str, float] = {str(r): 1.0 for r in recipe_ids}
        dist_lines_df = lines_df
        shell_recipe_ids: list[str] = []
        expansion_meta: dict[str, Any] = {"activated": False, "n_core": n_recipes}
        try:
            from neighborhood_expansion import expand_neighborhood, shell_lines_df
            from recipe_data_access import get_store

            store = get_store()
            core_leaf_sets = {
                str(rid): set(map(str, s)) for rid, s in recipe_leaf_sets.items()
            }
            result = expand_neighborhood(
                core_recipe_ids=[str(r) for r in recipe_ids],
                core_leaf_sets=core_leaf_sets,
                cut_nodes=set(cut_nodes),
                index=index,
                store=store,
                target_n=EXPANSION_TARGET_N,
                max_shell=EXPANSION_MAX_SHELL,
                min_similarity=EXPANSION_MIN_SIMILARITY,
                shell_weight=EXPANSION_SHELL_WEIGHT,
            )
            expansion_meta = dict(result.get("meta") or {})
            shell = result.get("shell") or []
            if shell:
                shell_recipe_ids = [str(e["recipe_id"]) for e in shell]
                for e in shell:
                    recipe_weight[str(e["recipe_id"])] = float(e["weight"])
                shell_df = shell_lines_df(store, shell_recipe_ids, canonical_recipe_id)
                if not shell_df.empty:
                    common = [c for c in lines_df.columns if c in shell_df.columns]
                    dist_lines_df = pd.concat(
                        [lines_df[common], shell_df[common]], ignore_index=True
                    )
        except Exception as exc:  # expansion is best-effort; fall back to core-only
            expansion_meta = {"activated": False, "n_core": n_recipes, "error": str(exc)}

        basis_share_rows: list[dict] = []
        for recipe_id, sub in dist_lines_df.groupby("recipe_nlg_id"):
            total_grams = float(sub["gram_weight"].sum())
            if total_grams <= 0:
                continue
            node_grams: dict[str, float] = {}
            for _, line in sub.iterrows():
                node_id = rollup_to_active(str(line["foodon_id"]), basis_nodes)
                if node_id is None:
                    continue
                node_grams[node_id] = node_grams.get(node_id, 0.0) + float(line["gram_weight"])
            w = float(recipe_weight.get(str(recipe_id), 1.0))
            for node_id, grams in node_grams.items():
                basis_share_rows.append(
                    {
                        "recipe_nlg_id": str(recipe_id),
                        "basis_node_id": node_id,
                        "share": grams / total_grams,
                        "weight": w,
                    }
                )
        basis_share_df = pd.DataFrame(basis_share_rows)

        starting_ingredients, x0, M = load_recipe_macro_problem(starting_recipe_id)
        leaf_by_fdc_idx: dict[tuple[int, int], str] = {}
        for _, row in lines_df.loc[lines_df["recipe_nlg_id"] == starting_recipe_id].iterrows():
            leaf_by_fdc_idx[(int(row["ingredient_idx"]), int(row["fdc_id"]))] = str(row["foodon_id"])

        ingredient_basis: list[str | None] = []
        for row in starting_ingredients.itertuples(index=False):
            leaf = leaf_by_fdc_idx.get((int(row.ingredient_idx), int(row.fdc_id)))
            ingredient_basis.append(rollup_to_active(leaf, basis_nodes) if leaf else None)

        recipe_basis_nodes = {nid for nid in ingredient_basis if nid is not None}
        basis_index = sorted(recipe_basis_nodes, key=lambda nid: index.labels.get(nid, nid).lower())
        has_weight_col = (not basis_share_df.empty) and ("weight" in basis_share_df.columns)
        basis_samples = {
            nid: basis_share_df.loc[basis_share_df["basis_node_id"] == nid, "share"].to_numpy(dtype=float)
            for nid in basis_index
        }
        basis_sample_weights = {
            nid: (
                basis_share_df.loc[basis_share_df["basis_node_id"] == nid, "weight"].to_numpy(dtype=float)
                if has_weight_col
                else np.ones(int((basis_share_df["basis_node_id"] == nid).sum()), dtype=float)
            )
            for nid in basis_index
        }

        fdc_catalog = (
            lines_df[["fdc_id", "fdc_description"]]
            .drop_duplicates(subset=["fdc_id"])
            .sort_values("fdc_id")
            .reset_index(drop=True)
        )

        return cls(
            canonical_recipe_id=canonical_recipe_id,
            title=title,
            lines_df=lines_df,
            recipe_ids=recipe_ids,
            n_recipes=n_recipes,
            subtree_df=subtree_df,
            cut_nodes=cut_nodes,
            best_nodes=best_nodes,
            recipe_sets=recipe_sets,
            starting_recipe_id=starting_recipe_id,
            rollup_chains=rollup_chains,
            basis_nodes=basis_nodes,
            basis_share_df=basis_share_df,
            basis_index=basis_index,
            basis_samples=basis_samples,
            basis_sample_weights=basis_sample_weights,
            ingredient_basis=ingredient_basis,
            starting_ingredients=starting_ingredients,
            x0=x0,
            M=M,
            fdc_catalog=fdc_catalog,
            shell_recipe_ids=shell_recipe_ids,
            expansion_meta=expansion_meta,
            from_cache=False,
        )

    def baseline_pfc_fractions(self) -> tuple[float, float, float]:
        macros = compute_macros(self.x0, self.M)
        return macro_calorie_fractions_from_grams(float(macros[0]), float(macros[1]), float(macros[2]))

    def ratio_loss(self, x: np.ndarray) -> float:
        kcal_target = atwater_kcal(self.x0, self.M)
        x_ref = atwater_scaled_baseline(self.x0, self.M, kcal_target)
        total_mass = float(x_ref.sum())
        return simple_empirical_obj_value(
            x,
            basis_index=self.basis_index,
            basis_samples=self.basis_samples,
            total_mass=total_mass,
            ingredient_basis=self.ingredient_basis,
        )


@dataclass
class OptimizationCaseResult:
    canonical_recipe_id: int
    canonical_title: str
    starting_recipe_id: str
    target_protein_frac: float
    target_carb_frac: float
    target_fat_frac: float
    kcal_target: float
    x_before: np.ndarray
    x_after: np.ndarray
    lp_status: str
    lp_feasible: bool
    macro_point_distance_before: float
    macro_point_distance_after: float
    ratio_loss_before: float
    ratio_loss_after: float
    portions_df: pd.DataFrame
    ingredients: pd.DataFrame
    macro_slack_loss_after: float | None = None
    ratio_objective_after: float | None = None


def run_canonical_optimization(
    neighborhood: CanonicalNeighborhood,
    *,
    target_protein_frac: float,
    target_carb_frac: float,
    target_fat_frac: float,
    kcal_target: float | None = None,
    use_slack_macros: bool = False,
    slack_weight: float = DEFAULT_SLACK_WEIGHT,
    target_box_half_width: float = 0.0,
) -> OptimizationCaseResult:
    x0 = neighborhood.x0
    M = neighborhood.M
    kcal = float(kcal_target if kcal_target is not None else atwater_kcal(x0, M))
    x_ref = atwater_scaled_baseline(x0, M, kcal)
    total_mass = float(x_ref.sum())

    region_ok, region_msg = check_optimizer_point_feasible(
        x0,
        M,
        kcal_target=kcal,
        target_protein_frac=target_protein_frac,
        target_carb_frac=target_carb_frac,
        target_fat_frac=target_fat_frac,
        total_mass=total_mass,
    )

    macros_before = compute_macros(x0, M)
    dist_before = macro_point_distance(
        float(macros_before[0]),
        float(macros_before[1]),
        float(macros_before[2]),
        target_protein_frac=target_protein_frac,
        target_carb_frac=target_carb_frac,
        target_fat_frac=target_fat_frac,
    )
    ratio_before = neighborhood.ratio_loss(x0)
    target_box = pfc_box_around_point(
        target_protein_frac,
        target_carb_frac,
        target_fat_frac,
        half_width=target_box_half_width,
    )

    if not region_ok and not use_slack_macros:
        portions = pd.DataFrame(
            {
                "ingredient_idx": neighborhood.starting_ingredients["ingredient_idx"].astype(int),
                "ingredient": neighborhood.starting_ingredients["ingredient"],
                "fdc_id": neighborhood.starting_ingredients["fdc_id"].astype(int),
                "grams_before": x0,
                "grams_after": x0,
            }
        )
        return OptimizationCaseResult(
            canonical_recipe_id=neighborhood.canonical_recipe_id,
            canonical_title=neighborhood.title,
            starting_recipe_id=neighborhood.starting_recipe_id,
            target_protein_frac=target_protein_frac,
            target_carb_frac=target_carb_frac,
            target_fat_frac=target_fat_frac,
            kcal_target=kcal,
            x_before=x0.copy(),
            x_after=x0.copy(),
            lp_status=f"infeasible:{region_msg}",
            lp_feasible=False,
            macro_point_distance_before=dist_before,
            macro_point_distance_after=dist_before,
            ratio_loss_before=ratio_before,
            ratio_loss_after=ratio_before,
            macro_slack_loss_after=None,
            ratio_objective_after=None,
            portions_df=portions,
            ingredients=neighborhood.starting_ingredients,
        )

    if use_slack_macros:
        opt = optimize_simple_empirical_obj_box_slack(
            x0,
            M,
            basis_index=neighborhood.basis_index,
            basis_samples=neighborhood.basis_samples,
            ingredient_basis=neighborhood.ingredient_basis,
            kcal_target=kcal,
            protein_frac_min=target_box["protein_min"],
            protein_frac_max=target_box["protein_max"],
            carb_frac_min=target_box["carb_min"],
            carb_frac_max=target_box["carb_max"],
            fat_frac_min=target_box["fat_min"],
            fat_frac_max=target_box["fat_max"],
            total_mass=total_mass,
            slack_weight=slack_weight,
        )
    else:
        opt = optimize_simple_empirical_obj(
        x0,
        M,
        basis_index=neighborhood.basis_index,
        basis_samples=neighborhood.basis_samples,
        ingredient_basis=neighborhood.ingredient_basis,
        kcal_target=kcal,
        protein_frac_min=target_protein_frac,
        protein_frac_max=target_protein_frac,
        carb_frac_min=target_carb_frac,
        carb_frac_max=target_carb_frac,
        fat_frac_min=target_fat_frac,
        fat_frac_max=target_fat_frac,
        total_mass=total_mass,
        )

    x_opt = np.asarray(opt["x_opt"], dtype=float)
    macros_after = compute_macros(x_opt, M)
    dist_after = macro_point_distance(
        float(macros_after[0]),
        float(macros_after[1]),
        float(macros_after[2]),
        target_protein_frac=target_protein_frac,
        target_carb_frac=target_carb_frac,
        target_fat_frac=target_fat_frac,
    )
    ratio_after = neighborhood.ratio_loss(x_opt)
    achieved_pfc = macro_calorie_fractions_from_grams(
        float(macros_after[0]), float(macros_after[1]), float(macros_after[2])
    )
    slack_after = (
        pfc_box_distance(
            *achieved_pfc,
            protein_min=target_box["protein_min"],
            protein_max=target_box["protein_max"],
            carb_min=target_box["carb_min"],
            carb_max=target_box["carb_max"],
            fat_min=target_box["fat_min"],
            fat_max=target_box["fat_max"],
        )
        if use_slack_macros
        else None
    )
    ratio_obj_after = (
        float(opt.get("ratio_objective", ratio_after)) if use_slack_macros else None
    )

    portions = pd.DataFrame(
        {
            "ingredient_idx": neighborhood.starting_ingredients["ingredient_idx"].astype(int),
            "ingredient": neighborhood.starting_ingredients["ingredient"],
            "fdc_id": neighborhood.starting_ingredients["fdc_id"].astype(int),
            "fdc_description": neighborhood.starting_ingredients["fdc_description"],
            "grams_before": x0,
            "grams_after": x_opt,
        }
    )

    return OptimizationCaseResult(
        canonical_recipe_id=neighborhood.canonical_recipe_id,
        canonical_title=neighborhood.title,
        starting_recipe_id=neighborhood.starting_recipe_id,
        target_protein_frac=target_protein_frac,
        target_carb_frac=target_carb_frac,
        target_fat_frac=target_fat_frac,
        kcal_target=kcal,
        x_before=x0.copy(),
        x_after=x_opt,
        lp_status=str(opt.get("status", "unknown")),
        lp_feasible=bool(opt.get("feasible")),
        macro_point_distance_before=dist_before,
        macro_point_distance_after=dist_after,
        ratio_loss_before=ratio_before,
        ratio_loss_after=ratio_after,
        macro_slack_loss_after=slack_after,
        ratio_objective_after=ratio_obj_after,
        portions_df=portions,
        ingredients=neighborhood.starting_ingredients,
    )
