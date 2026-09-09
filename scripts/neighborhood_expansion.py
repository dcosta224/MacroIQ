"""Expand thin canonical neighborhoods with a similarity-ranked recipe shell.

Small canonical neighborhoods (e.g. spaghetti carbonara ~24 recipes) do not give a
stable empirical mass-share distribution for the ratio/marginal loss. This module
finds additional recipes that are close to the core neighborhood -- by FoodOn
ingredient overlap and semantic embedding similarity -- and returns them as a
down-weighted "shell". Core recipes keep weight 1.0; shell recipes get a reduced
weight that scales with similarity so they enrich the distribution without
overriding the true dish.

Design notes:
- Operates over the local cap40 recipe store (``recipe_data_access.get_store``) so it
  works offline and does not hit Supabase. On a non-local backend it is skipped.
- Shell recipes must share the dish's anchor FoodOn nodes (rolled to the cut
  antichain) so we don't drift into a different dish.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from recipe_similarity import jaccard


def _build_candidate_leaf_sets(store, exclude_ids: set[int]) -> dict[str, set[str]]:
    """recipe_id (str) -> set of FoodOn leaf ids, over all resolved recipes in the store."""
    rr = store.resolved_recipes()
    rr = rr[rr["fdc_id"].notna() & rr["gram_weight"].notna()].copy()
    if rr.empty:
        return {}
    rr["fdc_id"] = rr["fdc_id"].astype(int)
    foodon = store.food_4macro_foodon()
    keep_cols = [c for c in ("fdc_id", "foodon_id") if c in foodon.columns]
    foodon = foodon[keep_cols].dropna(subset=["foodon_id"]).drop_duplicates(subset=["fdc_id"])
    foodon["fdc_id"] = foodon["fdc_id"].astype(int)
    merged = rr.merge(foodon, on="fdc_id", how="inner")
    if merged.empty:
        return {}
    out: dict[str, set[str]] = {}
    for rid, sub in merged.groupby("recipe_id"):
        if int(rid) in exclude_ids:
            continue
        out[str(rid)] = set(map(str, sub["foodon_id"].dropna().tolist()))
    return out


def _rollup_fn(index):
    cache: dict[str, tuple[str, ...]] = {}

    def chain(leaf_id: str) -> tuple[str, ...]:
        if leaf_id not in cache:
            cache[leaf_id] = (leaf_id, *index.ancestry_path(leaf_id))
        return cache[leaf_id]

    def rollup(leaf_id: str, active: set[str]) -> str | None:
        for node_id in chain(leaf_id):
            if node_id in active:
                return node_id
        return None

    return rollup


def expand_neighborhood(
    *,
    core_recipe_ids: list[str],
    core_leaf_sets: dict[str, set[str]],
    cut_nodes: set[str],
    index,
    store,
    target_n: int,
    max_shell: int,
    min_similarity: float,
    shell_weight: float,
) -> dict[str, Any]:
    """Return shell recipes to enrich a thin neighborhood.

    Returns a dict with ``shell`` (list of {recipe_id, weight, similarity, foodon,
    semantic}) and ``meta`` (diagnostics). ``shell`` is empty when expansion is not
    needed or not possible.
    """
    core_ids_str = {str(r) for r in core_recipe_ids}
    n_core = len(core_ids_str)
    meta: dict[str, Any] = {
        "n_core": n_core,
        "target_n": int(target_n),
        "n_shell": 0,
        "backend": getattr(store, "backend", "unknown"),
        "activated": False,
    }
    if n_core >= target_n:
        meta["reason"] = "core_already_at_target"
        return {"shell": [], "meta": meta}
    if getattr(store, "backend", None) != "local":
        meta["reason"] = "expansion_requires_local_store"
        return {"shell": [], "meta": meta}

    exclude = {int(r) for r in core_ids_str if str(r).isdigit()}
    candidates = _build_candidate_leaf_sets(store, exclude)
    if not candidates:
        meta["reason"] = "no_candidates"
        return {"shell": [], "meta": meta}

    rollup = _rollup_fn(index)

    # Anchor nodes: cut-antichain nodes present in a good fraction of core recipes.
    anchor_counts: Counter[str] = Counter()
    for leaves in core_leaf_sets.values():
        rolled = {rollup(l, cut_nodes) for l in leaves}
        rolled.discard(None)
        for node in rolled:
            anchor_counts[node] += 1
    anchor_floor = max(2, math.ceil(0.25 * n_core))
    anchors = {n for n, c in anchor_counts.items() if c >= anchor_floor}
    if not anchors:
        anchors = set(anchor_counts)
    min_anchor_overlap = min(2, len(anchors)) if anchors else 0
    meta["n_anchors"] = len(anchors)

    core_leaf_list = [s for s in core_leaf_sets.values() if s]

    # Semantic centroid from core embeddings (optional).
    centroid_unit: np.ndarray | None = None
    try:
        core_mat, _ = store.recipe_embeddings([int(r) for r in core_ids_str if str(r).isdigit()])
        if core_mat.size:
            centroid = core_mat.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid_unit = centroid / norm
    except Exception:
        centroid_unit = None

    cand_ids = list(candidates.keys())
    cand_cos: dict[str, float] = {}
    if centroid_unit is not None:
        try:
            mat, kept = store.recipe_embeddings([int(r) for r in cand_ids if str(r).isdigit()])
            if mat.size:
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                unit = mat / norms
                cos = unit @ centroid_unit
                for rid, c in zip(kept, cos.tolist()):
                    cand_cos[str(rid)] = float(np.clip(c, 0.0, 1.0))
        except Exception:
            cand_cos = {}

    w_foodon, w_semantic = (0.6, 0.4) if centroid_unit is not None else (1.0, 0.0)

    scored: list[dict[str, Any]] = []
    for rid, leaves in candidates.items():
        if not leaves:
            continue
        rolled = {rollup(l, cut_nodes) for l in leaves}
        rolled.discard(None)
        if anchors and len(rolled & anchors) < min_anchor_overlap:
            continue
        # FoodOn nearest-core similarity (max leaf-set Jaccard to any core recipe).
        fo = 0.0
        for cs in core_leaf_list:
            j = jaccard(leaves, cs)
            if j > fo:
                fo = j
                if fo >= 0.999:
                    break
        sem = cand_cos.get(rid, 0.0)
        combined = w_foodon * fo + w_semantic * sem
        if combined < min_similarity:
            continue
        scored.append(
            {
                "recipe_id": rid,
                "similarity": float(combined),
                "foodon": float(fo),
                "semantic": float(sem),
            }
        )

    scored.sort(key=lambda d: (-d["similarity"], d["recipe_id"]))
    room = max(0, int(target_n) - n_core)
    n_take = min(room, int(max_shell), len(scored))
    shell = scored[:n_take]
    for entry in shell:
        entry["weight"] = float(min(shell_weight, shell_weight * entry["similarity"] / max(min_similarity, 1e-6)))
        entry["weight"] = float(min(shell_weight, max(0.05, entry["weight"])))

    meta.update(
        {
            "n_candidates_scored": len(scored),
            "n_shell": len(shell),
            "activated": bool(shell),
            "shell_weight": float(shell_weight),
            "min_similarity": float(min_similarity),
        }
    )
    return {"shell": shell, "meta": meta}


def shell_lines_df(store, shell_recipe_ids: list[str], canonical_recipe_id: int) -> pd.DataFrame:
    """Resolved+FoodOn lines for shell recipes, shaped like ``load_canonical_lines`` output."""
    if not shell_recipe_ids:
        return pd.DataFrame()
    ids = [int(r) for r in shell_recipe_ids if str(r).isdigit()]
    rr = store.resolved_recipes(ids)
    rr = rr[rr["fdc_id"].notna() & rr["gram_weight"].notna()].copy()
    if rr.empty:
        return pd.DataFrame()
    rr["fdc_id"] = rr["fdc_id"].astype(int)
    foodon = store.food_4macro_foodon(rr["fdc_id"].unique().tolist())
    keep_cols = [c for c in ("fdc_id", "foodon_id", "foodon_label") if c in foodon.columns]
    foodon = foodon[keep_cols].drop_duplicates(subset=["fdc_id"])
    merged = rr.merge(foodon, on="fdc_id", how="left")
    merged = merged[merged["foodon_id"].notna()].copy()
    if merged.empty:
        return pd.DataFrame()
    merged["canonical_recipe_id"] = int(canonical_recipe_id)
    merged["recipe_nlg_id"] = merged["recipe_id"].astype(str)
    merged["fdc_id"] = merged["fdc_id"].astype(int)
    return merged
