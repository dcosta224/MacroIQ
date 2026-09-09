"""Recipe–recipe similarity: FoodOn (multi-level Jaccard) + semantic + cuisine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from mvp_recipe_ranker import cosine_similarity

# Combined score weights (FoodOn and semantic prioritized).
W_FOODON = 0.50
W_SEMANTIC = 0.35
W_CUISINE = 0.15

# Multi-level FoodOn Jaccard weights (leaf → coarse).
FOODON_LEVEL_WEIGHTS = (0.45, 0.30, 0.15, 0.10)

# Soft cuisine family map (symmetric pairs).
CUISINE_FAMILIES: dict[str, set[str]] = {
    "italian": {"italian", "mediterranean", "european"},
    "mediterranean": {"italian", "mediterranean", "greek", "spanish", "european"},
    "mexican": {"mexican", "latin", "tex-mex"},
    "chinese": {"chinese", "asian", "east asian"},
    "japanese": {"japanese", "asian", "east asian"},
    "indian": {"indian", "south asian", "asian"},
    "american": {"american", "southern", "cajun"},
    "french": {"french", "european", "mediterranean"},
}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


def multi_level_foodon_similarity(
    levels_q: list[set[str]],
    levels_r: list[set[str]],
    weights: tuple[float, ...] = FOODON_LEVEL_WEIGHTS,
) -> float:
    """Weighted Jaccard across abstraction levels (pad/truncate to weight length)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    n = len(w)
    score = 0.0
    for i in range(n):
        sq = levels_q[i] if i < len(levels_q) else set()
        sr = levels_r[i] if i < len(levels_r) else set()
        score += float(w[i]) * jaccard(sq, sr)
    return float(score)


def cuisine_similarity(cuisine_q: str | None, cuisine_r: str | None) -> float:
    if not cuisine_q or not cuisine_r:
        return 0.0
    a = cuisine_q.strip().lower()
    b = cuisine_r.strip().lower()
    if a == b:
        return 1.0
    fam_a = CUISINE_FAMILIES.get(a, {a})
    fam_b = CUISINE_FAMILIES.get(b, {b})
    if a in fam_b or b in fam_a or (fam_a & fam_b):
        return 0.5
    return 0.0


def semantic_similarity(emb_q: np.ndarray, emb_r: np.ndarray) -> float:
    sims = cosine_similarity(emb_q, np.asarray(emb_r, dtype=float).reshape(1, -1))
    return float(np.clip(sims[0], 0.0, 1.0))


@dataclass(frozen=True)
class SimilarityBreakdown:
    foodon: float
    semantic: float
    cuisine: float
    combined: float


def combined_similarity(
    *,
    foodon_levels_q: list[set[str]],
    foodon_levels_r: list[set[str]],
    emb_q: np.ndarray | None,
    emb_r: np.ndarray | None,
    cuisine_q: str | None,
    cuisine_r: str | None,
    w_foodon: float = W_FOODON,
    w_semantic: float = W_SEMANTIC,
    w_cuisine: float = W_CUISINE,
) -> SimilarityBreakdown:
    fo = multi_level_foodon_similarity(foodon_levels_q, foodon_levels_r)
    if emb_q is None or emb_r is None:
        sem = 0.0
        # Renormalize without semantic if missing.
        w_sum = w_foodon + w_cuisine
        w_f, w_c = (w_foodon / w_sum, w_cuisine / w_sum) if w_sum else (0.5, 0.5)
        w_s = 0.0
    else:
        sem = semantic_similarity(emb_q, emb_r)
        w_sum = w_foodon + w_semantic + w_cuisine
        w_f, w_s, w_c = w_foodon / w_sum, w_semantic / w_sum, w_cuisine / w_sum
    cui = cuisine_similarity(cuisine_q, cuisine_r)
    combined = w_f * fo + w_s * sem + w_c * cui
    return SimilarityBreakdown(foodon=fo, semantic=sem, cuisine=cui, combined=float(combined))


def rollup_to_levels(
    leaf_ids: Iterable[str],
    *,
    ancestry_fn,
    basis_nodes: set[str],
    mid_depth: int = 3,
) -> list[set[str]]:
    """Build 4 presence sets: leaf, basis-cut, mid-ancestor, coarse root-ish.

    ancestry_fn(node_id) -> list from leaf to root (inclusive).
    """
    leaves = {str(x) for x in leaf_ids if x}
    basis: set[str] = set()
    mid: set[str] = set()
    coarse: set[str] = set()
    for leaf in leaves:
        chain = list(ancestry_fn(leaf))
        if not chain:
            chain = [leaf]
        # deepest basis node on path
        mapped = None
        for node in chain:
            if node in basis_nodes:
                mapped = node
                break
        if mapped:
            basis.add(mapped)
        if len(chain) > mid_depth:
            mid.add(chain[min(mid_depth, len(chain) - 1)])
        else:
            mid.add(chain[-1])
        coarse.add(chain[-1])
    return [leaves, basis, mid, coarse]
