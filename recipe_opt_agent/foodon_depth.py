"""FoodOn aggregation depth limits.

Caps how many parent hops a leaf may roll up to a basis node. Default is half the
mean root-path depth across FoodOn leaves (computed once from the local index).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# Sensible fallback if the FoodOn index is unavailable.
_FALLBACK_MAX_LEVELS = 3


@lru_cache(maxsize=1)
def average_foodon_leaf_depth() -> float:
    """Mean number of parent hops from each leaf to the root (leaf depth)."""
    try:
        from pathlib import Path

        from foodon_index import FoodOnIndex

        root = Path(__file__).resolve().parents[1]
        cache = root / "foodon_web" / "cache" / "foodon_index.json"
        if not cache.exists():
            return float(2 * _FALLBACK_MAX_LEVELS)
        index = FoodOnIndex.from_json(cache.read_text(encoding="utf-8"))
        depths: list[int] = []
        for node_id in index.labels:
            if index.children.get(node_id):
                continue  # not a leaf
            depths.append(len(index.ancestry_path(node_id)))
        if not depths:
            return float(2 * _FALLBACK_MAX_LEVELS)
        return float(sum(depths) / len(depths))
    except Exception:
        return float(2 * _FALLBACK_MAX_LEVELS)


@lru_cache(maxsize=1)
def default_max_foodon_aggregation_levels() -> int:
    """Half the average FoodOn leaf depth, rounded down, at least 1."""
    avg = average_foodon_leaf_depth()
    return max(1, int(avg // 2))


def resolve_max_foodon_aggregation_levels(explicit: int | None = None) -> int:
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except (TypeError, ValueError):
            pass
    return default_max_foodon_aggregation_levels()


def capped_rollup_node(
    leaf_id: str | None,
    active_basis: set[str],
    rollup_chains: dict[str, Any],
    *,
    max_levels: int,
) -> str | None:
    """Roll leaf → first active basis within ``max_levels`` hops; else keep the leaf.

    Never abstracts past ``max_levels`` even if a coarser basis node exists higher up.
    """
    if not leaf_id:
        return None
    leaf = str(leaf_id)
    chain = rollup_chains.get(leaf) or rollup_chains.get(leaf_id)
    if not chain:
        return leaf if leaf in active_basis or not active_basis else leaf
    seq = [str(x) for x in chain]
    window = seq[: max(0, int(max_levels)) + 1]
    for node_id in window:
        if node_id in active_basis:
            return node_id
    # Cap exceeded for every active basis → stay at leaf (no deep abstraction).
    return leaf


def apply_foodon_aggregation_cap(
    problem: dict[str, Any],
    *,
    max_levels: int | None = None,
) -> dict[str, Any]:
    """Re-rollup ``ingredient_basis`` under the aggregation cap; annotate build_params."""
    problem = dict(problem)
    levels = resolve_max_foodon_aggregation_levels(max_levels)
    chains = problem.get("rollup_chains") or {}
    if not isinstance(chains, dict):
        chains = {}
    chains_norm = {
        str(k): [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
        for k, v in chains.items()
    }
    active = {str(n) for n in (problem.get("basis_nodes") or []) if n}
    # Also treat current basis list entries as active
    for n in problem.get("ingredient_basis") or []:
        if n:
            active.add(str(n))

    ingredients = list((problem.get("chosen_recipe") or {}).get("ingredients") or [])
    leaves = list(problem.get("ingredient_foodon_leaves") or [])
    while len(leaves) < len(ingredients):
        row = ingredients[len(leaves)]
        leaves.append(row.get("foodon_id") or row.get("foodon_leaf_id"))

    old_basis = list(problem.get("ingredient_basis") or [])
    new_basis: list[str | None] = []
    for i, leaf in enumerate(leaves):
        leaf_s = str(leaf) if leaf else None
        # Prefer rolling against the uncapped basis target when present
        if leaf_s and leaf_s in chains_norm:
            rolled = capped_rollup_node(leaf_s, active, chains_norm, max_levels=levels)
        else:
            prev = old_basis[i] if i < len(old_basis) else None
            rolled = str(prev) if prev else leaf_s
            # If previous rollup was deeper than cap, walk the chain and stop
            if leaf_s and prev and leaf_s in chains_norm:
                rolled = capped_rollup_node(leaf_s, {str(prev)} | active, chains_norm, max_levels=levels)
        new_basis.append(rolled)

    problem["ingredient_basis"] = new_basis
    bp = dict(problem.get("build_params") or {})
    bp["max_foodon_aggregation_levels"] = levels
    bp["max_foodon_aggregation_levels_default"] = default_max_foodon_aggregation_levels()
    bp["average_foodon_leaf_depth"] = average_foodon_leaf_depth()
    problem["build_params"] = bp
    return problem
