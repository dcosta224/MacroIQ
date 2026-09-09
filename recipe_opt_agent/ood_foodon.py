"""Map OOD / catalog foods onto real FoodOn nodes and enrich share samples.

Synthetic ``ood_lean_protein`` was a placeholder with no neighborhood samples, so
new ingredients showed blank losses. This module resolves real FoodOn leaves
(and a rollup basis) and harvests mass-share samples from query-expanded shell
recipes so every final ingredient can contribute to ratio/share loss.
"""

from __future__ import annotations

from typing import Any

# Preferred FoodOn leaves for known OOD lean proteins (verified in local FDC map).
OOD_FOODON_DEFAULTS: dict[str, dict[str, str]] = {
    "chicken": {
        "foodon_leaf_id": "FOODON_02020280",
        "foodon_leaf_label": "piece of chicken breast (skinless, boneless, raw)",
    },
    "turkey": {
        "foodon_leaf_id": "FOODON_02020542",
        "foodon_leaf_label": "piece of turkey breast (with skin, raw)",
    },
    "tofu": {
        "foodon_leaf_id": "FOODON_00005540",
        "foodon_leaf_label": "firm tofu",
    },
    "egg white": {
        "foodon_leaf_id": "FOODON_03542975",
        "foodon_leaf_label": "hen egg white, dried (efsa foodex2)",
    },
    "yogurt": {
        "foodon_leaf_id": "FOODON_03305085",
        "foodon_leaf_label": "yogurt (nonfat)",
    },
}

MIN_BASIS_HITS_FOR_LOSS = 5


def _label_for(node_id: str | None) -> str | None:
    if not node_id:
        return None
    try:
        from canonical_optimization import foodon_display_label

        return foodon_display_label(str(node_id)) or str(node_id)
    except Exception:
        return str(node_id)


def lookup_foodon_for_label(label: str) -> dict[str, str] | None:
    """Best-effort FoodOn leaf from local usda.food_4macro_foodon by description/label."""
    q = (label or "").lower().strip()
    if not q:
        return None
    # Keyword preference from known OOD defaults
    for key, defaults in OOD_FOODON_DEFAULTS.items():
        if key in q:
            # Try store lookup first for a more precise leaf
            hit = _store_foodon_search(q) or _store_foodon_search(key)
            if hit:
                return hit
            return {
                "foodon_leaf_id": defaults["foodon_leaf_id"],
                "foodon_leaf_label": defaults.get("foodon_leaf_label") or _label_for(defaults["foodon_leaf_id"]) or defaults["foodon_leaf_id"],
            }
    return _store_foodon_search(q)


def _store_foodon_search(query: str) -> dict[str, str] | None:
    try:
        from recipe_data_access import get_store

        store = get_store()
        fo = store.food_4macro_foodon()
        if fo is None or fo.empty:
            return None
        q_tokens = {t for t in query.lower().split() if len(t) > 2}
        best = None
        best_score = -1.0
        for row in fo.itertuples(index=False):
            lab = str(getattr(row, "foodon_label", "") or "").lower()
            desc = str(getattr(row, "description", "") or "").lower()
            text = f"{lab} {desc}"
            if not text.strip():
                continue
            toks = {t for t in text.split() if len(t) > 2}
            if not q_tokens or not toks:
                continue
            score = len(q_tokens & toks) / len(q_tokens | toks)
            if query in lab or query in desc:
                score += 0.5
            if "breast" in query and "breast" in lab:
                score += 0.35
            if "skinless" in query and "skinless" in lab:
                score += 0.15
            # Prefer known chicken-breast leaf when query is about chicken breast
            if "chicken" in query and "breast" in query and str(getattr(row, "foodon_id", "")) == "FOODON_02020280":
                score += 0.5
            if score > best_score:
                best_score = score
                nid = getattr(row, "foodon_id", None)
                if nid is None:
                    continue
                best = {
                    "foodon_leaf_id": str(nid),
                    "foodon_leaf_label": str(getattr(row, "foodon_label", "") or nid),
                }
        if best and best_score >= 0.08:
            return best
    except Exception:
        return None
    return None


def resolve_basis_for_leaf(
    leaf_id: str | None,
    problem: dict[str, Any],
    *,
    max_levels: int | None = None,
) -> tuple[str | None, str | None]:
    """Return (basis_node_id, basis_label). Prefer rollup into existing active basis."""
    if not leaf_id:
        return None, None
    leaf = str(leaf_id)
    chains = problem.get("rollup_chains") or {}
    chain = None
    if leaf in chains:
        chain = [str(x) for x in chains[leaf]]
    else:
        try:
            from canonical_optimization import _get_index

            index = _get_index()
            chain = [leaf, *index.ancestry_path(leaf)]
            # Persist for later reports
            chains = dict(chains)
            chains[leaf] = chain
            problem["rollup_chains"] = chains
        except Exception:
            chain = [leaf]

    active = {str(n) for n in (problem.get("basis_nodes") or []) if n}
    for n in problem.get("ingredient_basis") or []:
        if n:
            active.add(str(n))

    from recipe_opt_agent.foodon_depth import (
        capped_rollup_node,
        resolve_max_foodon_aggregation_levels,
    )

    levels = resolve_max_foodon_aggregation_levels(max_levels)
    basis = capped_rollup_node(leaf, active, {leaf: chain}, max_levels=levels)
    # If we stayed at leaf (not in active), treat leaf as its own basis node.
    if not basis:
        basis = leaf
    return basis, _label_for(basis)


def annotate_candidate_foodon(
    cand: dict[str, Any],
    problem: dict[str, Any],
    *,
    max_levels: int | None = None,
) -> dict[str, Any]:
    """Attach real FoodOn leaf/basis onto a candidate (mutates a copy)."""
    out = dict(cand)
    meta = dict(out.get("meta") or {})
    label = str(out.get("label") or meta.get("label") or "")
    leaf = meta.get("foodon_leaf_id") or meta.get("foodon_id") or out.get("foodon_id")
    leaf_label = meta.get("foodon_leaf_label")
    if not leaf:
        hit = lookup_foodon_for_label(label)
        if hit:
            leaf = hit["foodon_leaf_id"]
            leaf_label = hit.get("foodon_leaf_label")
    # Never keep the synthetic placeholder
    if str(meta.get("basis_node") or "") in {"ood_lean_protein", "ood_protein"}:
        meta.pop("basis_node", None)
    basis, basis_label = resolve_basis_for_leaf(str(leaf) if leaf else None, problem, max_levels=max_levels)
    if leaf:
        meta["foodon_leaf_id"] = str(leaf)
        meta["foodon_id"] = str(leaf)
        meta["foodon_leaf_label"] = leaf_label or _label_for(str(leaf))
    if basis:
        meta["basis_node"] = str(basis)
        meta["basis_node_label"] = basis_label or _label_for(str(basis))
    out["meta"] = meta
    if leaf:
        out["foodon_id"] = str(leaf)
    return out


def harvest_share_samples_for_nodes(
    recipe_ids: list[int],
    target_nodes: set[str],
    *,
    index=None,
    weight: float = 0.35,
) -> dict[str, list[float]]:
    """Mass-share samples for ``target_nodes`` from shell recipe FoodOn lines."""
    if not recipe_ids or not target_nodes:
        return {}
    try:
        from recipe_data_access import get_store

        store = get_store()
        rr = store.resolved_recipes([int(r) for r in recipe_ids])
        if rr is None or rr.empty:
            return {}
        rr = rr[rr["fdc_id"].notna() & rr["gram_weight"].notna()].copy()
        if rr.empty:
            return {}
        rr["fdc_id"] = rr["fdc_id"].astype(int)
        foodon = store.food_4macro_foodon(rr["fdc_id"].unique().tolist())
        keep = [c for c in ("fdc_id", "foodon_id") if c in foodon.columns]
        foodon = foodon[keep].dropna(subset=["foodon_id"]).drop_duplicates("fdc_id")
        foodon["fdc_id"] = foodon["fdc_id"].astype(int)
        merged = rr.merge(foodon, on="fdc_id", how="inner")
        if merged.empty:
            return {}
    except Exception:
        return {}

    if index is None:
        try:
            from canonical_optimization import _get_index

            index = _get_index()
        except Exception:
            index = None

    target = {str(n) for n in target_nodes}
    out: dict[str, list[float]] = {n: [] for n in target}

    def rollup(leaf: str) -> str | None:
        if leaf in target:
            return leaf
        if index is None:
            return None
        for anc in (leaf, *index.ancestry_path(leaf)):
            if anc in target:
                return anc
        return None

    for _, sub in merged.groupby("recipe_id"):
        total = float(sub["gram_weight"].sum())
        if total <= 0:
            continue
        grams: dict[str, float] = {}
        for _, line in sub.iterrows():
            leaf = str(line["foodon_id"])
            node = rollup(leaf)
            if not node:
                continue
            grams[node] = grams.get(node, 0.0) + float(line["gram_weight"])
        for node, g in grams.items():
            out.setdefault(node, []).append(g / total)

    return {k: v for k, v in out.items() if v}


def ensure_ingredient_nodes_in_loss(
    problem: dict[str, Any],
    *,
    min_hits: int = MIN_BASIS_HITS_FOR_LOSS,
) -> dict[str, Any]:
    """Make sure every ingredient basis node is in marginal_nodes and has samples when possible.

    Uses query-shell recipe ids on retrieval_context to harvest more share hits.
    """
    problem = dict(problem)
    basis_list = [str(n) for n in (problem.get("ingredient_basis") or []) if n]
    needed = set(basis_list)
    # Pending nodes: FoodOn basis of ingredients the agent is *considering* adding
    # (announced pre-expansion) so their share samples exist before the LP evaluates.
    for n in (problem.get("retrieval_context") or {}).get("pending_basis_nodes") or []:
        if n:
            needed.add(str(n))
    samples = {str(k): list(map(float, v)) for k, v in (problem.get("basis_samples") or {}).items()}
    weights = {
        str(k): list(map(float, v)) for k, v in (problem.get("basis_sample_weights") or {}).items()
    }

    short = {n for n in needed if len(samples.get(n) or []) < min_hits}
    ctx = problem.get("retrieval_context") or {}
    # Prefer structure-verified shell ids (pass/soft gram-share gate). When the
    # key is present (even empty), do NOT fall back to unverified shell recipes —
    # that would re-poison ratio loss with stretch-primary sides.
    verified_raw = ctx.get("structure_verified_shell_ids")
    shell_ids: list[int] = []
    if verified_raw is not None:
        for rid in verified_raw:
            if str(rid).isdigit():
                shell_ids.append(int(rid))
    else:
        for row in ctx.get("query_shell_recipes") or []:
            rid = row.get("recipe_id")
            if rid is not None and str(rid).isdigit():
                shell_ids.append(int(rid))
        for rid in problem.get("shell_recipe_ids") or []:
            if str(rid).isdigit():
                shell_ids.append(int(rid))
    shell_ids = sorted(set(shell_ids))

    if short and shell_ids:
        harvested = harvest_share_samples_for_nodes(shell_ids, short)
        for nid, vals in harvested.items():
            prev = list(samples.get(nid) or [])
            prev_w = list(weights.get(nid) or [1.0] * len(prev))
            samples[nid] = prev + vals
            weights[nid] = prev_w + [0.35] * len(vals)

    # Still short? Try a broader harvest over all target nodes from the same
    # (structure-verified when available) shell id set.
    still_short = {n for n in needed if len(samples.get(n) or []) < min_hits}
    if still_short and shell_ids:
        harvested = harvest_share_samples_for_nodes(shell_ids, still_short | needed)
        for nid, vals in harvested.items():
            if nid not in still_short:
                continue
            prev = list(samples.get(nid) or [])
            prev_w = list(weights.get(nid) or [1.0] * len(prev))
            samples[nid] = prev + vals
            weights[nid] = prev_w + [0.35] * len(vals)

    # Only nodes with enough share samples become loss terms. An under-sampled
    # node (e.g. 2 hits from shell harvest) produces a noisy RED zone that can
    # unfairly flip a good OOD edit into must_retry.
    marginal = list(problem.get("marginal_nodes") or [])
    for n in needed:
        if n in marginal:
            continue
        if len(samples.get(n) or []) >= min_hits:
            marginal.append(n)
    # Drop previously-registered nodes that still lack sample support.
    marginal = [n for n in marginal if len(samples.get(str(n)) or []) >= min_hits or str(n) not in needed]

    problem["basis_samples"] = samples
    problem["basis_sample_weights"] = weights
    problem["marginal_nodes"] = marginal
    problem["basis_hit_counts"] = {n: len(samples.get(n) or []) for n in needed}
    return problem
