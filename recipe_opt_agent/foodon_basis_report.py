"""FoodOn rollup / basis-hit reports for assessing neighborhood + ratio-loss quality.

For a recipe problem, records:
- which ingredients were aggregated up the FoodOn tree and by how many levels
- the active basis nodes and neighborhood hit counts per node
"""

from __future__ import annotations

from typing import Any


def _label_for(node_id: str | None) -> str | None:
    if not node_id:
        return None
    try:
        from canonical_optimization import foodon_display_label

        return foodon_display_label(str(node_id)) or str(node_id)
    except Exception:
        return str(node_id)


def aggregation_levels(
    leaf_id: str | None,
    basis_node_id: str | None,
    rollup_chains: dict[str, Any],
) -> int | None:
    """Hops from leaf → basis along the rollup chain (0 = leaf is already basis)."""
    if not leaf_id or not basis_node_id:
        return None
    chain = rollup_chains.get(str(leaf_id)) or rollup_chains.get(leaf_id)
    if not chain:
        return 0 if str(leaf_id) == str(basis_node_id) else None
    seq = [str(x) for x in chain]
    try:
        return int(seq.index(str(basis_node_id)))
    except ValueError:
        return None


def basis_hit_counts(problem: dict[str, Any]) -> dict[str, int]:
    """Neighborhood recipe hits per basis node (len of share samples)."""
    samples = problem.get("basis_samples") or {}
    out: dict[str, int] = {}
    for nid, vals in samples.items():
        try:
            out[str(nid)] = int(len(vals))
        except Exception:
            out[str(nid)] = 0
    # Also include neighborhood-wide basis nodes with 0 hits if listed.
    for nid in problem.get("basis_nodes") or []:
        out.setdefault(str(nid), 0)
    return out


def build_foodon_basis_report(problem: dict[str, Any] | None) -> dict[str, Any]:
    """Compact report attached to problems / pool entries / bundle next_problems."""
    problem = problem or {}
    ingredients = list((problem.get("chosen_recipe") or {}).get("ingredients") or [])
    basis_list = list(problem.get("ingredient_basis") or [])
    leaves = list(problem.get("ingredient_foodon_leaves") or [])
    chains = problem.get("rollup_chains") or {}
    if not isinstance(chains, dict):
        chains = {}
    # Normalize chain values to lists for JSON friendliness
    chains_norm = {
        str(k): [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
        for k, v in chains.items()
    }

    # Fill leaf ids from ingredients if parallel list missing / short
    while len(leaves) < len(ingredients):
        row = ingredients[len(leaves)]
        leaves.append(row.get("foodon_id") or row.get("foodon_leaf_id"))
    while len(basis_list) < len(ingredients):
        basis_list.append(None)

    hit_counts = basis_hit_counts(problem)
    build_params = problem.get("build_params") or {}
    if not isinstance(build_params, dict):
        build_params = {}

    ing_rows: list[dict[str, Any]] = []
    n_aggregated = 0
    n_unmapped = 0
    mapped_counts: dict[str, int] = {}
    for i, row in enumerate(ingredients):
        leaf = leaves[i] if i < len(leaves) else None
        leaf = str(leaf) if leaf else None
        basis = basis_list[i] if i < len(basis_list) else None
        basis = str(basis) if basis else None
        levels = aggregation_levels(leaf, basis, chains_norm)
        # Catalog adds often know basis but not leaf — treat as 0-level (no rollup).
        if levels is None and basis and not leaf:
            levels = 0
            leaf = basis
        if basis is None:
            n_unmapped += 1
        elif levels is not None and levels > 0:
            n_aggregated += 1
        if basis:
            mapped_counts[basis] = mapped_counts.get(basis, 0) + 1
        path = None
        if leaf and basis and leaf in chains_norm:
            chain = chains_norm[leaf]
            if basis in chain:
                path = chain[: chain.index(basis) + 1]
        ing_rows.append(
            {
                "index": i,
                "label": row.get("label") or row.get("name"),
                "fdc_id": row.get("fdc_id"),
                "foodon_leaf_id": leaf,
                "foodon_leaf_label": _label_for(leaf),
                "basis_node_id": basis,
                "basis_node_label": _label_for(basis),
                "aggregation_levels": levels,
                "rollup_path": path,
            }
        )

    # Prefer recipe-used basis nodes first, then remaining neighborhood basis nodes.
    used = {r["basis_node_id"] for r in ing_rows if r.get("basis_node_id")}
    all_basis_ids = sorted(set(hit_counts) | used | {str(x) for x in (problem.get("basis_nodes") or [])})
    basis_rows = []
    for nid in all_basis_ids:
        basis_rows.append(
            {
                "node_id": nid,
                "label": _label_for(nid),
                "n_hits": int(hit_counts.get(nid, 0)),
                "in_current_recipe": nid in used,
                "n_ingredients_mapped": int(mapped_counts.get(nid, 0)),
            }
        )
    # Stable: recipe-used first, then by hits desc
    basis_rows.sort(key=lambda r: (not r["in_current_recipe"], -r["n_hits"], r["node_id"]))

    return {
        "n_ingredients": len(ing_rows),
        "n_aggregated": n_aggregated,
        "n_unmapped": n_unmapped,
        "n_basis_nodes_in_recipe": len(used),
        "n_basis_nodes_neighborhood": len(problem.get("basis_nodes") or hit_counts),
        "neighborhood_n_recipes": problem.get("n_matches")
        or problem.get("neighborhood_n_recipes")
        or (problem.get("expansion_meta") or {}).get("n_core"),
        "min_basis_hits_target": build_params.get("adaptive_min_basis_hits")
        or build_params.get("min_basis_node_hits"),
        "ingredients": ing_rows,
        "basis_nodes": basis_rows,
        "aggregated_ingredients": [
            {
                "label": r["label"],
                "foodon_leaf_id": r["foodon_leaf_id"],
                "basis_node_id": r["basis_node_id"],
                "aggregation_levels": r["aggregation_levels"],
            }
            for r in ing_rows
            if (r.get("aggregation_levels") or 0) > 0
        ],
    }


def attach_foodon_basis_report(problem: dict[str, Any] | None) -> dict[str, Any] | None:
    """Mutate problem with a fresh ``foodon_basis_report``; return the report."""
    if not problem:
        return None
    report = build_foodon_basis_report(problem)
    problem["foodon_basis_report"] = report
    chosen = dict(problem.get("chosen_recipe") or {})
    if chosen:
        chosen["foodon_basis_report"] = report
        problem["chosen_recipe"] = chosen
    return report


def foodon_geometry_from_neighborhood(nb: Any) -> dict[str, Any]:
    """Extract rollup/basis geometry fields to embed on an agent problem dict."""
    rollup = getattr(nb, "rollup_chains", {}) or {}
    chains = {
        str(k): [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
        for k, v in rollup.items()
    }
    basis_nodes = sorted(str(x) for x in (getattr(nb, "basis_nodes", set()) or set()))
    leaves: list[str | None] = []
    leaf_by_fdc_idx: dict[tuple[int, int], str] = {}
    leaf_by_fdc: dict[int, str] = {}
    lines = getattr(nb, "lines_df", None)
    start_id = str(getattr(nb, "starting_recipe_id", "") or "")
    if lines is not None and hasattr(lines, "empty") and not lines.empty and start_id:
        sub = lines.loc[lines["recipe_nlg_id"] == start_id] if "recipe_nlg_id" in lines.columns else lines
        for _, row in sub.iterrows():
            try:
                fid = int(row["fdc_id"])
                leaf = str(row["foodon_id"])
                leaf_by_fdc_idx[(int(row["ingredient_idx"]), fid)] = leaf
                leaf_by_fdc.setdefault(fid, leaf)
            except Exception:
                continue
    starting = getattr(nb, "starting_ingredients", None)
    if starting is not None and hasattr(starting, "itertuples"):
        for row in starting.itertuples(index=False):
            try:
                fid = int(getattr(row, "fdc_id", 0) or 0)
                idx = int(getattr(row, "ingredient_idx", 0) or 0)
            except Exception:
                leaves.append(None)
                continue
            leaf = leaf_by_fdc_idx.get((idx, fid)) or leaf_by_fdc.get(fid)
            if leaf is None:
                leaf = getattr(row, "foodon_id", None) or getattr(row, "foodon_leaf_id", None)
                leaf = str(leaf) if leaf else None
            leaves.append(leaf)
    elif getattr(nb, "ingredient_basis", None):
        leaves = [None] * len(list(nb.ingredient_basis))

    build_params: dict[str, Any] = {}
    try:
        from canonical_optimization import adaptive_min_basis_hits

        build_params["adaptive_min_basis_hits"] = adaptive_min_basis_hits(
            int(getattr(nb, "n_recipes", 0) or 0)
        )
    except Exception:
        pass

    return {
        "rollup_chains": chains,
        "basis_nodes": basis_nodes,
        "ingredient_foodon_leaves": leaves,
        "build_params": build_params,
        "n_matches": int(getattr(nb, "n_recipes", 0) or 0),
    }
