"""Out-of-distribution (OOD) branch for high macro demand.

When the target protein box is aggressive relative to the current recipe /
neighborhood, open a parallel branch that:

1. Asks the LLM to justify OOD and propose culinary ideas + neighborhood search
   queries (so dishes like chicken+pasta enter the working neighborhood).
2. Expands retrieval co-occurrence via those queries.
3. Grounds ideas to FDC and scores them with the same LP as in-distribution
   bundles (hybrids included).

Hardcoded lean-protein foods remain a fallback when ideation finds nothing.
"""

from __future__ import annotations

from typing import Any

# Per-gram macros: [protein, fat, carbs, energy_kcal] approx USDA-ish.
OOD_PROTEIN_FOODS: list[dict[str, Any]] = [
    {
        "fdc_id": 900001,
        "label": "Chicken breast, raw, skinless",
        "fdc_description": "Chicken breast, raw, skinless",
        "role": "lean_protein",
        "ood": True,
        "macros_per_g": [0.31, 0.036, 0.0, 1.65],
    },
    {
        "fdc_id": 900002,
        "label": "Turkey breast, raw",
        "fdc_description": "Turkey breast, raw",
        "role": "lean_protein",
        "ood": True,
        "macros_per_g": [0.30, 0.02, 0.0, 1.35],
    },
    {
        "fdc_id": 900003,
        "label": "Tofu, firm",
        "fdc_description": "Tofu, firm, prepared with calcium sulfate",
        "role": "lean_protein",
        "ood": True,
        "macros_per_g": [0.17, 0.09, 0.03, 1.44],
    },
    {
        "fdc_id": 900004,
        "label": "Egg white, raw",
        "fdc_description": "Egg white, raw, fresh",
        "role": "lean_protein",
        "ood": True,
        "macros_per_g": [0.11, 0.002, 0.007, 0.52],
    },
    {
        "fdc_id": 900005,
        "label": "Greek yogurt, nonfat",
        "fdc_description": "Yogurt, Greek, plain, nonfat",
        "role": "lean_protein",
        "ood": True,
        "macros_per_g": [0.10, 0.004, 0.036, 0.59],
    },
]


def protein_demand_high(
    *,
    protein_min: float,
    pfc_after: dict[str, Any] | None = None,
    binding_macros: list[str] | None = None,
    requirement_tags: list[Any] | None = None,
    protein_gap_eps: float = 0.02,
) -> tuple[bool, str]:
    """Whether to open an OOD protein branch."""
    tags = requirement_tags or []
    for t in tags:
        tid = (t.get("tag_id") if isinstance(t, dict) else getattr(t, "tag_id", "")) or ""
        if str(tid).lower() in {"high_protein", "high-protein"}:
            return True, "requirement_tag:high_protein"
    if protein_min >= 0.28:
        return True, f"protein_min>={protein_min:.2f}"
    binding = {str(x) for x in (binding_macros or [])}
    if "protein_min" in binding:
        return True, "binding:protein_min"
    pfc = pfc_after or {}
    try:
        p = float(pfc.get("protein"))
        if p + protein_gap_eps < protein_min:
            return True, f"protein_gap:{protein_min - p:.3f}"
    except (TypeError, ValueError):
        pass
    return False, ""


def ensure_ood_macros_on_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Inject OOD FDC macros + catalog entries into retrieval_context."""
    from recipe_opt_agent.ood_foodon import annotate_candidate_foodon

    problem = dict(problem)
    ctx = dict(problem.get("retrieval_context") or {})
    macros = dict(ctx.get("fdc_macros") or {})
    catalog = list(ctx.get("fdc_catalog") or [])
    existing_ids = {int(c["fdc_id"]) for c in catalog if c.get("fdc_id") is not None}
    ood_catalog: list[dict[str, Any]] = []
    for food in OOD_PROTEIN_FOODS:
        fid = int(food["fdc_id"])
        macros[str(fid)] = [float(x) for x in food["macros_per_g"]]
        annotated = annotate_candidate_foodon(
            {
                "fdc_id": fid,
                "label": food["label"],
                "meta": {"ood": True, "role": food.get("role")},
            },
            problem,
        )
        meta = annotated.get("meta") or {}
        row = {
            "fdc_id": fid,
            "fdc_description": food["fdc_description"],
            "ood": True,
            "role": food.get("role"),
            "foodon_id": meta.get("foodon_leaf_id"),
            "basis_node": meta.get("basis_node"),
        }
        ood_catalog.append(row)
        if fid not in existing_ids:
            catalog.append(row)
            existing_ids.add(fid)
        if meta.get("basis_node"):
            fdc_basis = dict(ctx.get("fdc_basis") or {})
            fdc_basis[str(fid)] = str(meta["basis_node"])
            ctx["fdc_basis"] = fdc_basis
    ctx["fdc_macros"] = macros
    ctx["fdc_catalog"] = catalog
    ctx["ood_protein_catalog"] = ood_catalog
    problem["retrieval_context"] = ctx
    return problem


def build_ood_add_candidates(
    *,
    box_dict: dict[str, float],
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Fallback synthetic add candidates for the OOD protein slot."""
    ranked = sorted(
        OOD_PROTEIN_FOODS,
        key=lambda f: -float(f["macros_per_g"][0]),
    )[:top_k]
    out: list[dict[str, Any]] = []
    for i, food in enumerate(ranked):
        out.append(
            {
                "candidate_id": f"ood_add_{food['fdc_id']}",
                "action": "add",
                "label": food["label"],
                "fdc_id": food["fdc_id"],
                "branch": "ood_protein",
                "meta": {
                    "basis_node": None,  # filled by annotate_candidate_foodon
                    "ood": True,
                    "role": food.get("role"),
                    "macros_per_g": list(food["macros_per_g"]),
                    "delta_nutrient_proxy": float(food["macros_per_g"][0]),
                    "delta_ratio_proxy": 0.02,
                    "reason": "OOD lean protein fallback catalog",
                    "protein_frac_mid": 0.5 * (box_dict["protein_min"] + box_dict["protein_max"]),
                    "source": "ood_fallback_catalog",
                },
                "score": float(top_k - i),
            }
        )
    return out


def tag_bundles(bundles: list[dict[str, Any]], *, branch: str) -> list[dict[str, Any]]:
    out = []
    for b in bundles:
        bb = dict(b)
        bb["branch"] = branch
        bid = str(bb.get("bundle_id") or "")
        if branch and not bid.startswith(f"{branch}::"):
            bb["bundle_id"] = f"{branch}::{bid}" if bid else f"{branch}::bundle"
        out.append(bb)
    return out


def build_hybrid_bundles(
    id_bundles: list[dict[str, Any]],
    ood_bundles: list[dict[str, Any]],
    *,
    problem: dict[str, Any],
    box_dict: dict[str, float],
    max_hybrids: int = 4,
) -> list[dict[str, Any]]:
    """Combine top ID edit with top OOD add into jointly re-scored hybrids."""
    from recipe_opt_agent.bundle_scoring import apply_edits_to_problem, _lp_eval

    id_top = [b for b in id_bundles if b.get("edits") and not b.get("oscillation_blocked")][:3]
    ood_top = [b for b in ood_bundles if b.get("edits") and not b.get("oscillation_blocked")][:3]
    hybrids: list[dict[str, Any]] = []
    for ib in id_top:
        for ob in ood_top:
            edits = list(ib.get("edits") or []) + list(ob.get("edits") or [])
            labels = [str(e.get("label") or "").lower() for e in edits]
            if len(labels) != len(set(labels)):
                continue
            next_problem = apply_edits_to_problem(problem, edits)
            if next_problem is None:
                continue
            result = _lp_eval(next_problem, box_dict)
            if not result or not result.get("feasible"):
                continue
            L_before = None
            for src in (ib, ob):
                if src.get("L_star_before") is not None:
                    L_before = src["L_star_before"]
                    break
            L_after = result["objective"]
            delta = None if L_before is None else float(L_after) - float(L_before)
            hybrids.append(
                {
                    "bundle_id": f"hybrid::{ib.get('bundle_id')}++{ob.get('bundle_id')}",
                    "branch": "hybrid",
                    "edits": edits,
                    "lp_evaluated": True,
                    "L_star_before": L_before,
                    "L_star_after": L_after,
                    "delta_L_star": delta,
                    "ratio_term": result.get("ratio_term"),
                    "ratio_surrogate": result.get("ratio_surrogate"),
                    "nutrient_slack": result.get("nutrient_slack"),
                    "pfc_after": result.get("pfc_after"),
                    "next_problem": next_problem,
                    "parents": {
                        "in_distribution": ib.get("bundle_id"),
                        "ood_protein": ob.get("bundle_id"),
                    },
                }
            )
            if len(hybrids) >= max_hybrids:
                return hybrids
    return hybrids


def maybe_build_ood_branch(
    problem: dict[str, Any],
    *,
    box_dict: dict[str, float],
    diagnosis: dict[str, Any] | None,
    opt: dict[str, Any] | None,
    requirement_tags: list[Any] | None,
    id_bundles: list[dict[str, Any]],
    ideation_context: dict[str, Any] | None = None,
    ideation_model: str = "gpt-4.1-mini",
    n_ideas: int = 8,
) -> dict[str, Any]:
    """Return `{needed, reason, ood_candidates, ood_bundles, hybrid_bundles, problem, …}`."""
    from recipe_opt_agent.bundle_scoring import score_bundles
    from recipe_opt_agent.ideation import ground_ideas_to_candidates, ideate_ingredient_edits
    from recipe_opt_agent.neighborhood_query_expand import expand_neighborhood_by_queries

    needed, reason = protein_demand_high(
        protein_min=float(box_dict.get("protein_min") or 0.0),
        pfc_after=(opt or {}).get("pfc_after"),
        binding_macros=(diagnosis or {}).get("binding_macros"),
        requirement_tags=requirement_tags,
    )
    empty = {
        "needed": False,
        "reason": reason,
        "ood_candidates": [],
        "ood_bundles": [],
        "hybrid_bundles": [],
        "problem": problem,
        "ideation": None,
        "neighborhood_expansion": None,
    }
    if not needed:
        return empty

    problem = ensure_ood_macros_on_problem(problem)
    ctx_in = {
        **(ideation_context or {}),
        "diagnosis": diagnosis or {},
        "opt": opt or {},
        "macro_box": box_dict,
        "requirement_tags": requirement_tags or [],
        "current_ingredients": (problem.get("chosen_recipe") or {}).get("ingredients") or [],
    }
    ideation = ideate_ingredient_edits(ctx_in, model=ideation_model, n_ideas=n_ideas)
    queries = list(ideation.get("neighborhood_search_queries") or [])
    # Prefer queries attached to OOD ideas
    for idea in ideation.get("ideas") or []:
        if str(idea.get("branch") or "").startswith("ood"):
            for q in idea.get("neighborhood_search_queries") or []:
                if q and q not in queries:
                    queries.append(q)

    # Pre-announce pending FoodOn basis nodes so shell harvest targets them
    from recipe_opt_agent.ood_foodon import annotate_candidate_foodon, ensure_ingredient_nodes_in_loss

    pending_nodes: list[str] = []
    for idea in ideation.get("ideas") or []:
        stub = annotate_candidate_foodon(
            {"label": idea.get("ingredient"), "meta": {"ood": True}, "branch": idea.get("branch")},
            problem,
        )
        bn = (stub.get("meta") or {}).get("basis_node")
        if bn:
            pending_nodes.append(str(bn))
    ctx = dict(problem.get("retrieval_context") or {})
    ctx["pending_basis_nodes"] = pending_nodes
    problem["retrieval_context"] = ctx

    # Focus terms = the stretch ingredients themselves (chicken, turkey, …) so
    # the expansion retrieves recipes that actually CONTAIN them alongside the
    # dish identity — not more copies of the base dish.
    from recipe_opt_agent.neighborhood_query_expand import (
        derive_focus_terms,
        fallback_dish_structure,
        normalize_dish_structure,
    )

    current_labels = [
        str(r.get("label") or r.get("name") or "")
        for r in ((problem.get("chosen_recipe") or {}).get("ingredients") or [])
    ]
    idea_ingredient_texts = [
        str(idea.get("ingredient") or "")
        for idea in (ideation.get("ideas") or [])
        if str(idea.get("branch") or "").startswith("ood")
    ]
    title_texts = [str((ideation_context or {}).get("title") or "")]
    # Ingredient names are the authoritative stretch terms; query tokens only as
    # fallback (queries also contain identity words like "pasta" that would let
    # plain base-dish recipes through the focus gate).
    focus_terms = derive_focus_terms(idea_ingredient_texts, current_labels, extra_texts=title_texts)
    if not focus_terms:
        focus_terms = derive_focus_terms(queries, current_labels, extra_texts=title_texts)

    dish_structure = normalize_dish_structure(ideation.get("dish_structure"))
    if dish_structure is None:
        for idea in ideation.get("ideas") or []:
            if str(idea.get("branch") or "").startswith("ood") and isinstance(
                idea.get("dish_structure"), dict
            ):
                dish_structure = normalize_dish_structure(idea.get("dish_structure"))
                if dish_structure:
                    break
    if dish_structure is None:
        stretch = idea_ingredient_texts[0] if idea_ingredient_texts else None
        dish_structure = fallback_dish_structure(
            stretch_ingredient=stretch,
            identity_roles=list((ideation_context or {}).get("identity_roles") or []),
            current_labels=current_labels,
            stretch_role="accent",
        )

    expansion = expand_neighborhood_by_queries(
        problem,
        queries,
        focus_terms=focus_terms,
        dish_structure=dish_structure,
    )
    problem = expansion.get("problem") or problem

    ood_ideas = [
        idea
        for idea in (ideation.get("ideas") or [])
        if str(idea.get("branch") or "").startswith("ood") or idea.get("action") == "add"
    ]
    ood_cands = ground_ideas_to_candidates(
        ood_ideas,
        problem=problem,
        requirement_tags=requirement_tags,
        box_dict=box_dict,
    )
    # Keep only OOD-tagged grounded candidates; fall back to catalog
    ood_cands = [c for c in ood_cands if (c.get("branch") or "").startswith("ood") or (c.get("meta") or {}).get("ood")]
    if not ood_cands:
        ood_cands = build_ood_add_candidates(box_dict=box_dict)

    ood_cands = [annotate_candidate_foodon(c, problem) for c in ood_cands]
    for c in ood_cands:
        c["branch"] = c.get("branch") or "ood_protein"
        meta = dict(c.get("meta") or {})
        meta["ood"] = True
        c["meta"] = meta

    problem = ensure_ingredient_nodes_in_loss(problem, min_hits=5)

    per_slot = {"ood_protein_macro": ood_cands}
    try:
        raw = score_bundles(problem, per_slot, box_dict=box_dict)
    except Exception:
        raw = []
    ood_bundles = tag_bundles(raw, branch="ood_protein")
    for b in ood_bundles:
        for e in b.get("edits") or []:
            e["branch"] = "ood_protein"
            meta = dict(e.get("meta") or {})
            meta["ood"] = True
            e["meta"] = meta
    hybrids = build_hybrid_bundles(id_bundles, ood_bundles, problem=problem, box_dict=box_dict)
    return {
        "needed": True,
        "reason": reason,
        "ood_candidates": ood_cands,
        "ood_bundles": ood_bundles,
        "hybrid_bundles": hybrids,
        "problem": problem,
        "ideation": {
            "ood_justification": ideation.get("ood_justification"),
            "n_ideas": len(ideation.get("ideas") or []),
            "queries": queries,
            "trace_mode": (ideation.get("_llm_trace") or {}).get("mode"),
        },
        "neighborhood_expansion": expansion.get("meta"),
        "ideation_trace": ideation.get("_llm_trace"),
    }
