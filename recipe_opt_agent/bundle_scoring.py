"""Bundle enumeration + joint LP scoring for multi-ingredient edit sets.

A bundle is 1..2 atomic edits (add / swap / remove), one per slot. Bundles are
proxy-ranked cheaply (share dilution + nutrient direction), then the top few
get a joint LP re-optimization on the edited problem — this is the true
"does this edit set give the solver a better optimum" score. Every scored
bundle carries a fully materialized ``next_problem`` so apply is atomic.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

MAX_BUNDLE_SIZE = 2
MAX_BUNDLES_PROXY = 50
MAX_BUNDLES_LP = 10
DEFAULT_ADD_MASS_FRACTION = 0.1


def _macro_col_for_candidate(cand: dict[str, Any]) -> np.ndarray | None:
    """Per-gram macro column [protein_g, fat_g, carbs_g, kcal] for a candidate."""
    meta = cand.get("meta") or {}
    macros = meta.get("macros_per_g")
    if macros is not None:
        col = np.asarray(macros, dtype=float).ravel()
        if col.size >= 3 and float(np.abs(col[:3]).sum()) > 0:
            if col.size < 4:
                col = np.pad(col, (0, 4 - col.size))
            return col[:4]
    pfc = meta.get("pfc") or cand.get("pfc")
    if pfc is not None:
        p, c, f = (float(v) for v in np.asarray(pfc, dtype=float).ravel()[:3])
        # PFC calorie fractions → per-gram macro grams at ~1 kcal/g density.
        col = np.array([p / 4.0, f / 9.0, c / 4.0, 1.0], dtype=float)
        if float(np.abs(col[:3]).sum()) > 0:
            return col
    return None


def apply_edits_to_problem(
    problem: dict[str, Any],
    edits: list[dict[str, Any]],
    *,
    add_mass_fraction: float = DEFAULT_ADD_MASS_FRACTION,
) -> dict[str, Any] | None:
    """Materialize a next_problem dict for an edit set.

    Returns None when an edit cannot be applied (e.g. add without macro data).
    Keeps total_mass constant; the LP re-optimizes grams afterwards.
    """
    x0 = np.asarray(problem.get("x0") or [], dtype=float).copy()
    M = np.asarray(problem.get("M") or [], dtype=float)
    if M.ndim != 2 or M.size == 0 or x0.size == 0 or M.shape[1] != x0.size:
        return None
    M = M.copy()
    basis = list(problem.get("ingredient_basis") or [None] * x0.size)
    total_mass = float(problem.get("total_mass") or x0.sum())
    chosen = dict(problem.get("chosen_recipe") or {})
    ingredients = [dict(r) for r in (chosen.get("ingredients") or [])]

    remove_idxs: list[int] = []
    adds: list[tuple[dict[str, Any], np.ndarray]] = []

    for edit in edits:
        action = str(edit.get("action") or "")
        meta = edit.get("meta") or {}
        if action == "remove":
            idx = meta.get("remove_idx")
            if idx is None:
                return None
            remove_idxs.append(int(idx))
        elif action == "swap":
            idx = meta.get("swap_out_idx")
            if idx is None:
                return None
            remove_idxs.append(int(idx))
            col = _macro_col_for_candidate(edit)
            if col is None:
                return None
            adds.append((edit, col))
        elif action == "add":
            col = _macro_col_for_candidate(edit)
            if col is None:
                return None
            adds.append((edit, col))
        else:
            return None

    if len(set(remove_idxs)) != len(remove_idxs):
        return None  # two edits removing the same line
    for idx in remove_idxs:
        if idx < 0 or idx >= x0.size:
            return None

    keep = [i for i in range(x0.size) if i not in set(remove_idxs)]
    x_new = x0[keep]
    M_new = M[:, keep]
    basis_new = [basis[i] for i in keep]
    leaves = list(problem.get("ingredient_foodon_leaves") or [])
    while len(leaves) < x0.size:
        leaves.append(None)
    leaves_new = [leaves[i] for i in keep]
    ing_new = [ingredients[i] for i in keep if i < len(ingredients)] if ingredients else []

    add_grams = max(add_mass_fraction * total_mass, 1.0)
    fdc_basis = (problem.get("retrieval_context") or {}).get("fdc_basis") or {}
    for edit, col in adds:
        M_new = np.hstack([M_new, col.reshape(4, 1)])
        x_new = np.concatenate([x_new, [add_grams]])
        meta = edit.get("meta") or {}
        basis_node = meta.get("basis_node")
        fdc_id = edit.get("fdc_id")
        if basis_node is None and fdc_id is not None:
            basis_node = fdc_basis.get(str(fdc_id))
        basis_new.append(basis_node)
        # Best-effort leaf: if fdc maps to a rollup chain key via fdc_basis reverse, leave None
        # and let report treat basis-only rows as levels unknown/0 when leaf missing.
        leaf = meta.get("foodon_leaf_id") or meta.get("foodon_id")
        leaves_new.append(str(leaf) if leaf else None)
        ing_new.append(
            {
                "label": edit.get("label"),
                "grams": float(add_grams),
                "fdc_id": edit.get("fdc_id"),
                "added_by": edit.get("candidate_id"),
                "foodon_id": leaf,
            }
        )

    if x_new.size == 0:
        return None
    if x_new.sum() > 0:
        x_new = x_new * (total_mass / float(x_new.sum()))
    for row, grams in zip(ing_new, x_new, strict=False):
        row["grams"] = float(grams)

    chosen["ingredients"] = ing_new
    ctx = dict(problem.get("retrieval_context") or {})
    if ctx:
        ctx["starting_ingredients"] = ing_new
        ctx["starting_fdc"] = [int(r["fdc_id"]) for r in ing_new if r.get("fdc_id") is not None]
        ctx["starting_labels"] = [str(r.get("label") or "").lower() for r in ing_new if r.get("label")]

    next_problem = {
        **problem,
        "x0": x_new.tolist(),
        "M": M_new.tolist(),
        "ingredient_basis": basis_new,
        "ingredient_foodon_leaves": leaves_new,
        "chosen_recipe": chosen,
        "retrieval_context": ctx or problem.get("retrieval_context"),
        "last_applied_bundle": [e.get("candidate_id") for e in edits],
    }
    next_problem.pop("x_opt", None)
    # Resolve real FoodOn for any leftover placeholders and enrich share samples
    from recipe_opt_agent.ood_foodon import annotate_candidate_foodon, ensure_ingredient_nodes_in_loss

    fixed_basis = list(basis_new)
    fixed_leaves = list(leaves_new)
    n_old = len(keep)
    for j, (edit, _col) in enumerate(adds):
        idx = n_old + j
        annotated = annotate_candidate_foodon(edit, next_problem)
        meta = annotated.get("meta") or {}
        if meta.get("basis_node"):
            fixed_basis[idx] = str(meta["basis_node"])
        if meta.get("foodon_leaf_id"):
            fixed_leaves[idx] = str(meta["foodon_leaf_id"])
            if idx < len(ing_new):
                ing_new[idx]["foodon_id"] = str(meta["foodon_leaf_id"])
    next_problem["ingredient_basis"] = fixed_basis
    next_problem["ingredient_foodon_leaves"] = fixed_leaves
    next_problem["chosen_recipe"] = chosen
    next_problem = ensure_ingredient_nodes_in_loss(next_problem, min_hits=5)

    from recipe_opt_agent.foodon_basis_report import attach_foodon_basis_report

    attach_foodon_basis_report(next_problem)
    return next_problem


def enumerate_bundles(
    per_slot: dict[str, list[dict[str, Any]]],
    *,
    max_size: int = MAX_BUNDLE_SIZE,
    cap: int = MAX_BUNDLES_PROXY,
) -> list[list[dict[str, Any]]]:
    """Size 1..max_size bundles: one candidate per slot, pruned for conflicts."""
    slot_ids = [sid for sid, cands in per_slot.items() if cands]
    bundles: list[list[dict[str, Any]]] = []

    # Singletons from every slot.
    for sid in slot_ids:
        for cand in per_slot[sid]:
            bundles.append([cand])

    # Pairs across distinct slots.
    if max_size >= 2 and len(slot_ids) >= 2:
        for sa, sb in itertools.combinations(slot_ids, 2):
            for ca, cb in itertools.product(per_slot[sa][:5], per_slot[sb][:5]):
                if _conflicts(ca, cb):
                    continue
                bundles.append([ca, cb])

    return bundles[: max(1, int(cap))]


def _conflicts(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ma, mb = a.get("meta") or {}, b.get("meta") or {}
    # Same line targeted twice (remove/swap collisions).
    ta = ma.get("remove_idx", ma.get("swap_out_idx"))
    tb = mb.get("remove_idx", mb.get("swap_out_idx"))
    if ta is not None and tb is not None and int(ta) == int(tb):
        return True
    # Same food added twice.
    if a.get("action") in {"add", "swap"} and b.get("action") in {"add", "swap"}:
        if a.get("fdc_id") is not None and a.get("fdc_id") == b.get("fdc_id"):
            return True
        if str(a.get("label") or "").lower() == str(b.get("label") or "").lower():
            return True
    return False


def _proxy_score(bundle: list[dict[str, Any]]) -> float:
    """Lower is better: sum of ratio proxies minus nutrient alignment."""
    score = 0.0
    for cand in bundle:
        meta = cand.get("meta") or {}
        score += float(meta.get("delta_ratio_proxy") or 0.0)
        score -= float(meta.get("delta_nutrient_proxy") or 0.0)
    return score


def _lp_eval(problem: dict[str, Any], box_dict: dict[str, float]) -> dict[str, Any] | None:
    """Joint LP on a problem dict; returns objective + decomposition or None."""
    from recipe_opt_agent.kcal_utils import restore_kcal_target
    from weighted_empirical_opt import (
        optimize_weighted_empirical_obj,
        pfc_fractions_from_portions,
        term_losses,
    )

    problem = restore_kcal_target(problem)
    x0 = np.asarray(problem.get("x0") or [], dtype=float)
    M = np.asarray(problem.get("M") or [], dtype=float)
    if x0.size == 0 or M.ndim != 2 or M.shape[1] != x0.size:
        return None
    basis = list(problem.get("ingredient_basis") or [None] * x0.size)
    basis_samples = {
        k: np.asarray(v, dtype=float) for k, v in (problem.get("basis_samples") or {}).items()
    }
    ratio_samples = np.asarray(problem.get("ratio_samples") or [], dtype=float)
    marginal_nodes = list(problem.get("marginal_nodes") or [])
    total_mass = float(problem.get("total_mass") or x0.sum())
    kcal_target = float(problem.get("kcal_target") or 0.0)
    if kcal_target <= 0:
        from weighted_empirical_opt import atwater_kcal

        kcal_target = float(atwater_kcal(x0, M))

    try:
        opt = optimize_weighted_empirical_obj(
            x0,
            M,
            marginal_nodes=marginal_nodes,
            basis_samples=basis_samples,
            ratio_samples=ratio_samples,
            ingredient_basis=basis,
            kcal_target=kcal_target,
            protein_frac_min=box_dict["protein_min"],
            protein_frac_max=box_dict["protein_max"],
            carb_frac_min=box_dict["carb_min"],
            carb_frac_max=box_dict["carb_max"],
            fat_frac_min=box_dict["fat_min"],
            fat_frac_max=box_dict["fat_max"],
            total_mass=total_mass,
            nutrition_slack_weight=box_dict.get("nutrition_slack_weight"),
        )
    except Exception:
        return None
    if not opt.get("feasible"):
        return {"feasible": False, "objective": None}

    x_opt = np.asarray(opt["x_opt"], dtype=float)
    tl = term_losses(
        x_opt,
        marginal_nodes=marginal_nodes,
        basis_samples=basis_samples,
        ratio_samples=ratio_samples,
        total_mass=total_mass,
        ingredient_basis=basis,
    )
    p, c, f = pfc_fractions_from_portions(x_opt, M)

    def _box_slack(v: float, lo: float, hi: float) -> float:
        return max(lo - v, 0.0) + max(v - hi, 0.0)

    nutrient_slack = (
        _box_slack(p, box_dict["protein_min"], box_dict["protein_max"])
        + _box_slack(c, box_dict["carb_min"], box_dict["carb_max"])
        + _box_slack(f, box_dict["fat_min"], box_dict["fat_max"])
    )
    ratio_term = float(tl.get("ratio_surrogate", 0.0) or 0.0) if "ratio_surrogate" in tl else None
    if ratio_term is None:
        # Only fall back to ratio_value when surrogate absent — UI treats values >=1.5 as non-loss
        ratio_term = float(tl.get("ratio_value", 0.0) or 0.0)
    return {
        "feasible": True,
        "objective": float(opt["objective"]),
        "x_opt": x_opt.tolist(),
        "ratio_term": ratio_term,
        "ratio_surrogate": float(tl["ratio_surrogate"]) if tl.get("ratio_surrogate") is not None else None,
        "ratio_value": float(tl.get("ratio_value", 0.0) or 0.0),
        "nutrient_slack": float(nutrient_slack),
        "pfc_after": {"protein": p, "carbs": c, "fat": f},
        "term_losses": {k: float(v) for k, v in tl.items()},
    }


def score_bundles(
    problem: dict[str, Any],
    per_slot: dict[str, list[dict[str, Any]]],
    *,
    box_dict: dict[str, float],
    max_bundle_size: int = MAX_BUNDLE_SIZE,
    proxy_cap: int = MAX_BUNDLES_PROXY,
    lp_cap: int = MAX_BUNDLES_LP,
) -> list[dict[str, Any]]:
    """Enumerate → proxy-rank → joint LP top bundles. Returns scored bundle dicts."""
    bundles = enumerate_bundles(per_slot, max_size=max_bundle_size, cap=proxy_cap)
    if not bundles:
        return []

    baseline = _lp_eval(problem, box_dict)
    l_before = None if baseline is None else baseline.get("objective")

    ranked = sorted(bundles, key=_proxy_score)
    scored: list[dict[str, Any]] = []
    for i, bundle in enumerate(ranked):
        entry: dict[str, Any] = {
            "bundle_id": f"b{i}",
            "edits": [
                {
                    "action": c.get("action"),
                    "candidate_id": c.get("candidate_id"),
                    "label": c.get("label"),
                    "replace_label": c.get("replace_label")
                    or (c.get("meta") or {}).get("swap_out_label"),
                    "slot_id": (c.get("meta") or {}).get("slot_id"),
                }
                for c in bundle
            ],
            "size": len(bundle),
            "proxy_score": float(_proxy_score(bundle)),
            "L_star_before": l_before,
            "L_star_after": None,
            "delta_L_star": None,
            "ratio_term": None,
            "nutrient_slack": None,
            "churn_est": float(sum(1 for c in bundle if c.get("action") != "add")) / max(
                len(list((problem.get("chosen_recipe") or {}).get("ingredients") or [])), 1
            ),
            "lp_evaluated": False,
            "next_problem": None,
        }
        if i < lp_cap:
            next_problem = apply_edits_to_problem(problem, bundle)
            if next_problem is not None:
                result = _lp_eval(next_problem, box_dict)
                if result is not None and result.get("feasible"):
                    entry["lp_evaluated"] = True
                    entry["L_star_after"] = result["objective"]
                    entry["ratio_term"] = result["ratio_term"]
                    entry["ratio_surrogate"] = result.get("ratio_surrogate")
                    entry["nutrient_slack"] = result["nutrient_slack"]
                    entry["pfc_after"] = result.get("pfc_after")
                    if result.get("term_losses"):
                        entry["opt"] = {
                            "objective": result["objective"],
                            "pfc_after": result.get("pfc_after"),
                            "term_losses": result["term_losses"],
                            "x_opt": result.get("x_opt"),
                            "feasible": True,
                        }
                    if l_before is not None and result["objective"] is not None:
                        entry["delta_L_star"] = float(result["objective"] - l_before)
                    entry["next_problem"] = next_problem
        scored.append(entry)

    # Best (most negative delta) first among LP-evaluated; proxies after.
    scored.sort(
        key=lambda e: (
            not e["lp_evaluated"],
            e["delta_L_star"] if e["delta_L_star"] is not None else float("inf"),
            e["proxy_score"],
        )
    )
    return scored
