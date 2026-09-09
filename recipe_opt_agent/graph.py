"""LangGraph wiring for diagnose → band → decide → modify/expand loop."""

from __future__ import annotations

import time
from typing import Any, Callable, Literal

import numpy as np
from langgraph.graph import END, StateGraph

from hull_geometry import TargetBox, region_intersects_hull
from opt_diagnosis import FidelityBand, diagnose_optimizer_result
from recipe_opt_agent.config import AgentConfig, identity_roles_for_title
from recipe_opt_agent.context_builder import build_decision_context
from recipe_opt_agent.identity_roles import resolve_identity_roles
from recipe_opt_agent.llm import decide_action_llm, judge_finalists_llm, llm_draft_recipe
from recipe_opt_agent.model_policy import (
    select_decide_model,
    select_draft_model,
    select_judge_model,
    select_shadow_draft_model,
    select_tags_model,
)
from recipe_opt_agent.requirement_tags import (
    RequirementTag,
    deduce_requirement_tags,
    filter_candidates_by_tags,
    ingredient_passes_tags,
    tag_violations_for_ingredient,
)
from recipe_opt_agent.state import AgentState
from recipe_opt_agent.telemetry import (
    bump_telemetry,
    clear_favorite_bundle,
    delta_needles,
    edit_fingerprint,
    empty_telemetry,
    finalize_telemetry,
    snapshot_needles,
)
from weighted_empirical_opt import (
    MARGINAL_COLUMN_NODES,
    optimize_weighted_empirical_obj,
    recipe_ratio,
    term_losses,
)


def _box_from_config(cfg: AgentConfig) -> TargetBox:
    return TargetBox(
        protein_min=cfg.protein_min,
        protein_max=cfg.protein_max,
        carb_min=cfg.carb_min,
        carb_max=cfg.carb_max,
        fat_min=cfg.fat_min,
        fat_max=cfg.fat_max,
    )


def _cfg(state: AgentState) -> AgentConfig:
    raw = state.get("config") or {}
    return AgentConfig(**{k: v for k, v in raw.items() if k in AgentConfig.__dataclass_fields__})


def _agent_mode(state: AgentState) -> str:
    return str(state.get("agent_mode") or _cfg(state).agent_mode or "neighborhood")


def _strip_private_keys(obj: Any) -> Any:
    """Drop underscore-prefixed keys so model authorship stays out of judge / UI payloads."""
    if isinstance(obj, dict):
        return {
            k: _strip_private_keys(v)
            for k, v in obj.items()
            if not str(k).startswith("_")
        }
    if isinstance(obj, list):
        return [_strip_private_keys(v) for v in obj]
    return obj


def _public_survivor(survivor: dict[str, Any] | Any) -> Any:
    if hasattr(survivor, "to_dict"):
        survivor = survivor.to_dict()
    if isinstance(survivor, dict):
        return _strip_private_keys(survivor)
    return survivor


def _requirement_tags(state: AgentState) -> list[RequirementTag]:
    raw = state.get("requirement_tags") or []
    out: list[RequirementTag] = []
    for r in raw:
        if isinstance(r, RequirementTag):
            out.append(r)
        elif isinstance(r, dict):
            out.append(
                RequirementTag(
                    tag_id=str(r.get("tag_id") or ""),
                    kind=str(r.get("kind") or "preference"),
                    polarity=str(r.get("polarity") or "require"),
                    source_text=str(r.get("source_text") or ""),
                )
            )
    return out


def node_init(state: AgentState) -> dict[str, Any]:
    cfg = _cfg(state)
    from recipe_opt_agent.foodon_basis_report import attach_foodon_basis_report
    from recipe_opt_agent.foodon_depth import apply_foodon_aggregation_cap

    problem = apply_foodon_aggregation_cap(
        dict(state.get("problem") or {}),
        max_levels=getattr(cfg, "max_foodon_aggregation_levels", None),
    )
    attach_foodon_basis_report(problem)
    title = state.get("title") or problem.get("title") or f"canonical #{state.get('canonical_id')}"
    # Dropdown / problem title stands in for semantic taste text.
    taste_text = state.get("taste_text") or problem.get("taste_text") or title
    request = state.get("user_request") or taste_text
    chosen = problem.get("chosen_recipe") or {
        "title": title,
        "ingredients": [],
        "selection_note": "No chosen_recipe metadata on problem.",
    }
    roles = resolve_identity_roles(
        title=title,
        request=request,
        ingredients=chosen.get("ingredients") or [],
        templates=cfg.identity_templates,
        use_llm=False,  # offline-safe at init; LLM merge on draft/ground when keyed
        model=cfg.identity_extract_model,
    )
    if not roles:
        roles = identity_roles_for_title(title, cfg.identity_templates)
    return {
        "title": title,
        "taste_text": taste_text,
        "agent_mode": _agent_mode(state),
        "user_request": request,
        "identity_roles": roles,
        "iteration": int(state.get("iteration") or 0),
        "history": list(state.get("history") or []),
        "candidate_pool": list(state.get("candidate_pool") or []),
        "decision_outcomes": list(state.get("decision_outcomes") or []),
        "recent_edit_fingerprints": list(state.get("recent_edit_fingerprints") or []),
        "run_telemetry": state.get("run_telemetry") or empty_telemetry(),
        "neighbor_k": int(state.get("neighbor_k") or cfg.neighbor_k),
        "status": "running",
        "error": None,
        "chosen_recipe": chosen,
        # Frozen snapshot of the starting recipe for final-arbiter intent diffs.
        "original_ingredients": [dict(r) for r in (chosen.get("ingredients") or [])],
        "neighborhood_recipes": list(problem.get("neighborhood_recipes") or []),
        "foodon_basis_report": problem.get("foodon_basis_report")
        or chosen.get("foodon_basis_report"),
        "problem": problem,
        "candidates": [],  # filled live in propose
        "tools_used": [
            {
                "name": "select_canonical_recipe",
                "purpose": "Use match-ranked dropdown selection as the semantic/taste input and load its FoodOn neighborhood",
                "output_summary": {
                    "title": chosen.get("title"),
                    "recipe_nlg_id": chosen.get("recipe_nlg_id"),
                    "n_ingredients": len(chosen.get("ingredients") or []),
                    "n_neighborhood": len(problem.get("neighborhood_recipes") or []),
                    "taste_text": taste_text,
                    "identity_roles": roles,
                    "foodon_n_aggregated": (problem.get("foodon_basis_report") or {}).get("n_aggregated"),
                    "max_foodon_aggregation_levels": (problem.get("build_params") or {}).get(
                        "max_foodon_aggregation_levels"
                    ),
                    "note": "modification candidates are retrieved later in propose (live)",
                },
                "output": {
                    "chosen_recipe": chosen,
                    "neighborhood_recipes": problem.get("neighborhood_recipes"),
                    "taste_text": taste_text,
                    "identity_roles": roles,
                    "foodon_basis_report": problem.get("foodon_basis_report")
                    or chosen.get("foodon_basis_report"),
                    "build_params": problem.get("build_params"),
                },
            }
        ],
    }


def node_diagnose(state: AgentState) -> dict[str, Any]:
    """Requires state['problem'] with x0, M, ingredient_basis, basis_samples, etc."""
    cfg = _cfg(state)
    box = _box_from_config(cfg)
    problem = state.get("problem") or {}
    x0 = np.asarray(problem["x0"], dtype=float)
    M = np.asarray(problem["M"], dtype=float)
    ingredient_basis = list(problem["ingredient_basis"])
    # Include every ingredient basis node in the loss (not only pasta/egg defaults).
    from recipe_opt_agent.ood_foodon import ensure_ingredient_nodes_in_loss

    problem = ensure_ingredient_nodes_in_loss(dict(problem), min_hits=5)
    basis_samples = {k: np.asarray(v, dtype=float) for k, v in (problem.get("basis_samples") or {}).items()}
    basis_sample_weights = {
        k: np.asarray(v, dtype=float)
        for k, v in (problem.get("basis_sample_weights") or {}).items()
    } or None
    ratio_samples = np.asarray(problem.get("ratio_samples") or [], dtype=float)
    marginal_nodes = list(problem.get("marginal_nodes") or [nid for _, nid in MARGINAL_COLUMN_NODES])
    for nid in ingredient_basis:
        if nid and str(nid) not in marginal_nodes:
            marginal_nodes.append(str(nid))
    total_mass = float(problem.get("total_mass") or x0.sum())
    from recipe_opt_agent.kcal_utils import resolve_kcal_target, restore_kcal_target

    problem = restore_kcal_target(problem, cfg)
    kcal_target = float(resolve_kcal_target(cfg, problem) or 0.0)
    if kcal_target <= 0:
        from weighted_empirical_opt import atwater_kcal

        kcal_target = float(atwater_kcal(x0, M))
    # Keep mass consistent with the calorie target used by the LP.
    total_mass = float(problem.get("total_mass") or x0.sum())
    x0 = np.asarray(problem.get("x0") or x0, dtype=float)
    M = np.asarray(problem.get("M") or M, dtype=float)
    ingredient_basis = list(problem.get("ingredient_basis") or ingredient_basis)

    hull = region_intersects_hull(M, box, kcal_target=kcal_target)
    opt = optimize_weighted_empirical_obj(
        x0,
        M,
        marginal_nodes=marginal_nodes,
        basis_samples=basis_samples,
        ratio_samples=ratio_samples,
        ingredient_basis=ingredient_basis,
        kcal_target=kcal_target,
        protein_frac_min=box.protein_min,
        protein_frac_max=box.protein_max,
        carb_frac_min=box.carb_min,
        carb_frac_max=box.carb_max,
        fat_frac_min=box.fat_min,
        fat_frac_max=box.fat_max,
        total_mass=total_mass,
        basis_sample_weights=basis_sample_weights,
        nutrition_slack_weight=cfg.nutrition_slack_weight,
    )
    x_opt = np.asarray(opt["x_opt"], dtype=float)
    tl = term_losses(
        x_opt,
        marginal_nodes=marginal_nodes,
        basis_samples=basis_samples,
        ratio_samples=ratio_samples,
        total_mass=total_mass,
        ingredient_basis=ingredient_basis,
        basis_sample_weights=basis_sample_weights,
    )
    share_after = {
        nid: float(tl[f"{nid}__share"])
        for nid in marginal_nodes
        if f"{nid}__share" in tl
    }
    # Label shares with human names when available
    label_map = {nid: lab for lab, nid in MARGINAL_COLUMN_NODES}
    share_labeled = {label_map.get(k, k): v for k, v in share_after.items()}
    samples_labeled = {label_map.get(k, k): basis_samples.get(k, np.array([])) for k in share_after}

    identity_critical_names = set()
    roles = set(state.get("identity_roles") or [])
    for lab in share_labeled:
        key = lab.lower()
        if any(r.replace("_", " ") in key or key in r.replace("_", " ") for r in roles):
            identity_critical_names.add(lab)
        if "egg" in key and "egg" in roles:
            identity_critical_names.add(lab)
        if "cheese" in key and "cheese" in roles:
            identity_critical_names.add(lab)
        if ("pasta" in key or "spaghetti" in key) and "pasta" in roles:
            identity_critical_names.add(lab)
        if ("bacon" in key or "pork" in key) and "cured_pork" in roles:
            identity_critical_names.add(lab)

    binding = []
    # With soft bounds, report both near-active and violated sides so the next
    # proposal can add the missing macro vertex instead of treating slack as infeasibility.
    from weighted_empirical_opt import pfc_fractions_from_portions

    p, c, f = pfc_fractions_from_portions(x_opt, M)
    for value, lo, hi, name in (
        (p, box.protein_min, box.protein_max, "protein"),
        (c, box.carb_min, box.carb_max, "carb"),
        (f, box.fat_min, box.fat_max, "fat"),
    ):
        if value <= lo + 0.005:
            binding.append(f"{name}_min")
        if value >= hi - 0.005:
            binding.append(f"{name}_max")

    cfg = _cfg(state)
    diag = diagnose_optimizer_result(
        share_after=share_labeled,
        share_samples=samples_labeled,
        ratio_after=float(tl.get("ratio_value", float("nan"))),
        ratio_samples=ratio_samples,
        objective=float(opt["objective"]),
        macros_feasible=bool(opt.get("feasible")),
        hull_intersects=bool(hull.get("intersects")),
        binding_macros=binding,
        identity_critical_names=identity_critical_names,
        F_accept=cfg.F_accept,
        F_max=cfg.F_max,
    )

    # Optional loss-field summary (can be slow); skip if problem says so
    loss_summary: dict[str, Any] = {"skipped": True}
    if problem.get("compute_loss_field"):
        from loss_field import build_loss_field, summarize_loss_field

        field = build_loss_field(
            x0,
            M,
            marginal_nodes=marginal_nodes,
            basis_samples=basis_samples,
            ratio_samples=ratio_samples,
            ingredient_basis=ingredient_basis,
            kcal_target=kcal_target,
            total_mass=total_mass,
            grid_n=int(_cfg(state).loss_field_grid_n),
            cache_meta={"canonical_id": state.get("canonical_id")},
        )
        loss_summary = summarize_loss_field(field, box)

    # Serialize opt for state (drop huge arrays partially)
    opt_pub = {
        "status": opt.get("status"),
        "objective": float(opt["objective"]),
        "ratio_objective": float(opt.get("ratio_objective", opt["objective"])),
        "nutrient_slack": float(opt.get("nutrient_slack", 0.0)),
        "feasible": bool(opt.get("feasible")),
        "x_opt": x_opt.tolist(),
        "term_losses": {k: float(v) for k, v in tl.items()},
        "pfc_after": {"protein": p, "carbs": c, "fat": f},
    }
    tools_used = [
        {
            "name": "region_intersects_hull",
            "purpose": "Test whether target PFC box intersects the conical hull of current ingredients",
            "output_summary": {
                "intersects": hull.get("intersects"),
                "geometric_intersects": hull.get("geometric_intersects"),
                "lp_feasible": hull.get("lp_feasible"),
                "outside_score": (hull.get("distance") or {}).get("outside_score"),
                "interpretation": (hull.get("distance") or {}).get("interpretation"),
            },
            "output": {
                "intersects": hull.get("intersects"),
                "geometric_intersects": hull.get("geometric_intersects"),
                "lp_feasible": hull.get("lp_feasible"),
                "lp_message": hull.get("lp_message"),
                "distance": hull.get("distance"),
                "residual": hull.get("residual"),
                "n_samples": hull.get("n_samples"),
                "ingredient_pfc_vertices": hull.get("ingredient_pfc_vertices"),
            },
        },
        {
            "name": "optimize_weighted_empirical_obj",
            "purpose": "Solve weighted empirical fidelity LP under macro constraints",
            "output_summary": {
                "status": opt_pub["status"],
                "feasible": opt_pub["feasible"],
                "objective": opt_pub["objective"],
                "pfc_after": opt_pub["pfc_after"],
            },
            "output": opt_pub,
        },
        {
            "name": "diagnose_optimizer_result",
            "purpose": "IQR zones + three-band fidelity + retry triggers",
            "output_summary": {
                "diagnosis": diag.diagnosis.value,
                "fidelity_band": diag.fidelity_band.value,
                "L_max_norm": diag.L_max_norm,
                "n_red": diag.n_red,
                "n_triggers": len(diag.retry_triggers),
            },
            "output": diag.to_dict(),
        },
    ]
    if not loss_summary.get("skipped"):
        tools_used.append(
            {
                "name": "build_loss_field",
                "purpose": "Grid L(p)=min obj over PFC region",
                "output_summary": {k: loss_summary.get(k) for k in list(loss_summary)[:8]},
                "output": loss_summary,
            }
        )

    problem_out = {
        **problem,
        "x0": x0.tolist(),
        "M": np.asarray(M, dtype=float).tolist(),
        "total_mass": total_mass,
        "kcal_target": kcal_target,
        "user_kcal_target": resolve_kcal_target(cfg, problem) or problem.get("user_kcal_target"),
        "marginal_nodes": marginal_nodes,
        "ingredient_basis": ingredient_basis,
        "basis_samples": {k: np.asarray(v).tolist() for k, v in basis_samples.items()},
        "basis_sample_weights": {
            k: np.asarray(v).tolist()
            for k, v in (basis_sample_weights or {}).items()
        }
        if basis_sample_weights
        else problem.get("basis_sample_weights"),
        "ratio_samples": ratio_samples.tolist(),
        "x_opt": x_opt.tolist(),
        "basis_hit_counts": problem.get("basis_hit_counts"),
    }
    from recipe_opt_agent.foodon_basis_report import attach_foodon_basis_report

    foodon_report = attach_foodon_basis_report(problem_out) or {}
    foodon_summary = {
        "n_ingredients": foodon_report.get("n_ingredients"),
        "n_aggregated": foodon_report.get("n_aggregated"),
        "n_unmapped": foodon_report.get("n_unmapped"),
        "n_basis_nodes_in_recipe": foodon_report.get("n_basis_nodes_in_recipe"),
        "aggregated_ingredients": foodon_report.get("aggregated_ingredients"),
        "basis_nodes_in_recipe": [
            {"node_id": r["node_id"], "label": r.get("label"), "n_hits": r.get("n_hits")}
            for r in (foodon_report.get("basis_nodes") or [])
            if r.get("in_current_recipe")
        ],
    }
    tools_used.append(
        {
            "name": "foodon_basis_report",
            "purpose": "FoodOn leaf→basis aggregation levels + neighborhood hit counts per basis node",
            "output_summary": {
                "n_aggregated": foodon_summary.get("n_aggregated"),
                "n_unmapped": foodon_summary.get("n_unmapped"),
                "n_basis_in_recipe": foodon_summary.get("n_basis_nodes_in_recipe"),
            },
            "output": foodon_report,
        }
    )

    # Block accept when high-hit neighborhood staples are missing or grounding failed.
    from recipe_opt_agent.identity_gates import apply_identity_grounding_gates

    ratio_loss_val = None
    try:
        ratio_loss_val = float(tl.get("ratio_value") or tl.get("ratio_surrogate") or float("nan"))
        if ratio_loss_val != ratio_loss_val:  # NaN
            ratio_loss_val = None
    except (TypeError, ValueError):
        ratio_loss_val = None
    diag = apply_identity_grounding_gates(
        diag,
        problem={**problem_out, "opt": opt_pub, "foodon_basis_report": foodon_report},
        grounding_report=problem_out.get("grounding_report") or state.get("grounding_report"),
        identity_roles=list(state.get("identity_roles") or []),
        nutrient_slack=float(opt_pub.get("nutrient_slack") or 0.0),
        ratio_loss=ratio_loss_val,
    )

    out: dict[str, Any] = {
        "hull": hull,
        "opt": opt_pub,
        "loss_field_summary": loss_summary,
        "diagnosis": diag.to_dict(),
        "fidelity_band": diag.fidelity_band.value,
        "identity_critical": {n: True for n in identity_critical_names},
        "tools_used": tools_used,
        "foodon_basis_report": foodon_report,
        "chosen_recipe": problem_out.get("chosen_recipe") or state.get("chosen_recipe"),
        "problem": problem_out,
    }

    # Close pending decision outcome with after-needles + update telemetry.
    tel = dict(state.get("run_telemetry") or empty_telemetry())
    after = snapshot_needles({**state, **out})
    pending = state.get("pending_outcome")
    outcomes = list(state.get("decision_outcomes") or [])
    if pending:
        before = pending.get("before") or {}
        outcomes.append(
            {
                **pending,
                "after": after,
                "delta": delta_needles(before, after),
            }
        )
        out["pending_outcome"] = None
        out["decision_outcomes"] = outcomes[-8:]
        tel = bump_telemetry(
            tel,
            edges=[
                {
                    "kind": "post_apply",
                    "post_apply_delta_L_max_norm": (after.get("L_max_norm") or 0) - (before.get("L_max_norm") or 0)
                    if after.get("L_max_norm") is not None and before.get("L_max_norm") is not None
                    else None,
                    "post_apply_delta_ratio": (delta_needles(before, after) or {}).get("ratio_term"),
                    "post_apply_delta_nutrient": (delta_needles(before, after) or {}).get("nutrient_slack"),
                    "post_apply_delta_holistic": (delta_needles(before, after) or {}).get("holistic"),
                }
            ],
        )
    tel = bump_telemetry(
        tel,
        nodes={
            "diagnose": {
                "band": diag.fidelity_band.value,
                "L_max_norm": diag.L_max_norm,
                "n_red": diag.n_red,
                "hull_outside": not bool(hull.get("intersects")),
                "foodon_n_aggregated": foodon_summary.get("n_aggregated"),
                "foodon_n_unmapped": foodon_summary.get("n_unmapped"),
            }
        },
    )
    out["run_telemetry"] = tel

    from recipe_opt_agent.score_display import live_scores_from_state

    live_state = {**state, **out}
    live_scores = live_scores_from_state(live_state)
    history = list(state.get("score_history") or [])
    history.append(
        {
            "iteration": int(state.get("iteration") or 0),
            "branch": "current",
            "source": "diagnose",
            "ratio_loss": (live_scores.get("ratio_loss") or {}).get("value"),
            "nutrient_loss": (live_scores.get("nutrient_loss") or {}).get("value"),
            "L_max_norm": diag.L_max_norm,
        }
    )
    out["live_scores"] = live_scores
    out["score_history"] = history[-40:]
    return out


def node_save_candidate(state: AgentState) -> dict[str, Any]:
    """Save feasible macro snapshots to candidate_pool (moderate + selected must_retry cases)."""
    band = state.get("fidelity_band")
    cfg = _cfg(state)
    opt = state.get("opt") or {}
    diag = state.get("diagnosis") or {}
    macros_feasible = bool(opt.get("feasible"))
    should_save = False
    branch = ""
    if band == FidelityBand.MODERATE.value:
        should_save = True
        branch = "moderate"
    elif (
        band == FidelityBand.MUST_RETRY.value
        and macros_feasible
        and cfg.save_on_must_retry_feasible
    ):
        should_save = True
        branch = "must_retry_feasible"
    elif band == FidelityBand.ACCEPT.value and _agent_mode(state) == "creative":
        should_save = True
        branch = "accept_polish"

    if not should_save:
        return {}

    problem = state.get("problem") or {}
    foodon_report = (
        problem.get("foodon_basis_report")
        or state.get("foodon_basis_report")
        or (problem.get("chosen_recipe") or {}).get("foodon_basis_report")
    )
    entry = {
        "candidate_id": f"pool_{state.get('iteration', 0)}_{len(state.get('candidate_pool') or [])}",
        "iteration": state.get("iteration", 0),
        "branch": branch,
        "objective": opt.get("objective"),
        "L_total": diag.get("L_total"),
        "L_max_norm": diag.get("L_max_norm"),
        "n_red": diag.get("n_red"),
        "x_opt": opt.get("x_opt"),
        "pfc_after": opt.get("pfc_after"),
        "diagnosis": diag.get("diagnosis"),
        "fidelity_band": band,
        "ingredients": (problem.get("chosen_recipe") or {}).get("ingredients"),
        "foodon_basis_report": foodon_report,
        "opt": opt,
        "diagnosis_full": diag,
    }
    pool = list(state.get("candidate_pool") or [])
    pool.append(entry)
    interesting = list(state.get("interesting_candidates") or [])
    interesting.append({**entry, "source": "save_candidate", "branch": entry.get("branch") or "in_distribution"})
    return {
        "candidate_pool": pool,
        "interesting_candidates": interesting[-24:],
        "tools_used": [
            {
                "name": "save_candidate",
                "purpose": "Soft-band pool save (moderate / feasible must_retry / creative accept)",
                "output_summary": {"branch": branch, "pool_size": len(pool), "L_max_norm": diag.get("L_max_norm")},
                "output": entry,
            }
        ],
    }


def node_save_moderate(state: AgentState) -> dict[str, Any]:
    """Backward-compatible alias."""
    return node_save_candidate(state)


def _filter_candidates(
    cands: list[dict[str, Any]],
    *,
    critical: set[str],
    tags: list[RequirementTag],
    problem: dict[str, Any] | None = None,
    title: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from recipe_opt_agent.clash_gates import filter_candidates_by_clash_gates
    from recipe_opt_agent.edit_grounding import filter_candidates_by_neighborhood_support

    filtered: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for c in cands:
        if c.get("action") == "remove" and c.get("label") in critical:
            dropped.append({"candidate": c, "reason": "identity_critical_remove"})
            continue
        if c.get("identity_critical_target") and c.get("action") == "remove":
            dropped.append({"candidate": c, "reason": "identity_critical_target"})
            continue
        filtered.append(c)
    if tags:
        filtered, tag_dropped = filter_candidates_by_tags(filtered, tags)
        dropped.extend(tag_dropped)
    filtered, clash_dropped = filter_candidates_by_clash_gates(
        filtered, problem=problem, title=title
    )
    dropped.extend(clash_dropped)
    filtered, nb_dropped = filter_candidates_by_neighborhood_support(
        filtered, problem=problem
    )
    dropped.extend(nb_dropped)
    return filtered, dropped


def node_propose(state: AgentState) -> dict[str, Any]:
    """Plan slots → per-slot retrieval → joint bundle scoring (all live, no pre-baked lists)."""
    from recipe_opt_agent.bundle_scoring import score_bundles
    from recipe_opt_agent.problem_loader import (
        _build_modification_candidates_from_problem,
        _build_slot_candidates_from_problem,
    )
    from recipe_opt_agent.slot_planner import plan_slots, slots_to_dicts

    problem = dict(state.get("problem") or {})
    cfg = _cfg(state)
    box_dict = {
        "protein_min": cfg.protein_min,
        "protein_max": cfg.protein_max,
        "carb_min": cfg.carb_min,
        "carb_max": cfg.carb_max,
        "fat_min": cfg.fat_min,
        "fat_max": cfg.fat_max,
        "nutrition_slack_weight": cfg.nutrition_slack_weight,
    }
    tags = _requirement_tags(state)
    critical = set((state.get("identity_critical") or {}).keys())
    tools_used: list[dict[str, Any]] = []

    # 1) Plan edit slots from the diagnosis.
    current_ingredients = list(
        (state.get("chosen_recipe") or (problem.get("chosen_recipe") or {})).get("ingredients") or []
    )
    slots = plan_slots(
        state.get("diagnosis") or {},
        identity_critical=state.get("identity_critical") or {},
        requirement_tags=tags,
        current_ingredients=current_ingredients,
    )
    slot_dicts = slots_to_dicts(slots)
    tools_used.append(
        {
            "name": "plan_slots",
            "purpose": "Derive up to 2 structured edit slots (open_hull / fix_share / macro_gap / dietary_swap / remove_outlier) from the diagnosis",
            "output_summary": {
                "n_slots": len(slot_dicts),
                "slots": [{"slot_id": s["slot_id"], "kind": s["kind"], "reason": s["reason"]} for s in slot_dicts],
            },
            "output": {"slots": slot_dicts},
        }
    )

    # 2) LLM ideation (5–10 ideas) → ground to FDC → verify with proxies/LP.
    #    Nutrient/co-occurrence retrieval remains a fallback / supplement.
    from recipe_opt_agent.ideation import ground_ideas_to_candidates, ideate_ingredient_edits

    ideation_ctx = {
        "title": state.get("title"),
        "user_request": state.get("user_request") or state.get("taste_text"),
        "taste_text": state.get("taste_text"),
        "taste_preference": (state.get("problem") or {}).get("taste_preference")
        or state.get("taste_preference"),
        "identity_roles": state.get("identity_roles"),
        "requirement_tags": [t.to_dict() if hasattr(t, "to_dict") else t for t in tags],
        "fidelity_band": state.get("fidelity_band"),
        "diagnosis": state.get("diagnosis") or {},
        "opt": state.get("opt") or {},
        "macro_box": box_dict,
        "current_ingredients": current_ingredients,
        "slots": slot_dicts,
    }
    ideation = ideate_ingredient_edits(
        ideation_ctx,
        model=getattr(cfg, "ideation_model", None) or cfg.model_escalate,
        n_ideas=int(getattr(cfg, "n_ideation_candidates", 8) or 8),
    )
    # Expand neighborhood using LLM search queries before grounding/scoring
    queries = list(ideation.get("neighborhood_search_queries") or [])
    if queries:
        from recipe_opt_agent.neighborhood_query_expand import (
            derive_focus_terms,
            expand_neighborhood_by_queries,
        )

        from recipe_opt_agent.neighborhood_query_expand import (
            fallback_dish_structure,
            normalize_dish_structure,
        )

        idea_ingredients = [
            str(i.get("ingredient") or "")
            for i in (ideation.get("ideas") or [])
            if str(i.get("branch") or "").startswith("ood") or i.get("neighborhood_search_queries")
        ]
        cur_labels = [str(r.get("label") or r.get("name") or "") for r in current_ingredients]
        title_texts = [str(state.get("title") or "")]
        # Ingredient names first; query tokens only as fallback (see ood_branch).
        focus_terms = derive_focus_terms(idea_ingredients, cur_labels, extra_texts=title_texts)
        if not focus_terms:
            focus_terms = derive_focus_terms(queries, cur_labels, extra_texts=title_texts)
        dish_structure = normalize_dish_structure(ideation.get("dish_structure"))
        if dish_structure is None:
            for idea in ideation.get("ideas") or []:
                if isinstance(idea.get("dish_structure"), dict):
                    dish_structure = normalize_dish_structure(idea.get("dish_structure"))
                    if dish_structure:
                        break
        if dish_structure is None:
            dish_structure = fallback_dish_structure(
                stretch_ingredient=idea_ingredients[0] if idea_ingredients else None,
                identity_roles=list(state.get("identity_roles") or []),
                current_labels=cur_labels,
                stretch_role="accent",
            )
        exp = expand_neighborhood_by_queries(
            problem,
            queries,
            focus_terms=focus_terms,
            dish_structure=dish_structure,
        )
        problem = exp.get("problem") or problem
        # Attest expansion shell ingredient labels for edit-support gates.
        ctx = dict(problem.get("retrieval_context") or {})
        harvest_labels: list[str] = list(ctx.get("expansion_harvest_labels") or [])
        for shell in exp.get("shell_recipes") or []:
            for lab in shell.get("ingredient_labels") or shell.get("labels") or []:
                s = str(lab).strip()
                if s and s not in harvest_labels:
                    harvest_labels.append(s)
            title = str(shell.get("title") or "").strip()
            if title and title not in harvest_labels:
                harvest_labels.append(title)
        if harvest_labels:
            ctx["expansion_harvest_labels"] = harvest_labels
            problem["retrieval_context"] = ctx
        tools_used.append(
            {
                "name": "expand_neighborhood_by_queries",
                "purpose": (
                    "Widen co-occurrence neighborhood using LLM culinary search phrases; "
                    "verify gram-share structure (anchor vs stretch) before harvesting shares"
                ),
                "output_summary": exp.get("meta"),
                "output": {
                    "queries": queries,
                    "dish_structure": dish_structure,
                    "meta": exp.get("meta"),
                    "n_shell": len(exp.get("shell_recipes") or []),
                },
            }
        )

    idea_cands = ground_ideas_to_candidates(
        ideation.get("ideas") or [],
        problem=problem,
        requirement_tags=tags,
        box_dict=box_dict,
    )
    tools_used.append(
        {
            "name": "ideate_ingredient_edits",
            "purpose": "LLM proposes 5–10 contextual ingredient ideas; FDC grounding + LP verify next",
            "mode": (ideation.get("_llm_trace") or {}).get("mode"),
            "model": (ideation.get("_llm_trace") or {}).get("model"),
            "output_summary": {
                "n_ideas": len(ideation.get("ideas") or []),
                "n_grounded": len(idea_cands),
                "ood_justification": ideation.get("ood_justification"),
                "n_queries": len(queries),
            },
            "output": {
                "ideas": ideation.get("ideas"),
                "grounded": idea_cands,
                "queries": queries,
            },
            "llm_trace": ideation.get("_llm_trace"),
        }
    )

    # 2b) Per-slot retrieval with cheap nutrient/ratio proxies (supplement).
    per_slot: dict[str, list[dict[str, Any]]] = {}
    retrieval_error = None
    if problem.get("retrieval_context"):
        try:
            per_slot = _build_slot_candidates_from_problem(problem, slot_dicts, box_dict=box_dict)
        except Exception as exc:
            retrieval_error = str(exc)
            per_slot = {}
    if not any(per_slot.values()):
        # Fallbacks: legacy flat retrieval, then fixture candidates already on the problem.
        flat: list[dict[str, Any]] = []
        if problem.get("retrieval_context"):
            try:
                flat = _build_modification_candidates_from_problem(problem, box_dict=box_dict)
            except Exception as exc:
                retrieval_error = retrieval_error or str(exc)
        if not flat:
            flat = list(problem.get("modification_candidates") or [])
        if flat:
            sid = slot_dicts[0]["slot_id"] if slot_dicts else "improve"
            per_slot = {sid: flat}

    # Prefer LLM ideas: put them first in each slot (or a dedicated ideation slot).
    if idea_cands:
        sid = slot_dicts[0]["slot_id"] if slot_dicts else "ideation"
        existing = list(per_slot.get(sid) or [])
        # Dedup by label
        seen_labels = {str(c.get("label") or "").lower() for c in idea_cands}
        merged = list(idea_cands) + [
            c for c in existing if str(c.get("label") or "").lower() not in seen_labels
        ]
        per_slot[sid] = merged

    # Filter each slot's candidates (identity + tags + clash + neighborhood attestation).
    dropped: list[dict[str, Any]] = []
    for sid in list(per_slot.keys()):
        kept, slot_dropped = _filter_candidates(
            per_slot[sid],
            critical=critical,
            tags=tags,
            problem=problem,
            title=state.get("title"),
        )
        per_slot[sid] = kept
        dropped.extend(slot_dropped)
    flat_candidates = [c for cands in per_slot.values() for c in cands]
    tools_used.append(
        {
            "name": "retrieve_slots",
            "purpose": "Slot-conditioned candidate retrieval (supplement to LLM ideation; cooc + geometry proxies)",
            "output_summary": {
                "per_slot_counts": {sid: len(cands) for sid, cands in per_slot.items()},
                "n_kept": len(flat_candidates),
                "n_dropped": len(dropped),
                "retrieval_error": retrieval_error,
                "n_from_ideation": len(idea_cands),
            },
            "output": {"per_slot": per_slot, "dropped": dropped, "retrieval_error": retrieval_error},
        }
    )

    # 3) Enumerate bundles and score the top ones with a joint LP.
    bundles: list[dict[str, Any]] = []
    bundle_error = None
    if flat_candidates:
        try:
            bundles = score_bundles(
                problem,
                per_slot,
                box_dict=box_dict,
                proxy_cap=int(getattr(cfg, "bundle_proxy_cap", 50) or 50),
                lp_cap=int(getattr(cfg, "bundle_lp_cap", 10) or 10),
            )
        except Exception as exc:
            bundle_error = str(exc)
            bundles = []
    # Propagate candidate branch onto ID bundles when missing
    for b in bundles:
        if b.get("branch"):
            continue
        branches = {
            str(e.get("branch") or (e.get("meta") or {}).get("branch") or "")
            for e in (b.get("edits") or [])
        }
        branches.discard("")
        if any(br.startswith("ood") for br in branches):
            b["branch"] = "ood_protein"
        elif "hybrid" in branches:
            b["branch"] = "hybrid"
        else:
            b["branch"] = "in_distribution"

    # Anti-oscillation: flag / demote fingerprints matching recent applies.
    recent_fps = list(state.get("recent_edit_fingerprints") or [])[-2:]
    osc_hits = 0
    if recent_fps and bundles:
        improve_eps = float(cfg.oscillation_improve_eps)
        # Best historical delta among recent is unknown; require absolute improvement vs 0 by eps
        # when fingerprint matches — mark blocked unless delta_L_star < -improve_eps relative to 0.
        for b in bundles:
            fp = edit_fingerprint(b.get("edits"))
            if fp and fp in recent_fps:
                d = b.get("delta_L_star")
                if d is None or float(d) > -improve_eps:
                    b["oscillation_blocked"] = True
                    b["oscillation_fingerprint"] = fp
                    osc_hits += 1
                else:
                    b["oscillation_flagged"] = True
                    b["oscillation_fingerprint"] = fp
        bundles = sorted(
            bundles,
            key=lambda b: (
                1 if b.get("oscillation_blocked") else 0,
                float(b["delta_L_star"]) if b.get("delta_L_star") is not None else 99.0,
            ),
        )

    bundles_pub = [{k: v for k, v in b.items() if k != "next_problem"} for b in bundles]
    tools_used.append(
        {
            "name": "score_bundles",
            "purpose": "Enumerate 1-2 edit bundles, proxy-rank, then joint LP re-optimization (L*_before vs L*_after) on the top set",
            "output_summary": {
                "n_bundles": len(bundles),
                "n_lp_evaluated": sum(1 for b in bundles if b.get("lp_evaluated")),
                "oscillation_hits": osc_hits,
                "best": (
                    {
                        "bundle_id": bundles_pub[0].get("bundle_id"),
                        "edits": bundles_pub[0].get("edits"),
                        "delta_L_star": bundles_pub[0].get("delta_L_star"),
                    }
                    if bundles_pub
                    else None
                ),
                "bundle_error": bundle_error,
            },
            "output": {"bundles": bundles_pub, "bundle_error": bundle_error},
        }
    )

    # 4) OOD protein branch when the box / tags demand lean protein beyond neighborhood.
    from recipe_opt_agent.ood_branch import maybe_build_ood_branch

    for b in bundles:
        b.setdefault("branch", "in_distribution")
    ood_info = maybe_build_ood_branch(
        problem,
        box_dict=box_dict,
        diagnosis=state.get("diagnosis") or {},
        opt=state.get("opt") or {},
        requirement_tags=tags,
        id_bundles=bundles,
        ideation_context={
            "title": state.get("title"),
            "user_request": state.get("user_request") or state.get("taste_text"),
            "identity_roles": state.get("identity_roles"),
        },
        ideation_model=getattr(cfg, "ideation_model", None) or cfg.model_escalate,
        n_ideas=int(getattr(cfg, "n_ideation_candidates", 8) or 8),
    )
    problem = ood_info.get("problem") or problem
    ood_bundles = list(ood_info.get("ood_bundles") or [])
    hybrid_bundles = list(ood_info.get("hybrid_bundles") or [])
    ood_cands = list(ood_info.get("ood_candidates") or [])
    if ood_info.get("needed"):
        flat_candidates = flat_candidates + ood_cands
        bundles = list(bundles) + ood_bundles + hybrid_bundles
        bundles = sorted(
            bundles,
            key=lambda b: (
                1 if b.get("oscillation_blocked") else 0,
                float(b["delta_L_star"]) if b.get("delta_L_star") is not None else 99.0,
            ),
        )
        tools_used.append(
            {
                "name": "ood_protein_branch",
                "purpose": "LLM-ideated OOD lean-protein candidates + query-expanded neighborhood + hybrid ID⊕OOD bundles",
                "output_summary": {
                    "needed": True,
                    "reason": ood_info.get("reason"),
                    "n_ood_candidates": len(ood_cands),
                    "n_ood_bundles": len(ood_bundles),
                    "n_hybrid_bundles": len(hybrid_bundles),
                    "ideation": ood_info.get("ideation"),
                    "neighborhood_expansion": ood_info.get("neighborhood_expansion"),
                },
                "output": {
                    "reason": ood_info.get("reason"),
                    "ood_candidates": ood_cands,
                    "ood_bundles": [{k: v for k, v in b.items() if k != "next_problem"} for b in ood_bundles],
                    "hybrid_bundles": [
                        {k: v for k, v in b.items() if k != "next_problem"} for b in hybrid_bundles
                    ],
                    "ideation": ood_info.get("ideation"),
                    "neighborhood_expansion": ood_info.get("neighborhood_expansion"),
                },
                "llm_trace": ood_info.get("ideation_trace"),
            }
        )

    # Park strong / OOD / hybrid hits for end-of-run finalist evaluation.
    interesting = list(state.get("interesting_candidates") or [])
    for b in bundles:
        if b.get("oscillation_blocked"):
            continue
        d = b.get("delta_L_star")
        branch = b.get("branch") or "in_distribution"
        keep = branch in {"ood_protein", "hybrid"} or (d is not None and float(d) < -1e-4)
        if not keep:
            continue
        interesting.append(
            {
                "candidate_id": f"bundle::{b.get('bundle_id')}",
                "source": "propose_bundle",
                "branch": branch,
                "iteration": state.get("iteration", 0),
                "delta_L_star": d,
                "edits": b.get("edits"),
                "nutrient_slack": b.get("nutrient_slack"),
                "ratio_term": b.get("ratio_term"),
                "ingredients": (b.get("next_problem") or {}).get("chosen_recipe", {}).get("ingredients")
                if isinstance(b.get("next_problem"), dict)
                else None,
                "L_max_norm": None,
                "opt": {
                    "objective": b.get("L_star_after"),
                    "pfc_after": b.get("pfc_after"),
                    "feasible": True,
                },
                "foodon_basis_report": (b.get("next_problem") or {}).get("foodon_basis_report")
                if isinstance(b.get("next_problem"), dict)
                else None,
            }
        )
    # Deduplicate by candidate_id keeping latest
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in reversed(interesting):
        cid = str(row.get("candidate_id") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        deduped.append(row)
    interesting = list(reversed(deduped))[-24:]

    problem["modification_candidates"] = flat_candidates
    best_delta = None
    lp_n = sum(1 for b in bundles if b.get("lp_evaluated"))
    for b in bundles:
        if b.get("delta_L_star") is not None and not b.get("oscillation_blocked"):
            best_delta = float(b["delta_L_star"])
            break
    tel = bump_telemetry(
        state.get("run_telemetry"),
        oscillation_hits=osc_hits,
        nodes={
            "propose": {
                "n_slots": len(slot_dicts),
                "n_bundles_lp": lp_n,
                "best_delta_L_star": best_delta,
                "ood_needed": bool(ood_info.get("needed")),
                "n_ood_bundles": len(ood_bundles),
                "n_hybrid_bundles": len(hybrid_bundles),
            }
        },
    )
    from recipe_opt_agent.score_display import best_branch_scores_from_bundles, live_scores_from_state

    cfg = _cfg(state)
    box = {
        "protein_min": float(cfg.protein_min),
        "protein_max": float(cfg.protein_max),
        "carb_min": float(cfg.carb_min),
        "carb_max": float(cfg.carb_max),
        "fat_min": float(cfg.fat_min),
        "fat_max": float(cfg.fat_max),
    }
    it = int(state.get("iteration") or 0)
    branch_pts = best_branch_scores_from_bundles(bundles, iteration=it, box=box)
    history = list(state.get("score_history") or [])
    history.extend(branch_pts)
    live_scores = live_scores_from_state({**state, "problem": problem, "score_history": history})
    # Prefer best in-distribution (or overall) bundle losses for the live cards when available
    if branch_pts:
        prefer = next((p for p in branch_pts if p.get("branch") == "in_distribution"), branch_pts[0])
        if prefer.get("ratio_loss") is not None and live_scores.get("ratio_loss"):
            live_scores = {
                **live_scores,
                "ratio_loss": {
                    **live_scores["ratio_loss"],
                    "value": prefer["ratio_loss"],
                    "band": live_scores["ratio_loss"].get("band"),
                    "source": f"best_bundle:{prefer.get('branch')}",
                },
            }
            from recipe_opt_agent.score_display import band_for_loss, RATIO_LOSS_BANDS

            live_scores["ratio_loss"]["band"] = band_for_loss(
                prefer["ratio_loss"],
                good_max=RATIO_LOSS_BANDS["good_max"],
                warn_max=RATIO_LOSS_BANDS["warn_max"],
            )
        if prefer.get("nutrient_loss") is not None and live_scores.get("nutrient_loss"):
            from recipe_opt_agent.score_display import band_for_loss, NUTRIENT_LOSS_BANDS

            live_scores = {
                **live_scores,
                "nutrient_loss": {
                    **live_scores["nutrient_loss"],
                    "value": prefer["nutrient_loss"],
                    "band": band_for_loss(
                        prefer["nutrient_loss"],
                        good_max=NUTRIENT_LOSS_BANDS["good_max"],
                        warn_max=NUTRIENT_LOSS_BANDS["warn_max"],
                    ),
                    "source": f"best_bundle:{prefer.get('branch')}",
                },
            }
    live_scores["score_history"] = history[-80:]
    live_scores["iteration"] = it

    return {
        "candidates": flat_candidates,
        "candidates_dropped": dropped,
        "planned_slots": slot_dicts,
        "bundles": bundles,
        "problem": problem,
        "interesting_candidates": interesting,
        "tools_used": tools_used,
        "run_telemetry": tel,
        "live_scores": live_scores,
        "score_history": history[-80:],
    }


def node_decide(state: AgentState) -> dict[str, Any]:
    from recipe_opt_agent.context_builder import count_adds_so_far

    cfg = _cfg(state)
    ctx = build_decision_context(state)
    adds_so_far = count_adds_so_far(state)
    marginal_eps = float(getattr(cfg, "marginal_add_delta_eps", 0.02) or 0.02)
    max_adds = int(getattr(cfg, "max_total_adds", 2) or 2)

    def _n_adds(b: dict[str, Any]) -> int:
        return sum(1 for e in b.get("edits") or [] if str(e.get("action") or "") == "add")

    def _has_add(b: dict[str, Any]) -> bool:
        return _n_adds(b) > 0

    def _bundle_ok(b: dict[str, Any]) -> bool:
        """Gate for LLM-skipping auto-apply — stricter than what the LLM may pick."""
        from recipe_opt_agent.llm import _bundle_passes_tags

        if not _bundle_passes_tags(b, ctx):
            return False
        # Stop-adding policy: bundle adds must fit the remaining budget, and
        # marginal add-bundles never auto-apply.
        n_adds = _n_adds(b)
        if n_adds:
            if adds_so_far + n_adds > max_adds:
                return False
            d = b.get("delta_L_star")
            if d is None or float(d) > -marginal_eps:
                return False
            # Only LLM-ideated adds may skip the LLM. Raw slot-retrieval adds
            # (macro_gap/fix_share catalog hits like lemon, syrup, wine) need an
            # explicit LLM identity/taste check before entering the dish.
            for e in b.get("edits") or []:
                if str(e.get("action") or "") != "add":
                    continue
                cid = str(e.get("candidate_id") or "")
                src = str(((e.get("meta") or {}).get("source")) or "")
                if not (cid.startswith("idea_") or src == "llm_ideation"):
                    return False
        return True

    favorite = None
    if state.get("fidelity_band") != FidelityBand.ACCEPT.value:
        favorite = clear_favorite_bundle(
            list(state.get("bundles") or []),
            delta_eps=cfg.auto_apply_delta_eps,
            margin=cfg.auto_apply_margin,
            passes_tags_fn=_bundle_ok,
            ood_delta_handicap=float(getattr(cfg, "ood_delta_handicap", 0.015) or 0.0),
        )
        # Never auto-apply a clash ingredient for a tiny LP gain.
        if favorite is not None:
            from recipe_opt_agent.clash_gates import bundle_clash_detail

            clashes = bundle_clash_detail(
                favorite,
                problem=state.get("problem"),
                title=state.get("title"),
            )
            if clashes:
                favorite = None

    decide_mode = "llm_mini"
    llm_trace: dict[str, Any] = {}
    if favorite is not None:
        decision = {
            "action": "apply_bundle",
            "chosen_candidate_id": None,
            "chosen_bundle_id": favorite.get("bundle_id"),
            "edits": list(favorite.get("edits") or []),
            "rationale": "auto_clear_favorite",
            "identity": {
                "preserves_dish": True,
                "acceptable_variant": True,
                "roles_retained": state.get("identity_roles") or [],
                "role_change": ", ".join(str(e.get("label")) for e in favorite.get("edits") or []),
                "rationale": "clear LP favorite; skipped LLM",
            },
            "expand_directive": None,
        }
        decide_mode = "auto"
        tools_used = [
            {
                "name": "decide_auto",
                "purpose": "Auto-apply clear LP favorite bundle (skip LLM)",
                "mode": "auto",
                "model": None,
                "output_summary": {
                    "action": decision["action"],
                    "chosen_bundle_id": decision["chosen_bundle_id"],
                    "delta_L_star": favorite.get("delta_L_star"),
                    "edits": decision["edits"],
                    "rationale": decision["rationale"],
                },
                "output": decision,
            }
        ]
        tel = bump_telemetry(state.get("run_telemetry"), n_auto_applies=1)
    else:
        model = select_decide_model(ctx, cfg=cfg)
        decide_mode = "llm_escalate" if model == cfg.model_escalate else "llm_mini"
        decision = decide_action_llm(ctx, model=model)
        llm_trace = decision.pop("_llm_trace", None) or {}
        # Hard add-budget guard: veto choices whose adds exceed the remaining budget.
        n_new_adds = 0
        if decision.get("action") == "add":
            n_new_adds = 1
        elif decision.get("action") == "apply_bundle":
            chosen_b = next(
                (
                    b
                    for b in (state.get("bundles") or [])
                    if str(b.get("bundle_id")) == str(decision.get("chosen_bundle_id"))
                ),
                None,
            )
            n_new_adds = _n_adds(chosen_b) if chosen_b is not None else 0
        if n_new_adds and adds_so_far + n_new_adds > max_adds:
            fallback = "accept_pool_best" if (state.get("candidate_pool") or []) else "accept"
            decision = {
                **decision,
                "action": fallback,
                "chosen_candidate_id": None,
                "chosen_bundle_id": None,
                "rationale": (
                    f"Add budget: {adds_so_far} adds so far + {n_new_adds} in choice exceeds "
                    f"max_total_adds={max_adds}; vetoed. Original LLM rationale: "
                    f"{decision.get('rationale')}"
                ),
                "add_budget_veto": True,
            }
        # Identity clash veto: reject apply_bundle / add that introduce denylist or
        # family-stack clashes unless nutrient slack improvement is large.
        if decision.get("action") in {"apply_bundle", "add"} and not decision.get("add_budget_veto"):
            from recipe_opt_agent.clash_gates import bundle_clash_detail, clash_reason_for_label

            clash_hits: list[dict[str, Any]] = []
            if decision.get("action") == "apply_bundle":
                chosen_b = next(
                    (
                        b
                        for b in (state.get("bundles") or [])
                        if str(b.get("bundle_id")) == str(decision.get("chosen_bundle_id"))
                    ),
                    None,
                )
                clash_hits = bundle_clash_detail(
                    chosen_b,
                    problem=state.get("problem"),
                    title=state.get("title"),
                )
                nutrient_delta = None
                if chosen_b is not None:
                    try:
                        nutrient_delta = -float(chosen_b.get("delta_L_star") or 0.0)
                    except (TypeError, ValueError):
                        nutrient_delta = None
                # Only allow clash if the bundle clearly fixes a large loss (≥0.05).
                allow = nutrient_delta is not None and nutrient_delta >= 0.05
                # Denylist never allowed.
                if any(str(h.get("detail") or "").startswith("denylist") for h in clash_hits):
                    allow = False
                if clash_hits and not allow:
                    decision = {
                        **decision,
                        "action": "accept_pool_best"
                        if (state.get("candidate_pool") or [])
                        else "accept",
                        "chosen_candidate_id": None,
                        "chosen_bundle_id": None,
                        "rationale": (
                            "Clash veto: edits "
                            + ", ".join(
                                f"{h.get('label')} ({h.get('detail')})" for h in clash_hits[:3]
                            )
                            + f" not justified by LP gain. Original: {decision.get('rationale')}"
                        ),
                        "clash_veto": True,
                        "clash_details": clash_hits,
                    }
            elif decision.get("action") == "add":
                # Direct add action — check label if present on decision.
                add_label = str(
                    decision.get("add_label")
                    or ((decision.get("identity") or {}).get("role_change"))
                    or ""
                )
                reason = clash_reason_for_label(
                    add_label,
                    current_labels=[
                        str(r.get("label") or "")
                        for r in (
                            ((state.get("problem") or {}).get("chosen_recipe") or {}).get(
                                "ingredients"
                            )
                            or []
                        )
                    ],
                    title=state.get("title"),
                )
                if reason and str(reason).startswith("denylist"):
                    decision = {
                        **decision,
                        "action": "accept_pool_best"
                        if (state.get("candidate_pool") or [])
                        else "accept",
                        "rationale": (
                            f"Clash veto on add {add_label!r}: {reason}. "
                            f"Original: {decision.get('rationale')}"
                        ),
                        "clash_veto": True,
                    }
        tools_used = [
            {
                "name": "decide_action_llm",
                "purpose": "Choose accept / modify / expand given DecisionContext",
                "mode": llm_trace.get("mode") or decide_mode,
                "model": llm_trace.get("model") or model,
                "output_summary": {
                    "action": decision.get("action"),
                    "chosen_candidate_id": decision.get("chosen_candidate_id"),
                    "chosen_bundle_id": decision.get("chosen_bundle_id"),
                    "rationale": decision.get("rationale"),
                    "decide_mode": decide_mode,
                },
                "output": decision,
                "llm_trace": llm_trace,
            }
        ]
        tel = bump_telemetry(state.get("run_telemetry"), n_llm_calls=1 if llm_trace.get("mode") == "openai" else 0)

    # On iteration 0, never settle for accept when an improving LP bundle exists —
    # force at least one proportion-improving apply before early exit.
    it_now = int(state.get("iteration") or 0)
    if it_now < 1 and decision.get("action") in {"accept", "accept_pool_best"}:
        improve_ranked = [
            b
            for b in (state.get("bundles") or [])
            if b.get("delta_L_star") is not None
            and not b.get("oscillation_blocked")
            and float(b["delta_L_star"]) < -1e-6
            and b.get("next_problem")
        ]
        if improve_ranked:
            best_improve = min(improve_ranked, key=lambda b: float(b["delta_L_star"]))
            decision = {
                **decision,
                "action": "apply_bundle",
                "chosen_bundle_id": best_improve.get("bundle_id"),
                "chosen_candidate_id": None,
                "edits": list(best_improve.get("edits") or []),
                "rationale": (
                    "iter0_force_improve: blocked early accept; applying best LP bundle "
                    f"(delta_L_star={best_improve.get('delta_L_star')}). "
                    f"Original: {decision.get('rationale')}"
                ),
                "iter0_force_improve": True,
            }

    # LP agreement
    lp_best_id = None
    ranked = [
        b
        for b in (state.get("bundles") or [])
        if b.get("lp_evaluated") and b.get("delta_L_star") is not None and not b.get("oscillation_blocked")
    ]
    if ranked:
        lp_best_id = min(ranked, key=lambda b: float(b["delta_L_star"])).get("bundle_id")
    agreed = decision.get("action") == "apply_bundle" and str(decision.get("chosen_bundle_id")) == str(lp_best_id)

    hist = list(state.get("history") or [])
    hist.append(
        {
            "iteration": state.get("iteration", 0),
            "decision": decision,
            "band": state.get("fidelity_band"),
            "llm_mode": llm_trace.get("mode") or decide_mode,
            "decide_mode": decide_mode,
            "agreed_with_lp_best": agreed,
        }
    )
    tel = bump_telemetry(
        tel,
        nodes={
            "decide": {
                "mode": decide_mode,
                "agreed_with_lp_best": agreed,
                "action": decision.get("action"),
            }
        },
    )

    # Pending outcome filled on next diagnose after apply
    pending = {
        "iteration": int(state.get("iteration") or 0),
        "decision": {
            "action": decision.get("action"),
            "chosen_bundle_id": decision.get("chosen_bundle_id"),
            "chosen_candidate_id": decision.get("chosen_candidate_id"),
            "edits": decision.get("edits")
            or (
                (favorite or {}).get("edits")
                if favorite is not None
                else next(
                    (
                        b.get("edits")
                        for b in (state.get("bundles") or [])
                        if str(b.get("bundle_id")) == str(decision.get("chosen_bundle_id"))
                    ),
                    None,
                )
            ),
            "rationale": decision.get("rationale"),
        },
        "before": snapshot_needles(state),
        "bundles_considered_top": [
            {
                "bundle_id": b.get("bundle_id"),
                "edits": b.get("edits"),
                "delta_L_star": b.get("delta_L_star"),
                "branch": b.get("branch"),
            }
            for b in ranked[:5]
        ],
    }

    # Persist LLM-interesting / applied-target bundles for end evaluation.
    interesting = list(state.get("interesting_candidates") or [])
    noted_ids = set()
    for key in ("chosen_bundle_id", "alternative_bundle_id"):
        bid = decision.get(key)
        if bid:
            noted_ids.add(str(bid))
    for alt in decision.get("shortlisted_bundle_ids") or []:
        noted_ids.add(str(alt))
    for b in state.get("bundles") or []:
        if str(b.get("bundle_id")) not in noted_ids:
            continue
        interesting.append(
            {
                "candidate_id": f"decide::{b.get('bundle_id')}",
                "source": "decide_interest",
                "branch": b.get("branch") or "in_distribution",
                "iteration": state.get("iteration", 0),
                "delta_L_star": b.get("delta_L_star"),
                "edits": b.get("edits"),
                "ingredients": (b.get("next_problem") or {}).get("chosen_recipe", {}).get("ingredients")
                if isinstance(b.get("next_problem"), dict)
                else None,
                "opt": {
                    "objective": b.get("L_star_after"),
                    "pfc_after": b.get("pfc_after"),
                    "feasible": True,
                },
                "rationale": decision.get("rationale"),
            }
        )

    # Always attach concrete edits onto the decision for UI progress copy.
    if not decision.get("edits"):
        bid = decision.get("chosen_bundle_id")
        if bid is not None:
            for b in state.get("bundles") or []:
                if str(b.get("bundle_id")) == str(bid) and b.get("edits"):
                    decision["edits"] = list(b.get("edits") or [])
                    break
        if not decision.get("edits") and decision.get("chosen_candidate_id") is not None:
            cid = str(decision.get("chosen_candidate_id"))
            for c in state.get("candidates") or []:
                if str(c.get("candidate_id")) != cid:
                    continue
                decision["edits"] = [
                    {
                        "action": c.get("action"),
                        "label": c.get("label"),
                        "replace_label": c.get("replace_label")
                        or ((c.get("meta") or {}).get("swap_out_label")),
                    }
                ]
                break
    # Keep tool summaries in sync so SSE clients that only read output_summary see edits.
    for tool in tools_used:
        if not isinstance(tool, dict):
            continue
        summary = tool.setdefault("output_summary", {})
        if isinstance(summary, dict) and decision.get("edits") and not summary.get("edits"):
            summary["edits"] = list(decision.get("edits") or [])

    return {
        "decision": decision,
        "history": hist,
        "llm_trace": llm_trace,
        "decision_context": ctx,
        "interesting_candidates": interesting[-24:],
        "tools_used": tools_used,
        "run_telemetry": tel,
        "pending_outcome": pending,
    }


def node_apply_or_expand(state: AgentState) -> dict[str, Any]:
    decision = state.get("decision") or {}
    action = decision.get("action")
    problem = dict(state.get("problem") or {})
    iteration = int(state.get("iteration") or 0) + 1

    if action in {"accept", "accept_pool_best"}:
        return {
            "iteration": iteration,
            "status": "done_pending_finalize",
            "pending_outcome": None,  # no apply → no after needles
            "tools_used": [
                {
                    "name": "apply_or_expand",
                    "purpose": "No recipe change; proceed to finalize",
                    "output_summary": {"action": action},
                    "output": {"action": action},
                }
            ],
        }

    if action == "expand":
        directive = decision.get("expand_directive") or {}
        delta = int(directive.get("delta_k") or 20)
        k = int(state.get("neighbor_k") or 40) + delta
        queries = list(directive.get("neighborhood_search_queries") or [])
        problem_out = dict(problem)
        expand_meta = None
        if queries:
            from recipe_opt_agent.neighborhood_query_expand import (
                derive_focus_terms,
                expand_neighborhood_by_queries,
                fallback_dish_structure,
                normalize_dish_structure,
            )

            cur_labels = [
                str(r.get("label") or r.get("name") or "")
                for r in ((problem_out.get("chosen_recipe") or {}).get("ingredients") or [])
            ]
            dish_structure = normalize_dish_structure(directive.get("dish_structure"))
            if dish_structure is None:
                dish_structure = fallback_dish_structure(
                    stretch_ingredient=None,
                    identity_roles=list(state.get("identity_roles") or []),
                    current_labels=cur_labels,
                    stretch_role="accent",
                )
            # Prefer stretch from structure for focus gating
            stretch_texts = []
            if dish_structure and dish_structure.get("stretch_ingredient"):
                stretch_texts = [dish_structure["stretch_ingredient"]]
            focus_terms = derive_focus_terms(stretch_texts, cur_labels, extra_texts=[str(state.get("title") or "")])
            if not focus_terms:
                focus_terms = derive_focus_terms(queries, cur_labels, extra_texts=[str(state.get("title") or "")])
            exp = expand_neighborhood_by_queries(
                problem_out,
                queries,
                focus_terms=focus_terms,
                dish_structure=dish_structure,
            )
            problem_out = exp.get("problem") or problem_out
            expand_meta = exp.get("meta")
        tel = bump_telemetry(state.get("run_telemetry"), expand_count=1, nodes={"apply": {"status": "expanded"}})
        return {
            "iteration": iteration,
            "neighbor_k": k,
            "expand_directive": directive,
            "problem": problem_out,
            "status": "expanded",
            "run_telemetry": tel,
            "tools_used": [
                {
                    "name": "expand_neighborhood",
                    "purpose": (
                        "Widen neighbor retrieval via LLM search queries + dish_structure "
                        "gram-share verification; bump neighbor_k"
                    ),
                    "output_summary": {
                        "neighbor_k": k,
                        "delta_k": delta,
                        "n_queries": len(queries),
                        "query_expansion": expand_meta,
                    },
                    "output": {
                        "neighbor_k": k,
                        "directive": directive,
                        "query_expansion": expand_meta,
                    },
                }
            ],
        }

    tags = _requirement_tags(state)

    # --- apply a scored bundle atomically ---
    if action == "apply_bundle":
        bundle_id = decision.get("chosen_bundle_id")
        bundles = {str(b.get("bundle_id")): b for b in (state.get("bundles") or [])}
        bundle = bundles.get(str(bundle_id)) if bundle_id is not None else None
        if bundle is None or not bundle.get("next_problem"):
            tel = bump_telemetry(state.get("run_telemetry"), nodes={"apply": {"status": "expand_needed"}})
            return {
                "iteration": iteration,
                "status": "expand_needed",
                "run_telemetry": tel,
                "tools_used": [
                    {
                        "name": "apply_bundle",
                        "purpose": "Apply chosen edit bundle",
                        "output_summary": {
                            "error": "bundle_not_found_or_no_next_problem",
                            "chosen_bundle_id": bundle_id,
                        },
                        "output": {"error": "bundle_not_found_or_no_next_problem", "chosen_bundle_id": bundle_id},
                    }
                ],
            }
        if tags:
            for edit in bundle.get("edits") or []:
                if edit.get("action") in {"add", "swap"}:
                    label = str(edit.get("label") or "")
                    if not ingredient_passes_tags(label, tags):
                        tel = bump_telemetry(
                            state.get("run_telemetry"), nodes={"apply": {"status": "tag_rollback"}}
                        )
                        return {
                            "iteration": iteration,
                            "status": "tag_violation_rollback",
                            "run_telemetry": tel,
                            "tools_used": [
                                {
                                    "name": "apply_bundle",
                                    "purpose": "Post-apply tag validation",
                                    "output_summary": {
                                        "error": "tag_violation",
                                        "label": label,
                                        "violations": tag_violations_for_ingredient(label, tags),
                                    },
                                    "output": {"error": "tag_violation", "bundle": {k: v for k, v in bundle.items() if k != "next_problem"}},
                                }
                            ],
                        }
        next_problem = bundle["next_problem"]
        bundle_pub = {k: v for k, v in bundle.items() if k != "next_problem"}
        fps = list(state.get("recent_edit_fingerprints") or [])
        fps.append(edit_fingerprint(bundle.get("edits")))
        tel = bump_telemetry(state.get("run_telemetry"), nodes={"apply": {"status": "applied", "kind": "bundle"}})
        return {
            "iteration": iteration,
            "problem": next_problem,
            "chosen_recipe": next_problem.get("chosen_recipe") or state.get("chosen_recipe"),
            "status": "modified",
            "last_applied_candidate": bundle_pub,
            "recent_edit_fingerprints": fps[-6:],
            "run_telemetry": tel,
            "tools_used": [
                {
                    "name": "apply_bundle",
                    "purpose": "Replace the LP problem with the bundle's jointly re-optimized next_problem",
                    "output_summary": {
                        "bundle_id": bundle_pub.get("bundle_id"),
                        "edits": bundle_pub.get("edits"),
                        "delta_L_star": bundle_pub.get("delta_L_star"),
                        "n_ingredients_after": len(np.asarray(next_problem.get("x0") or []).ravel()),
                    },
                    "output": bundle_pub,
                }
            ],
        }

    # --- apply a single modification candidate ---
    cand_id = decision.get("chosen_candidate_id")
    cands = {c["candidate_id"]: c for c in (state.get("candidates") or [])}
    cand = cands.get(str(cand_id)) if cand_id is not None else None
    if cand is None:
        tel = bump_telemetry(state.get("run_telemetry"), nodes={"apply": {"status": "expand_needed"}})
        return {
            "iteration": iteration,
            "status": "expand_needed",
            "run_telemetry": tel,
            "tools_used": [
                {
                    "name": "apply_modification",
                    "purpose": "Apply chosen candidate",
                    "output_summary": {"error": "candidate_not_found", "chosen_candidate_id": cand_id},
                    "output": {"error": "candidate_not_found", "chosen_candidate_id": cand_id},
                }
            ],
        }

    if tags and cand.get("action") in {"add", "swap"}:
        label = str(cand.get("label") or "")
        if not ingredient_passes_tags(label, tags):
            tel = bump_telemetry(state.get("run_telemetry"), nodes={"apply": {"status": "tag_rollback"}})
            return {
                "iteration": iteration,
                "status": "tag_violation_rollback",
                "run_telemetry": tel,
                "tools_used": [
                    {
                        "name": "apply_modification",
                        "purpose": "Post-apply tag validation",
                        "output_summary": {
                            "error": "tag_violation",
                            "label": label,
                            "violations": tag_violations_for_ingredient(label, tags),
                        },
                        "output": {"error": "tag_violation", "candidate": cand},
                    }
                ],
            }

    # Use the candidate's next_problem when present; otherwise materialize one live.
    next_problem = (cand.get("meta") or {}).get("next_problem")
    apply_note = "candidate_meta"
    if next_problem is None:
        from recipe_opt_agent.bundle_scoring import apply_edits_to_problem

        next_problem = apply_edits_to_problem(problem, [cand])
        apply_note = "materialized_live"
    if next_problem is not None:
        problem = next_problem
        status = "modified"
        apply_status = "applied"
    else:
        # Cannot mutate the LP (no macro data for the edit) — keep the stub for traceability.
        problem = {**problem, "last_applied": cand}
        status = "modified"
        apply_note = "stub_no_macro_data"
        apply_status = "stub"

    fps = list(state.get("recent_edit_fingerprints") or [])
    fps.append(edit_fingerprint([cand]))
    tel = bump_telemetry(state.get("run_telemetry"), nodes={"apply": {"status": apply_status, "kind": "single"}})
    cand_edits = [
        {
            "action": cand.get("action"),
            "label": cand.get("label"),
            "replace_label": cand.get("replace_label")
            or ((cand.get("meta") or {}).get("swap_out_label")),
            "candidate_id": cand.get("candidate_id"),
        }
    ]
    cand_pub = {**cand, "edits": cand.get("edits") or cand_edits}

    return {
        "iteration": iteration,
        "problem": problem,
        "chosen_recipe": (problem.get("chosen_recipe") or state.get("chosen_recipe")),
        "status": status,
        "last_applied_candidate": cand_pub,
        "recent_edit_fingerprints": fps[-6:],
        "run_telemetry": tel,
        "tools_used": [
            {
                "name": "apply_modification",
                "purpose": f"Apply {cand.get('action')} candidate to the recipe problem",
                "output_summary": {
                    "candidate_id": cand.get("candidate_id"),
                    "label": cand.get("label"),
                    "action": cand.get("action"),
                    "edits": cand_pub.get("edits"),
                    "apply_note": apply_note,
                    "n_ingredients_after": len(np.asarray(problem.get("x0") or []).ravel()),
                },
                "output": {"candidate": cand_pub, "apply_note": apply_note},
            }
        ],
    }


def _enrich_final_display(state: AgentState, final_payload: dict[str, Any]) -> dict[str, Any]:
    """Attach authoritative display_scores + sync telemetry from the chosen recipe."""
    from recipe_opt_agent.score_display import (
        build_display_scores,
        extract_judge_rationale,
        extract_ratio_and_nutrient,
        filter_display_ingredients,
        prepare_browse_candidates,
        proportion_typicality_from_ingredients,
        select_path_finalists,
    )

    cfg = _cfg(state)
    final_payload = {
        **final_payload,
        "config": state.get("config") or {},
        "macro_targets": cfg.target_box_dict(),
        "problem": state.get("problem"),
        "opt": (final_payload.get("chosen") or {}).get("opt") or state.get("opt"),
        "iteration": state.get("iteration"),
        "score_history": state.get("score_history") or [],
        "chosen_recipe": state.get("chosen_recipe"),
    }
    chosen = final_payload.get("chosen") or {}
    entry = chosen.get("entry") if isinstance(chosen.get("entry"), dict) else None
    if entry and isinstance(entry.get("opt"), dict):
        final_payload["opt"] = entry["opt"]
    final_payload["last_applied_candidate"] = (
        final_payload.get("last_applied_candidate") or state.get("last_applied_candidate")
    )
    final_payload["decision_outcomes"] = (
        final_payload.get("decision_outcomes") or state.get("decision_outcomes") or []
    )
    final_payload["original_ingredients"] = (
        final_payload.get("original_ingredients")
        or state.get("original_ingredients")
        or []
    )
    display = build_display_scores(final_payload)
    try:
        from recipe_opt_agent.portion_display import (
            consolidate_duplicate_ingredients,
            enrich_ingredients_with_usda_portions,
        )

        display = dict(display)
        display["ingredients"] = enrich_ingredients_with_usda_portions(
            list(display.get("ingredients") or [])
        )
        merged, problem_update = consolidate_duplicate_ingredients(
            list(display.get("ingredients") or []),
            problem=final_payload.get("problem")
            if isinstance(final_payload.get("problem"), dict)
            else None,
        )
        merged = filter_display_ingredients(merged)
        display["ingredients"] = merged
        if problem_update is not None:
            final_payload["problem"] = problem_update
        # Refresh proportion copy after merge so the verdict matches the rows shown.
        try:
            typicality = proportion_typicality_from_ingredients(merged)
            ratio = dict(display.get("ratio_loss") or {})
            ratio["band"] = typicality.get("band") or ratio.get("band")
            ratio["band_summary"] = typicality.get("summary") or ratio.get("band_summary")
            ratio["proportion_key"] = typicality.get("key")
            ratio["proportion_css"] = typicality.get("css")
            ratio["outside_iqr_frac"] = typicality.get("outside_iqr_frac")
            ratio["outside_iqr_pct"] = typicality.get("outside_iqr_pct")
            ratio["outside_iqr_calorie_frac"] = typicality.get("outside_iqr_calorie_frac")
            ratio["outside_iqr_calorie_pct"] = typicality.get("outside_iqr_calorie_pct")
            display["ratio_loss"] = ratio
        except Exception:
            pass
    except Exception:
        pass

    # Enforce the user calorie target on the primary displayed recipe.
    from recipe_opt_agent.kcal_utils import resolve_user_kcal_target, scale_candidate_to_kcal

    user_kcal = resolve_user_kcal_target(
        final_payload.get("config"),
        state.get("config"),
        final_payload.get("problem"),
        state.get("problem"),
        cfg,
    )
    if user_kcal is not None:
        problem_scaled, display_scaled = scale_candidate_to_kcal(
            problem=final_payload.get("problem") if isinstance(final_payload.get("problem"), dict) else {},
            display=display,
            kcal_target=user_kcal,
        )
        if display_scaled is not None:
            display = display_scaled
        if problem_scaled is not None:
            final_payload["problem"] = problem_scaled
    final_payload["display_scores"] = display

    ratio, ratio_src, nutrient, _nut_src = extract_ratio_and_nutrient(final_payload)
    tel = dict(final_payload.get("run_telemetry") or {})
    tel["final_ratio_term"] = ratio
    tel["final_ratio_source"] = ratio_src
    tel["final_nutrient_slack"] = nutrient
    if display.get("holistic_0_10", {}).get("value") is not None:
        h = display["holistic_0_10"]["value"]
        if display["holistic_0_10"].get("source") == "llm_judge":
            tel["final_holistic"] = float(h) / 10.0
        elif tel.get("final_holistic") is None:
            tel["final_holistic"] = float(h) / 10.0
    final_payload["run_telemetry"] = tel
    final_payload["path_finals"] = select_path_finalists(
        state,
        ood_handicap=float(getattr(cfg, "ood_delta_handicap", 0.015) or 0.0),
    )
    enable_judge = bool(getattr(cfg, "enable_llm_judge", False))
    judge_mode = str(getattr(cfg, "llm_judge_mode", "demote_weird") or "demote_weird").lower()
    if enable_judge and judge_mode == "full" and not final_payload.get("judge_rationale"):
        final_payload["judge_rationale"] = extract_judge_rationale(final_payload)
    elif not final_payload.get("weird_candidate_ids"):
        # Demote mode only surfaces rationale when something was actually demoted.
        if judge_mode != "full":
            final_payload["judge_rationale"] = None
    browse = prepare_browse_candidates(
        final_payload,
        state,
        max_candidates=int(getattr(cfg, "max_finalists", 4) or 4),
    )
    final_payload["browse_candidates"] = browse
    # Primary result = best card after proportion rank + weird demotion.
    if browse:
        best = browse[0]
        if best.get("display_scores"):
            final_payload["display_scores"] = best["display_scores"]
        if isinstance(best.get("problem"), dict) and best["problem"]:
            final_payload["problem"] = best["problem"]
        if isinstance(best.get("macro_targets"), dict) and best["macro_targets"]:
            final_payload["macro_targets"] = best["macro_targets"]
        chosen = dict(final_payload.get("chosen") or {})
        if judge_mode != "full" or chosen.get("source") != "final_arbiter":
            chosen["source"] = "proportion_rank"
        chosen["proportion_rank"] = 1
        chosen["candidate_id"] = best.get("candidate_id")
        entry = dict(chosen.get("entry") or {}) if isinstance(chosen.get("entry"), dict) else {}
        entry["candidate_id"] = best.get("candidate_id")
        chosen["entry"] = entry
        if judge_mode != "full":
            chosen.pop("arbiter_rationale", None)
            chosen.pop("arbiter_winner_id", None)
            chosen.pop("judge", None)
        final_payload["chosen"] = chosen
        # Refresh telemetry ratio from the promoted card.
        ratio, ratio_src, nutrient, _nut_src = extract_ratio_and_nutrient(final_payload)
        tel = dict(final_payload.get("run_telemetry") or {})
        tel["final_ratio_term"] = ratio
        tel["final_ratio_source"] = ratio_src
        tel["final_nutrient_slack"] = nutrient
        tel["proportion_rank_winner_id"] = best.get("candidate_id")
        tel["n_weird_demoted"] = sum(1 for c in browse if c.get("is_weird"))
        final_payload["run_telemetry"] = tel
    return final_payload


def _attach_final_gpt4o_evaluation(
    state: AgentState,
    final_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    """Optional GPT holistic evaluation — only in full judge mode."""
    cfg = _cfg(state)
    if not bool(getattr(cfg, "enable_llm_judge", False)):
        return final_payload, None, []
    if str(getattr(cfg, "llm_judge_mode", "demote_weird") or "demote_weird").lower() != "full":
        return final_payload, None, []

    from recipe_opt_agent.final_evaluator import evaluate_final_recipe
    from recipe_opt_agent.score_display import band_for_holistic_0_10

    tools: list[dict[str, Any]] = []
    try:
        evaluation = evaluate_final_recipe(state, final_payload, model="gpt-5.5")
    except Exception as exc:
        evaluation = {"error": str(exc)}
    if not evaluation or evaluation.get("error"):
        return final_payload, evaluation, tools

    public = {k: v for k, v in evaluation.items() if k not in {"_llm_trace", "briefing"}}
    final_payload["final_evaluation"] = public
    judge_model = (evaluation.get("_llm_trace") or {}).get("model") or "gpt-5.5"
    if evaluation.get("overall_score_0_10") is not None:
        h = float(evaluation["overall_score_0_10"])
        display = dict(final_payload.get("display_scores") or {})
        display["holistic_0_10"] = {
            "value": h,
            "band": band_for_holistic_0_10(h),
            "source": f"{judge_model}_final_evaluator",
        }
        final_payload["display_scores"] = display
        tel = dict(final_payload.get("run_telemetry") or {})
        tel["final_holistic"] = h / 10.0
        final_payload["run_telemetry"] = tel

    tools.append(
        {
            "name": "final_evaluator_gpt55",
            "purpose": (
                "GPT-5.5 holistic evaluation: ingredient plausibility calibrated by neighborhood "
                "hull stretch, user needs, dietary restriction flag"
            ),
            "mode": (evaluation.get("_llm_trace") or {}).get("mode"),
            "model": judge_model,
            "output_summary": {
                "overall_score_0_10": evaluation.get("overall_score_0_10"),
                "ingredients_make_sense": evaluation.get("ingredients_make_sense"),
                "meets_user_needs": evaluation.get("meets_user_needs"),
                "dietary_violation_flag": evaluation.get("dietary_violation_flag"),
                "target_stretch_level": (
                    (evaluation.get("briefing") or {})
                    .get("macro_target_context", {})
                    .get("target_stretch_level")
                ),
            },
            "output": public,
            "llm_trace": evaluation.get("_llm_trace"),
        }
    )
    return final_payload, evaluation, tools


def _run_llm_judge_pass(
    state: AgentState,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Run demote-weird screen or full arbiter. Returns (judgment, tools)."""
    cfg = _cfg(state)
    if not bool(getattr(cfg, "enable_llm_judge", False)):
        return None, []
    mode = str(getattr(cfg, "llm_judge_mode", "demote_weird") or "demote_weird").lower()
    tools: list[dict[str, Any]] = []
    if mode == "full":
        try:
            from recipe_opt_agent.final_arbiter import arbitrate_final_recipe

            judgment = arbitrate_final_recipe(state)
        except Exception as exc:
            judgment = {"error": str(exc)}
        if judgment and not judgment.get("error"):
            public_judgment = {
                k: v
                for k, v in judgment.items()
                if k not in {"_llm_trace", "winner_entry", "_shadow_consideration"}
            }
            tools.append(
                {
                    "name": "final_arbiter_llm",
                    "purpose": (
                        "Final judgment across saved candidates: relative loss gaps vs intent drift "
                        "(clash ingredients need overwhelming numeric case)"
                    ),
                    "mode": (judgment.get("_llm_trace") or {}).get("mode"),
                    "model": (judgment.get("_llm_trace") or {}).get("model"),
                    "output_summary": {
                        "winner_id": judgment.get("winner_id"),
                        "ranking": judgment.get("ranking"),
                        "rationale": judgment.get("rationale"),
                        "holistic_0_10": judgment.get("holistic_0_10"),
                        "n_candidates": len(judgment.get("comparison") or []),
                    },
                    "output": public_judgment,
                    "llm_trace": judgment.get("_llm_trace"),
                }
            )
        return judgment, tools

    # Default: demote_weird — flag disasters only; do not pick a winner.
    try:
        from recipe_opt_agent.final_arbiter import flag_weird_candidates, weird_ids_from_judgment

        judgment = flag_weird_candidates(state)
    except Exception as exc:
        judgment = {"error": str(exc)}
    if judgment and not judgment.get("error"):
        flags = weird_ids_from_judgment(judgment)
        judgment = {
            **judgment,
            "weird_flags": flags,
            "weird_candidate_ids": list(judgment.get("weird_candidate_ids") or list(flags.keys())),
        }
        tools.append(
            {
                "name": "weirdness_flagger_llm",
                "purpose": (
                    "Screen candidates for culinary disasters only; demote weird recipes "
                    "without ranking the rest"
                ),
                "mode": (judgment.get("_llm_trace") or {}).get("mode"),
                "model": (judgment.get("_llm_trace") or {}).get("model"),
                "output_summary": {
                    "n_weird": len(judgment.get("weird_candidate_ids") or []),
                    "weird_candidate_ids": judgment.get("weird_candidate_ids"),
                    "rationale": judgment.get("rationale"),
                },
                "output": {
                    k: v
                    for k, v in judgment.items()
                    if k not in {"_llm_trace", "comparison"}
                },
                "llm_trace": judgment.get("_llm_trace"),
            }
        )
    return judgment, tools


def _apply_judge_to_final_payload(
    final_payload: dict[str, Any],
    judgment: dict[str, Any] | None,
    *,
    mode: str,
) -> dict[str, Any]:
    """Attach judge outputs for browse demotion / optional full-mode pick metadata."""
    if not judgment or judgment.get("error"):
        return final_payload
    mode = (mode or "demote_weird").lower()
    if mode == "full":
        final_payload["final_judgment"] = {
            k: v for k, v in judgment.items() if not str(k).startswith("_")
        }
        if judgment.get("rationale"):
            final_payload["judge_rationale"] = judgment.get("rationale")
        # Also demote clash verdicts in full mode so browse stays consistent.
        from recipe_opt_agent.final_arbiter import weird_ids_from_judgment

        flags = weird_ids_from_judgment(judgment)
        if flags:
            final_payload["weird_flags"] = flags
            final_payload["weird_candidate_ids"] = list(flags.keys())
        return final_payload

    flags = judgment.get("weird_flags") or {}
    if not flags:
        from recipe_opt_agent.final_arbiter import weird_ids_from_judgment

        flags = weird_ids_from_judgment(judgment)
    final_payload["weird_flags"] = flags
    final_payload["weird_candidate_ids"] = list(
        judgment.get("weird_candidate_ids") or list(flags.keys())
    )
    final_payload["final_judgment"] = {
        k: v
        for k, v in judgment.items()
        if k in {"flags", "weird_flags", "weird_candidate_ids", "rationale"}
        or (not str(k).startswith("_") and k not in {"comparison"})
    }
    if judgment.get("rationale") and final_payload.get("weird_candidate_ids"):
        final_payload["judge_rationale"] = judgment.get("rationale")
    else:
        final_payload["judge_rationale"] = None
    return final_payload


def node_finalize(state: AgentState) -> dict[str, Any]:
    shadow_patch = _await_shadow_candidate(state)
    if shadow_patch:
        state = {**state, **{k: v for k, v in shadow_patch.items() if not str(k).startswith("_")}}
        # Keep private collect meta on state for arbiter audit.
        if shadow_patch.get("_shadow_collect_meta") is not None:
            state = {**state, "_shadow_collect_meta": shadow_patch["_shadow_collect_meta"]}

    band = state.get("fidelity_band")
    decision = state.get("decision") or {}
    action = decision.get("action")
    pool = list(state.get("candidate_pool") or [])
    mode = _agent_mode(state)
    tel = finalize_telemetry(state, status="")

    # Creative mode: use scored finalists if present
    if mode == "creative" and state.get("scored_finalists"):
        scored = list(state.get("scored_finalists") or [])
        # Deterministic order only — LLM judge is off; browse rank will re-order
        # by proportion quality and promote the best card as Recommended.
        max_n = max(1, int((_cfg(state).max_finalists)))
        ordered = scored[:max_n]
        winner = ordered[0] if ordered else None
        alts = ordered[1:]
        foodon_report = (winner or {}).get("foodon_basis_report") or (
            (state.get("problem") or {}).get("foodon_basis_report")
            or state.get("foodon_basis_report")
        )
        chosen = {
            "source": "creative_finalist",
            "entry": winner,
            "foodon_basis_report": foodon_report,
        }
        status = "accepted_creative"
        tel = finalize_telemetry(state, status=status)
        final_payload = {
            "status": status,
            "chosen": chosen,
            "foodon_basis_report": foodon_report,
            "alternatives": alts,
            "scored_finalists": ordered,
            "interesting_candidates": state.get("interesting_candidates") or [],
            "history": state.get("history"),
            "decision_outcomes": state.get("decision_outcomes"),
            "run_telemetry": tel,
            "title": state.get("title"),
            "taste_text": state.get("taste_text"),
            "user_request": state.get("user_request"),
            "requirement_tags": state.get("requirement_tags"),
            "judge_result": None,
        }
        judge_mode = str(getattr(_cfg(state), "llm_judge_mode", "demote_weird") or "demote_weird")
        judgment, judge_tools = _run_llm_judge_pass(state)
        final_payload = _apply_judge_to_final_payload(final_payload, judgment, mode=judge_mode)
        final_payload = _enrich_final_display(state, final_payload)
        final_payload, final_evaluation, eval_tools = _attach_final_gpt4o_evaluation(state, final_payload)
        return {
            "final": final_payload,
            "status": status,
            "run_telemetry": final_payload.get("run_telemetry") or tel,
            "live_scores": final_payload.get("display_scores"),
            "final_evaluation": final_evaluation,
            "tools_used": list(judge_tools) + list(eval_tools),
        }

    if band == FidelityBand.ACCEPT.value or action == "accept":
        chosen = {
            "source": "current",
            "opt": state.get("opt"),
            "diagnosis": state.get("diagnosis"),
        }
        status = "accepted"
    elif pool:
        best = min(pool, key=lambda p: (p.get("n_red", 99), p.get("L_max_norm", 99.0), p.get("L_total", 99.0)))
        chosen = {"source": "candidate_pool", "entry": best}
        status = "accepted_pool_best"
    else:
        chosen = {"source": "best_effort", "opt": state.get("opt"), "diagnosis": state.get("diagnosis")}
        status = "failed_or_best_effort"

    # LLM judge: demote weird recipes (default) or full arbiter pick (llm_judge_mode=full).
    cfg = _cfg(state)
    judge_mode = str(getattr(cfg, "llm_judge_mode", "demote_weird") or "demote_weird").lower()
    final_judgment, judge_tools = _run_llm_judge_pass(state)
    if (
        judge_mode == "full"
        and final_judgment
        and final_judgment.get("winner_id")
        and final_judgment.get("winner_entry")
    ):
        winner = final_judgment["winner_entry"]
        chosen = {
            "source": "final_arbiter",
            "entry": winner,
            "opt": winner.get("opt"),
            "ingredients": winner.get("ingredients"),
            "arbiter_winner_id": final_judgment["winner_id"],
            "arbiter_rationale": final_judgment.get("rationale"),
        }
        status = "accepted_final_judgment"

    problem = state.get("problem") or {}
    foodon_report = (
        problem.get("foodon_basis_report")
        or state.get("foodon_basis_report")
        or (problem.get("chosen_recipe") or {}).get("foodon_basis_report")
    )
    if isinstance(chosen, dict) and foodon_report and "foodon_basis_report" not in chosen:
        chosen = {**chosen, "foodon_basis_report": foodon_report}
    tel = finalize_telemetry(state, status=status)
    final_payload = {
        "status": status,
        "chosen": chosen,
        "foodon_basis_report": foodon_report,
        "interesting_candidates": state.get("interesting_candidates") or [],
        "history": state.get("history"),
        "decision_outcomes": state.get("decision_outcomes"),
        "run_telemetry": tel,
        "title": state.get("title"),
        "taste_text": state.get("taste_text"),
        "judge_result": state.get("judge_result") if judge_mode == "full" else None,
        "final_judgment": None,
    }
    final_payload = _apply_judge_to_final_payload(final_payload, final_judgment, mode=judge_mode)
    # Prefer live chosen_recipe ingredients when pool/current lacks them
    if not (chosen.get("ingredients") or (isinstance(chosen.get("entry"), dict) and chosen["entry"].get("ingredients"))):
        cr = state.get("chosen_recipe") or (state.get("problem") or {}).get("chosen_recipe") or {}
        if cr.get("ingredients"):
            chosen = {**chosen, "ingredients": cr["ingredients"]}
            final_payload["chosen"] = chosen
    final_payload = _enrich_final_display(state, final_payload)
    final_payload, final_evaluation, eval_tools = _attach_final_gpt4o_evaluation(state, final_payload)
    out: dict[str, Any] = {
        "final": final_payload,
        "status": status,
        "run_telemetry": final_payload.get("run_telemetry") or tel,
        "live_scores": final_payload.get("display_scores"),
        "final_judgment": (
            {k: v for k, v in final_judgment.items() if not str(k).startswith("_")}
            if isinstance(final_judgment, dict)
            else final_judgment
        ),
        "final_evaluation": final_evaluation,
    }
    tools_used: list[dict[str, Any]] = list(judge_tools) + list(eval_tools)
    # Persist shadow/arbiter consideration on telemetry only (not tools_used / UI).
    consideration = (final_judgment or {}).get("_shadow_consideration") if isinstance(final_judgment, dict) else None
    if consideration:
        tel = dict(final_payload.get("run_telemetry") or tel or {})
        nodes = dict(tel.get("nodes") or {})
        nodes["shadow_arbiter_consideration"] = consideration
        tel["nodes"] = nodes
        final_payload["run_telemetry"] = tel
        out["run_telemetry"] = tel
    if tools_used:
        out["tools_used"] = tools_used
    if shadow_patch:
        if shadow_patch.get("candidate_pool") is not None:
            out["candidate_pool"] = shadow_patch["candidate_pool"]
        if shadow_patch.get("interesting_candidates") is not None:
            out["interesting_candidates"] = shadow_patch["interesting_candidates"]
        if "shadow_job_id" in shadow_patch:
            out["shadow_job_id"] = shadow_patch["shadow_job_id"]
    return out


def _route_after_diagnose(state: AgentState) -> Literal["save_candidate", "propose", "finalize", "build_finalists"]:
    band = state.get("fidelity_band")
    it = int(state.get("iteration") or 0)
    cfg = _cfg(state)
    mode = _agent_mode(state)

    if mode == "creative":
        if it >= cfg.max_iterations:
            return "build_finalists"
        if band in {FidelityBand.MODERATE.value, FidelityBand.MUST_RETRY.value, FidelityBand.ACCEPT.value}:
            return "save_candidate"
        return "propose"

    if band == FidelityBand.ACCEPT.value:
        # Never finalize on the first diagnose pass — always score at least one
        # propose/decide cycle so proportions can improve before early accept.
        if it < 1:
            return "propose"
        return "finalize"
    if it >= cfg.max_iterations:
        return "finalize"
    if band == FidelityBand.MODERATE.value:
        return "save_candidate"
    if band == FidelityBand.MUST_RETRY.value and cfg.save_on_must_retry_feasible:
        # node_save_candidate itself verifies macro feasibility before saving.
        return "save_candidate"
    return "propose"


def _route_after_save(state: AgentState) -> Literal["propose", "build_finalists"]:
    it = int(state.get("iteration") or 0)
    cfg = _cfg(state)
    if _agent_mode(state) == "creative" and it >= cfg.max_iterations:
        return "build_finalists"
    return "propose"


def _route_after_decide(state: AgentState) -> Literal["finalize", "apply", "build_finalists"]:
    action = (state.get("decision") or {}).get("action")
    it = int(state.get("iteration") or 0)
    cfg = _cfg(state)
    mode = _agent_mode(state)
    if mode == "creative":
        if it + 1 >= cfg.max_iterations and action in {"accept", "accept_pool_best"}:
            return "build_finalists"
        return "apply"
    if action in {"accept", "accept_pool_best"}:
        # Route through apply on iter 0 so iteration advances before finalize.
        if it < 1:
            return "apply"
        return "finalize"
    if it + 1 >= cfg.max_iterations and action != "expand":
        return "apply"
    return "apply"


def _route_after_apply(state: AgentState) -> Literal["diagnose", "finalize", "build_finalists"]:
    status = state.get("status")
    it = int(state.get("iteration") or 0)
    cfg = _cfg(state)
    mode = _agent_mode(state)
    if status in {"done_pending_finalize"}:
        return "finalize" if mode != "creative" else "build_finalists"
    if mode == "creative" and it >= cfg.max_iterations:
        return "build_finalists"
    if it >= cfg.max_iterations:
        return "finalize"
    return "diagnose"


def node_deduce_tags(state: AgentState) -> dict[str, Any]:
    cfg = _cfg(state)
    request = state.get("user_request") or state.get("taste_text") or ""
    draft = state.get("recipe_draft") or {}
    tags = deduce_requirement_tags(
        request,
        draft_tags=draft.get("requirement_tags"),
        model=select_tags_model(cfg),
    )
    return {
        "requirement_tags": [t.to_dict() for t in tags],
        "tools_used": [
            {
                "name": "deduce_requirement_tags",
                "purpose": "Hard dietary/macro tags from request + draft",
                "model": select_tags_model(cfg),
                "output_summary": {"n_tags": len(tags), "tag_ids": [t.tag_id for t in tags]},
                "output": [t.to_dict() for t in tags],
            }
        ],
    }


def node_llm_draft(state: AgentState) -> dict[str, Any]:
    cfg = _cfg(state)
    request = state.get("user_request") or state.get("taste_text") or ""
    model = select_draft_model(cfg)
    example = (state.get("problem") or {}).get("example_recipe")
    from recipe_opt_agent.kcal_utils import resolve_kcal_target

    kcal_target = resolve_kcal_target(cfg, state.get("problem"))
    draft, trace = llm_draft_recipe(
        request,
        macro_box=cfg.target_box_dict(),
        example_recipe=example if isinstance(example, dict) else None,
        canonical_title=state.get("title") or None,
        kcal_target=kcal_target,
        model=model,
    )
    tel = bump_telemetry(
        state.get("run_telemetry"),
        n_llm_calls=1 if (trace or {}).get("mode") == "openai" else 0,
        nodes={
            "draft": {
                "model": trace.get("model") or model,
                "n_ingredients": len(draft.get("ingredients") or []),
                "used_example_recipe": bool(example),
            }
        },
    )
    return {
        "recipe_draft": draft,
        "title": draft.get("title") or state.get("title"),
        "llm_trace": trace,
        "run_telemetry": tel,
        "tools_used": [
            {
                "name": "llm_draft_recipe",
                "purpose": (
                    "Structured gram-level recipe JSON from user request"
                    + (" + neighborhood example recipe" if example else "")
                ),
                "mode": trace.get("mode"),
                "model": trace.get("model") or model,
                "output_summary": {
                    "title": draft.get("title"),
                    "n_ingredients": len(draft.get("ingredients") or []),
                    "used_example_recipe": bool(example),
                    "example_title": (example or {}).get("title") if isinstance(example, dict) else None,
                },
                "output": draft,
                "llm_trace": trace,
            }
        ],
    }


def node_shadow_gpt_candidate(state: AgentState) -> dict[str, Any]:
    """Start GPT-5.5 draft→optimize in a background thread (non-blocking)."""
    from recipe_opt_agent.shadow_gpt_candidate import start_shadow_gpt_job
    from recipe_opt_agent.telemetry import bump_telemetry

    cfg = _cfg(state)
    if not bool(getattr(cfg, "enable_shadow_gpt_candidate", True)):
        return {}
    if state.get("shadow_job_id"):
        return {}
    job_id = start_shadow_gpt_job(
        state,
        model=select_shadow_draft_model(cfg),
        enabled=True,
    )
    if not job_id:
        return {}
    tel = bump_telemetry(
        state.get("run_telemetry"),
        nodes={
            "shadow_gpt_candidate": {
                "async": True,
                "job_started": True,
                "job_id": job_id,
                # Model stays in backend telemetry only (not tools_used / UI steps).
                "_shadow_model": select_shadow_draft_model(cfg),
            }
        },
    )
    # Intentionally omit tools_used / model so UIs do not surface authorship.
    return {
        "shadow_job_id": job_id,
        "run_telemetry": tel,
    }


def _await_shadow_candidate(state: AgentState) -> dict[str, Any]:
    """Join the async shadow job and merge into pools before finalists/arbiter."""
    from recipe_opt_agent.shadow_gpt_candidate import merge_shadow_into_pools
    from recipe_opt_agent.telemetry import bump_telemetry

    if not state.get("shadow_job_id") and any(
        str((e or {}).get("candidate_id") or "").startswith("pool_shadow")
        for e in list(state.get("candidate_pool") or [])
        + list(state.get("interesting_candidates") or [])
    ):
        return {}
    merged = merge_shadow_into_pools(state)
    meta = merged.get("shadow_collect_meta") or {}
    tel = bump_telemetry(
        state.get("run_telemetry"),
        nodes={
            "shadow_gpt_collect": {
                "async": True,
                "collected": bool(meta.get("collected") or meta.get("skipped")),
                "has_entry": bool(meta.get("has_entry")),
                "timed_out": bool(meta.get("timed_out")),
                "elapsed_s": meta.get("elapsed_s"),
                "error": meta.get("error"),
                "_shadow_model": meta.get("model"),
            }
        },
    )
    out = {k: v for k, v in merged.items() if k != "shadow_collect_meta"}
    out["run_telemetry"] = tel
    out["_shadow_collect_meta"] = meta
    return out


def node_ground_recipe(state: AgentState) -> dict[str, Any]:
    from recipe_opt_agent.draft_schema import parse_draft
    from recipe_opt_agent.grounding import ground_draft_to_problem

    draft = parse_draft(state.get("recipe_draft") or {})
    tags = _requirement_tags(state)
    problem_stub = state.get("problem") or {}
    ctx = problem_stub.get("retrieval_context") or {}
    nb_cat = list(ctx.get("fdc_catalog") or [])

    problem, report, chosen = ground_draft_to_problem(
        draft,
        requirement_tags=tags,
        neighborhood_catalog=nb_cat,
        broader_catalog=nb_cat,
        basis_samples=problem_stub.get("basis_samples"),
        ratio_samples=problem_stub.get("ratio_samples"),
        retrieval_context=ctx,
        offline=bool(problem_stub.get("grounding_offline") or problem_stub.get("creative_offline")),
        use_dequant_cache=True,
    )
    from recipe_opt_agent.kcal_utils import restore_kcal_target

    problem = restore_kcal_target(problem, _cfg(state), problem_stub)
    # Keep neighborhood geometry / hull context from the stub when present.
    for key in (
        "basis_samples",
        "basis_sample_weights",
        "ratio_samples",
        "marginal_nodes",
        "neighborhood_hull_context",
        "foodon_basis_report",
        "neighborhood_recipes",
        "retrieval_context",
    ):
        if problem_stub.get(key) is not None and problem.get(key) is None:
            problem[key] = problem_stub[key]
    roles = resolve_identity_roles(
        title=chosen.get("title") or state.get("title") or draft.title or "",
        request=state.get("user_request") or state.get("taste_text") or "",
        ingredients=chosen.get("ingredients") or [],
        templates=_cfg(state).identity_templates,
        use_llm=True,
        model=_cfg(state).identity_extract_model,
        foodon_basis_report=problem.get("foodon_basis_report"),
        prefer_title_priors=True,
    )
    report_dict = report.to_dict()
    resolve_rate = None
    try:
        n_req = len((state.get("recipe_draft") or {}).get("ingredients") or [])
        n_ok = len(chosen.get("ingredients") or [])
        resolve_rate = (n_ok / n_req) if n_req else None
    except Exception:
        resolve_rate = None
    tel = bump_telemetry(
        state.get("run_telemetry"),
        nodes={"ground": {"resolve_rate": resolve_rate}},
    )
    return {
        "problem": problem,
        "chosen_recipe": chosen,
        "grounding_report": report_dict,
        "grounded_r0": problem.get("grounded_r0") or [],
        "identity_roles": roles,
        "run_telemetry": tel,
        "tools_used": [
            {
                "name": "ground_draft_to_problem",
                "purpose": "Neighborhood-first FDC resolve → x0/M problem",
                "output_summary": {
                    "resolve_rate": resolve_rate,
                    "n_ingredients": len(chosen.get("ingredients") or []),
                    "identity_roles": roles,
                    "dequant_cache_used": bool(report_dict.get("dequant_cache_used")),
                    "dequant_cache_hits": int(report_dict.get("dequant_cache_hits") or 0),
                },
                "output": {"report": report_dict, "chosen_recipe": chosen},
            }
        ],
    }


def node_build_finalists(state: AgentState) -> dict[str, Any]:
    cfg = _cfg(state)
    from recipe_opt_agent.shadow_gpt_candidate import force_include_shadow, is_shadow_candidate

    shadow_patch = _await_shadow_candidate(state)
    if shadow_patch:
        state = {**state, **{k: v for k, v in shadow_patch.items() if not str(k).startswith("_")}}

    pool = list(state.get("candidate_pool") or [])
    interesting = list(state.get("interesting_candidates") or [])
    # Merge pool + interesting archive (OOD / hybrid / LLM shortlist / strong deltas).
    by_id: dict[str, dict[str, Any]] = {}
    for e in pool + interesting:
        cid = str(e.get("candidate_id") or "")
        if not cid:
            continue
        prev = by_id.get(cid)
        if prev is None:
            by_id[cid] = e
            continue
        # Prefer entries with richer opt / ingredients
        if e.get("ingredients") and not prev.get("ingredients"):
            by_id[cid] = e
        elif e.get("opt") and not prev.get("opt"):
            by_id[cid] = e
    pool = list(by_id.values())
    if not pool:
        pool = [
            {
                "candidate_id": "current",
                "iteration": state.get("iteration", 0),
                "branch": "current",
                "opt": state.get("opt"),
                "diagnosis_full": state.get("diagnosis"),
                "ingredients": (state.get("chosen_recipe") or {}).get("ingredients"),
                "foodon_basis_report": (state.get("problem") or {}).get("foodon_basis_report")
                or state.get("foodon_basis_report")
                or (state.get("chosen_recipe") or {}).get("foodon_basis_report"),
                "x_opt": (state.get("opt") or {}).get("x_opt"),
                "L_max_norm": (state.get("diagnosis") or {}).get("L_max_norm"),
                "n_red": (state.get("diagnosis") or {}).get("n_red"),
            }
        ]
    # Prefer diversity of branches among top finalists
    pool = sorted(
        pool,
        key=lambda p: (
            0 if is_shadow_candidate(p) else 1,
            0 if p.get("branch") in {"ood_protein", "hybrid"} else 1,
            p.get("n_red", 99),
            p.get("L_max_norm", 99.0) if p.get("L_max_norm") is not None else 99.0,
            p.get("objective", 99.0) if p.get("objective") is not None else 99.0,
            float(p.get("delta_L_star") or 0.0),
        ),
    )
    max_n = max(1, int(cfg.max_finalists))
    pool = force_include_shadow(pool, max_n=max_n)
    out: dict[str, Any] = {
        "finalist_pool": pool,
        "interesting_candidates": interesting,
        "tools_used": [
            {
                "name": "build_finalists",
                "purpose": f"Collect pool + interesting OOD/hybrid/shortlist hits (≤{max_n})",
                "output_summary": {
                    "n_finalists": len(pool),
                    "n_interesting": len(interesting),
                    "branches": sorted({str(p.get("branch") or "in_distribution") for p in pool}),
                },
                "output": pool,
            }
        ],
    }
    if shadow_patch:
        if shadow_patch.get("candidate_pool") is not None:
            out["candidate_pool"] = shadow_patch["candidate_pool"]
        if shadow_patch.get("interesting_candidates") is not None:
            out["interesting_candidates"] = shadow_patch["interesting_candidates"]
        if "shadow_job_id" in shadow_patch:
            out["shadow_job_id"] = shadow_patch["shadow_job_id"]
        if shadow_patch.get("run_telemetry") is not None:
            out["run_telemetry"] = shadow_patch["run_telemetry"]
        if shadow_patch.get("_shadow_collect_meta") is not None:
            out["_shadow_collect_meta"] = shadow_patch["_shadow_collect_meta"]
    return out


def node_pareto_and_rank(state: AgentState) -> dict[str, Any]:
    from mvp_nutrient_fit import nutrient_range_fit_from_totals
    from recipe_opt_agent.candidate_scoring import (
        compute_churn,
        compute_intent_gap,
        score_finalist_pool,
        top_survivors_for_judge,
    )
    from weighted_empirical_opt import pfc_fractions_from_portions

    cfg = _cfg(state)
    pool = list(state.get("finalist_pool") or state.get("candidate_pool") or [])
    request = state.get("user_request") or state.get("taste_text") or ""
    grounded_r0 = state.get("grounded_r0") or []
    problem = state.get("problem") or {}
    M = np.asarray(problem.get("M") or [], dtype=float)
    entries: list[dict[str, Any]] = []

    for e in pool:
        opt = e.get("opt") or {}
        x_opt = np.asarray(e.get("x_opt") or opt.get("x_opt") or [], dtype=float)
        pfc = opt.get("pfc_after") or {}
        if not pfc and x_opt.size and M.size:
            p, c, f = pfc_fractions_from_portions(x_opt, M)
            pfc = {"protein": p, "carbs": c, "fat": f}
        kcal = float(problem.get("kcal_target") or 500.0)
        prot_g = pfc.get("protein", 0.2) * kcal / 4.0
        carb_g = pfc.get("carbs", 0.4) * kcal / 4.0
        fat_g = pfc.get("fat", 0.3) * kcal / 9.0
        nutrient_dist = nutrient_range_fit_from_totals(
            prot_g,
            fat_g,
            carb_g,
            kcal,
            cfg.fat_min,
            cfg.fat_max,
            cfg.carb_min,
            cfg.carb_max,
            cfg.protein_min,
            cfg.protein_max,
        )
        ings = e.get("ingredients") or (state.get("chosen_recipe") or {}).get("ingredients") or []
        entries.append(
            {
                "candidate_id": e.get("candidate_id"),
                "branch": e.get("branch") or "in_distribution",
                "metrics": {
                    "nutrient_dist": float(nutrient_dist),
                    "ratio_badness": float(
                        e.get("L_max_norm") or (e.get("diagnosis_full") or {}).get("L_max_norm") or 0.0
                    ),
                    "intent_gap": compute_intent_gap(request, state.get("title") or "", ings),
                    "churn": compute_churn(
                        ings,
                        grounded_r0,
                        x_opt=x_opt.tolist() if x_opt.size else None,
                        x_ref=problem.get("x0"),
                    ),
                },
                "entry": e,
            }
        )

    scored = score_finalist_pool(entries, weights=cfg.score_weights())
    scored_dicts = [s.to_dict() for s in scored]
    survivors, need_judge = top_survivors_for_judge(
        scored, epsilon=cfg.judge_epsilon, max_survivors=max(1, int(cfg.max_finalists))
    )
    shadow_scored = [s for s in scored if str(s.candidate_id).startswith("pool_shadow")]
    if shadow_scored:
        shadow = shadow_scored[0]
        if not any(s.candidate_id == shadow.candidate_id for s in survivors):
            survivors = list(survivors) + [shadow]
            max_s = max(1, int(cfg.max_finalists))
            if len(survivors) > max_s:
                # Keep shadow + best non-shadow fillers.
                others = [s for s in survivors if s.candidate_id != shadow.candidate_id]
                survivors = [shadow] + others[: max_s - 1]
        if len(survivors) >= 2:
            need_judge = True
    if not bool(getattr(cfg, "enable_llm_judge", False)):
        need_judge = False
    elif str(getattr(cfg, "llm_judge_mode", "demote_weird") or "demote_weird").lower() != "full":
        # Demote-weird runs at finalize; no mid-graph ranking judge.
        need_judge = False
    return {
        "scored_finalists": scored_dicts,
        "_need_judge": need_judge,
        "_survivors": [s.to_dict() for s in survivors],
        "tools_used": [
            {
                "name": "pareto_and_rank",
                "purpose": "Pareto filter + weighted composite (0.4/0.3/0.2/0.1)",
                "output_summary": {
                    "n_scored": len(scored),
                    "top_composite": scored[0].composite if scored else None,
                    "need_judge": need_judge,
                },
                "output": scored_dicts,
            }
        ],
    }


def node_judge_final(state: AgentState) -> dict[str, Any]:
    cfg = _cfg(state)
    survivors = state.get("_survivors") or state.get("scored_finalists") or []
    need_judge = bool(state.get("_need_judge")) and bool(getattr(cfg, "enable_llm_judge", False))
    if not need_judge or len(survivors) <= 1:
        winner_id = survivors[0].get("candidate_id") if survivors else None
        return {
            "judge_result": {
                "winner_id": winner_id,
                "rationale": "LLM judge disabled — ranking by proportion quality",
                "skipped": True,
            },
            "tools_used": [
                {
                    "name": "judge_final",
                    "purpose": "Skip LLM judge (disabled or no tie)",
                    "output_summary": {"winner_id": winner_id, "skipped": True},
                    "output": {"winner_id": winner_id},
                }
            ],
        }

    ctx = {
        "user_request": state.get("user_request"),
        "requirement_tags": state.get("requirement_tags"),
        "survivors": [_public_survivor(s) for s in survivors],
        "title": state.get("title"),
    }
    result, trace = judge_finalists_llm(ctx, model=select_judge_model(cfg))
    # Clamp / fill holistic 0–10
    if isinstance(result, dict):
        hs = result.get("holistic_score_0_10")
        if hs is None and result.get("winner_id") and isinstance(result.get("scores_0_10"), dict):
            hs = result["scores_0_10"].get(result["winner_id"])
        if hs is not None:
            try:
                result["holistic_score_0_10"] = max(0.0, min(10.0, float(hs)))
            except (TypeError, ValueError):
                pass
    tel = bump_telemetry(
        state.get("run_telemetry"),
        n_llm_calls=1 if (trace or {}).get("mode") == "openai" else 0,
        nodes={"judge": {"model": (trace or {}).get("model") or select_judge_model(cfg)}},
    )
    return {
        "judge_result": result,
        "llm_trace": trace,
        "run_telemetry": tel,
        "tools_used": [
            {
                "name": "judge_final",
                "purpose": "LLM pick among tied Pareto survivors",
                "mode": trace.get("mode"),
                "model": trace.get("model") or cfg.judge_model,
                "output_summary": result,
                "output": result,
                "llm_trace": trace,
            }
        ],
    }


def _route_after_pareto(state: AgentState) -> Literal["judge_final", "finalize"]:
    if state.get("_need_judge"):
        return "judge_final"
    return "finalize"


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("init", node_init)
    g.add_node("shadow_gpt_candidate", node_shadow_gpt_candidate)
    g.add_node("diagnose", node_diagnose)
    g.add_node("save_candidate", node_save_candidate)
    g.add_node("save_moderate", node_save_moderate)
    g.add_node("propose", node_propose)
    g.add_node("decide", node_decide)
    g.add_node("apply", node_apply_or_expand)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("init")
    g.add_edge("init", "shadow_gpt_candidate")
    g.add_edge("shadow_gpt_candidate", "diagnose")
    g.add_conditional_edges(
        "diagnose",
        _route_after_diagnose,
        {
            "finalize": "finalize",
            "save_candidate": "save_candidate",
            "propose": "propose",
            "build_finalists": "finalize",
        },
    )
    g.add_conditional_edges(
        "save_candidate",
        _route_after_save,
        {"propose": "propose", "build_finalists": "finalize"},
    )
    g.add_edge("propose", "decide")
    g.add_conditional_edges(
        "decide",
        _route_after_decide,
        {"finalize": "finalize", "apply": "apply", "build_finalists": "finalize"},
    )
    g.add_conditional_edges(
        "apply",
        _route_after_apply,
        {"diagnose": "diagnose", "finalize": "finalize", "build_finalists": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def build_creative_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("init", node_init)
    g.add_node("deduce_tags", node_deduce_tags)
    g.add_node("llm_draft", node_llm_draft)
    g.add_node("ground_recipe", node_ground_recipe)
    g.add_node("shadow_gpt_candidate", node_shadow_gpt_candidate)
    g.add_node("diagnose", node_diagnose)
    g.add_node("save_candidate", node_save_candidate)
    g.add_node("propose", node_propose)
    g.add_node("decide", node_decide)
    g.add_node("apply", node_apply_or_expand)
    g.add_node("build_finalists", node_build_finalists)
    g.add_node("pareto_and_rank", node_pareto_and_rank)
    g.add_node("judge_final", node_judge_final)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("init")
    g.add_edge("init", "deduce_tags")
    g.add_edge("deduce_tags", "llm_draft")
    g.add_edge("llm_draft", "ground_recipe")
    g.add_edge("ground_recipe", "shadow_gpt_candidate")
    g.add_edge("shadow_gpt_candidate", "diagnose")
    g.add_conditional_edges(
        "diagnose",
        _route_after_diagnose,
        {
            "save_candidate": "save_candidate",
            "propose": "propose",
            "build_finalists": "build_finalists",
            "finalize": "build_finalists",
        },
    )
    g.add_conditional_edges(
        "save_candidate",
        _route_after_save,
        {"propose": "propose", "build_finalists": "build_finalists"},
    )
    g.add_edge("propose", "decide")
    g.add_conditional_edges(
        "decide",
        _route_after_decide,
        {"apply": "apply", "finalize": "build_finalists", "build_finalists": "build_finalists"},
    )
    g.add_conditional_edges(
        "apply",
        _route_after_apply,
        {"diagnose": "diagnose", "finalize": "build_finalists", "build_finalists": "build_finalists"},
    )
    g.add_edge("build_finalists", "pareto_and_rank")
    g.add_conditional_edges(
        "pareto_and_rank",
        _route_after_pareto,
        {"judge_final": "judge_final", "finalize": "finalize"},
    )
    g.add_edge("judge_final", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def run_agent(initial: AgentState, *, creative: bool = False, config: dict[str, Any] | None = None) -> AgentState:
    app = build_creative_graph() if creative else build_graph()
    if config:
        return app.invoke(initial, config=config)
    return app.invoke(initial)


CREATIVE_FLOW_NODES = (
    "init",
    "deduce_tags",
    "llm_draft",
    "ground_recipe",
    "shadow_gpt_candidate",
    "diagnose",
    "save_candidate",
    "propose",
    "decide",
    "apply",
    "build_finalists",
    "pareto_and_rank",
    "judge_final",
    "finalize",
)

CREATIVE_FLOW_EDGES = (
    ("init", "deduce_tags"),
    ("deduce_tags", "llm_draft"),
    ("llm_draft", "ground_recipe"),
    ("ground_recipe", "shadow_gpt_candidate"),
    ("shadow_gpt_candidate", "diagnose"),
    ("diagnose", "save_candidate"),
    ("diagnose", "propose"),
    ("diagnose", "build_finalists"),
    ("save_candidate", "propose"),
    ("save_candidate", "build_finalists"),
    ("propose", "decide"),
    ("decide", "apply"),
    ("decide", "build_finalists"),
    ("apply", "diagnose"),
    ("apply", "build_finalists"),
    ("build_finalists", "pareto_and_rank"),
    ("pareto_and_rank", "judge_final"),
    ("pareto_and_rank", "finalize"),
    ("judge_final", "finalize"),
)


# Node order used by UIs for flow highlighting.
FLOW_NODES = (
    "init",
    "shadow_gpt_candidate",
    "diagnose",
    "save_candidate",
    "propose",
    "decide",
    "apply",
    "finalize",
)

FLOW_EDGES = (
    ("init", "shadow_gpt_candidate"),
    ("shadow_gpt_candidate", "diagnose"),
    ("diagnose", "save_candidate"),
    ("diagnose", "propose"),
    ("diagnose", "finalize"),
    ("save_candidate", "propose"),
    ("propose", "decide"),
    ("decide", "apply"),
    ("decide", "finalize"),
    ("apply", "diagnose"),
    ("apply", "finalize"),
)

# Human-readable docs for the flow diagram (also returned by /api/flow).
# Each entry: title, summary, detail, tools=[{name, purpose}], compute=
#   deterministic | llm_content | llm_controller
# compute drives WebUI node color-coding:
#   deterministic  — pure Python tools (no LLM)
#   llm_content    — LLM generates structured content (tags, draft, judgment)
#   llm_controller — LLM chooses which action / tool to invoke next
FLOW_NODE_DOCS: dict[str, dict[str, Any]] = {
    "init": {
        "title": "Initialize",
        "summary": "Bind the dropdown-selected canonical recipe (and its ingredients) as the semantic input; load its FoodOn neighborhood.",
        "detail": (
            "The match-ranked canonical dropdown defines the neighborhood — there is no separate embedding "
            "similarity search. taste_text is set to the dish title. The starting NLG recipe inside that "
            "neighborhood is chosen closest to the target macro box. Modification candidates are retrieved "
            "later, live, inside propose."
        ),
        "compute": "deterministic",
        "tools": [
            {
                "name": "select_canonical_recipe",
                "purpose": "Bind dropdown selection + FoodOn neighborhood; pick start by nutrition",
                "detail": (
                    "Loads the selected canonical dish and its FoodOn co-occurrence neighborhood "
                    "(Jaccard cache for the neighbor *set* only). The starting NLG recipe inside "
                    "that neighborhood is chosen by nutrition proximity: L1 PFC to the macro box "
                    "(or loss projection). No LLM."
                ),
            },
        ],
    },
    "shadow_gpt_candidate": {
        "title": "Extra draft candidate",
        "summary": "Quietly starts one additional optimized draft in the background for later comparison.",
        "detail": (
            "Kicks off an alternate gram-level draft (GPT-5.5 by default) from the user request, "
            "dish title, and macro box on a background thread so diagnose/propose can proceed in "
            "parallel. The job is joined before finalists / the final arbiter; the result is saved "
            "as a pool entry and does not replace the working recipe. Authorship is not shown in "
            "product UI."
        ),
        "compute": "llm_content",
        "tools": [
            {
                "name": "shadow_optimized_draft",
                "purpose": "Async draft → ground → LP; merge into candidate_pool before arbiter",
            },
        ],
    },
    "diagnose": {
        "title": "Diagnose (+ optimizer)",
        "summary": "This is where the optimizer runs, together with hull geometry and fidelity bands.",
        "detail": (
            "Inside this single LangGraph node the agent calls, in order:\n"
            "1) region_intersects_hull — conical hull of ingredient PFC vectors vs the target macro box (H∩T), "
            "including how far outside the hull the box sits;\n"
            "2) optimize_weighted_empirical_obj — the convex / LP optimizer that reweights ingredient grams "
            "to minimize neighborhood fidelity loss subject to protein/carb/fat constraints;\n"
            "3) diagnose_optimizer_result — IQR green/yellow/red zones, L_max_norm, three-band gate "
            "(accept / moderate / must_retry) and retry_triggers.\n"
            "The optimizer is intentionally not a separate graph node: diagnose always means "
            "“run geometry + solve + score”, then route on the fidelity band."
        ),
        "compute": "deterministic",
        "tools": [
            {
                "name": "region_intersects_hull",
                "purpose": "Target PFC box vs conical ingredient hull (H∩T)",
                "detail": "Geometric + LP check that the target macro box intersects the cone spanned by current ingredient PFC vectors.",
            },
            {
                "name": "optimize_weighted_empirical_obj",
                "purpose": "LP: reweight grams to minimize fidelity loss under macro bounds",
                "detail": "Convex / HiGHS solve for ingredient grams minimizing neighborhood share + ratio loss subject to protein/carb/fat calorie-fraction bounds.",
            },
            {
                "name": "diagnose_optimizer_result",
                "purpose": "IQR zones + three-band gate + retry triggers",
                "detail": "Maps each share/ratio term into green/yellow/red IQR zones and sets fidelity_band (accept / moderate / must_retry) with explicit retry_triggers.",
            },
            {
                "name": "build_loss_field",
                "purpose": "Optional grid of L(p) over the PFC region",
                "detail": "Expensive optional grid; only runs when problem.compute_loss_field is true.",
            },
        ],
    },
    "save_candidate": {
        "title": "Save candidate",
        "summary": "Park feasible snapshots in candidate_pool (moderate / feasible must_retry / creative accept).",
        "detail": (
            "Extended soft-band save: moderate, must_retry with feasible macros (config flag), "
            "and creative-mode accept polish. Never discards pool entries on retry."
        ),
        "compute": "deterministic",
        "tools": [
            {
                "name": "save_candidate",
                "purpose": "Append feasible snapshot to candidate_pool",
                "detail": "Stores opt + diagnosis + ingredients for later pool-best / creative finalist selection.",
            },
        ],
    },
    "save_moderate": {
        "title": "Save candidate (alias)",
        "summary": "Alias for save_candidate.",
        "detail": "Backward-compatible name for the same pool-save node.",
        "compute": "deterministic",
        "tools": [],
    },
    "propose": {
        "title": "Propose edits (slots → bundles)",
        "summary": "Plan up to 2 diagnosis-driven slots, retrieve per-slot candidates with cheap proxies, then joint-LP score 1-2 edit bundles.",
        "detail": (
            "Three internal stages, each emitted as a tool event:\n"
            "1) plan_slots — turn the diagnosis (RED/YELLOW share terms, binding macros, hull misses, dietary "
            "violations) into at most 2 structured edit slots (open_hull / fix_share / macro_gap / dietary_swap / "
            "remove_outlier), each preferring add, swap or remove;\n"
            "2) retrieve_slots — per-slot shortlists from the FoodOn neighborhood catalog with co-occurrence, "
            "geometry, a share-dilution ratio proxy and a nutrient-direction proxy; real swap candidates replace "
            "specific current lines;\n"
            "3) score_bundles — enumerate compatible 1-2 edit bundles, proxy-rank, then re-run the joint LP on the "
            "top set to get L*_before → L*_after with ratio/nutrient decomposition. Every LP-scored bundle carries "
            "a materialized next_problem so apply is atomic. The LLM may only choose from these bundles/candidates."
        ),
        "compute": "deterministic",
        "tools": [
            {
                "name": "plan_slots",
                "purpose": "Diagnosis → up to 2 structured edit slots",
                "detail": "Maps RED share terms, hull misses, binding macros, and dietary violations into ≤2 slots with preferred add/swap/remove actions.",
            },
            {
                "name": "retrieve_slots",
                "purpose": "Per-slot candidate shortlists + dilution/nutrient proxies + real swaps",
                "detail": "Co-occurrence + geometry shortlist with share-dilution and nutrient-direction proxies; generates real swap candidates for target lines.",
            },
            {
                "name": "score_bundles",
                "purpose": "Enumerate bundles, proxy-rank, joint LP on top 10 (L*_before vs L*_after)",
                "detail": "Builds size-1/2 edit bundles, proxy-ranks ≤50, then jointly re-optimizes the top 10 and attaches a materialized next_problem.",
            },
        ],
    },
    "decide": {
        "title": "Decide action (LLM)",
        "summary": "gpt-4o-mini (or heuristic) chooses accept / apply_bundle / add / swap / remove / expand.",
        "detail": (
            "Builds DecisionContext (diagnosis, hull, slots, scored bundle table, flat candidates, identity, pool "
            "summary) and calls decide_action_llm. apply_bundle + chosen_bundle_id selects a jointly scored edit "
            "set; single add/swap/remove with chosen_candidate_id still works for size-1 edits. Without "
            "OPENAI_API_KEY a deterministic heuristic picks the tag-safe bundle with the best delta_L_star."
        ),
        "compute": "llm_controller",
        "tools": [
            {
                "name": "decide_action_llm",
                "purpose": "Choose accept / apply_bundle / add / swap / remove / expand from DecisionContext",
                "detail": (
                    "LLM controller (gpt-4o-mini) or deterministic heuristic when OPENAI_API_KEY is missing. "
                    "May only pick candidate/bundle ids from the propose shortlist."
                ),
            },
        ],
    },
    "apply": {
        "title": "Apply or expand",
        "summary": "Atomically swap in the bundle's next_problem (new x0/M/basis), or widen the neighbor search, then re-diagnose.",
        "detail": (
            "apply_bundle: replace the LP problem with the bundle's materialized next_problem (updated x0, M "
            "columns, ingredient_basis, chosen_recipe ingredients) after re-validating dietary tags. Single "
            "add/swap/remove: use the candidate's next_problem or materialize one live via "
            "apply_edits_to_problem. expand: bump neighbor_k per expand_directive. Accept actions skip mutation "
            "and go to finalize; otherwise the graph loops back to diagnose, which re-runs the optimizer on the "
            "mutated basis."
        ),
        "compute": "deterministic",
        "tools": [
            {
                "name": "apply_bundle",
                "purpose": "Swap in bundle next_problem (atomic multi-edit)",
                "detail": "Replaces x0 / M / ingredient_basis / chosen_recipe with the jointly scored next_problem after tag validation.",
            },
            {
                "name": "apply_modification",
                "purpose": "Apply single candidate; materialize next_problem live",
                "detail": "Uses candidate meta.next_problem or builds one via apply_edits_to_problem when macros are available.",
            },
            {
                "name": "expand_neighborhood",
                "purpose": "Widen neighbor retrieval (neighbor_k += delta)",
                "detail": "Increments neighbor_k from expand_directive so the next diagnose rebuilds a broader FoodOn neighborhood.",
            },
        ],
    },
    "finalize": {
        "title": "Finalize",
        "summary": "Emit the accepted solution or best-effort / pool-best fallback.",
        "detail": (
            "If band=accept (or action=accept): return the current opt+diagnosis. "
            "Else if candidate_pool is non-empty: return the best moderate snapshot. "
            "Else: best-effort current state with status failed_or_best_effort."
        ),
        "compute": "deterministic",
        "tools": [],
    },
    "deduce_tags": {
        "title": "Deduce requirement tags",
        "summary": "Extract hard dietary/macro tags (vegetarian, no_pork, high_protein, …) from the user request.",
        "detail": (
            "LLM (or lexical heuristic) turns the free-text request into structured requirement_tags. These are "
            "hard constraints: retrieval, bundle scoring and apply all filter against them."
        ),
        "compute": "llm_content",
        "tools": [
            {
                "name": "deduce_tags_llm",
                "purpose": "Free text → structured requirement_tags",
                "detail": "LLM (or lexical heuristic) extracts hard dietary / macro tags used as filters for the rest of the run.",
            },
        ],
    },
    "llm_draft": {
        "title": "LLM draft recipe",
        "summary": "Warm-start OOD requests: draft a structured recipe (title, ingredients, grams, roles) from the request.",
        "detail": (
            "Creative mode only. The draft seeds the problem when no canonical dish is selected; it is then "
            "grounded against FDC/FoodOn data before optimization."
        ),
        "compute": "llm_content",
        "tools": [
            {
                "name": "llm_draft_recipe",
                "purpose": "Request → structured recipe draft JSON",
                "detail": "Produces title, ingredients with starting grams/roles, and requirement_tags sized for the target macro box.",
            },
        ],
    },
    "ground_recipe": {
        "title": "Ground draft",
        "summary": "Resolve draft ingredient names to FDC foods and build the initial optimization problem.",
        "detail": (
            "Maps each draft line to an FDC id + macro column, drops unresolvable lines, and assembles x0/M/basis "
            "so diagnose can run the optimizer."
        ),
        "compute": "deterministic",
        "tools": [
            {
                "name": "ground_recipe_draft",
                "purpose": "Draft lines → FDC ids + macro columns (x0, M)",
                "detail": "Resolves each draft ingredient name to an FDC food and builds the initial LP problem matrices.",
            },
        ],
    },
    "build_finalists": {
        "title": "Build finalists",
        "summary": "Collect the candidate pool + current state into a finalist set for creative judging.",
        "detail": "Creative mode only: dedupes pool snapshots and prepares them for Pareto ranking.",
        "compute": "deterministic",
        "tools": [],
    },
    "pareto_and_rank": {
        "title": "Pareto + rank",
        "summary": "Score finalists on loss/feasibility/churn axes and keep the Pareto-efficient set.",
        "detail": "Creative mode only: normalizes metrics, computes composite scores, drops dominated finalists.",
        "compute": "deterministic",
        "tools": [
            {
                "name": "pareto_rank_finalists",
                "purpose": "Multi-axis Pareto filter + composite ranking",
                "detail": "Normalizes nutrient/ratio/intent/churn goodness, drops dominated finalists, ranks by composite score.",
            },
        ],
    },
    "judge_final": {
        "title": "Judge finalists (LLM)",
        "summary": "LLM picks the winner among Pareto survivors for the user request.",
        "detail": "Creative mode only: judge_finalists_llm compares survivors and returns winner + rationale.",
        "compute": "llm_content",
        "tools": [
            {
                "name": "judge_finalists_llm",
                "purpose": "Pick winner among Pareto survivors",
                "detail": "LLM compares survivors against the user request and returns winner_id + rationale.",
            },
        ],
    },
}

FLOW_COMPUTE_KINDS = {
    "deterministic": {
        "label": "Deterministic tools",
        "blurb": "Pure Python / LP — no LLM call",
    },
    "llm_content": {
        "label": "LLM content",
        "blurb": "LLM generates structured content (tags, draft, judgment)",
    },
    "llm_controller": {
        "label": "LLM chooses tools",
        "blurb": "LLM selects the next action / tool to invoke",
    },
}


def _transcript_for_node(node: str, update: dict[str, Any]) -> list[dict[str, Any]]:
    """Build clearly marked transcript entries for the side panel."""
    entries: list[dict[str, Any]] = []
    for tool in update.get("tools_used") or []:
        entries.append(
            {
                "kind": "tool",
                "node": node,
                "name": tool.get("name"),
                "purpose": tool.get("purpose"),
                "summary": tool.get("output_summary"),
                "output": tool.get("output"),
            }
        )
        if tool.get("llm_trace"):
            trace = tool["llm_trace"]
            for msg in trace.get("messages") or []:
                entries.append(
                    {
                        "kind": "prompt",
                        "node": node,
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                        "mode": trace.get("mode"),
                        "model": trace.get("model"),
                    }
                )
            entries.append(
                {
                    "kind": "llm_response",
                    "node": node,
                    "mode": trace.get("mode"),
                    "model": trace.get("model"),
                    "content": trace.get("raw_response"),
                    "usage": trace.get("usage"),
                }
            )
            if trace.get("rationale"):
                entries.append(
                    {
                        "kind": "reasoning",
                        "node": node,
                        "content": trace.get("rationale"),
                        "mode": trace.get("mode"),
                    }
                )
    # Also surface retry triggers as transcript notes on diagnose
    diag = update.get("diagnosis") or {}
    if diag.get("retry_triggers") and update.get("fidelity_band") in {"must_retry", "moderate"}:
        for trig in diag["retry_triggers"]:
            entries.append(
                {
                    "kind": "retry_trigger",
                    "node": node,
                    "metric": trig.get("metric"),
                    "reason": trig.get("reason"),
                    "current_value": trig.get("current_value"),
                    "threshold_to_clear": trig.get("threshold_to_clear"),
                    "clearance": trig.get("clearance"),
                    "primary": trig.get("primary"),
                }
            )
    if update.get("candidates") is not None:
        entries.append(
            {
                "kind": "candidates",
                "node": node,
                "candidates": update.get("candidates"),
                "dropped": update.get("candidates_dropped"),
            }
        )
    return entries


def _step_payload(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """UI-friendly payload: compact summary fields + full `detail` for dropdowns."""
    from recipe_opt_agent.portion_display import build_progress_detail

    out: dict[str, Any] = {"node": node}
    if "fidelity_band" in update:
        out["fidelity_band"] = update["fidelity_band"]
    if "diagnosis" in update:
        d = update["diagnosis"] or {}
        out["diagnosis"] = {
            "diagnosis": d.get("diagnosis"),
            "meaning": d.get("meaning"),
            "n_red": d.get("n_red"),
            "L_max_norm": d.get("L_max_norm"),
            "L_total": d.get("L_total"),
            "terms": d.get("terms"),
            "binding_macros": d.get("binding_macros"),
            "retry_triggers": d.get("retry_triggers"),
            "band_thresholds": d.get("band_thresholds"),
            "recommended_action_class": d.get("recommended_action_class"),
        }
    if "decision" in update:
        out["decision"] = update["decision"]
    if "candidates" in update:
        out["candidates"] = update["candidates"]
    if "candidates_dropped" in update:
        out["candidates_dropped"] = update["candidates_dropped"]
    if "planned_slots" in update:
        out["planned_slots"] = update["planned_slots"]
    if "bundles" in update:
        out["bundles"] = [
            {k: v for k, v in (b or {}).items() if k != "next_problem"}
            for b in (update["bundles"] or [])
        ]
    if "opt" in update:
        opt = update["opt"] or {}
        out["opt"] = {
            "status": opt.get("status"),
            "objective": opt.get("objective"),
            "feasible": opt.get("feasible"),
            "pfc_after": opt.get("pfc_after"),
            "term_losses": opt.get("term_losses"),
            "x_opt": opt.get("x_opt"),
        }
    if "hull" in update:
        h = update["hull"] or {}
        out["hull"] = {
            "intersects": h.get("intersects"),
            "geometric_intersects": h.get("geometric_intersects"),
            "lp_feasible": h.get("lp_feasible"),
            "lp_message": h.get("lp_message"),
            "residual": h.get("residual"),
            "distance": h.get("distance"),
            "ingredient_pfc_vertices": h.get("ingredient_pfc_vertices"),
            "n_samples": h.get("n_samples"),
        }
    if "candidate_pool" in update:
        out["candidate_pool"] = update["candidate_pool"]
        out["candidate_pool_n"] = len(update["candidate_pool"] or [])
    if "final" in update:
        out["final"] = update["final"]
    if "status" in update:
        out["status"] = update["status"]
    if "iteration" in update:
        out["iteration"] = update["iteration"]
    if "identity_roles" in update:
        out["identity_roles"] = update["identity_roles"]
    if "chosen_recipe" in update:
        out["chosen_recipe"] = update["chosen_recipe"]
    if "foodon_basis_report" in update:
        out["foodon_basis_report"] = update["foodon_basis_report"]
    elif isinstance(update.get("problem"), dict) and update["problem"].get("foodon_basis_report"):
        out["foodon_basis_report"] = update["problem"]["foodon_basis_report"]
    if "neighborhood_recipes" in update:
        out["neighborhood_recipes"] = update["neighborhood_recipes"]
        out["neighborhood_n"] = len(update["neighborhood_recipes"] or [])
    if "tools_used" in update:
        out["tools_used"] = [
            {
                "name": t.get("name"),
                "purpose": t.get("purpose"),
                "output_summary": t.get("output_summary"),
                "mode": t.get("mode"),
                "model": t.get("model"),
            }
            for t in (update["tools_used"] or [])
        ]
    if "llm_trace" in update:
        out["llm_trace"] = update["llm_trace"]
    if "decision_context" in update:
        out["decision_context"] = update["decision_context"]
    if "last_applied_candidate" in update:
        out["last_applied_candidate"] = update["last_applied_candidate"]
    if "expand_directive" in update:
        out["expand_directive"] = update["expand_directive"]
    if "run_telemetry" in update:
        out["run_telemetry"] = update["run_telemetry"]
    if "decision_outcomes" in update:
        out["decision_outcomes"] = update["decision_outcomes"]
    if "live_scores" in update:
        out["live_scores"] = update["live_scores"]
    if "score_history" in update:
        out["score_history"] = update["score_history"]

    progress_detail = build_progress_detail(node, update)
    if progress_detail:
        out["progress_detail"] = progress_detail

    # Full detail blob for dropdowns (includes tool outputs / prompts)
    detail: dict[str, Any] = {
        "update_keys": sorted(update.keys()),
        "tools_used": update.get("tools_used"),
        "chosen_recipe": update.get("chosen_recipe"),
        "neighborhood_recipes": update.get("neighborhood_recipes"),
        "diagnosis": update.get("diagnosis"),
        "hull": update.get("hull"),
        "opt": update.get("opt"),
        "candidates": update.get("candidates"),
        "candidates_dropped": update.get("candidates_dropped"),
        "decision": update.get("decision"),
        "llm_trace": update.get("llm_trace"),
        "decision_context": update.get("decision_context"),
        "candidate_pool": update.get("candidate_pool"),
        "loss_field_summary": update.get("loss_field_summary"),
        "last_applied_candidate": update.get("last_applied_candidate"),
        "final": update.get("final"),
        "identity_roles": update.get("identity_roles"),
        "identity_critical": update.get("identity_critical"),
        "status": update.get("status"),
        "taste_text": update.get("taste_text"),
        "title": update.get("title"),
        "run_telemetry": update.get("run_telemetry"),
        "decision_outcomes": update.get("decision_outcomes"),
    }
    out["detail"] = {k: v for k, v in detail.items() if v is not None}
    out["transcript"] = _transcript_for_node(node, update)
    return out


def stream_agent(
    initial: AgentState,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    creative: bool = False,
    config: dict[str, Any] | None = None,
) -> AgentState:
    """Run the graph in update-stream mode; emit step events for UIs/notebooks."""
    app = build_creative_graph() if creative else build_graph()
    flow_nodes = CREATIVE_FLOW_NODES if creative else FLOW_NODES
    flow_edges = CREATIVE_FLOW_EDGES if creative else FLOW_EDGES
    last: AgentState = dict(initial)
    seq = 0
    if on_event:
        on_event({"type": "graph_meta", "seq": seq, "nodes": list(flow_nodes), "edges": list(flow_edges), "creative": creative})
    stream_iter = app.stream(initial, stream_mode="updates", config=config) if config else app.stream(initial, stream_mode="updates")
    last_step_at = time.perf_counter()
    for update in stream_iter:
        completed_at = time.perf_counter()
        duration_ms = round((completed_at - last_step_at) * 1000.0, 1)
        # update: {node_name: partial_state}
        for node, patch in update.items():
            seq += 1
            if isinstance(patch, dict):
                last = {**last, **patch}
            payload = _step_payload(node, patch if isinstance(patch, dict) else {})
            event = {
                "type": "step",
                "seq": seq,
                "node": node,
                "payload": payload,
                "fidelity_band": last.get("fidelity_band"),
                "iteration": last.get("iteration"),
                # LangGraph's update stream emits after a node completes. This
                # interval is therefore the completed node's runtime, not the
                # time spent in the previously displayed UI node.
                "duration_ms": duration_ms,
                "transcript": payload.get("transcript") or [],
            }
            # Silent background candidate — still mutates state, but skip UI step noise.
            if node == "shadow_gpt_candidate":
                last_step_at = completed_at
                continue
            if on_event:
                on_event(event)
                for entry in payload.get("transcript") or []:
                    on_event(
                        {
                            "type": "transcript",
                            "seq": seq,
                            "node": node,
                            "entry": entry,
                            "iteration": last.get("iteration"),
                        }
                    )
        last_step_at = completed_at
    seq += 1
    if on_event:
        on_event(
            {
                "type": "done",
                "seq": seq,
                "final": last.get("final") or last,
                "status": last.get("status"),
            }
        )
    return last
