"""Build curated DecisionContext briefing for the LLM controller."""

from __future__ import annotations

from typing import Any

from recipe_opt_agent.telemetry import reflection_digest


TRADEOFF_FRAME = (
    "Ratio fidelity (mass-share / neighborhood shape), nutrient/macro fit (PFC box), "
    "and holistic dish similarity to the user request can conflict. The LP may excel on "
    "1–2 axes while the third suffers slightly — judge which tradeoffs are acceptable "
    "for this query; do not chase one metric to zero if identity or taste breaks."
)


def _compact_diagnosis(diagnosis: dict[str, Any]) -> dict[str, Any]:
    terms = diagnosis.get("terms") or []
    hot = []
    for t in terms:
        zone = str(t.get("zone") or t.get("iqr_zone") or "").upper()
        if zone in {"RED", "YELLOW"} or t.get("is_red") or t.get("is_yellow"):
            hot.append(
                {
                    "name": t.get("name") or t.get("term"),
                    "zone": zone or ("RED" if t.get("is_red") else "YELLOW"),
                    "L_norm": t.get("L_norm") or t.get("loss_norm"),
                }
            )
    triggers = list(diagnosis.get("retry_triggers") or [])[:5]
    return {
        "diagnosis": diagnosis.get("diagnosis"),
        "meaning": diagnosis.get("meaning"),
        "n_red": diagnosis.get("n_red"),
        "n_yellow": diagnosis.get("n_yellow"),
        "L_max_norm": diagnosis.get("L_max_norm"),
        "L_total": diagnosis.get("L_total"),
        "recommended_action_class": diagnosis.get("recommended_action_class"),
        "binding_macros": diagnosis.get("binding_macros"),
        "terms_hot": hot[:12],
        "retry_triggers": triggers,
        "band_thresholds": diagnosis.get("band_thresholds"),
    }


def _compact_neighborhood_expansion(state: dict[str, Any]) -> dict[str, Any] | None:
    """Surface structure-verification stats so the LLM can judge expansion quality."""
    problem = state.get("problem") or {}
    ctx = problem.get("retrieval_context") or {}
    meta = ctx.get("neighborhood_structure_meta")
    if not meta and not ctx.get("query_shell_recipes"):
        # Fall back to last tool-trace expansion meta if present on state
        for tool in reversed(state.get("tools_used") or []):
            name = str(tool.get("name") or "")
            if "expand" in name:
                summary = tool.get("output_summary") or tool.get("output") or {}
                if isinstance(summary, dict) and (
                    summary.get("structure_applied") is not None
                    or summary.get("query_expansion")
                    or summary.get("n_structure_checked") is not None
                ):
                    meta = summary.get("query_expansion") or summary
                    break
    if not meta and not ctx.get("dish_structure"):
        return None
    meta = dict(meta or {})
    return {
        "dish_structure": ctx.get("dish_structure") or meta.get("dish_structure"),
        "structure_applied": meta.get("structure_applied"),
        "n_structure_checked": meta.get("n_structure_checked"),
        "n_structure_passed": meta.get("n_structure_passed"),
        "n_structure_soft": meta.get("n_structure_soft"),
        "n_rejected_wrong_dominance": meta.get("n_rejected_wrong_dominance"),
        "n_context_only": meta.get("n_context_only"),
        "n_harvest_eligible": len(ctx.get("structure_verified_shell_ids") or []),
        "anchor_share_median": meta.get("anchor_share_median"),
        "stretch_share_median": meta.get("stretch_share_median"),
        "stretch_share_iqr": meta.get("stretch_share_iqr"),
        "n_shell_recipes": len(ctx.get("query_shell_recipes") or []),
        "note": (
            "System verified retrieved recipes by FoodOn gram shares. "
            "Rejected recipes had the stretch as primary and the anchor as a side "
            "(or otherwise failed stretch_role). Prefer expand again with clearer "
            "anchor-primary queries if n_structure_passed is low."
        ),
    }


def _compact_hull(hull: dict[str, Any] | None) -> dict[str, Any] | None:
    if not hull:
        return None
    dist = hull.get("distance") or {}
    return {
        "intersects": hull.get("intersects"),
        "outside_score": dist.get("outside_score"),
        "interpretation": dist.get("interpretation"),
    }


def _recipe_lines(state: dict[str, Any]) -> dict[str, Any]:
    chosen = state.get("chosen_recipe") or (state.get("problem") or {}).get("chosen_recipe") or {}
    lines = []
    for row in chosen.get("ingredients") or []:
        lines.append(
            {
                "label": row.get("label") or row.get("name"),
                "grams": row.get("grams"),
                "role": row.get("role"),
            }
        )
    return {"title": chosen.get("title") or state.get("title"), "ingredients": lines}


def _pfc_vs_box(state: dict[str, Any]) -> str:
    opt = state.get("opt") or {}
    pfc = opt.get("pfc_after") or {}
    cfg = state.get("config") or {}
    if not pfc:
        return "PFC unavailable"
    return (
        f"PFC protein={pfc.get('protein'):.3f} carb={pfc.get('carbs'):.3f} fat={pfc.get('fat'):.3f} "
        f"vs box P[{cfg.get('protein_min')}-{cfg.get('protein_max')}] "
        f"C[{cfg.get('carb_min')}-{cfg.get('carb_max')}] "
        f"F[{cfg.get('fat_min')}-{cfg.get('fat_max')}]"
    )


def _slot_kind_map(planned_slots: list[dict[str, Any]]) -> dict[str, str]:
    return {str(s.get("slot_id")): str(s.get("kind") or "") for s in planned_slots or []}


def _annotate_bundle(
    bundle: dict[str, Any],
    *,
    slot_kinds: dict[str, str],
    identity_roles: list[str],
) -> dict[str, Any]:
    """Deterministic role/effect annotations for uncertain-path prompting."""
    roles = {r.lower() for r in identity_roles or []}
    edit_ann = []
    for edit in bundle.get("edits") or []:
        label = str(edit.get("label") or "").lower()
        action = edit.get("action")
        slot_id = str((edit.get("meta") or {}).get("slot_id") or edit.get("slot_id") or "")
        kind = slot_kinds.get(slot_id) or str((edit.get("meta") or {}).get("slot_kind") or "")
        component_role = "other"
        if any(r.replace("_", " ") in label or label in r.replace("_", " ") for r in roles):
            component_role = "identity"
        elif kind == "macro_gap":
            component_role = "macro_filler"
        elif kind == "fix_share":
            component_role = "share_diluent"
        elif kind == "dietary_swap":
            component_role = "dietary_swap"
        elif kind == "open_hull":
            component_role = "hull_opener"
        elif kind == "remove_outlier":
            component_role = "outlier_removal"
        effect = {
            "ratio": "improve" if bundle.get("delta_ratio_term", 0) not in (None, 0) and float(bundle.get("delta_ratio_term") or 0) < 0 else "mixed",
            "nutrient": "improve" if float(bundle.get("nutrient_slack") or 1) < 0.05 or float(bundle.get("delta_nutrient_slack") or 0) < 0 else "mixed",
            "holistic": "unknown",
        }
        if action == "remove":
            effect["holistic"] = "risk" if component_role == "identity" else "neutral"
        edit_ann.append(
            {
                "action": action,
                "label": edit.get("label"),
                "slot_kind": kind or None,
                "component_role": component_role,
                "likely_effect": effect,
            }
        )
    pub = {k: v for k, v in bundle.items() if k != "next_problem"}
    pub["edit_annotations"] = edit_ann
    pub["branch"] = bundle.get("branch") or "in_distribution"
    return pub


def _identity_tension(state: dict[str, Any], bundles: list[dict[str, Any]]) -> bool:
    critical = set((state.get("identity_critical") or {}).keys())
    roles = set(state.get("identity_roles") or [])
    if not critical and not roles:
        return False
    for b in bundles[:5]:
        for edit in b.get("edits") or []:
            if edit.get("action") != "remove":
                continue
            lab = str(edit.get("label") or "")
            if lab in critical:
                return True
            low = lab.lower()
            if any(r.replace("_", " ") in low for r in roles):
                return True
    return False


def count_adds_so_far(state: dict[str, Any]) -> int:
    """New ingredients this run already introduced (lines tagged added_by)."""
    chosen = state.get("chosen_recipe") or (state.get("problem") or {}).get("chosen_recipe") or {}
    return sum(1 for r in (chosen.get("ingredients") or []) if r.get("added_by"))


def _bundle_is_add_only(bundle: dict[str, Any]) -> bool:
    edits = bundle.get("edits") or []
    return bool(edits) and all(str(e.get("action") or "") == "add" for e in edits)


def build_decision_context(state: dict[str, Any]) -> dict[str, Any]:
    diagnosis = state.get("diagnosis") or {}
    raw_bundles = list(state.get("bundles") or [])
    planned_slots = list(state.get("planned_slots") or [])
    slot_kinds = _slot_kind_map(planned_slots)
    identity_roles = list(state.get("identity_roles") or [])
    cfg = state.get("config") or {}
    adds_so_far = count_adds_so_far(state)
    max_adds = int(cfg.get("max_total_adds") or 2)
    adds_exhausted = adds_so_far >= max_adds
    marginal_eps = float(cfg.get("marginal_add_delta_eps") or 0.02)

    # Prefer non-blocked bundles; ensure OOD / hybrid branches appear in the briefing.
    ranked = sorted(
        raw_bundles,
        key=lambda b: (
            1 if b.get("oscillation_blocked") else 0,
            float(b["delta_L_star"]) if b.get("delta_L_star") is not None else 99.0,
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_bids: set[str] = set()

    def _take(pred, n: int) -> None:
        count = 0
        for b in ranked:
            if count >= n:
                break
            bid = str(b.get("bundle_id") or "")
            if bid in seen_bids or b.get("oscillation_blocked"):
                continue
            if not pred(b):
                continue
            seen_bids.add(bid)
            selected.append(b)
            count += 1

    _take(lambda b: (b.get("branch") or "in_distribution") == "in_distribution", 3)
    _take(lambda b: b.get("branch") == "ood_protein", 2)
    _take(lambda b: b.get("branch") == "hybrid", 2)
    _take(lambda _b: True, 8 - len(selected))
    top = [_annotate_bundle(b, slot_kinds=slot_kinds, identity_roles=identity_roles) for b in selected]
    for pub in top:
        n_adds = sum(1 for e in pub.get("edits") or [] if str(e.get("action") or "") == "add")
        d = pub.get("delta_L_star")
        pub["marginal_improvement"] = bool(
            n_adds and (d is None or float(d) > -marginal_eps)
        )
        pub["blocked_by_add_budget"] = bool(n_adds and adds_so_far + n_adds > max_adds)

    pool = state.get("candidate_pool") or []
    best_pool = None
    if pool:
        best_pool = min(
            pool,
            key=lambda p: (
                p.get("n_red", 99),
                p.get("L_max_norm", 99.0),
                p.get("L_total", 99.0),
            ),
        )

    outcomes = list(state.get("decision_outcomes") or [])[-3:]
    iteration = int(state.get("iteration") or 0)
    revisit = None
    if iteration >= 1 and outcomes:
        last = outcomes[-1]
        revisit = {
            "ask": (
                "Given past decisions and before/after movement on ratio, nutrient, and holistic, "
                "what alternatives remain and how should strategy change? "
                "Do not repeat a failed edit fingerprint without a new theory."
            ),
            "past_outcomes": outcomes,
            "recent_edit_fingerprints": state.get("recent_edit_fingerprints") or [],
        }

    candidates_compact = []
    for c in (state.get("candidates") or [])[:12]:
        candidates_compact.append(
            {
                "candidate_id": c.get("candidate_id"),
                "action": c.get("action"),
                "label": c.get("label"),
                "L_star": c.get("L_star"),
                "cooccurrence": c.get("cooccurrence"),
                "branch": c.get("branch") or "in_distribution",
            }
        )

    return {
        "fidelity_band": state.get("fidelity_band"),
        "diagnosis": _compact_diagnosis(diagnosis),
        "hull": _compact_hull(state.get("hull")),
        "recipe": _recipe_lines(state),
        "pfc_vs_box": _pfc_vs_box(state),
        "tradeoff_frame": TRADEOFF_FRAME,
        "taste_text": state.get("taste_text"),
        "user_request": state.get("user_request"),
        "title": state.get("title"),
        "identity_roles": identity_roles,
        "identity_critical": state.get("identity_critical"),
        "identity_tension": _identity_tension(state, ranked),
        "candidates": candidates_compact,
        "planned_slots": [
            {"slot_id": s.get("slot_id"), "kind": s.get("kind"), "reason": s.get("reason")}
            for s in planned_slots
        ],
        "bundles": top,
        "candidate_pool_summary": {
            "n": len(pool),
            "best": {
                "L_total": None if best_pool is None else best_pool.get("L_total"),
                "L_max_norm": None if best_pool is None else best_pool.get("L_max_norm"),
                "n_red": None if best_pool is None else best_pool.get("n_red"),
            },
        },
        "iteration": iteration,
        "adds_so_far": adds_so_far,
        "max_total_adds": max_adds,
        "adds_exhausted": adds_exhausted,
        "stop_adding_note": (
            f"This run already added {adds_so_far} new ingredient(s) (budget {max_adds}). "
            + (
                "Add budget EXHAUSTED — do not choose another add; prefer accept / "
                "accept_pool_best / swap / remove."
                if adds_exhausted
                else "Each further add must clear a meaningful delta_L_star improvement "
                f"(≥ {marginal_eps}); marginal add bundles are flagged marginal_improvement."
            )
        ),
        "decision_outcomes": outcomes,
        "reflection_digest": reflection_digest(outcomes),
        "revisit_reflection": revisit,
        "allowed_actions": (
            ["accept", "accept_pool_best", "apply_bundle", "add", "swap", "remove", "expand"]
            if top
            else ["accept", "accept_pool_best", "add", "swap", "remove", "expand"]
        ),
        "requirement_tags": state.get("requirement_tags") or [],
        "agent_mode": state.get("agent_mode"),
        "neighborhood_expansion": _compact_neighborhood_expansion(state),
    }
