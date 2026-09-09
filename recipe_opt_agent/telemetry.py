"""Lean run telemetry + needle snapshots for outcome history."""

from __future__ import annotations

from typing import Any


def empty_telemetry() -> dict[str, Any]:
    return {
        "final_status": None,
        "final_L_max_norm": None,
        "final_n_red": None,
        "feasible": None,
        "final_ratio_term": None,
        "final_nutrient_slack": None,
        "final_holistic": None,
        "iterations_used": 0,
        "n_llm_calls": 0,
        "n_auto_applies": 0,
        "tag_violations_final": 0,
        "expand_count": 0,
        "oscillation_hits": 0,
        "nodes": {},
        "edges": [],
    }


def snapshot_needles(state: dict[str, Any]) -> dict[str, Any]:
    """Compact 3-way tradeoff + fidelity needles from current diagnose state."""
    diag = state.get("diagnosis") or {}
    opt = state.get("opt") or {}
    tl = opt.get("term_losses") or {}
    ratio_term = None
    ratio_source = None
    if tl.get("ratio_surrogate") is not None:
        ratio_term = tl.get("ratio_surrogate")
        ratio_source = "ratio_surrogate"
    elif tl.get("ratio_loss") is not None:
        ratio_term = tl.get("ratio_loss")
        ratio_source = "ratio_loss"
    elif tl.get("ratio") is not None:
        ratio_term = tl.get("ratio")
        ratio_source = "ratio"
    # Do NOT fall back to summing __share levels — that is not a loss and produces
    # false "perfect" / nonsense telemetry for the UI.
    nutrient_slack = _nutrient_slack_from_state(state)
    holistic = _holistic_from_state(state)
    return {
        "L_total": diag.get("L_total") if diag.get("L_total") is not None else opt.get("objective"),
        "L_max_norm": diag.get("L_max_norm"),
        "n_red": diag.get("n_red"),
        "ratio_term": float(ratio_term) if ratio_term is not None else None,
        "ratio_source": ratio_source,
        "nutrient_slack": nutrient_slack,
        "holistic": holistic,
    }


def _nutrient_slack_from_state(state: dict[str, Any]) -> float | None:
    opt = state.get("opt") or {}
    pfc = opt.get("pfc_after") or {}
    cfg = state.get("config") or {}
    if not pfc:
        return None
    try:
        p = float(pfc.get("protein"))
        c = float(pfc.get("carbs"))
        f = float(pfc.get("fat"))
    except (TypeError, ValueError):
        return None
    pmin = float(cfg.get("protein_min", 0.0))
    pmax = float(cfg.get("protein_max", 1.0))
    cmin = float(cfg.get("carb_min", 0.0))
    cmax = float(cfg.get("carb_max", 1.0))
    fmin = float(cfg.get("fat_min", 0.0))
    fmax = float(cfg.get("fat_max", 1.0))
    # Distance outside box (0 if inside)
    slack = 0.0
    if p < pmin:
        slack += pmin - p
    elif p > pmax:
        slack += p - pmax
    if c < cmin:
        slack += cmin - c
    elif c > cmax:
        slack += c - cmax
    if f < fmin:
        slack += fmin - f
    elif f > fmax:
        slack += f - fmax
    return float(slack)


def _holistic_from_state(state: dict[str, Any]) -> float | None:
    from recipe_opt_agent.candidate_scoring import compute_intent_gap

    request = state.get("user_request") or state.get("taste_text") or ""
    title = state.get("title") or ""
    chosen = state.get("chosen_recipe") or {}
    ings = list(chosen.get("ingredients") or [])
    if not request and not title:
        return None
    gap = compute_intent_gap(request or title, title, ings)
    return float(1.0 - gap)


def delta_needles(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("L_total", "L_max_norm", "n_red", "ratio_term", "nutrient_slack", "holistic"):
        a, b = after.get(k), before.get(k)
        if a is None or b is None:
            out[k] = None
        else:
            out[k] = float(a) - float(b)
    return out


def reflection_digest(outcomes: list[dict[str, Any]]) -> str:
    if not outcomes:
        return "No prior decisions yet."
    last = outcomes[-1]
    d = last.get("delta") or {}
    parts = []
    for axis, label in (
        ("ratio_term", "ratio"),
        ("nutrient_slack", "nutrient"),
        ("holistic", "holistic"),
    ):
        v = d.get(axis)
        if v is None:
            parts.append(f"{label}=n/a")
        elif axis == "holistic":
            # higher holistic is better
            parts.append(f"{label}={'up' if v > 1e-6 else 'down' if v < -1e-6 else 'flat'}")
        else:
            # lower ratio_term / nutrient_slack is better
            parts.append(f"{label}={'down' if v < -1e-6 else 'up' if v > 1e-6 else 'flat'}")
    action = (last.get("decision") or {}).get("action")
    return f"After last {action}: " + ", ".join(parts) + "."


def bump_telemetry(tel: dict[str, Any] | None, **kwargs: Any) -> dict[str, Any]:
    out = dict(tel or empty_telemetry())
    for k, v in kwargs.items():
        if k == "nodes" and isinstance(v, dict):
            nodes = dict(out.get("nodes") or {})
            nodes.update(v)
            out["nodes"] = nodes
        elif k == "edges" and isinstance(v, list):
            edges = list(out.get("edges") or [])
            edges.extend(v)
            out["edges"] = edges
        elif k in {"n_llm_calls", "n_auto_applies", "expand_count", "oscillation_hits"} and isinstance(v, int):
            out[k] = int(out.get(k) or 0) + v
        else:
            out[k] = v
    return out


def finalize_telemetry(state: dict[str, Any], *, status: str) -> dict[str, Any]:
    tel = dict(state.get("run_telemetry") or empty_telemetry())
    needles = snapshot_needles(state)
    opt = state.get("opt") or {}
    diag = state.get("diagnosis") or {}
    tel["final_status"] = status
    tel["final_L_max_norm"] = needles.get("L_max_norm")
    tel["final_n_red"] = needles.get("n_red")
    tel["feasible"] = opt.get("feasible")
    tel["final_ratio_term"] = needles.get("ratio_term")
    tel["final_ratio_source"] = needles.get("ratio_source")
    tel["final_nutrient_slack"] = needles.get("nutrient_slack")
    tel["final_holistic"] = needles.get("holistic")
    tel["iterations_used"] = int(state.get("iteration") or 0)
    # Tag violations: count from requirement tags vs chosen ingredients if available
    tel["tag_violations_final"] = _count_tag_vios(state)
    return tel


def _count_tag_vios(state: dict[str, Any]) -> int:
    from recipe_opt_agent.requirement_tags import RequirementTag, tag_violations_for_ingredient

    raw = state.get("requirement_tags") or []
    tags: list[RequirementTag] = []
    for r in raw:
        if isinstance(r, dict):
            tags.append(
                RequirementTag(
                    tag_id=str(r.get("tag_id") or ""),
                    kind=str(r.get("kind") or "preference"),
                    polarity=str(r.get("polarity") or "require"),
                    source_text=str(r.get("source_text") or ""),
                )
            )
    if not tags:
        return 0
    ings = (state.get("chosen_recipe") or {}).get("ingredients") or []
    n = 0
    for row in ings:
        label = str(row.get("label") or row.get("name") or "")
        if tag_violations_for_ingredient(label, tags):
            n += 1
    return n


def edit_fingerprint(edits: list[dict[str, Any]] | None) -> str:
    """Stable fingerprint for anti-oscillation (sorted action+label)."""
    parts = []
    for e in edits or []:
        parts.append(f"{e.get('action')}:{str(e.get('label') or '').lower().strip()}")
    return "|".join(sorted(parts))


def clear_favorite_bundle(
    bundles: list[dict[str, Any]],
    *,
    delta_eps: float = 0.01,
    margin: float = 0.02,
    passes_tags_fn=None,
    ood_delta_handicap: float = 0.015,
) -> dict[str, Any] | None:
    """Return the unique clear LP favorite, or None if uncertain.

    OOD / hybrid bundles get a small ``delta_L_star`` handicap so a modest ratio
    cost does not auto-lose to in-distribution when nutrient lift is real. The
    stored ``delta_L_star`` on the returned bundle is unchanged — only ranking
    uses the adjusted score.
    """

    def _adj(b: dict[str, Any]) -> float:
        d = float(b.get("delta_L_star") or 0.0)
        branch = str(b.get("branch") or "in_distribution")
        if branch in {"ood_protein", "ood_other", "hybrid"}:
            # More negative is better; subtract handicap to favor OOD slightly
            nutrient = b.get("nutrient_slack")
            # Extra credit when nutrient slack is near zero (macros actually hit)
            extra = 0.0
            try:
                if nutrient is not None and float(nutrient) <= 1e-6:
                    extra = 0.5 * float(ood_delta_handicap)
            except (TypeError, ValueError):
                pass
            return d - float(ood_delta_handicap) - extra
        return d

    eligible = []
    for b in bundles or []:
        if not b.get("lp_evaluated"):
            continue
        d = b.get("delta_L_star")
        if d is None:
            continue
        if passes_tags_fn is not None and not passes_tags_fn(b):
            continue
        if b.get("oscillation_blocked"):
            continue
        if _adj(b) < -delta_eps:
            eligible.append(b)
    if not eligible:
        return None
    eligible.sort(key=_adj)
    best = eligible[0]
    if len(eligible) == 1:
        return best
    second = eligible[1]
    if _adj(best) - _adj(second) < -margin:
        return best
    return None
