"""LangSmith / local observability helpers for the recipe opt agent.

Enable tracing with env vars (prefer LANGSMITH_* ; LANGCHAIN_* still work):

  LANGSMITH_TRACING=true
  LANGSMITH_API_KEY=lsv2_...
  LANGSMITH_PROJECT=recipe-opt-agent

Optional: LANGSMITH_WORKSPACE_ID, LANGSMITH_ENDPOINT (EU).

Metric classes
--------------
* ``ACTIONABLE_METRICS`` — primary knobs for accept rate, fidelity, controller mix, cost.
* ``OBSERVABILITY_METRICS`` — auxiliary diagnostics (FoodOn rollup depth / hit quality,
  neighborhood size, apply churn, grounding, etc.). Posted to LangSmith with
  ``comment=class=observability`` so they stay filterable without crowding the
  main decision dashboard.
"""

from __future__ import annotations

import math
import os
from typing import Any


ACTIONABLE_METRICS = (
    # End-of-run — decide whether to change F_accept / max_iterations / retrieval
    "final_status_ok",
    "final_L_max_norm",
    "final_n_red",
    "final_nutrient_slack",
    "final_holistic",
    "tag_violations_final",
    "iterations_used",
    # Controller mix — widen auto-apply margin if agreement high but auto≈0
    "n_llm_calls",
    "n_auto_applies",
    "lp_agreement_rate",
    "escalate_rate",
    # Escape hatches / loops
    "expand_count",
    "oscillation_hits",
    # Cost / latency
    "elapsed_s",
    "estimated_cost_usd",
)

# Auxiliary / diagnostic — neighborhood geometry, FoodOn signal quality, path shape.
OBSERVABILITY_METRICS = (
    # FoodOn leaf→basis aggregation (neighborhood coarseness)
    "foodon_n_aggregated",
    "foodon_n_unmapped",
    "foodon_n_basis_in_recipe",
    "foodon_n_basis_neighborhood",
    "foodon_mean_aggregation_levels",
    "foodon_max_aggregation_levels",
    "foodon_frac_aggregated",
    # Hit-count quality for ratio / share terms
    "foodon_min_hits_in_recipe",
    "foodon_mean_hits_in_recipe",
    "foodon_n_low_hit_basis",
    "foodon_min_basis_hits_target",
    # Neighborhood / recipe shape
    "neighborhood_n_recipes",
    "n_ingredients_final",
    "final_ratio_term",
    # Path / retrieval shape
    "pool_size_final",
    "n_decision_outcomes",
    "n_applies",
    "hull_outside_final",
    "grounding_resolve_rate",
)


def tracing_enabled() -> bool:
    for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        val = (os.environ.get(key) or "").strip().lower()
        if val in {"1", "true", "yes", "on"}:
            return True
    return False


def langsmith_api_key() -> str | None:
    return os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY") or None


def langsmith_project(default: str = "recipe-opt-agent") -> str:
    return (
        os.environ.get("LANGSMITH_PROJECT")
        or os.environ.get("LANGCHAIN_PROJECT")
        or default
    )


def ensure_tracing_env(*, project: str | None = None) -> dict[str, str]:
    """Normalize env so LangGraph + langsmith pick up tracing.

    Does not invent an API key — returns status for callers/logs.
    """
    status: dict[str, str] = {}
    proj = project or langsmith_project()
    os.environ.setdefault("LANGSMITH_PROJECT", proj)
    os.environ.setdefault("LANGCHAIN_PROJECT", proj)
    if tracing_enabled():
        # Dual-write legacy + modern flags
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        status["tracing"] = "on"
    else:
        status["tracing"] = "off"
    key = langsmith_api_key()
    if key:
        os.environ.setdefault("LANGSMITH_API_KEY", key)
        os.environ.setdefault("LANGCHAIN_API_KEY", key)
        status["api_key"] = "set"
    else:
        status["api_key"] = "missing"
    status["project"] = proj
    return status


def get_openai_client():
    """OpenAI client for recipe_opt_agent — **only** ``OPENAI_API_KEY``.

    Never use OPENAI_API_KEY_2 / _3 / _FOODON (those are for batch/FoodOn pipelines).
    Wrapped for LangSmith when tracing is on.
    """
    from openai import OpenAI

    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for recipe_opt_agent LLM calls "
            "(do not fall back to OPENAI_API_KEY_2/_3/_FOODON)."
        )
    client = OpenAI(api_key=key)
    if tracing_enabled() and langsmith_api_key():
        try:
            from langsmith.wrappers import wrap_openai

            return wrap_openai(client)
        except Exception:
            return client
    return client


def _truncate(text: str | None, n: int = 500) -> str:
    s = str(text or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def agent_graph_meta(*, creative: bool) -> dict[str, Any]:
    """Static node/edge layout for the active graph mode."""
    from recipe_opt_agent.graph import (
        CREATIVE_FLOW_EDGES,
        CREATIVE_FLOW_NODES,
        FLOW_EDGES,
        FLOW_NODES,
    )

    nodes = list(CREATIVE_FLOW_NODES if creative else FLOW_NODES)
    edges = list(CREATIVE_FLOW_EDGES if creative else FLOW_EDGES)
    return {
        "creative": bool(creative),
        "nodes": nodes,
        "edges": [{"from": a, "to": b} for a, b in edges],
        "n_nodes": len(nodes),
        "n_edges": len(edges),
    }


def models_by_stage(config: Any) -> dict[str, str]:
    """Which model string is configured for each LLM stage."""
    return {
        "decide_routine": str(getattr(config, "model", "") or ""),
        "decide_escalate": str(getattr(config, "model_escalate", "") or ""),
        "draft": str(getattr(config, "creative_model", "") or ""),
        "tags": str(getattr(config, "tags_model", "") or ""),
        "judge": str(getattr(config, "judge_model", "") or ""),
        "identity_extract": str(getattr(config, "identity_extract_model", "") or ""),
    }


def build_run_metadata(
    *,
    config: Any | None = None,
    agent_mode: str = "neighborhood",
    user_request: str = "",
    taste_text: str = "",
    title: str = "",
    case_name: str = "",
    suite_id: str = "",
    canonical_id: int | None = None,
    problem: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rich metadata attached to LangSmith / eval runs (filterable in the UI)."""
    from recipe_opt_agent.config import AgentConfig

    cfg = config if config is not None else AgentConfig()
    mode = (agent_mode or getattr(cfg, "agent_mode", None) or "neighborhood").strip().lower()
    creative = mode == "creative"
    problem = problem or {}
    request = user_request or taste_text or title
    foodon = problem.get("foodon_basis_report") or (problem.get("chosen_recipe") or {}).get(
        "foodon_basis_report"
    )

    meta: dict[str, Any] = {
        "agent": "recipe_opt",
        "agent_mode": mode,
        "case_name": case_name,
        "suite_id": suite_id,
        "canonical_id": int(canonical_id) if canonical_id is not None else problem.get("canonical_id"),
        "title": _truncate(title or problem.get("title") or "", 200),
        "user_request": _truncate(request, 500),
        "taste_text": _truncate(taste_text or problem.get("taste_text") or "", 300),
        "macro_targets": {
            "protein_min": float(getattr(cfg, "protein_min", 0.0)),
            "protein_max": float(getattr(cfg, "protein_max", 1.0)),
            "carb_min": float(getattr(cfg, "carb_min", 0.0)),
            "carb_max": float(getattr(cfg, "carb_max", 1.0)),
            "fat_min": float(getattr(cfg, "fat_min", 0.0)),
            "fat_max": float(getattr(cfg, "fat_max", 1.0)),
        },
        "F_accept": float(getattr(cfg, "F_accept", 1.0)),
        "F_max": float(getattr(cfg, "F_max", 1.5)),
        "max_iterations": int(getattr(cfg, "max_iterations", 3)),
        "neighbor_k": int(getattr(cfg, "neighbor_k", 40)),
        "auto_apply_delta_eps": float(getattr(cfg, "auto_apply_delta_eps", 0.01)),
        "auto_apply_margin": float(getattr(cfg, "auto_apply_margin", 0.02)),
        "oscillation_improve_eps": float(getattr(cfg, "oscillation_improve_eps", 0.02)),
        "models": models_by_stage(cfg),
        "graph": agent_graph_meta(creative=creative),
        "score_weights": cfg.score_weights() if creative and hasattr(cfg, "score_weights") else None,
        "neighborhood_n_recipes": (
            problem.get("n_matches")
            or len(problem.get("neighborhood_recipes") or [])
            or None
        ),
        "neighborhood_from_cache": problem.get("neighborhood_from_cache"),
        "foodon_n_aggregated_start": (foodon or {}).get("n_aggregated") if isinstance(foodon, dict) else None,
        "foodon_n_basis_in_recipe_start": (foodon or {}).get("n_basis_nodes_in_recipe")
        if isinstance(foodon, dict)
        else None,
    }
    if extra:
        meta.update(extra)
    return meta


def run_config(
    *,
    case_name: str = "",
    agent_mode: str = "neighborhood",
    suite_id: str = "",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    config: Any | None = None,
    user_request: str = "",
    taste_text: str = "",
    title: str = "",
    canonical_id: int | None = None,
    problem: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """LangGraph RunnableConfig with tags/metadata visible in LangSmith."""
    ensure_tracing_env()
    base = build_run_metadata(
        config=config,
        agent_mode=agent_mode,
        user_request=user_request,
        taste_text=taste_text,
        title=title,
        case_name=case_name,
        suite_id=suite_id,
        canonical_id=canonical_id,
        problem=problem,
        extra=metadata,
    )
    tag_list = ["recipe-opt-agent", agent_mode or "neighborhood"]
    if case_name:
        tag_list.append(f"case:{case_name}")
    if suite_id:
        tag_list.append(f"suite:{suite_id}")
    for t in tags or []:
        if t not in tag_list:
            tag_list.append(t)
    return {
        "tags": tag_list,
        "metadata": base,
        "run_name": case_name or f"recipe_opt_{agent_mode or 'neighborhood'}",
    }


def _foodon_observability(foodon: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(foodon, dict):
        return {
            "foodon_n_aggregated": None,
            "foodon_n_unmapped": None,
            "foodon_n_basis_in_recipe": None,
            "foodon_n_basis_neighborhood": None,
            "foodon_mean_aggregation_levels": None,
            "foodon_max_aggregation_levels": None,
            "foodon_frac_aggregated": None,
            "foodon_min_hits_in_recipe": None,
            "foodon_mean_hits_in_recipe": None,
            "foodon_n_low_hit_basis": None,
            "foodon_min_basis_hits_target": None,
        }
    ings = list(foodon.get("ingredients") or [])
    levels = [
        float(r["aggregation_levels"])
        for r in ings
        if r.get("aggregation_levels") is not None
    ]
    n_ing = int(foodon.get("n_ingredients") or len(ings) or 0)
    n_agg = foodon.get("n_aggregated")
    used = [r for r in (foodon.get("basis_nodes") or []) if r.get("in_current_recipe")]
    hits = [float(r.get("n_hits") or 0) for r in used]
    target = foodon.get("min_basis_hits_target")
    try:
        target_f = float(target) if target is not None else None
    except (TypeError, ValueError):
        target_f = None
    n_low = None
    if target_f is not None and used:
        n_low = sum(1 for h in hits if h < target_f)
    return {
        "foodon_n_aggregated": n_agg,
        "foodon_n_unmapped": foodon.get("n_unmapped"),
        "foodon_n_basis_in_recipe": foodon.get("n_basis_nodes_in_recipe"),
        "foodon_n_basis_neighborhood": foodon.get("n_basis_nodes_neighborhood"),
        "foodon_mean_aggregation_levels": (sum(levels) / len(levels)) if levels else None,
        "foodon_max_aggregation_levels": max(levels) if levels else None,
        "foodon_frac_aggregated": (float(n_agg) / n_ing) if n_ing and n_agg is not None else None,
        "foodon_min_hits_in_recipe": min(hits) if hits else None,
        "foodon_mean_hits_in_recipe": (sum(hits) / len(hits)) if hits else None,
        "foodon_n_low_hit_basis": n_low,
        "foodon_min_basis_hits_target": target_f,
    }


def metrics_from_result(result: dict[str, Any], *, elapsed_s: float | None = None) -> dict[str, Any]:
    """Flatten actionable + observability metrics from a run final payload."""
    tel = result.get("run_telemetry") or {}
    hist = result.get("history") or []
    decide_modes = [h.get("decide_mode") for h in hist if h.get("decide_mode")]
    agreements = [h.get("agreed_with_lp_best") for h in hist if "agreed_with_lp_best" in h]
    n_decide = len(decide_modes) or 1
    status = result.get("status") or tel.get("final_status")
    foodon = (
        result.get("foodon_basis_report")
        or (result.get("chosen") or {}).get("foodon_basis_report")
        or ((result.get("chosen") or {}).get("entry") or {}).get("foodon_basis_report")
    )
    foodon_obs = _foodon_observability(foodon if isinstance(foodon, dict) else None)

    chosen = result.get("chosen") or {}
    entry = chosen.get("entry") if isinstance(chosen, dict) else None
    ings = None
    if isinstance(entry, dict):
        ings = entry.get("ingredients")
    if not ings and isinstance(chosen, dict):
        ings = chosen.get("ingredients")
    pool = result.get("candidate_pool") or (result.get("scored_finalists") or [])
    outcomes = result.get("decision_outcomes") or []
    n_applies = sum(
        1
        for o in outcomes
        if (o.get("decision") or {}).get("action")
        in {"apply_bundle", "add", "swap", "remove", "apply"}
    )
    nodes = tel.get("nodes") or {}
    ground = nodes.get("ground") or {}
    diagnose = nodes.get("diagnose") or {}
    hull_outside = diagnose.get("hull_outside")
    if hull_outside is None and isinstance(result.get("hull"), dict):
        hull_outside = not bool(result["hull"].get("intersects"))

    actionable = {
        "final_status_ok": 1.0 if status and str(status).startswith("accepted") else 0.0,
        "final_L_max_norm": tel.get("final_L_max_norm"),
        "final_n_red": tel.get("final_n_red"),
        "final_nutrient_slack": tel.get("final_nutrient_slack"),
        "final_holistic": tel.get("final_holistic"),
        "tag_violations_final": tel.get("tag_violations_final"),
        "iterations_used": tel.get("iterations_used"),
        "n_llm_calls": tel.get("n_llm_calls"),
        "n_auto_applies": tel.get("n_auto_applies"),
        "lp_agreement_rate": (
            sum(1 for a in agreements if a) / len(agreements) if agreements else None
        ),
        "escalate_rate": (
            sum(1 for m in decide_modes if m == "llm_escalate") / n_decide if decide_modes else None
        ),
        "expand_count": tel.get("expand_count"),
        "oscillation_hits": tel.get("oscillation_hits"),
        "elapsed_s": elapsed_s,
        "estimated_cost_usd": tel.get("estimated_cost_usd"),
    }
    observability = {
        **foodon_obs,
        "neighborhood_n_recipes": foodon.get("neighborhood_n_recipes")
        if isinstance(foodon, dict)
        else None,
        "n_ingredients_final": len(ings) if isinstance(ings, list) else None,
        "final_ratio_term": tel.get("final_ratio_term"),
        "pool_size_final": len(pool) if isinstance(pool, list) else None,
        "n_decision_outcomes": len(outcomes) if isinstance(outcomes, list) else None,
        "n_applies": n_applies,
        "hull_outside_final": 1.0 if hull_outside else (0.0 if hull_outside is False else None),
        "grounding_resolve_rate": ground.get("resolve_rate"),
    }

    return {
        **actionable,
        **observability,
        "status": status,
        "metric_classes": {
            "actionable": list(ACTIONABLE_METRICS),
            "observability": list(OBSERVABILITY_METRICS),
        },
        "actionable": {k: actionable.get(k) for k in ACTIONABLE_METRICS},
        "observability": {k: observability.get(k) for k in OBSERVABILITY_METRICS},
        "foodon_basis_report": foodon,
    }


def _post_metric_class(
    client: Any,
    *,
    run_id: str,
    metrics: dict[str, Any],
    keys: tuple[str, ...],
    metric_class: str,
    comment: str = "",
) -> list[str]:
    posted: list[str] = []
    class_comment = f"class={metric_class}"
    if comment:
        class_comment = f"{class_comment}; {comment}"
    for key in keys:
        val = metrics.get(key)
        if val is None or isinstance(val, str):
            continue
        try:
            score = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        # LangSmith rejects scores outside ±1e5; clamp runaway LP/telemetry values.
        score = float(max(-99999.9999, min(99999.9999, score)))
        try:
            client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=class_comment,
            )
            posted.append(key)
        except Exception:
            continue
    return posted


def post_run_feedback(
    *,
    run_id: str | None,
    metrics: dict[str, Any],
    comment: str = "",
    include_observability: bool = True,
) -> list[str]:
    """Attach numeric feedback scores to a LangSmith root run.

    Posts ``ACTIONABLE_METRICS`` (and optionally ``OBSERVABILITY_METRICS``) with
    ``comment=class=…`` so filters can separate the two classes.
    """
    if not run_id or not langsmith_api_key():
        return []
    try:
        from langsmith import Client
    except Exception:
        return []
    client = Client()
    posted = _post_metric_class(
        client,
        run_id=run_id,
        metrics=metrics,
        keys=ACTIONABLE_METRICS,
        metric_class="actionable",
        comment=comment,
    )
    if include_observability:
        posted.extend(
            _post_metric_class(
                client,
                run_id=run_id,
                metrics=metrics,
                keys=OBSERVABILITY_METRICS,
                metric_class="observability",
                comment=comment,
            )
        )
    return posted


def get_current_run_id() -> str | None:
    try:
        from langsmith import get_current_run_tree

        tree = get_current_run_tree()
        if tree is None:
            return None
        return str(getattr(tree, "id", None) or getattr(tree, "trace_id", None) or "")
    except Exception:
        return None
