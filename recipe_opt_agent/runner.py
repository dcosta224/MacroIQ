"""Public runner for the recipe optimization agent."""

from __future__ import annotations

from typing import Any, Callable

from recipe_opt_agent.config import AgentConfig
from recipe_opt_agent.graph import run_agent, stream_agent
from recipe_opt_agent.state import AgentState
from recipe_opt_agent.telemetry import empty_telemetry


def _initial_state(
    *,
    problem: dict[str, Any],
    taste_text: str,
    title: str,
    canonical_id: int | None,
    config: AgentConfig,
    agent_mode: str = "neighborhood",
    user_request: str = "",
) -> AgentState:
    return {
        "canonical_id": int(canonical_id or 0),
        "taste_text": taste_text,
        "title": title,
        "user_request": user_request or taste_text,
        "agent_mode": agent_mode,
        "config": {
            "model": config.model,
            "model_escalate": config.model_escalate,
            "creative_model": config.creative_model,
            "shadow_draft_model": config.shadow_draft_model,
            "enable_shadow_gpt_candidate": config.enable_shadow_gpt_candidate,
            "tags_model": config.tags_model,
            "judge_model": config.judge_model,
            "enable_llm_judge": config.enable_llm_judge,
            "llm_judge_mode": config.llm_judge_mode,
            "identity_extract_model": config.identity_extract_model,
            "max_iterations": config.max_iterations,
            "F_accept": config.F_accept,
            "F_max": config.F_max,
            "protein_min": config.protein_min,
            "protein_max": config.protein_max,
            "carb_min": config.carb_min,
            "carb_max": config.carb_max,
            "fat_min": config.fat_min,
            "fat_max": config.fat_max,
            "kcal_target": config.kcal_target,
            "nutrition_slack_weight": config.nutrition_slack_weight,
            "neighbor_k": config.neighbor_k,
            "loss_field_grid_n": config.loss_field_grid_n,
            "agent_mode": agent_mode,
            "save_on_must_retry_feasible": config.save_on_must_retry_feasible,
            "w_score_nutrient": config.w_score_nutrient,
            "w_score_ratio": config.w_score_ratio,
            "w_score_intent": config.w_score_intent,
            "w_score_churn": config.w_score_churn,
            "judge_epsilon": config.judge_epsilon,
            "min_finalists": config.min_finalists,
            "max_finalists": config.max_finalists,
            "auto_apply_delta_eps": config.auto_apply_delta_eps,
            "auto_apply_margin": config.auto_apply_margin,
            "oscillation_improve_eps": config.oscillation_improve_eps,
            "max_foodon_aggregation_levels": config.max_foodon_aggregation_levels,
            "n_ideation_candidates": config.n_ideation_candidates,
            "ood_delta_handicap": config.ood_delta_handicap,
            "ideation_model": config.ideation_model,
            "marginal_add_delta_eps": config.marginal_add_delta_eps,
            "max_total_adds": config.max_total_adds,
            "bundle_proxy_cap": config.bundle_proxy_cap,
            "bundle_lp_cap": config.bundle_lp_cap,
        },
        "problem": problem,
        "candidate_pool": [],
        "history": [],
        "decision_outcomes": [],
        "recent_edit_fingerprints": [],
        "run_telemetry": empty_telemetry(),
        "iteration": 0,
    }


def run_recipe_opt_agent(
    *,
    problem: dict[str, Any],
    taste_text: str = "",
    title: str = "",
    canonical_id: int | None = None,
    config: AgentConfig | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    agent_mode: str = "neighborhood",
    user_request: str = "",
    langgraph_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the LangGraph loop on a pre-built optimization problem dict.

    `problem` must include: x0, M, ingredient_basis, basis_samples, ratio_samples,
    and optionally modification_candidates, marginal_nodes, total_mass, kcal_target.

    If `on_event` is provided, streams step events (for web UI / notebook / eval artifacts).
    Pass `langgraph_config` (tags/metadata) for LangSmith filtering.
    """
    from recipe_opt_agent.observability import ensure_tracing_env

    ensure_tracing_env()
    cfg = config or AgentConfig()
    mode = agent_mode or cfg.agent_mode
    state = _initial_state(
        problem=problem,
        taste_text=taste_text,
        title=title,
        canonical_id=canonical_id,
        config=cfg,
        agent_mode=mode,
        user_request=user_request or taste_text,
    )
    creative = mode == "creative"
    if on_event is not None:
        out = stream_agent(state, on_event=on_event, creative=creative, config=langgraph_config)
    else:
        out = run_agent(state, creative=creative, config=langgraph_config)
    return dict(out.get("final") or out)
