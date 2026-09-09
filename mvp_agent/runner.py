"""Strands agent runner for the MVP recipe pipeline."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from mvp_log import finish_run, start_run
from mvp_pipeline import PipelineEvent, UserQuery, run_pipeline

from mvp_agent.config import AgentConfig
from mvp_agent.context import AgentSession, Phase, set_active_session
from mvp_agent.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from mvp_agent.router import DEFAULT_TOOL_ORDER, LinearRouter
from mvp_agent.tools import ALL_TOOLS, TOOL_REGISTRY

logger = logging.getLogger(__name__)


def _build_user_message(query: UserQuery) -> str:
    return (
        "Recommend a recipe for this request.\n"
        f"taste_text: {query.taste_text}\n"
        f"kcal: {query.kcal_min}-{query.kcal_max}\n"
        f"PFC fractions — fat: {query.fat_frac_min}-{query.fat_frac_max}, "
        f"carb: {query.carb_frac_min}-{query.carb_frac_max}, "
        f"protein: {query.protein_frac_min}-{query.protein_frac_max}\n"
        f"top_k: {query.top_k}\n"
        "Call tools 1 through 5 in order: embed_taste_query, rank_recipes_by_fit, "
        "optimize_top_candidates, judge_final_recipe, finalize_recommendation."
    )


def _run_deterministic(session: AgentSession) -> None:
    """Sequential tool execution when Bedrock agent does not complete all phases."""
    logger.warning("Strands agent did not finish pipeline; running deterministic fallback")
    for name in DEFAULT_TOOL_ORDER:
        if session.phase == Phase.FINALIZED:
            break
        tool_fn = TOOL_REGISTRY[name]
        if name == "embed_taste_query":
            result = tool_fn(session.query.taste_text)
        elif name == "optimize_top_candidates":
            result = tool_fn(session.query.top_k)
        else:
            result = tool_fn()
        if not result.get("ok"):
            raise RuntimeError(f"Tool {name} failed: {result.get('message')}")


def build_agent(cfg: AgentConfig | None = None):
    """Construct a Strands Agent with Bedrock model and MVP tools."""
    from strands import Agent
    from strands.models import BedrockModel

    cfg = cfg or AgentConfig.from_env()
    if cfg.aws_profile:
        os.environ.setdefault("AWS_PROFILE", cfg.aws_profile)

    model = BedrockModel(
        model_id=cfg.bedrock_model_id,
        region_name=cfg.aws_region,
    )
    return Agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    )


def run_agent_pipeline(
    query: UserQuery,
    *,
    on_event: Callable[[PipelineEvent], None] | None = None,
    log_to_db: bool = True,
    corpus: dict[str, Any] | None = None,
    cfg: AgentConfig | None = None,
) -> dict[str, Any]:
    """Run MVP pipeline via Strands Agent (Bedrock orchestrator + OpenAI judge)."""
    cfg = cfg or AgentConfig.from_env()
    if not cfg.enabled:
        return run_pipeline(
            query,
            on_event=on_event,
            log_to_db=log_to_db,
            corpus=corpus,
        )

    run_id = None
    if log_to_db:
        try:
            run_id = start_run(query.taste_text, {
                "taste_text": query.taste_text,
                "kcal_min": query.kcal_min,
                "kcal_max": query.kcal_max,
                "fat_frac_min": query.fat_frac_min,
                "fat_frac_max": query.fat_frac_max,
                "carb_frac_min": query.carb_frac_min,
                "carb_frac_max": query.carb_frac_max,
                "protein_frac_min": query.protein_frac_min,
                "protein_frac_max": query.protein_frac_max,
                "w_semantic": query.w_semantic,
                "w_nutrient": query.w_nutrient,
                "top_k": query.top_k,
            })
        except Exception:
            run_id = None

    session = AgentSession(
        query=query,
        run_id=run_id,
        log_to_db=log_to_db,
        corpus=corpus,
        on_event=on_event,
    )
    set_active_session(session)

    try:
        try:
            agent = build_agent(cfg)
            user_msg = _build_user_message(query)
            router = LinearRouter()
            remaining = router.next_tools(session)
            if remaining:
                user_msg += f"\nRemaining tools: {', '.join(remaining)}."

            try:
                agent(user_msg)
            except Exception as exc:
                logger.warning(
                    "Strands agent error (%s); using deterministic fallback", exc
                )
        except Exception as exc:
            logger.warning(
                "Could not run Strands agent (%s); using deterministic fallback", exc
            )

        if session.phase != Phase.FINALIZED:
            _run_deterministic(session)

        if session.final_payload is None:
            raise RuntimeError("Pipeline did not produce a final recommendation")

        return session.final_payload

    except Exception as exc:
        if run_id:
            try:
                finish_run(run_id, status="error", error_message=str(exc))
            except Exception:
                pass
        raise
    finally:
        set_active_session(None)
