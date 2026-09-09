"""Routing hooks for MVP agent tool selection (linear today, extensible later)."""

from __future__ import annotations

from typing import Protocol

from mvp_agent.context import AgentSession


DEFAULT_TOOL_ORDER = [
    "embed_taste_query",
    "rank_recipes_by_fit",
    "optimize_top_candidates",
    "judge_final_recipe",
    "finalize_recommendation",
]


class AgentRouter(Protocol):
    def next_tools(self, session: AgentSession) -> list[str]:
        """Return tool names the agent should call next (advisory for prompts)."""
        ...


class LinearRouter:
    """Fixed pipeline order; future routers may branch on query or session state."""

    def next_tools(self, session: AgentSession) -> list[str]:
        from mvp_agent.context import Phase, PHASE_ORDER

        idx = PHASE_ORDER.index(session.phase)
        remaining = DEFAULT_TOOL_ORDER[idx:]
        return remaining
