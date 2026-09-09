"""Shared session state and phase gates for MVP agent tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from mvp_pipeline import PipelineEvent, UserQuery
from mvp_recipe_judge import JudgeResult
from mvp_recipe_ranker import RankedRecipe


class Phase(str, Enum):
    INIT = "init"
    EMBEDDED = "embedded"
    RANKED = "ranked"
    OPTIMIZED = "optimized"
    JUDGED = "judged"
    FINALIZED = "finalized"


PHASE_ORDER = [
    Phase.INIT,
    Phase.EMBEDDED,
    Phase.RANKED,
    Phase.OPTIMIZED,
    Phase.JUDGED,
    Phase.FINALIZED,
]

TOOL_REQUIRED_PHASE: dict[str, Phase] = {
    "embed_taste_query": Phase.INIT,
    "rank_recipes_by_fit": Phase.EMBEDDED,
    "optimize_top_candidates": Phase.RANKED,
    "judge_final_recipe": Phase.OPTIMIZED,
    "finalize_recommendation": Phase.JUDGED,
}

TOOL_NEXT_PHASE: dict[str, Phase] = {
    "embed_taste_query": Phase.EMBEDDED,
    "rank_recipes_by_fit": Phase.RANKED,
    "optimize_top_candidates": Phase.OPTIMIZED,
    "judge_final_recipe": Phase.JUDGED,
    "finalize_recommendation": Phase.FINALIZED,
}


@dataclass
class AgentSession:
    query: UserQuery
    run_id: str | None = None
    log_to_db: bool = True
    corpus: dict[str, Any] | None = None
    kcal_target: float = 0.0
    query_emb: np.ndarray | None = None
    ranked: list[RankedRecipe] = field(default_factory=list)
    optimized: list[dict[str, Any]] = field(default_factory=list)
    pick_candidates: list[Any] = field(default_factory=list)
    judge_result: JudgeResult | None = None
    final_payload: dict[str, Any] | None = None
    phase: Phase = Phase.INIT
    seq: int = 0
    events: list[PipelineEvent] = field(default_factory=list)
    on_event: Callable[[PipelineEvent], None] | None = None

    def require_phase(self, tool_name: str) -> str | None:
        """Return error message if tool called out of order."""
        required = TOOL_REQUIRED_PHASE[tool_name]
        if self.phase != required:
            return (
                f"Tool {tool_name} requires phase {required.value} "
                f"but session is at {self.phase.value}. "
                f"Call the previous pipeline tool first."
            )
        return None

    def advance(self, tool_name: str) -> None:
        self.phase = TOOL_NEXT_PHASE[tool_name]

    def emit(
        self,
        stage: str,
        payload: dict[str, Any],
        *,
        log: bool = True,
    ) -> PipelineEvent:
        from mvp_log import log_stage

        self.seq += 1
        ev = PipelineEvent(stage=stage, seq=self.seq, payload=payload)
        self.events.append(ev)
        if self.on_event:
            self.on_event(ev)
        if log and self.log_to_db and self.run_id:
            try:
                log_stage(self.run_id, stage, self.seq, payload)
            except Exception:
                pass
        return ev


_active_session: AgentSession | None = None


def set_active_session(session: AgentSession | None) -> None:
    global _active_session
    _active_session = session


def get_active_session() -> AgentSession:
    if _active_session is None:
        raise RuntimeError("No active AgentSession; call set_active_session first")
    return _active_session
