"""Agent state for LangGraph recipe optimization loop."""

from __future__ import annotations

from typing import Any, TypedDict


class DishStructure(TypedDict, total=False):
    """LLM-declared structure for role-aware neighborhood expansion."""

    anchor_ingredients: list[str]  # mass/identity base (e.g. rice, pasta)
    stretch_ingredient: str  # ingredient being added / underrepresented
    stretch_role: str  # accent | co_main


class ExpandDirective(TypedDict, total=False):
    delta_k: int
    relax_cuisine: bool
    foodon_weight_shift: str  # coarser | finer
    notes: str
    neighborhood_search_queries: list[str]
    dish_structure: DishStructure


class AgentState(TypedDict, total=False):
    # inputs
    canonical_id: int
    taste_text: str
    title: str
    config: dict[str, Any]
    agent_mode: str
    user_request: str
    requirement_tags: list[dict[str, Any]]
    # frozen starting-recipe snapshot for final-arbiter intent diffs
    original_ingredients: list[dict[str, Any]]
    # final LLM arbitration result (winner, verdicts, comparison table)
    final_judgment: dict[str, Any]
    # GPT-4o holistic evaluation of the selected recipe
    final_evaluation: dict[str, Any]
    recipe_draft: dict[str, Any]
    grounding_report: dict[str, Any]
    grounded_r0: list[dict[str, Any]]

    # retrieve
    neighbor_ids: list[str]
    neighbor_k: int
    cuisine: str | None
    expand_directive: ExpandDirective | None

    # basis / problem
    identity_roles: list[str]
    identity_critical: dict[str, bool]
    basis_nodes: list[str]
    ingredient_labels: list[str]
    # serialized numpy-ish payloads stored as lists for graph friendliness in tests
    problem: dict[str, Any]
    foodon_basis_report: dict[str, Any]

    # diagnose
    hull: dict[str, Any]
    opt: dict[str, Any]
    loss_field_summary: dict[str, Any]
    diagnosis: dict[str, Any]
    fidelity_band: str
    chosen_recipe: dict[str, Any]
    neighborhood_recipes: list[dict[str, Any]]
    tools_used: list[dict[str, Any]]
    llm_trace: dict[str, Any]
    decision_context: dict[str, Any]
    candidates_dropped: list[dict[str, Any]]
    last_applied_candidate: dict[str, Any]

    # propose: slot/bundle pipeline
    planned_slots: list[dict[str, Any]]
    bundles: list[dict[str, Any]]

    # decide / act
    candidates: list[dict[str, Any]]
    decision: dict[str, Any]
    history: list[dict[str, Any]]
    candidate_pool: list[dict[str, Any]]
    iteration: int
    decision_outcomes: list[dict[str, Any]]
    pending_outcome: dict[str, Any]
    recent_edit_fingerprints: list[str]
    run_telemetry: dict[str, Any]
    live_scores: dict[str, Any]
    score_history: list[dict[str, Any]]

    interesting_candidates: list[dict[str, Any]]

    # creative finalize
    finalist_pool: list[dict[str, Any]]
    scored_finalists: list[dict[str, Any]]
    judge_result: dict[str, Any]

    # output
    final: dict[str, Any]
    status: str
    error: str | None
    # async GPT-5.5 shadow draft (joined before finalists / arbiter)
    shadow_job_id: str | None
    _shadow_collect_meta: dict[str, Any]
