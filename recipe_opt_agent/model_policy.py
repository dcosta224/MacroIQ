"""Model selection by node complexity for the recipe opt agent."""

from __future__ import annotations

from typing import Any

from recipe_opt_agent.config import AgentConfig

# Markers that suggest lexical tag extraction may be incomplete / ambiguous.
# Avoid bare substrings that appear inside ordinary words (e.g. "ish" ⊂ "dish").
_AMBIGUOUS_TAG_MARKERS = (
    "maybe",
    "kinda",
    "sort of",
    "-ish",
    " prefer ",
    "if possible",
    "optional",
    "mostly",
    "flexitarian",
    "pescatarian",
    "almost vegetarian",
    "no red meat",
    "plant forward",
)

# Diet-ish cues that lexical may have missed (still require explicit language).
_DIET_CUE_MARKERS = (
    "allerg",
    "intoleran",
    "dairy-free",
    "dairy free",
    "gluten-free",
    "gluten free",
    "plant-based",
    "plant based",
    "meatless",
    "without meat",
    "no meat",
    "vegetarian",
    "vegan",
    "pescatarian",
    "halal",
    "kosher",
)


def tags_need_llm(request: str, lexical_tags: list[Any]) -> bool:
    """Use LLM tags only for ambiguous / explicit diet language — not every request.

    Previously any ≥12-char request with empty lexical tags called the tags LLM,
    which invented false dietary restrictions (e.g. vegetarian on bobotie).
    """
    text = f" {(request or '').lower()} "
    if any(m in text for m in _AMBIGUOUS_TAG_MARKERS):
        return True
    # also allow unpadded prefer at start
    raw = (request or "").lower()
    if raw.startswith("prefer ") or " prefer " in f" {raw} ":
        return True
    has_dietary = any(
        getattr(t, "kind", None) == "dietary_restriction"
        or (isinstance(t, dict) and t.get("kind") == "dietary_restriction")
        for t in (lexical_tags or [])
    )
    if any(m in raw for m in _DIET_CUE_MARKERS) and not has_dietary:
        return True
    return False


def select_tags_model(cfg: AgentConfig) -> str:
    return cfg.tags_model


def select_draft_model(cfg: AgentConfig) -> str:
    return cfg.creative_model


def select_shadow_draft_model(cfg: AgentConfig) -> str:
    return getattr(cfg, "shadow_draft_model", None) or "gpt-5.5"


def select_judge_model(cfg: AgentConfig) -> str:
    return cfg.judge_model


def _improving_bundles(context: dict[str, Any], *, eps: float) -> list[dict[str, Any]]:
    out = []
    for b in context.get("bundles") or []:
        if not b.get("lp_evaluated"):
            continue
        d = b.get("delta_L_star")
        if d is None:
            continue
        if float(d) < -eps:
            out.append(b)
    out.sort(key=lambda x: float(x["delta_L_star"]))
    return out


def decide_should_escalate(
    context: dict[str, Any],
    *,
    cfg: AgentConfig | None = None,
) -> bool:
    """True → use model_escalate; False → routine mini."""
    cfg = cfg or AgentConfig()
    iteration = int(context.get("iteration") or 0)
    if iteration >= 1:
        return True
    if context.get("revisit_reflection"):
        return True
    if context.get("identity_tension"):
        return True

    improving = _improving_bundles(context, eps=cfg.auto_apply_delta_eps)
    if not improving:
        band = context.get("fidelity_band")
        if band == "accept":
            return False
        return True
    if len(improving) >= 2:
        best = float(improving[0]["delta_L_star"])
        second = float(improving[1]["delta_L_star"])
        if abs(best - second) <= cfg.auto_apply_margin:
            return True

    tags = context.get("requirement_tags") or []
    dietary = [
        t
        for t in tags
        if (t.get("kind") if isinstance(t, dict) else getattr(t, "kind", None)) == "dietary_restriction"
    ]
    if dietary and improving:
        for b in improving[:3]:
            for edit in b.get("edits") or []:
                if edit.get("action") in {"swap", "remove", "add"}:
                    return True
    return False


def select_decide_model(context: dict[str, Any], *, cfg: AgentConfig | None = None) -> str:
    cfg = cfg or AgentConfig()
    if decide_should_escalate(context, cfg=cfg):
        return cfg.model_escalate
    return cfg.model


def estimate_context_tokens(context: dict[str, Any]) -> int:
    """Rough token estimate (~4 chars/token) for complexity smokes."""
    import json

    raw = json.dumps(context, default=str)
    return max(1, len(raw) // 4)
