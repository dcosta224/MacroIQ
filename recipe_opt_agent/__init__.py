"""Recipe optimization agent (LangGraph): hull + loss + retrieval loop."""

from __future__ import annotations

__all__ = ["run_recipe_opt_agent"]


def __getattr__(name: str):
    if name == "run_recipe_opt_agent":
        from recipe_opt_agent.runner import run_recipe_opt_agent

        return run_recipe_opt_agent
    raise AttributeError(name)
