"""MVP Strands agent — Bedrock orchestrator for recipe recommendation."""

__all__ = ["build_agent", "run_agent_pipeline"]


def __getattr__(name: str):
    if name in __all__:
        from mvp_agent.runner import build_agent, run_agent_pipeline

        return build_agent if name == "build_agent" else run_agent_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
