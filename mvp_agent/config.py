"""Environment configuration for the MVP Strands agent."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class AgentConfig:
    enabled: bool
    aws_region: str
    aws_profile: str | None
    bedrock_model_id: str
    max_iterations: int

    @classmethod
    def from_env(cls) -> AgentConfig:
        return cls(
            enabled=_env_bool("MVP_AGENT_ENABLED", True),
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            aws_profile=os.environ.get("AWS_PROFILE") or None,
            bedrock_model_id=os.environ.get(
                "BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0"
            ),
            max_iterations=int(os.environ.get("MVP_AGENT_MAX_ITERATIONS", "12")),
        )


def bedrock_status() -> dict[str, str | bool]:
    """Lightweight Bedrock readiness check for /health (non-fatal)."""
    cfg = AgentConfig.from_env()
    out: dict[str, str | bool] = {
        "region": cfg.aws_region,
        "model_id": cfg.bedrock_model_id,
        "reachable": False,
        "detail": "",
    }
    try:
        import boto3

        session_kwargs: dict[str, str] = {"region_name": cfg.aws_region}
        if cfg.aws_profile:
            session = boto3.Session(profile_name=cfg.aws_profile)
        else:
            session = boto3.Session()
        client = session.client("bedrock", **session_kwargs)
        client.list_foundation_models(byOutputModality="TEXT")
        out["reachable"] = True
        out["detail"] = "ok"
    except Exception as exc:
        out["detail"] = f"{type(exc).__name__}: {exc}"
    return out
