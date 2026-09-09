"""Launch-readiness checks for the MVP web app (no live hosting required).

Reports which env vars / features are ready before flipping on a public runtime.
Does not contact AWS or create paid resources.
"""

from __future__ import annotations

import os
from typing import Any


# Required to talk to Supabase via the pooler (scripts/db.py).
REQUIRED_PG = (
    "PG_DATABASE",
    "PG_PASSWORD",
    "PG_SSL_MODE",
    "PG_POOL_USER",
    "PG_POOL_HOST",
    "PG_POOL_TRANSACTION_PORT",
    "PG_POOL_SESSION_PORT",
)

# Nice to have for full demo behavior; app can still boot without some of these.
OPTIONAL_RUNTIME = (
    "HF_TOKEN",
    "OPENAI_API_KEY",
    "AWS_REGION",
    "BEDROCK_MODEL_ID",
)

# Cognito placeholders — unused until AUTH_ENABLED=1 and JWT verify is wired.
COGNITO_VARS = (
    "COGNITO_REGION",
    "COGNITO_USER_POOL_ID",
    "COGNITO_APP_CLIENT_ID",
)


def _present(name: str) -> bool:
    value = os.environ.get(name)
    return bool(value and value.strip() and not value.strip().startswith("<"))


def auth_enabled() -> bool:
    return os.environ.get("AUTH_ENABLED", "0").strip() == "1"


def missing(names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not _present(name)]


def launch_status() -> dict[str, Any]:
    """Structured readiness for /health and pre-launch checklists."""
    pg_missing = missing(REQUIRED_PG)
    optional_missing = missing(OPTIONAL_RUNTIME)
    cognito_missing = missing(COGNITO_VARS)
    auth_on = auth_enabled()
    supabase_ready = not pg_missing
    cognito_configured = not cognito_missing

    if not auth_on:
        auth_state = "disabled"
    elif not cognito_configured:
        auth_state = "enabled_but_cognito_incomplete"
    else:
        auth_state = "enabled_pending_jwt_verify"

    blockers: list[str] = []
    if pg_missing:
        blockers.append(f"missing_supabase_env:{','.join(pg_missing)}")
    if auth_on and cognito_missing:
        blockers.append(f"missing_cognito_env:{','.join(cognito_missing)}")
    if auth_on and cognito_configured:
        blockers.append("cognito_jwt_verification_not_wired")

    # Safe to attach a public runtime when DB is configured and auth is still off
    # (or auth work is finished — JWT verify not yet, so auth_on blocks "ready").
    site_ready = supabase_ready and not auth_on

    return {
        "supabase_ready": supabase_ready,
        "supabase_missing": pg_missing,
        "optional_missing": optional_missing,
        "auth": {
            "enabled": auth_on,
            "state": auth_state,
            "cognito_configured": cognito_configured,
            "cognito_missing": cognito_missing,
        },
        "site_ready_without_auth": site_ready,
        "blockers": blockers,
        "go_live_when": [
            "Inject PG_* (and optional API keys) into the runtime env",
            "Point App Runner/ECS/EC2 at ECR image macroiq:deployment (or a main tag)",
            "Keep AUTH_ENABLED=0 until Cognito pool exists and JWT verify is implemented",
            "Then set COGNITO_* + AUTH_ENABLED=1",
        ],
    }
