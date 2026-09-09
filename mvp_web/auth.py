"""Auth hooks for a future Cognito gate (inactive by default).

UI is unchanged. With AUTH_ENABLED=0 (default) this middleware is a no-op.
When AUTH_ENABLED=1, protected API routes require a Bearer token and configured
Cognito env vars; full JWT verification against Cognito JWKS is intentionally
not implemented yet so the hook can land without a live user pool.
"""

from __future__ import annotations

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .launch_ready import COGNITO_VARS, auth_enabled, missing

# Paths that stay public even when auth is enabled (health + static UI assets).
PUBLIC_PREFIXES = ("/health", "/static", "/docs", "/openapi.json", "/redoc")
PUBLIC_EXACT = {"/", "/favicon.ico"}


def _is_public(path: str) -> bool:
    if path in PUBLIC_EXACT:
        return True
    return any(path == p or path.startswith(p + "/") for p in PUBLIC_PREFIXES)


class CognitoAuthStubMiddleware(BaseHTTPMiddleware):
    """No-op until AUTH_ENABLED=1; then enforces stub Cognito requirements."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if not auth_enabled() or _is_public(request.url.path):
            return await call_next(request)

        incomplete = missing(COGNITO_VARS)
        if incomplete:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "auth_not_configured",
                    "message": "AUTH_ENABLED=1 but Cognito env is incomplete",
                    "missing": incomplete,
                },
            )

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or len(auth_header.split(" ", 1)[1].strip()) < 1:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "missing_bearer_token",
                    "message": "Authorization: Bearer <token> required when AUTH_ENABLED=1",
                },
            )

        # Token present + Cognito env set, but JWKS verification not wired yet.
        return JSONResponse(
            status_code=503,
            content={
                "error": "cognito_jwt_verification_not_wired",
                "message": (
                    "Cognito hooks are present; implement JWKS JWT verify before "
                    "serving authenticated traffic."
                ),
                "pool_id": os.environ.get("COGNITO_USER_POOL_ID", ""),
                "client_id": os.environ.get("COGNITO_APP_CLIENT_ID", ""),
                "region": os.environ.get("COGNITO_REGION", ""),
            },
        )
