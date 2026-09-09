"""Unit tests for launch readiness + auth stubs (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import Response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mvp_web.auth import CognitoAuthStubMiddleware
from mvp_web.launch_ready import launch_status


@pytest.fixture(autouse=True)
def _clear_launch_env(monkeypatch):
    for key in (
        "AUTH_ENABLED",
        "PG_DATABASE",
        "PG_PASSWORD",
        "PG_SSL_MODE",
        "PG_POOL_USER",
        "PG_POOL_HOST",
        "PG_POOL_TRANSACTION_PORT",
        "PG_POOL_SESSION_PORT",
        "COGNITO_REGION",
        "COGNITO_USER_POOL_ID",
        "COGNITO_APP_CLIENT_ID",
        "HF_TOKEN",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_launch_status_reports_missing_supabase():
    status = launch_status()
    assert status["supabase_ready"] is False
    assert "PG_PASSWORD" in status["supabase_missing"]
    assert status["site_ready_without_auth"] is False
    assert status["auth"]["enabled"] is False


def test_launch_status_ready_when_pg_set(monkeypatch):
    for key, value in {
        "PG_DATABASE": "postgres",
        "PG_PASSWORD": "secret",
        "PG_SSL_MODE": "require",
        "PG_POOL_USER": "postgres.x",
        "PG_POOL_HOST": "example.pooler.supabase.com",
        "PG_POOL_TRANSACTION_PORT": "6543",
        "PG_POOL_SESSION_PORT": "5432",
    }.items():
        monkeypatch.setenv(key, value)

    status = launch_status()
    assert status["supabase_ready"] is True
    assert status["site_ready_without_auth"] is True
    assert status["blockers"] == []


def test_auth_middleware_noop_when_disabled():
    app = FastAPI()
    app.add_middleware(CognitoAuthStubMiddleware)

    @app.get("/api/recommend")
    def recommend():
        return Response("ok", media_type="text/plain")

    client = TestClient(app)
    assert client.get("/api/recommend").status_code == 200


def test_auth_middleware_blocks_when_enabled_without_cognito(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "1")
    app = FastAPI()
    app.add_middleware(CognitoAuthStubMiddleware)

    @app.get("/api/recommend")
    def recommend():
        return Response("ok", media_type="text/plain")

    client = TestClient(app)
    response = client.get("/api/recommend")
    assert response.status_code == 503
    assert response.json()["error"] == "auth_not_configured"


def test_health_public_when_auth_enabled(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "1")
    app = FastAPI()
    app.add_middleware(CognitoAuthStubMiddleware)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    client = TestClient(app)
    assert client.get("/health").status_code == 200
