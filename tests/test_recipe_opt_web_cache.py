"""Tests that the web server prefers neighborhood cache and falls back live on miss."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


def _collect_sse_events(response) -> list[dict]:
    body = getattr(response, "text", None)
    if body is None and hasattr(response, "body"):
        body = response.body.decode("utf-8")
    if body is None:
        body = response.content.decode("utf-8")
    events: list[dict] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        data_line = next((line for line in block.splitlines() if line.startswith("data: ")), "")
        if data_line:
            events.append(json.loads(data_line.removeprefix("data: ")))
    return events


def test_run_endpoint_reports_cache_hit(monkeypatch):
    from fastapi.testclient import TestClient

    from recipe_opt_web import server

    fake_problem = {
        "title": "Carbonara",
        "taste_text": "Carbonara",
        "neighborhood_from_cache": True,
        "neighborhood_recipes": [{"recipe_nlg_id": "r1"}],
        "chosen_recipe": {
            "recipe_nlg_id": "r1",
            "ingredients": [{"label": "Pasta", "grams": 100.0}],
            "selection": {
                "method": "l1_pfc",
                "neighborhood_from_cache": True,
            },
        },
    }

    monkeypatch.setattr(server, "load_canonical_problem", lambda *args, **kwargs: fake_problem)
    monkeypatch.setattr(
        server,
        "run_recipe_opt_agent",
        lambda **kwargs: {"status": "ok", "chosen": {"entry": {}}},
    )

    client = TestClient(server.app)
    res = client.post(
        "/api/run",
        json={
            "mode": "neighborhood",
            "canonical_id": 42,
            "max_iterations": 1,
        },
    )
    assert res.status_code == 200
    events = _collect_sse_events(res)
    selected = next(e for e in events if e.get("phase") == "selected_start")
    assert selected["neighborhood_from_cache"] is True
    assert "Jaccard cache hit" in selected["message"]


def test_run_endpoint_allows_live_rebuild_on_cache_miss(monkeypatch):
    from fastapi.testclient import TestClient

    from recipe_opt_web import server

    fake_problem = {
        "title": "Carbonara",
        "taste_text": "Carbonara",
        "neighborhood_from_cache": False,
        "neighborhood_recipes": [{"recipe_nlg_id": "r1"}],
        "chosen_recipe": {
            "recipe_nlg_id": "r1",
            "ingredients": [{"label": "Pasta", "grams": 100.0}],
            "selection": {
                "method": "l1_pfc",
                "neighborhood_from_cache": False,
            },
        },
    }
    seen: dict = {}

    def _load(*args, **kwargs):
        seen["require_cache"] = kwargs.get("require_cache")
        return fake_problem

    monkeypatch.setattr(server, "load_canonical_problem", _load)
    monkeypatch.setattr(
        server,
        "run_recipe_opt_agent",
        lambda **kwargs: {"status": "ok", "chosen": {"entry": {}}},
    )

    client = TestClient(server.app)
    res = client.post(
        "/api/run",
        json={
            "mode": "neighborhood",
            "canonical_id": 42,
            "max_iterations": 1,
        },
    )
    assert res.status_code == 200
    events = _collect_sse_events(res)
    assert not any(e.get("type") == "error" for e in events)
    selected = next(e for e in events if e.get("phase") == "selected_start")
    assert selected["neighborhood_from_cache"] is False
    assert "live neighborhood rebuild" in selected["message"]
    assert seen.get("require_cache") is False
    assert any(e.get("type") == "result" for e in events)


def test_flow_api_includes_tools_and_edge_lists():
    """/api/flow docs must carry tools + incoming/outgoing for the node popover."""
    from fastapi.testclient import TestClient

    from recipe_opt_web import server

    client = TestClient(server.app)
    for mode in ("neighborhood", "creative"):
        res = client.get(f"/api/flow?mode={mode}")
        assert res.status_code == 200
        data = res.json()
        assert data["nodes"]
        assert data["edges"]
        docs = data["docs"]
        for name in data["nodes"]:
            if name not in docs:
                continue
            doc = docs[name]
            assert "tools" in doc, f"{name} missing tools"
            assert "incoming" in doc and "outgoing" in doc, f"{name} missing edge lists"
        # propose must document the slot/bundle pipeline tools
        propose_tools = {t["name"] for t in docs["propose"]["tools"]}
        assert {"plan_slots", "retrieve_slots", "score_bundles"} <= propose_tools
        # decide docs list apply_bundle-capable tool; apply lists apply_bundle
        apply_tools = {t["name"] for t in docs["apply"]["tools"]}
        assert "apply_bundle" in apply_tools
        # compute kinds color-code LLM vs deterministic nodes
        assert "compute_kinds" in data
        assert set(data["compute_kinds"]) >= {"deterministic", "llm_content", "llm_controller"}
        assert docs["propose"]["compute"] == "deterministic"
        assert docs["decide"]["compute"] == "llm_controller"
        assert docs["diagnose"]["compute"] == "deterministic"
        decide_tool = next(t for t in docs["decide"]["tools"] if t["name"] == "decide_action_llm")
        assert decide_tool.get("detail")
        assert len(decide_tool.get("prompts") or []) >= 2
        roles = {p["role"] for p in decide_tool["prompts"]}
        assert roles >= {"system", "user"}
        assert all(p.get("content") for p in decide_tool["prompts"])
        if mode == "creative":
            assert docs["llm_draft"]["compute"] == "llm_content"
            assert docs["judge_final"]["compute"] == "llm_content"
            assert docs["ground_recipe"]["compute"] == "deterministic"
            draft_tool = next(t for t in docs["llm_draft"]["tools"] if t["name"] == "llm_draft_recipe")
            assert draft_tool.get("prompts")
        # edge lists match the returned edges
        edges = {(e["from"], e["to"]) for e in data["edges"]}
        for name in data["nodes"]:
            if name not in docs:
                continue
            for src in docs[name]["incoming"]:
                assert (src, name) in edges
            for dst in docs[name]["outgoing"]:
                assert (name, dst) in edges
            assert docs[name]["compute"] in data["compute_kinds"]
