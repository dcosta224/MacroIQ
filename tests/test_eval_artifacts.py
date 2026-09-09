"""Tests for eval artifact capture (offline)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)


def test_eval_suite_writes_full_artifacts(tmp_path):
    from recipe_opt_agent.config import AgentConfig
    from recipe_opt_agent.eval_artifacts import EvalSuiteRecorder, run_agent_with_artifacts
    import numpy as np

    suite = EvalSuiteRecorder(name="unit", root=tmp_path)
    suite.start()
    problem = {
        "x0": [200.0, 200.0],
        "M": [[0.25, 0.0], [0.0, 0.0], [0.0, 0.25], [0.0, 0.0]],
        "ingredient_basis": ["pasta", "cheese"],
        "basis_samples": {"pasta": [0.45, 0.5, 0.55], "cheese": [0.45, 0.5, 0.55]},
        "ratio_samples": [],
        "marginal_nodes": ["pasta", "cheese"],
        "kcal_target": 400.0,
        "total_mass": 400.0,
        "modification_candidates": [],
        "chosen_recipe": {"title": "Cheese Pasta", "ingredients": [{"label": "pasta", "grams": 200}]},
    }
    cfg = AgentConfig(
        max_iterations=1,
        F_accept=5.0,
        F_max=10.0,
        protein_min=0.0,
        protein_max=1.0,
        carb_min=0.0,
        carb_max=1.0,
        fat_min=0.0,
        fat_max=1.0,
    )
    result, rec = run_agent_with_artifacts(
        problem=problem,
        case_name="unit_nb",
        agent_mode="neighborhood",
        suite=suite,
        taste_text="cheese pasta",
        title="Cheese Pasta",
        config=cfg,
    )
    suite.finish()
    assert rec.run_dir.exists()
    for name in (
        "manifest.json",
        "final.json",
        "steps.json",
        "tools.json",
        "llm_calls.json",
        "retrieval.json",
        "metrics.json",
        "run_metadata.json",
        "flow.json",
        "events.jsonl",
    ):
        assert (rec.run_dir / name).exists(), name
    steps = json.loads((rec.run_dir / "steps.json").read_text())
    assert len(steps) >= 2
    metrics = json.loads((rec.run_dir / "metrics.json").read_text())
    assert "final_status_ok" in metrics
    assert "actionable" in metrics and "observability" in metrics
    run_meta = json.loads((rec.run_dir / "run_metadata.json").read_text())
    assert run_meta.get("agent_mode") == "neighborhood"
    assert "models" in run_meta and "graph" in run_meta
    assert "F_accept" in run_meta and "macro_targets" in run_meta
    assert (suite.suite_dir / "suite_summary.json").exists()
    assert result.get("status") is not None
