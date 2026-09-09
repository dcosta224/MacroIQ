"""Fast-demo latency profile for MacroIQ production."""

from __future__ import annotations

import os

from recipe_opt_agent.config import AgentConfig, fast_demo_from_env


def test_fast_demo_env_flag(monkeypatch):
    monkeypatch.delenv("MACROIQ_FAST_DEMO", raising=False)
    assert fast_demo_from_env() is False
    monkeypatch.setenv("MACROIQ_FAST_DEMO", "1")
    assert fast_demo_from_env() is True


def test_apply_fast_demo_cuts_llm_and_lp():
    cfg = AgentConfig(max_iterations=3, n_ideation_candidates=8).apply_fast_demo()
    assert cfg.enable_shadow_gpt_candidate is False
    assert cfg.enable_llm_judge is False
    assert cfg.ideation_model == "gpt-4o-mini"
    assert cfg.model_escalate == "gpt-4o-mini"
    assert cfg.n_ideation_candidates == 4
    assert cfg.max_iterations == 2
    assert cfg.bundle_lp_cap == 5


def test_for_request_respects_env(monkeypatch):
    monkeypatch.setenv("MACROIQ_FAST_DEMO", "1")
    cfg = AgentConfig.for_request(max_iterations=3)
    assert cfg.max_iterations == 2
    assert cfg.enable_shadow_gpt_candidate is False
    monkeypatch.setenv("MACROIQ_FAST_DEMO", "0")
    # "0" is not in the truthy set
    assert fast_demo_from_env() is False
