"""Unit tests for MVP Strands agent tools and phase gates."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mvp_agent.context import AgentSession, Phase, set_active_session
from mvp_agent.tools import (
    embed_taste_query,
    finalize_recommendation,
    judge_final_recipe,
    optimize_top_candidates,
    rank_recipes_by_fit,
)
from mvp_pipeline import UserQuery
from mvp_recipe_judge import JudgeResult
from mvp_recipe_ranker import RankedRecipe


def _query() -> UserQuery:
    return UserQuery(
        taste_text="chicken stir fry",
        kcal_min=400,
        kcal_max=600,
        fat_frac_min=0.2,
        fat_frac_max=0.35,
        carb_frac_min=0.35,
        carb_frac_max=0.55,
        protein_frac_min=0.15,
        protein_frac_max=0.35,
        top_k=2,
    )


def _ranked() -> list[RankedRecipe]:
    return [
        RankedRecipe(
            recipe_id=1,
            recipe_name="Chicken A",
            semantic_sim=0.9,
            semantic_dist=0.1,
            semantic_score=90.0,
            nutrient_fit=0.0,
            nutrient_score=100.0,
            combined_score=95.0,
            rank=1,
            pfc_in_range=True,
            kcal_target=500.0,
            recipe_kcal=480.0,
        ),
        RankedRecipe(
            recipe_id=2,
            recipe_name="Chicken B",
            semantic_sim=0.8,
            semantic_dist=0.2,
            semantic_score=80.0,
            nutrient_fit=0.1,
            nutrient_score=90.0,
            combined_score=85.0,
            rank=2,
            pfc_in_range=True,
            kcal_target=500.0,
            recipe_kcal=470.0,
        ),
    ]


def _optimized() -> list[dict]:
    return [
        {
            "recipe_id": 1,
            "recipe_name": "Chicken A",
            "portion_score": 0.1,
            "avg_pct_change": 5.0,
            "macro_feasible": True,
            "used_fallback": False,
            "already_feasible": False,
            "macros_before": {"energy_kcal": 400},
            "macros_after": {"energy_kcal": 500},
            "ingredients": [{"ingredient": "chicken"}],
        },
        {
            "recipe_id": 2,
            "recipe_name": "Chicken B",
            "portion_score": 0.2,
            "avg_pct_change": 8.0,
            "macro_feasible": True,
            "used_fallback": False,
            "already_feasible": False,
            "macros_before": {"energy_kcal": 390},
            "macros_after": {"energy_kcal": 500},
            "ingredients": [{"ingredient": "chicken breast"}],
        },
    ]


@pytest.fixture
def session():
    s = AgentSession(query=_query(), log_to_db=False)
    s.corpus = {
        "recipe_ids": [1, 2],
        "recipe_names": {1: "Chicken A", 2: "Chicken B"},
        "embeddings": np.zeros((2, 384), dtype=np.float32),
        "nutrient_rows": {},
        "features": {
            1: {
                "title_clean": "Chicken A",
                "semantic_text": "stir fry chicken",
                "nlg_ingredients": "chicken, soy sauce",
            },
            2: {
                "title_clean": "Chicken B",
                "semantic_text": "chicken bowl",
                "nlg_ingredients": "chicken breast, rice",
            },
        },
    }
    set_active_session(s)
    yield s
    set_active_session(None)


def test_phase_gate_rejects_out_of_order(session):
    result = rank_recipes_by_fit()
    assert result["ok"] is False
    assert "embed_taste_query" in result["message"] or "embedded" in result["message"]


@patch("mvp_agent.tools.encode_query")
def test_embed_advances_phase(mock_encode, session):
    mock_encode.return_value = np.zeros(384, dtype=np.float32)
    result = embed_taste_query("chicken stir fry")
    assert result["ok"] is True
    assert session.phase == Phase.EMBEDDED
    assert session.query_emb is not None


@patch("mvp_agent.tools.rank_recipes")
@patch("mvp_agent.tools.encode_query")
def test_full_tool_chain(mock_encode, mock_rank, session):
    mock_encode.return_value = np.ones(384, dtype=np.float32)
    mock_rank.return_value = _ranked()

    with patch("mvp_agent.tools.optimize_recipe") as mock_opt:
        mock_opt.side_effect = lambda rid, q, corpus=None: next(
            o for o in _optimized() if o["recipe_id"] == rid
        )
        with patch("mvp_agent.tools.select_final_candidate") as mock_judge:
            mock_judge.return_value = JudgeResult(
                chosen_recipe_id=1,
                rationale="Best chicken match.",
                portion_summary="Minor portion adjustments.",
                runner_up_notes="",
            )

            assert embed_taste_query(session.query.taste_text)["ok"]
            assert rank_recipes_by_fit()["ok"]
            assert optimize_top_candidates(session.query.top_k)["ok"]
            assert judge_final_recipe()["ok"]
            assert finalize_recommendation()["ok"]

    assert session.phase == Phase.FINALIZED
    assert session.final_payload is not None
    assert session.final_payload["chosen_recipe_id"] == 1
