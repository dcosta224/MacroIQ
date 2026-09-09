"""Tests for candidate_scoring normalization and Pareto filter."""

from __future__ import annotations

from recipe_opt_agent.candidate_scoring import (
    CandidateMetrics,
    pareto_filter,
    score_finalist_pool,
    weighted_composite,
)


def test_pareto_drops_dominated():
    metrics = [
        CandidateMetrics("a", 0.1, 0.2, 0.3, 0.1),
        CandidateMetrics("b", 0.5, 0.6, 0.7, 0.8),  # dominated by a
    ]
    keep = pareto_filter(metrics)
    assert keep == [0]


def test_weighted_composite_prefers_nutrient():
    good_a = {"nutrient_dist": 0.9, "ratio_badness": 0.5, "intent_gap": 0.5, "churn": 0.5}
    good_b = {"nutrient_dist": 0.5, "ratio_badness": 0.9, "intent_gap": 0.5, "churn": 0.5}
    assert weighted_composite(good_a) > weighted_composite(good_b)


def test_score_finalist_pool_orders_by_composite():
    entries = [
        {"candidate_id": "x", "metrics": {"nutrient_dist": 0.0, "ratio_badness": 1.0, "intent_gap": 0.5, "churn": 0.5}},
        {"candidate_id": "y", "metrics": {"nutrient_dist": 0.0, "ratio_badness": 0.1, "intent_gap": 0.5, "churn": 0.5}},
    ]
    scored = score_finalist_pool(entries)
    assert scored[0].candidate_id == "y"
    assert all(0.0 <= s.composite <= 1.0 for s in scored)
