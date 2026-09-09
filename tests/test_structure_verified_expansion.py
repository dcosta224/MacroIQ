"""Tests for role-aware / structure-verified neighborhood expansion."""

from __future__ import annotations

from recipe_opt_agent.neighborhood_query_expand import (
    classify_structure_fit,
    compute_recipe_structure_shares,
    expand_neighborhood_by_queries,
    fallback_dish_structure,
    normalize_dish_structure,
    verify_shell_structure,
)
from recipe_opt_agent.ood_foodon import ensure_ingredient_nodes_in_loss


def _lines(items: list[tuple[str, float]]) -> list[dict]:
    return [
        {"fdc_description": desc, "gram_weight": grams, "foodon_id": None}
        for desc, grams in items
    ]


def test_normalize_and_fallback_dish_structure():
    assert normalize_dish_structure(None) is None
    ds = normalize_dish_structure(
        {
            "anchor_ingredients": ["rice"],
            "stretch_ingredient": "chicken breast",
            "stretch_role": "accent",
        }
    )
    assert ds["stretch_role"] == "accent"
    fb = fallback_dish_structure(
        stretch_ingredient="chicken",
        identity_roles=["pasta", "egg"],
        current_labels=["Spaghetti, dry", "Egg, whole"],
    )
    assert "pasta" in [a.lower() for a in fb["anchor_ingredients"]]
    assert fb["stretch_ingredient"] == "chicken"


def test_accent_rejects_stretch_primary_without_anchor():
    shares = compute_recipe_structure_shares(
        _lines([("Chicken breast, raw", 200.0), ("Broccoli", 50.0)]),
        anchor_terms=["rice"],
        stretch_terms=["chicken"],
    )
    assert shares["stretch_share"] > 0.5
    assert shares["anchor_share"] == 0.0
    fit = classify_structure_fit(shares, stretch_role="accent")
    assert fit["verdict"] == "reject"
    assert fit["reason"] == "accent_stretch_without_anchor"


def test_accent_passes_rice_primary_with_chicken():
    shares = compute_recipe_structure_shares(
        _lines(
            [
                ("White rice, cooked", 250.0),
                ("Chicken breast, raw", 80.0),
                ("Soy sauce", 10.0),
            ]
        ),
        anchor_terms=["rice"],
        stretch_terms=["chicken"],
    )
    assert shares["anchor_share"] > shares["stretch_share"]
    fit = classify_structure_fit(shares, stretch_role="accent")
    assert fit["verdict"] == "pass"


def test_accent_soft_when_anchor_present_but_not_dominant():
    shares = compute_recipe_structure_shares(
        _lines(
            [
                ("Chicken breast, raw", 220.0),
                ("White rice, cooked", 40.0),  # side
            ]
        ),
        anchor_terms=["rice"],
        stretch_terms=["chicken"],
    )
    assert shares["stretch_share"] > shares["anchor_share"] > 0
    fit = classify_structure_fit(shares, stretch_role="accent")
    assert fit["verdict"] == "soft"
    assert 0 < fit["weight_scale"] < 1


def test_verify_shell_structure_filters_and_marks_context_only():
    shell = [
        {"recipe_id": 1, "title": "chicken fried rice", "similarity": 0.9, "weight": 0.35, "labels": []},
        {"recipe_id": 2, "title": "roast chicken with rice side", "similarity": 0.85, "weight": 0.35, "labels": []},
        {"recipe_id": 3, "title": "chicken stew (no grams)", "similarity": 0.8, "weight": 0.35, "labels": []},
    ]
    lines_by_id = {
        1: _lines([("White rice, cooked", 250.0), ("Chicken breast, raw", 70.0)]),
        2: _lines([("Chicken breast, raw", 300.0), ("White rice, cooked", 30.0)]),
        # 3: missing → context_only
    }
    result = verify_shell_structure(
        shell,
        {
            "anchor_ingredients": ["rice"],
            "stretch_ingredient": "chicken",
            "stretch_role": "accent",
        },
        recipe_lines_by_id=lines_by_id,
    )
    accepted_ids = {r["recipe_id"] for r in result["accepted"]}
    context_ids = {r["recipe_id"] for r in result["context_only"]}
    rejected_ids = {r["recipe_id"] for r in result["rejected"]}

    assert 1 in accepted_ids
    # recipe 2 is soft (anchor present not dominant) — still harvest-eligible
    assert 2 in accepted_ids
    assert (result["accepted"][0]["structure"]["verdict"] == "pass") or any(
        (r.get("structure") or {}).get("verdict") == "pass" for r in result["accepted"]
    )
    soft = [r for r in result["accepted"] if (r.get("structure") or {}).get("verdict") == "soft"]
    assert soft and soft[0]["recipe_id"] == 2
    assert soft[0]["weight"] < 0.35
    assert 3 in context_ids
    assert result["meta"]["structure_applied"] is True
    assert result["meta"]["n_structure_passed"] >= 1
    assert result["meta"]["n_context_only"] == 1
    # No hard reject expected for soft case; stretch-without-anchor would reject
    assert 2 not in rejected_ids


def test_expand_sets_structure_verified_ids(monkeypatch):
    corpus = [
        {
            "recipe_id": 11,
            "title": "chicken fried rice",
            "text": "chicken fried rice chicken breast white rice",
            "labels": ["chicken breast", "white rice"],
        },
        {
            "recipe_id": 22,
            "title": "grilled chicken plate with rice",
            "text": "grilled chicken breast rice side",
            "labels": ["chicken breast", "rice"],
        },
    ]
    monkeypatch.setattr(
        "recipe_opt_agent.neighborhood_query_expand._load_recipe_text_corpus",
        lambda _exclude, limit=8000: corpus,
    )
    monkeypatch.setattr(
        "recipe_opt_agent.neighborhood_query_expand._embed_queries_and_corpus",
        lambda q, t: (None, None),
    )
    # Skip DB harvest during expand
    monkeypatch.setattr(
        "recipe_opt_agent.ood_foodon.ensure_ingredient_nodes_in_loss",
        lambda problem, min_hits=5: problem,
    )

    lines_by_id = {
        11: _lines([("White rice, cooked", 250.0), ("Chicken breast, raw", 70.0)]),
        22: _lines([("Chicken breast, raw", 280.0), ("White rice, cooked", 25.0)]),
    }
    problem = {
        "title": "fried rice",
        "chosen_recipe": {
            "ingredients": [{"label": "White rice, cooked", "grams": 200.0}]
        },
        "retrieval_context": {"core_recipe_ids": [], "neighbor_label_sets": [], "fdc_catalog": []},
        "ingredient_basis": [],
    }
    out = expand_neighborhood_by_queries(
        problem,
        ["chicken fried rice"],
        focus_terms=["chicken"],
        dish_structure={
            "anchor_ingredients": ["rice"],
            "stretch_ingredient": "chicken breast",
            "stretch_role": "accent",
        },
        recipe_lines_by_id=lines_by_id,
    )
    ctx = out["problem"]["retrieval_context"]
    verified = set(ctx["structure_verified_shell_ids"])
    assert 11 in verified
    assert 22 in verified  # soft, still harvest-eligible
    assert out["meta"]["structure_applied"] is True
    assert out["meta"]["n_structure_passed"] >= 1
    assert ctx.get("neighborhood_structure_meta", {}).get("n_structure_passed") >= 1


def test_ensure_ingredient_nodes_uses_only_verified_ids(monkeypatch):
    harvested_ids: list[list[int]] = []

    def fake_harvest(recipe_ids, target_nodes, **kwargs):
        harvested_ids.append(list(recipe_ids))
        return {n: [0.1] * len(recipe_ids) for n in target_nodes}

    monkeypatch.setattr(
        "recipe_opt_agent.ood_foodon.harvest_share_samples_for_nodes",
        fake_harvest,
    )
    problem = {
        "ingredient_basis": ["NODE_CHICKEN"],
        "basis_samples": {"NODE_CHICKEN": [0.05]},  # short of 5
        "basis_sample_weights": {"NODE_CHICKEN": [1.0]},
        "marginal_nodes": [],
        "retrieval_context": {
            "pending_basis_nodes": ["NODE_CHICKEN"],
            "query_shell_recipes": [
                {"recipe_id": 1},
                {"recipe_id": 2},
                {"recipe_id": 99},  # unverified — must NOT be harvested
            ],
            "structure_verified_shell_ids": [1, 2],
        },
    }
    out = ensure_ingredient_nodes_in_loss(problem, min_hits=5)
    assert harvested_ids, "expected harvest call"
    for call_ids in harvested_ids:
        assert 99 not in call_ids
        assert set(call_ids) <= {1, 2}
    assert len(out["basis_samples"]["NODE_CHICKEN"]) >= 5 or out["basis_hit_counts"]["NODE_CHICKEN"] >= 1


def test_no_structure_keeps_backward_compatible_behavior():
    shell = [
        {"recipe_id": 1, "title": "x", "similarity": 0.5, "weight": 0.3, "labels": []},
    ]
    result = verify_shell_structure(shell, None, recipe_lines_by_id={})
    assert result["meta"]["structure_applied"] is False
    assert len(result["accepted"]) == 1
    assert result["rejected"] == []
