"""Tests for silent shadow GPT draft → optimize candidate."""

from __future__ import annotations

from recipe_opt_agent.config import AgentConfig
from recipe_opt_agent.model_policy import select_shadow_draft_model
from recipe_opt_agent.prompts import draft_user_message
from recipe_opt_agent.shadow_gpt_candidate import (
    SHADOW_CANDIDATE_ID,
    SHADOW_SOURCE,
    force_include_shadow,
    is_shadow_candidate,
)


def test_shadow_draft_model_default():
    cfg = AgentConfig()
    assert cfg.shadow_draft_model == "gpt-5.5"
    assert cfg.enable_shadow_gpt_candidate is True
    assert select_shadow_draft_model(cfg) == "gpt-5.5"


def test_draft_user_message_includes_canonical_title():
    msg = draft_user_message(
        "high protein please",
        macro_box={
            "protein_min": 0.3,
            "protein_max": 0.4,
            "carb_min": 0.3,
            "carb_max": 0.4,
            "fat_min": 0.2,
            "fat_max": 0.3,
        },
        canonical_title="Spaghetti Carbonara",
    )
    assert "Spaghetti Carbonara" in msg
    assert "high protein please" in msg
    assert "protein" in msg


def test_force_include_shadow_reserves_slot():
    shadow = {
        "candidate_id": SHADOW_CANDIDATE_ID,
        "source": SHADOW_SOURCE,
        "branch": "in_distribution",
        "n_red": 5,
        "L_max_norm": 9.0,
    }
    others = [
        {
            "candidate_id": f"pool_0_{i}",
            "branch": "ood_protein",
            "n_red": 0,
            "L_max_norm": 0.1,
        }
        for i in range(6)
    ]
    ordered = others + [shadow]
    out = force_include_shadow(ordered, max_n=4)
    assert len(out) == 4
    assert is_shadow_candidate(out[0])
    assert out[0]["candidate_id"] == SHADOW_CANDIDATE_ID
    assert "gpt-5.5" not in str(out)


def test_build_shadow_candidate_shape(monkeypatch):
    from recipe_opt_agent import shadow_gpt_candidate as mod

    def fake_draft(*_a, **_k):
        return (
            {
                "title": "Shadow dish",
                "servings": 2,
                "requirement_tags": [],
                "ingredients": [
                    {"name": "chicken breast", "grams": 120.0},
                    {"name": "rice", "grams": 100.0},
                ],
                "notes": "",
            },
            {"mode": "heuristic", "model": "gpt-5.5"},
        )

    def fake_ground(draft, **_k):
        ings = [
            {"name": "chicken breast", "label": "chicken breast", "grams": 120.0},
            {"name": "rice", "label": "rice", "grams": 100.0},
        ]
        problem = {
            "x0": [120.0, 100.0],
            "M": [[20.0, 0.0, 1.0], [2.0, 25.0, 0.5]],
            "ingredient_basis": ["meat", "grain"],
            "basis_samples": {"meat": [0.3, 0.4], "grain": [0.4, 0.5]},
            "ratio_samples": [1.0, 1.1],
            "marginal_nodes": ["meat", "grain"],
            "kcal_target": 500.0,
            "chosen_recipe": {"title": "Shadow dish", "ingredients": ings},
        }
        report = type("R", (), {"to_dict": lambda self: {}})()
        return problem, report, problem["chosen_recipe"]

    def fake_opt(problem, box, **_k):
        x = list(problem["x0"])
        ings = [dict(r) for r in (problem.get("chosen_recipe") or {}).get("ingredients") or []]
        for i, row in enumerate(ings):
            row["grams"] = float(x[i])
            ings[i] = row
        chosen = {"title": "Shadow dish", "ingredients": ings}
        opt = {
            "status": "optimal",
            "objective": 0.2,
            "nutrient_slack": 0.0,
            "feasible": True,
            "x_opt": x,
            "term_losses": {"meat__share": 0.1, "grain__share": 0.1},
            "pfc_after": {"protein": 0.35, "carbs": 0.4, "fat": 0.25},
        }
        problem = dict(problem)
        problem["chosen_recipe"] = chosen
        return {"problem": problem, "opt": opt, "chosen_recipe": chosen}

    import recipe_opt_agent.grounding as ground_mod
    import recipe_opt_agent.llm as llm_mod

    monkeypatch.setattr(llm_mod, "llm_draft_recipe", fake_draft)
    monkeypatch.setattr(ground_mod, "ground_draft_to_problem", fake_ground)
    monkeypatch.setattr(mod, "optimize_grounded_problem", fake_opt)

    state = {
        "title": "Carbonara",
        "user_request": "higher protein carbonara",
        "config": {
            "shadow_draft_model": "gpt-5.5",
            "enable_shadow_gpt_candidate": True,
            "protein_min": 0.3,
            "protein_max": 0.4,
            "carb_min": 0.3,
            "carb_max": 0.45,
            "fat_min": 0.2,
            "fat_max": 0.35,
        },
        "problem": {"retrieval_context": {"fdc_catalog": []}, "basis_samples": {}},
        "candidate_pool": [],
        "interesting_candidates": [],
    }
    entry = mod.build_shadow_gpt_candidate(state, model="gpt-5.5")
    assert entry is not None
    assert entry["candidate_id"] == SHADOW_CANDIDATE_ID
    assert entry["source"] == SHADOW_SOURCE
    assert entry["branch"] == "in_distribution"
    assert entry.get("ingredients")
    assert entry.get("opt", {}).get("feasible") is True
    assert entry.get("_shadow_model") == "gpt-5.5"
    public = {k: v for k, v in entry.items() if not str(k).startswith("_")}
    assert "gpt-5.5" not in str(public)


def test_async_shadow_job_start_and_collect(monkeypatch):
    from recipe_opt_agent import shadow_gpt_candidate as mod

    def fake_build(state, *, model=None, enabled=True):
        return {
            "candidate_id": SHADOW_CANDIDATE_ID,
            "source": SHADOW_SOURCE,
            "branch": "in_distribution",
            "ingredients": [{"label": "egg", "grams": 50}],
            "_shadow_model": model,
        }

    monkeypatch.setattr(mod, "build_shadow_gpt_candidate", fake_build)
    state = {
        "title": "Pad Thai",
        "user_request": "high protein",
        "config": {"shadow_draft_model": "gpt-5.5", "enable_shadow_gpt_candidate": True},
        "problem": {},
        "candidate_pool": [],
        "interesting_candidates": [],
    }
    job_id = mod.start_shadow_gpt_job(state, model="gpt-5.5")
    assert job_id
    entry, meta = mod.collect_shadow_gpt_job(job_id, timeout=5.0)
    assert entry is not None
    assert entry["candidate_id"] == SHADOW_CANDIDATE_ID
    assert meta["collected"] is True
    assert meta["has_entry"] is True
    assert meta["model"] == "gpt-5.5"


def test_shadow_arbiter_consideration_flags_presence():
    from recipe_opt_agent.shadow_gpt_candidate import shadow_arbiter_consideration

    shadow = {
        "candidate_id": SHADOW_CANDIDATE_ID,
        "source": SHADOW_SOURCE,
        "_shadow_model": "gpt-5.5",
    }
    other = {"candidate_id": "pool_0_1", "branch": "in_distribution"}
    audit = shadow_arbiter_consideration(
        candidates=[shadow, other],
        winner_id=SHADOW_CANDIDATE_ID,
        collect_meta={"collected": True, "has_entry": True},
        model="gpt-5.5",
    )
    assert audit["shadow_in_arbiter_set"] is True
    assert audit["winner_is_shadow"] is True
    assert audit["shadow_model"] == "gpt-5.5"
    assert SHADOW_CANDIDATE_ID in audit["arbiter_candidate_ids"]


def test_arbitrate_attaches_backend_shadow_consideration(monkeypatch):
    from recipe_opt_agent import final_arbiter as fa

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fake_collect(state, *, max_candidates=4):
        return [
            {
                "candidate_id": SHADOW_CANDIDATE_ID,
                "source": SHADOW_SOURCE,
                "branch": "in_distribution",
                "ingredients": [
                    {"label": "rice noodles", "grams": 100},
                    {"label": "egg", "grams": 50},
                ],
                "opt": {
                    "feasible": True,
                    "nutrient_slack": 0.0,
                    "objective": 0.2,
                    "term_losses": {"FOODON:x__share": 0.02},
                },
                "diagnosis_full": {},
                "_shadow_model": "gpt-5.5",
            },
            {
                "candidate_id": "pool_0_1",
                "branch": "in_distribution",
                "ingredients": [
                    {"label": "rice noodles", "grams": 120},
                    {"label": "tofu", "grams": 80},
                ],
                "opt": {
                    "feasible": True,
                    "nutrient_slack": 0.0,
                    "objective": 0.25,
                    "term_losses": {"FOODON:x__share": 0.03},
                },
                "diagnosis_full": {},
            },
        ]

    monkeypatch.setattr(fa, "collect_arbiter_candidates", fake_collect)
    result = fa.arbitrate_final_recipe(
        {
            "title": "Pad Thai",
            "user_request": "high protein",
            "config": {"shadow_draft_model": "gpt-5.5", "max_finalists": 4},
            "problem": {},
            "original_ingredients": [{"label": "rice noodles", "grams": 100}],
            "_shadow_collect_meta": {"collected": True, "has_entry": True, "model": "gpt-5.5"},
        }
    )
    assert result is not None
    audit = result.get("_shadow_consideration") or {}
    assert audit.get("shadow_in_arbiter_set") is True
    assert audit.get("shadow_model") == "gpt-5.5"
    assert SHADOW_CANDIDATE_ID in (audit.get("shadow_candidate_ids") or [])
    # Public fields must not advertise the model name
    public = {
        k: v
        for k, v in result.items()
        if not str(k).startswith("_") and k not in {"winner_entry", "comparison"}
    }
    assert "gpt-5.5" not in str(public)


def test_node_build_finalists_force_includes_shadow():
    from recipe_opt_agent.graph import node_build_finalists

    shadow = {
        "candidate_id": SHADOW_CANDIDATE_ID,
        "source": SHADOW_SOURCE,
        "branch": "in_distribution",
        "n_red": 9,
        "L_max_norm": 9.0,
        "objective": 9.0,
        "ingredients": [{"label": "tofu", "grams": 100}],
        "opt": {
            "feasible": True,
            "x_opt": [100.0],
            "pfc_after": {"protein": 0.3, "carbs": 0.4, "fat": 0.3},
        },
    }
    pool = [
        {
            "candidate_id": f"pool_0_{i}",
            "branch": "ood_protein",
            "n_red": 0,
            "L_max_norm": 0.05,
            "objective": 0.05,
            "ingredients": [{"label": f"ing{i}", "grams": 50}],
            "opt": {"feasible": True},
        }
        for i in range(5)
    ] + [shadow]
    out = node_build_finalists(
        {
            "config": {"max_finalists": 4},
            "candidate_pool": pool,
            "interesting_candidates": [],
        }
    )
    finals = out["finalist_pool"]
    assert len(finals) <= 4
    assert any(is_shadow_candidate(e) for e in finals)
