"""Tests for hardened FDC grounding, identity priors, and edit support gates."""

from __future__ import annotations

from recipe_opt_agent.culinary_types import families_compatible, families_for_text
from recipe_opt_agent.draft_schema import DraftIngredient
from recipe_opt_agent.edit_grounding import (
    filter_candidates_by_neighborhood_support,
    missing_high_hit_basis_nodes,
)
from recipe_opt_agent.grounding import _match_score, resolve_line_to_fdc
from recipe_opt_agent.identity_gates import apply_identity_grounding_gates
from recipe_opt_agent.identity_roles import lexical_roles_from_text, resolve_identity_roles
from recipe_opt_agent.model_policy import tags_need_llm
from recipe_opt_agent.requirement_tags import (
    RequirementTag,
    deduce_requirement_tags,
    dietary_tag_supported_by_request,
)
from opt_diagnosis import Diagnosis, DiagnosisResult, FidelityBand


def test_rejects_bbq_sauce_to_soy_sauce():
    assert _match_score("low-sugar BBQ sauce", "Soy sauce made from soy and wheat (shoyu)") == 0.0


def test_rejects_rice_to_wine():
    assert _match_score("long grain white rice", "Alcoholic beverage, wine, table, white") == 0.0


def test_rejects_milk_to_chocolate_dessert():
    assert _match_score("whole milk", "Milk dessert, frozen, milk-fat free, chocolate") == 0.0


def test_rejects_butter_to_apple_butter():
    assert _match_score("butter", "Fruit butters, apple") == 0.0


def test_rejects_turkey_to_diced_tomatoes():
    assert _match_score("smoked turkey breast, diced", "DICED TOMATOES") == 0.0


def test_accepts_rice_to_rice():
    assert _match_score("long grain white rice", "Rice, white, long-grain, regular, raw") >= 0.35


def test_resolve_line_leaves_bad_sauce_unresolved():
    catalog = [
        {"fdc_id": 1, "fdc_description": "Soy sauce made from soy and wheat (shoyu)"},
        {"fdc_id": 2, "fdc_description": "Sauce, barbecue"},
    ]
    gl = resolve_line_to_fdc(
        DraftIngredient(name="low-sugar BBQ sauce", grams=60, role="sauce"),
        neighborhood_catalog=catalog,
        broader_catalog=None,
        tags=[],
    )
    assert gl.status == "matched"
    assert "barbecue" in gl.label.lower() or "bbq" in gl.label.lower() or gl.fdc_id == 2


def test_resolve_line_rejects_only_soy_in_catalog():
    catalog = [
        {"fdc_id": 1, "fdc_description": "Soy sauce made from soy and wheat (shoyu)"},
    ]
    gl = resolve_line_to_fdc(
        DraftIngredient(name="low-sugar BBQ sauce", grams=60, role="sauce"),
        neighborhood_catalog=catalog,
        broader_catalog=None,
        tags=[],
    )
    assert gl.status == "unresolved"
    assert gl.fdc_id is None


def test_grape_leaves_identity_includes_rice():
    roles = lexical_roles_from_text("Stuffed Grape Leaves", "higher-protein stuffed grape leaves")
    assert "rice" in roles
    assert "grape_leaf" in roles


def test_bbq_ribs_identity_includes_sauce():
    roles = lexical_roles_from_text("BBQ Ribs", "")
    assert "bbq_sauce" in roles
    assert "pork_rib" in roles


def test_identity_prefers_title_over_mangled_ingredients():
    roles = resolve_identity_roles(
        title="Higher-Protein Stuffed Grape Leaves",
        request="boost protein",
        ingredients=[{"label": "Alcoholic beverage, wine, table, white"}],
        use_llm=False,
        prefer_title_priors=True,
    )
    assert "rice" in roles


def test_no_false_vegetarian_on_high_protein_request():
    req = (
        "Higher-protein Bobotie: about 32% protein. Keep the dish recognizable "
        "but boost protein relative to a typical neighborhood version."
    )
    assert not tags_need_llm(req, [])
    tags = deduce_requirement_tags(req, force_llm=False)
    assert not any(t.tag_id == "vegetarian" for t in tags)


def test_dietary_tag_requires_explicit_evidence():
    assert dietary_tag_supported_by_request("vegetarian", "make it vegetarian please")
    assert not dietary_tag_supported_by_request(
        "vegetarian", "Keep the dish recognizable but boost protein"
    )


def test_filter_drops_unsupported_add():
    problem = {
        "retrieval_context": {
            "fdc_catalog": [
                {"fdc_id": 1, "fdc_description": "Chicken, broilers or fryers, breast"},
                {"fdc_id": 2, "fdc_description": "Onions, raw"},
            ]
        }
    }
    cands = [
        {"candidate_id": "a", "action": "add", "label": "Chocolate syrup"},
        {"candidate_id": "b", "action": "add", "label": "Chicken breast"},
    ]
    kept, dropped = filter_candidates_by_neighborhood_support(cands, problem=problem)
    assert len(kept) == 1
    assert kept[0]["candidate_id"] == "b"
    assert dropped[0]["reason"] == "no_neighborhood_support"


def test_missing_high_hit_basis_detected():
    report = {
        "basis_nodes": [
            {"label": "barbeque sauce", "n_hits": 22, "in_current_recipe": False},
            {"label": "pork ribs", "n_hits": 6, "in_current_recipe": True},
        ]
    }
    missing = missing_high_hit_basis_nodes(report, min_hits=8)
    assert len(missing) == 1
    assert "barbeque" in missing[0]["label"]


def test_identity_gate_escalates_accept():
    diag = DiagnosisResult(
        diagnosis=Diagnosis.OK,
        fidelity_band=FidelityBand.ACCEPT,
        meaning="ok",
        terms=[],
        n_red=0,
        n_yellow=0,
        L_max_norm=0.0,
        L_total=0.5,
        macros_feasible=True,
        hull_intersects=True,
    )
    problem = {
        "foodon_basis_report": {
            "basis_nodes": [
                {"label": "long grain white rice", "n_hits": 20, "in_current_recipe": False}
            ]
        },
        "grounding_report": {"unresolved": []},
    }
    out = apply_identity_grounding_gates(
        diag,
        problem=problem,
        identity_roles=["rice", "grape_leaf"],
        nutrient_slack=0.0,
    )
    assert out.fidelity_band == FidelityBand.MUST_RETRY


def test_turkey_tomato_families_conflict():
    assert not families_compatible(families_for_text("smoked turkey"), families_for_text("DICED TOMATOES"))
