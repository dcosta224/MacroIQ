"""Tests for requirement_tags hard dietary filters."""

from __future__ import annotations

from recipe_opt_agent.requirement_tags import (
    RequirementTag,
    deduce_tags_from_text,
    filter_candidates_by_tags,
    ingredient_passes_tags,
    tag_violations_for_ingredient,
)


def test_deduce_vegetarian_from_text():
    tags = deduce_tags_from_text("vegetarian carbonara please")
    assert any(t.tag_id == "vegetarian" for t in tags)


def test_no_pork_forbids_bacon():
    tags = [RequirementTag("no_pork", "dietary_restriction", "forbid", "no pork")]
    assert not ingredient_passes_tags("guanciale, cured", tags)
    assert ingredient_passes_tags("mushrooms, raw", tags)


def test_filter_candidates_drops_pork_add():
    tags = [RequirementTag("vegetarian", "dietary_restriction", "require", "vegetarian")]
    cands = [
        {"candidate_id": "a1", "action": "add", "label": "chicken breast"},
        {"candidate_id": "a2", "action": "add", "label": "mushrooms, white"},
    ]
    kept, dropped = filter_candidates_by_tags(cands, tags)
    assert len(kept) == 1
    assert kept[0]["candidate_id"] == "a2"
    assert dropped[0]["reason"] == "tag_violation"


def test_high_protein_tag_from_request():
    tags = deduce_tags_from_text("40% protein carbonara")
    assert any(t.tag_id == "high_protein" for t in tags)
