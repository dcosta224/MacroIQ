"""Structured LLM recipe draft schema for creative mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DraftIngredient:
    name: str
    grams: float
    role: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecipeDraft:
    title: str
    servings: int = 2
    requirement_tags: list[dict[str, Any]] = field(default_factory=list)
    ingredients: list[DraftIngredient] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "servings": self.servings,
            "requirement_tags": list(self.requirement_tags),
            "ingredients": [i.to_dict() for i in self.ingredients],
            "notes": self.notes,
        }


def parse_draft(data: dict[str, Any]) -> RecipeDraft:
    ings: list[DraftIngredient] = []
    for raw in data.get("ingredients") or []:
        if isinstance(raw, str):
            ings.append(DraftIngredient(name=raw, grams=0.0))
            continue
        ings.append(
            DraftIngredient(
                name=str(raw.get("name") or raw.get("label") or ""),
                grams=float(raw.get("grams") or 0.0),
                role=str(raw.get("role") or ""),
                notes=str(raw.get("notes") or ""),
            )
        )
    return RecipeDraft(
        title=str(data.get("title") or "Untitled"),
        servings=int(data.get("servings") or 2),
        requirement_tags=list(data.get("requirement_tags") or []),
        ingredients=ings,
        notes=str(data.get("notes") or ""),
    )


def draft_json_schema() -> dict[str, Any]:
    return {
        "title": "string",
        "servings": 2,
        "requirement_tags": [
            {"tag_id": "high_protein", "kind": "macro_intent", "polarity": "require", "source_text": ""}
        ],
        "ingredients": [{"name": "string", "grams": 120.0, "role": "pasta", "notes": ""}],
        "notes": "string",
    }
