"""Hard dietary / macro requirement tags for creative mode."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol


TagKind = str  # dietary_restriction | preference | macro_intent
TagPolarity = str  # require | forbid


@dataclass(frozen=True)
class RequirementTag:
    tag_id: str
    kind: TagKind
    polarity: TagPolarity
    source_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Keyword patterns for heuristic / offline tag deduction.
_TAG_PATTERNS: list[tuple[str, TagKind, TagPolarity, tuple[str, ...]]] = [
    ("vegetarian", "dietary_restriction", "require", ("vegetarian", "veggie", "no meat")),
    ("vegan", "dietary_restriction", "require", ("vegan", "plant-based", "plant based")),
    ("no_pork", "dietary_restriction", "forbid", ("no pork", "without pork", "pork-free", "pork free")),
    ("no_beef", "dietary_restriction", "forbid", ("no beef", "without beef", "beef-free")),
    ("no_dairy", "dietary_restriction", "forbid", ("no dairy", "dairy-free", "dairy free", "lactose-free")),
    ("gluten_free", "dietary_restriction", "require", ("gluten-free", "gluten free", "no gluten")),
    ("high_protein", "macro_intent", "require", ("high protein", "high-protein", "40% protein", "40 percent protein")),
    ("low_carb", "macro_intent", "require", ("low carb", "low-carb", "keto")),
]

# Ingredient label substrings that violate dietary restriction tags.
_VIOLATION_SUBSTR: dict[str, tuple[str, ...]] = {
    "vegetarian": (
        "pork",
        "bacon",
        "guanciale",
        "pancetta",
        "prosciutto",
        "ham",
        "sausage",
        "beef",
        "chicken",
        "turkey",
        "fish",
        "salmon",
        "tuna",
        "shrimp",
        "anchov",
    ),
    "vegan": (
        "egg",
        "cheese",
        "milk",
        "cream",
        "butter",
        "yogurt",
        "pork",
        "bacon",
        "guanciale",
        "pancetta",
        "chicken",
        "beef",
        "fish",
        "honey",
    ),
    "no_pork": ("pork", "bacon", "guanciale", "pancetta", "prosciutto", "ham", "sausage", "salami", "pepperoni"),
    "no_beef": ("beef", "steak", "brisket", "ground beef"),
    "no_dairy": ("cheese", "milk", "cream", "butter", "yogurt", "parmesan", "mozzarella", "ricotta", "pecorino"),
}


class RequirementTagMapper(Protocol):
    """Future hook: map RequirementTags to attribute-tag IDs."""

    def map_to_attribute_tags(self, tags: list[RequirementTag]) -> list[str]: ...


class StubRequirementTagMapper:
    """Pass-through until attribute-tag system lands."""

    def map_to_attribute_tags(self, tags: list[RequirementTag]) -> list[str]:
        return [t.tag_id for t in tags]


def _normalize_tag(raw: dict[str, Any]) -> RequirementTag | None:
    tag_id = str(raw.get("tag_id") or "").strip().lower().replace(" ", "_")
    if not tag_id:
        return None
    kind = str(raw.get("kind") or "preference")
    polarity = str(raw.get("polarity") or "require")
    if polarity not in {"require", "forbid"}:
        polarity = "require"
    if kind not in {"dietary_restriction", "preference", "macro_intent"}:
        kind = "preference"
    return RequirementTag(
        tag_id=tag_id,
        kind=kind,
        polarity=polarity,
        source_text=str(raw.get("source_text") or ""),
    )


def deduce_tags_from_text(request: str) -> list[RequirementTag]:
    """Heuristic tag deduction from free-text request (no LLM)."""
    text = request.lower()
    out: list[RequirementTag] = []
    seen: set[str] = set()
    for tag_id, kind, polarity, phrases in _TAG_PATTERNS:
        if tag_id in seen:
            continue
        if any(p in text for p in phrases):
            out.append(
                RequirementTag(
                    tag_id=tag_id,
                    kind=kind,
                    polarity=polarity,
                    source_text=next(p for p in phrases if p in text),
                )
            )
            seen.add(tag_id)
    # Explicit macro percents in request
    m = re.search(r"(\d+)\s*%\s*protein", text)
    if m and "high_protein" not in seen:
        out.append(
            RequirementTag(
                tag_id="high_protein",
                kind="macro_intent",
                polarity="require",
                source_text=m.group(0),
            )
        )
    return out


_DIETARY_EVIDENCE: dict[str, tuple[str, ...]] = {
    "vegetarian": ("vegetarian", "veggie", "no meat", "meatless", "without meat", "plant-based", "plant based"),
    "vegan": ("vegan", "plant-based", "plant based"),
    "no_pork": ("no pork", "without pork", "pork-free", "pork free"),
    "no_beef": ("no beef", "without beef", "beef-free", "beef free"),
    "no_dairy": ("no dairy", "dairy-free", "dairy free", "lactose-free", "lactose free"),
    "gluten_free": ("gluten-free", "gluten free", "no gluten", "without gluten"),
}


def dietary_tag_supported_by_request(tag_id: str, request: str) -> bool:
    """True only when the request text explicitly evidences this dietary tag."""
    text = (request or "").lower()
    phrases = _DIETARY_EVIDENCE.get(tag_id, ())
    if not phrases:
        # Unknown dietary tag: require the tag_id itself as a word-ish cue
        return tag_id.replace("_", " ") in text or tag_id in text
    return any(p in text for p in phrases)


def deduce_requirement_tags(
    request: str,
    *,
    draft_tags: list[dict[str, Any]] | None = None,
    llm: Any | None = None,
    model: str | None = None,
    force_llm: bool | None = None,
) -> list[RequirementTag]:
    """Merge tags from draft JSON and request text; optional LLM pass when needed."""
    from recipe_opt_agent.model_policy import tags_need_llm

    merged: dict[str, RequirementTag] = {}
    for raw in draft_tags or []:
        t = _normalize_tag(raw)
        if t:
            merged[t.tag_id] = t
    lexical = deduce_tags_from_text(request)
    for t in lexical:
        merged.setdefault(t.tag_id, t)

    want_llm = force_llm if force_llm is not None else tags_need_llm(request, lexical)
    if want_llm and (llm is not None or os.environ.get("OPENAI_API_KEY")):
        try:
            from recipe_opt_agent.llm import deduce_tags_llm

            tag_model = model or getattr(llm, "model", None) or "gpt-4.1-nano"
            llm_tags = deduce_tags_llm(request, model=tag_model)
            for raw in llm_tags:
                t = _normalize_tag(raw)
                if not t:
                    continue
                # Never accept dietary restrictions the request does not explicitly support.
                if t.kind == "dietary_restriction" and not dietary_tag_supported_by_request(
                    t.tag_id, request
                ):
                    continue
                merged[t.tag_id] = t
        except Exception:
            pass

    # Also strip draft/LLM dietary tags that lack request evidence (draft may invent them).
    cleaned: dict[str, RequirementTag] = {}
    for tag_id, t in merged.items():
        if t.kind == "dietary_restriction" and not dietary_tag_supported_by_request(tag_id, request):
            # Keep if lexical deduction found it (heuristic phrases already in request)
            if not any(x.tag_id == tag_id for x in lexical):
                continue
        cleaned[tag_id] = t
    return list(cleaned.values())


def tag_violations_for_ingredient(
    label: str,
    tags: list[RequirementTag],
    *,
    fdc_description: str | None = None,
) -> list[str]:
    """Return tag_ids violated by this ingredient line (dietary_restriction only)."""
    text = f"{label} {fdc_description or ''}".lower()
    violated: list[str] = []
    dietary = [t for t in tags if t.kind == "dietary_restriction"]
    for tag in dietary:
        substrs = _VIOLATION_SUBSTR.get(tag.tag_id, ())
        if tag.polarity == "forbid":
            if any(s in text for s in substrs):
                violated.append(tag.tag_id)
        elif tag.polarity == "require":
            if tag.tag_id == "vegetarian" and any(s in text for s in _VIOLATION_SUBSTR["vegetarian"]):
                violated.append(tag.tag_id)
            elif tag.tag_id == "vegan" and any(s in text for s in _VIOLATION_SUBSTR["vegan"]):
                violated.append(tag.tag_id)
            elif tag.tag_id == "gluten_free" and any(
                s in text for s in ("wheat", "flour", "bread", "pasta", "spaghetti", "noodle", "barley", "rye")
            ):
                # Gluten-free is complex; flag obvious gluten carriers unless labeled gf
                if "gluten-free" not in text and "gluten free" not in text:
                    violated.append(tag.tag_id)
    return violated


def ingredient_passes_tags(
    label: str,
    tags: list[RequirementTag],
    *,
    fdc_description: str | None = None,
) -> bool:
    return len(tag_violations_for_ingredient(label, tags, fdc_description=fdc_description)) == 0


def filter_ingredients_by_tags(
    ingredients: list[dict[str, Any]],
    tags: list[RequirementTag],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ingredients into kept vs rejected with violation reasons."""
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in ingredients:
        label = str(row.get("label") or row.get("name") or "")
        desc = row.get("fdc_description")
        vios = tag_violations_for_ingredient(label, tags, fdc_description=desc)
        if vios:
            rejected.append({**row, "tag_violations": vios})
        else:
            kept.append(row)
    return kept, rejected


def filter_candidates_by_tags(
    candidates: list[dict[str, Any]],
    tags: list[RequirementTag],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop modification candidates that would violate dietary tags."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for c in candidates:
        action = c.get("action")
        label = str(c.get("label") or "")
        if action in {"add", "swap"} and not ingredient_passes_tags(label, tags):
            dropped.append({"candidate": c, "reason": "tag_violation", "tag_violations": tag_violations_for_ingredient(label, tags)})
            continue
        kept.append(c)
    return kept, dropped


def tags_checklist(tags: list[RequirementTag]) -> list[dict[str, str]]:
    return [{"tag_id": t.tag_id, "kind": t.kind, "polarity": t.polarity} for t in tags]


def tags_to_json(tags: list[RequirementTag]) -> str:
    return json.dumps([t.to_dict() for t in tags], indent=2)
