"""Identity role extraction: templates ∪ lexical/LLM merge."""

from __future__ import annotations

import os
from typing import Any

from recipe_opt_agent.config import IDENTITY_TEMPLATES, identity_roles_for_title

# Lightweight dish → roles for free-text titles not in IDENTITY_TEMPLATES.
# These are title/neighborhood priors — do NOT derive solely from a mangled draft.
_LEXICAL_DISH_ROLES: dict[str, list[str]] = {
    "shakshuka": ["egg", "tomato", "pepper", "onion"],
    "ramen": ["noodle", "broth", "protein"],
    "risotto": ["rice", "stock", "cheese"],
    "pho": ["noodle", "broth", "herb"],
    "pad thai": ["noodle", "protein", "tamarind"],
    "burrito": ["tortilla", "bean", "protein"],
    "taco": ["tortilla", "protein"],
    "curry": ["sauce", "protein", "aromatic"],
    "chili": ["bean", "protein", "chili"],
    "salad": ["greens", "dressing"],
    "soup": ["broth", "vegetable"],
    "stew": ["protein", "vegetable", "broth"],
    "omelette": ["egg"],
    "omelet": ["egg"],
    "pancake": ["batter", "egg"],
    "stir fry": ["vegetable", "protein", "sauce"],
    "stir-fry": ["vegetable", "protein", "sauce"],
    "bbq ribs": ["pork_rib", "bbq_sauce", "spice"],
    "barbeque ribs": ["pork_rib", "bbq_sauce", "spice"],
    "barbecue ribs": ["pork_rib", "bbq_sauce", "spice"],
    "ribs": ["pork_rib", "bbq_sauce", "spice"],
    "bobotie": ["ground_meat", "egg", "curry", "bread", "milk"],
    "stuffed grape leaves": ["grape_leaf", "rice", "protein", "herb"],
    "grape leaves": ["grape_leaf", "rice", "protein", "herb"],
    "dolma": ["grape_leaf", "rice", "protein"],
    "dolmades": ["grape_leaf", "rice", "protein"],
    "focaccia": ["flour", "yeast", "olive_oil", "salt"],
    "fried rice": ["rice", "egg", "protein", "aromatic"],
    "biryani": ["rice", "protein", "spice"],
    "cheeseburger": ["bun", "beef", "cheese"],
    "hamburger": ["bun", "beef"],
    "buffalo wings": ["chicken_wing", "sauce"],
    "chicken wings": ["chicken_wing", "sauce"],
    "al pastor": ["pork", "chili", "aromatic"],
    "carne asada": ["beef", "citrus", "aromatic"],
    "ceviche": ["fish", "citrus", "onion"],
}

# Map neighborhood FoodOn / basis labels → identity roles
_BASIS_LABEL_TO_ROLE: list[tuple[str, str]] = [
    ("barbeque sauce", "bbq_sauce"),
    ("barbecue sauce", "bbq_sauce"),
    ("pork ribs", "pork_rib"),
    ("long grain white rice", "rice"),
    ("white rice", "rice"),
    ("rice", "rice"),
    ("grape leaf", "grape_leaf"),
    ("curry powder", "curry"),
    ("ground beef", "ground_meat"),
    ("beef", "protein"),
    ("hen egg", "egg"),
    ("egg", "egg"),
    ("white bread", "bread"),
    ("olive oil", "olive_oil"),
    ("bread flour", "flour"),
    ("yeast", "yeast"),
]


def lexical_roles_from_text(title: str, request: str = "") -> list[str]:
    text = f"{title} {request}".lower()
    # Prefer longer dish keys first
    for key in sorted(_LEXICAL_DISH_ROLES.keys(), key=len, reverse=True):
        if key in text:
            return list(_LEXICAL_DISH_ROLES[key])
    # Ingredient-ish cues
    roles: list[str] = []
    cues = [
        ("egg", "egg"),
        ("tomato", "tomato"),
        ("cheese", "cheese"),
        ("pasta", "pasta"),
        ("noodle", "noodle"),
        ("rice", "rice"),
        ("chicken", "protein"),
        ("beef", "protein"),
        ("tofu", "protein"),
        ("pork", "cured_pork"),
        ("bacon", "cured_pork"),
        ("crust", "crust"),
        ("dough", "crust"),
        ("bean", "bean"),
        ("lentil", "bean"),
        ("grape leaf", "grape_leaf"),
        ("bbq", "bbq_sauce"),
        ("barbeque", "bbq_sauce"),
    ]
    for needle, role in cues:
        if needle in text and role not in roles:
            roles.append(role)
    return roles


def roles_from_neighborhood_basis(
    foodon_basis_report: dict[str, Any] | None,
    *,
    min_hits: int = 10,
) -> list[str]:
    """Derive identity roles from high-hit neighborhood FoodOn basis nodes."""
    roles: list[str] = []
    seen: set[str] = set()
    nodes = list((foodon_basis_report or {}).get("basis_nodes") or [])
    nodes.sort(key=lambda n: -int(n.get("n_hits") or 0))
    for n in nodes:
        if int(n.get("n_hits") or 0) < min_hits:
            continue
        label = str(n.get("label") or "").lower()
        for needle, role in _BASIS_LABEL_TO_ROLE:
            if needle in label and role not in seen:
                seen.add(role)
                roles.append(role)
                break
    return roles


def merge_roles(*role_lists: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for lst in role_lists:
        for r in lst or []:
            key = str(r).strip().lower().replace(" ", "_")
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def resolve_identity_roles(
    *,
    title: str,
    request: str = "",
    ingredients: list[dict[str, Any]] | None = None,
    templates: dict[str, list[str]] | None = None,
    use_llm: bool = True,
    model: str = "gpt-4o-mini",
    foodon_basis_report: dict[str, Any] | None = None,
    prefer_title_priors: bool = True,
) -> list[str]:
    """Template ∪ lexical ∪ neighborhood basis ∪ optional LLM roles.

    When ``prefer_title_priors`` is True (default), ingredient-derived roles are
    only used to *add* cues — they never replace title/neighborhood priors.
    This prevents mangled FDC grounds (wine-as-rice) from rewriting identity.
    """
    template = identity_roles_for_title(title or request, templates or IDENTITY_TEMPLATES)
    lexical = lexical_roles_from_text(title or "", request or "")
    nb_roles = roles_from_neighborhood_basis(foodon_basis_report)
    ingredient_roles: list[str] = []
    if ingredients and not prefer_title_priors:
        labels = " ".join(str(r.get("label") or r.get("name") or "") for r in ingredients)
        ingredient_roles = lexical_roles_from_text(labels, "")
    extracted: list[str] = []
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        try:
            # Pass title/request only — avoid poisoning LLM with bad FDC labels
            extracted = extract_identity_roles_llm(
                title=title,
                request=request,
                ingredients=None if prefer_title_priors else ingredients,
                model=model,
            )
        except Exception:
            extracted = []
    return merge_roles(template, lexical, nb_roles, ingredient_roles, extracted)


def extract_identity_roles_llm(
    *,
    title: str,
    request: str = "",
    ingredients: list[dict[str, Any]] | None = None,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """LLM JSON list of stable role ids (pasta, egg, cheese, …)."""
    from recipe_opt_agent.llm import _call_json_llm

    ings = ingredients or []
    labels = [str(r.get("label") or r.get("name") or "") for r in ings][:20]
    system = (
        "Extract dish-identity ingredient ROLES that must remain filled for this dish to stay itself. "
        "Return JSON: {\"identity_roles\": [\"role\", ...]}. "
        "Use short stable ids like pasta, egg, cheese, cured_pork, crust, tomato, protein, broth, "
        "noodle, rice, grape_leaf, bbq_sauce, pork_rib, ground_meat, curry, flour, yeast. "
        "Prefer 2–6 roles grounded in the dish name / cuisine, not in a possibly-wrong ingredient list. "
        "Do not invent roles unrelated to the dish."
    )
    user = (
        f"Title: {title}\nRequest: {request}\n"
        + (f"Ingredients: {labels}\n" if labels else "")
        + "Respond with identity_roles JSON only."
    )

    def _heuristic() -> dict[str, Any]:
        return {"identity_roles": lexical_roles_from_text(title, request)}

    data, _ = _call_json_llm(system=system, user=user, model=model, heuristic_fn=_heuristic)
    roles = data.get("identity_roles") or []
    return [str(r).strip().lower().replace(" ", "_") for r in roles if str(r).strip()]
