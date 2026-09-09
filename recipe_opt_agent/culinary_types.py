"""Culinary type families for grounding and edit support checks.

Prevents catastrophic FDC mismatches (BBQ sauce→soy sauce, rice→wine,
milk→frozen dessert, butter→apple butter) by requiring query and candidate
labels to share a compatible culinary family when either is typed.
"""

from __future__ import annotations

# family_id → content tokens / phrases (lowercase). Longer phrases checked first.
CULINARY_FAMILIES: dict[str, tuple[str, ...]] = {
    "pork_rib": ("babyback", "baby back", "spare rib", "sparerib", "pork rib", "pork ribs"),
    "bbq_sauce": (
        "bbq sauce",
        "barbeque sauce",
        "barbecue sauce",
        "low-sugar bbq",
        "low sugar bbq",
        "barbeque",
        "barbecue",
        " bbq",
    ),
    "soy_sauce": ("soy sauce", "shoyu", "tamari", "soya sauce"),
    "turkey": ("turkey breast", "smoked turkey", "turkey"),
    "tomato": ("diced tomato", "diced tomatoes", "tomato", "tomatoes"),
    "cornbread": ("cornbread", "corn bread", "cornmeal bread"),
    "olive_oil": ("olive oil", "extra virgin olive"),
    "rice": (
        "long grain white rice",
        "brown rice",
        "white rice",
        "arborio",
        "basmati",
        "jasmine",
        "long grain",
        "short grain",
        " rice",
    ),
    "wine": ("white wine", "red wine", "table wine", "cooking wine", " wine", "vino", "marsala", "sherry"),
    "dessert_dairy": (
        "milk dessert",
        "ice cream",
        "frozen dessert",
        "pudding",
        "chocolate milk dessert",
        "milk-fat free, chocolate",
    ),
    "milk": ("whole milk", "skim milk", "2% milk", "lowfat milk", "low-fat milk", "cow milk", " milk"),
    "fruit_butter": ("fruit butter", "fruit butters", "apple butter"),
    "butter": ("unsalted butter", "salted butter", "butter,", " butter", "ghee"),
    "salt": ("table salt", "kosher salt", "sea salt", "salt, table"),
    "grape_leaf": ("grape leaf", "grape leaves", "vine leaf"),
    "ground_beef": ("ground beef", "lean ground beef", "minced beef", "beef mince"),
    "curry": ("curry powder", "curry paste"),
    "bread": ("white bread", "wheat bread", "sandwich bread", " bread"),
    "egg": ("egg white", "egg yolk", "whole egg", " egg"),
    "onion_rings": (
        "onion ring",
        "onion rings",
        "fast foods, onion",
    ),
    "onion": ("onion", "onions", "shallot", "scallion"),
    "flour": ("bread flour", "wheat flour", "all-purpose flour", " flour"),
    "yeast": ("active dry yeast", "instant yeast", "yeast"),
    "coffee": ("brewed coffee", "coffee, brewed", "coffee", "espresso"),
    "cheese": (
        "blue cheese",
        "cheddar",
        "mozzarella",
        "parmesan",
        "pecorino",
        "feta",
        "ricotta",
        "cheese",
    ),
    "yogurt": ("greek yogurt", "greek yoghurt", "yogurt", "yoghurt"),
    "cream": ("heavy cream", "sour cream", "whipping cream", "cream,"),
    "tofu": ("tofu", "bean curd"),
    "chicken": ("chicken breast", "chicken thigh", "chicken,"),
    "beef": ("beef,", "beef brisket", "beef chuck", "steak"),
    "pork": ("pork loin", "pork shoulder", "pork tenderloin", "guanciale", "pancetta", "bacon"),
}

# Pairs that must never match even if tokens overlap.
CONFLICTING_FAMILIES: set[tuple[str, str]] = {
    ("bbq_sauce", "soy_sauce"),
    ("soy_sauce", "bbq_sauce"),
    ("rice", "wine"),
    ("wine", "rice"),
    ("milk", "dessert_dairy"),
    ("dessert_dairy", "milk"),
    ("butter", "fruit_butter"),
    ("fruit_butter", "butter"),
    ("turkey", "tomato"),
    ("tomato", "turkey"),
    ("salt", "butter"),
    ("butter", "salt"),
    ("onion", "onion_rings"),
    ("onion_rings", "onion"),
    ("yeast", "coffee"),
    ("coffee", "yeast"),
    ("flour", "coffee"),
    ("coffee", "flour"),
    ("bread", "coffee"),
    ("coffee", "bread"),
}


def families_for_text(text: str) -> set[str]:
    raw = (text or "").lower()
    padded = f" {raw} "
    hit: set[str] = set()
    # Prefer more specific families when phrases overlap
    priority = [
        "dessert_dairy",
        "fruit_butter",
        "onion_rings",
        "bbq_sauce",
        "soy_sauce",
        "pork_rib",
        "ground_beef",
        "grape_leaf",
        "olive_oil",
        "cornbread",
        "turkey",
        "tomato",
        "wine",
        "rice",
        "yogurt",
        "cream",
        "tofu",
        "milk",
        "butter",
        "salt",
        "curry",
        "bread",
        "egg",
        "chicken",
        "beef",
        "pork",
        "onion",
        "flour",
        "yeast",
        "coffee",
        "cheese",
    ]
    for fam in priority:
        tokens = CULINARY_FAMILIES.get(fam, ())
        for tok in sorted(tokens, key=len, reverse=True):
            needle = tok if tok.startswith(" ") else f" {tok.strip()} "
            if needle in padded or tok.strip() in raw:
                # Avoid classifying "butter, without salt" as salt-primary
                if fam == "salt" and "butter" in raw:
                    continue
                # Avoid classifying dessert milk as milk
                if fam == "milk" and any(x in raw for x in ("dessert", "ice cream", "pudding", "chocolate")):
                    continue
                # Avoid fruit butter as butter
                if fam == "butter" and "fruit" in raw and "butter" in raw:
                    continue
                # Plain onion must not also claim onion_rings hits
                if fam == "onion" and any(x in raw for x in ("onion ring", "onion rings", "fast foods")):
                    continue
                hit.add(fam)
                break
    return hit


def families_compatible(query_families: set[str], label_families: set[str]) -> bool:
    if not query_families:
        return True  # untyped query: rely on score alone
    for q in query_families:
        for lab in label_families:
            if (q, lab) in CONFLICTING_FAMILIES:
                return False
    if query_families & label_families:
        return True
    if not label_families:
        return True  # untyped label — let lexical score decide
    return False  # both typed, disjoint families


def content_tokens(text: str) -> set[str]:
    stop = {
        "and",
        "the",
        "with",
        "from",
        "for",
        "raw",
        "fresh",
        "made",
        "prepared",
        "table",
        "low",
        "sugar",
        "diced",
        "chopped",
        "lean",
        "trimmed",
        "additional",
        "main",
        "side",
        "without",
        "salad",
        "cooking",
    }
    return {
        t
        for t in "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in (text or "").lower()).split()
        if len(t) > 2 and t not in stop
    }
