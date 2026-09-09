#!/usr/bin/env python3
"""Agent vs GPT-5.5 evaluation suites (A/B/C/D/E) with resume + full disk persistence.

Suites
------
A  Macro precision — high-protein ±2% box, creative_example agent
B  Dietary hard constraints — explicit diet tags in the request
C  Identity under stretch — high-protein box, creative_example agent
D  Cookability under constraint — kitchen-plausible grams + staples under stretch
E  Taste preference × macros — explicit taste ask + tight protein box

Both systems get the same user request + macro box. GPT-5.5 is a one-shot
structured draft (no tools). Shared gpt-5.5 judge. Win rules use nutrient loss,
ratio loss, dietary safety, identity/cookability/taste proxies, and holistic score.

Resume
------
  PYTHONPATH=scripts:. uv run python tests/run_agent_vs_gpt55_eval.py \\
      --resume scratch/recipe_opt_runs/eval_suites/agent_vs_gpt55_<id>

Progress is checkpointed after every case to ``progress.jsonl`` + ``checkpoint.json``.
Completed ``case_id`` values are skipped on resume.

Usage
-----
  PYTHONPATH=scripts:. uv run python tests/run_agent_vs_gpt55_eval.py
  PYTHONPATH=scripts:. uv run python tests/run_agent_vs_gpt55_eval.py --suites A,B --max-iterations 2
  PYTHONPATH=scripts:. uv run python tests/run_agent_vs_gpt55_eval.py --suites D,E
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

DEFAULT_COMPETITOR_MODEL = "gpt-5.5"
DEFAULT_JUDGE_MODEL = "gpt-5.5"
SUITE_NAME = "agent_vs_gpt55"

# Dietary cases for Suite B (canonical_id may be None → title-only creative).
DIETARY_CASES: list[dict[str, Any]] = [
    {
        "case_key": "vegetarian_carbonara",
        "title": "Carbonara",
        "canonical_id": None,
        "request": (
            "Vegetarian carbonara: about 28% protein, 40% carbs, 32% fat "
            "(calorie shares). No meat or fish. Keep it recognizable as carbonara."
        ),
        "tags": ["vegetarian", "high_protein"],
        "box": {
            "protein_min": 0.26,
            "protein_max": 0.30,
            "carb_min": 0.38,
            "carb_max": 0.42,
            "fat_min": 0.30,
            "fat_max": 0.34,
        },
    },
    {
        "case_key": "vegetarian_bobotie",
        "title": "Bobotie",
        "canonical_id": 67,
        "request": (
            "Vegetarian bobotie: about 28% protein, 28% carbs, 44% fat. "
            "No meat. Keep South African bobotie identity (curry, egg custard, fruit)."
        ),
        "tags": ["vegetarian", "high_protein"],
        "use_high_protein_from_neighborhood": False,
        "box": {
            "protein_min": 0.26,
            "protein_max": 0.30,
            "carb_min": 0.26,
            "carb_max": 0.30,
            "fat_min": 0.42,
            "fat_max": 0.46,
        },
    },
    {
        "case_key": "no_pork_bbq_ribs",
        "title": "BBQ Ribs",
        "canonical_id": 35,
        "request": (
            "Higher-protein BBQ-style ribs without pork: about 32% protein, 39% carbs, "
            "29% fat. Use beef or poultry ribs/meat. No pork."
        ),
        "tags": ["no_pork", "high_protein"],
        "use_high_protein_from_neighborhood": True,
    },
    {
        "case_key": "no_pork_al_pastor",
        "title": "Al Pastor",
        "canonical_id": 10,
        "request": (
            "Higher-protein al pastor without pork: about 32% protein. "
            "Keep chili-pineapple profile. No pork."
        ),
        "tags": ["no_pork", "high_protein"],
        "use_high_protein_from_neighborhood": True,
    },
    {
        "case_key": "gluten_free_focaccia",
        "title": "Focaccia",
        "canonical_id": 187,
        "request": (
            "Gluten-free focaccia: about 18% protein, 40% carbs, 42% fat. "
            "No wheat/gluten. Keep olive-oil bread identity."
        ),
        "tags": ["gluten_free"],
        "box": {
            "protein_min": 0.16,
            "protein_max": 0.20,
            "carb_min": 0.38,
            "carb_max": 0.42,
            "fat_min": 0.40,
            "fat_max": 0.44,
        },
    },
    {
        "case_key": "gluten_free_fried_rice",
        "title": "Fried Rice",
        "canonical_id": 193,
        "request": (
            "Higher-protein gluten-free fried rice: about 26% protein, 44% carbs, 30% fat. "
            "No wheat soy sauce / gluten. Keep fried-rice identity."
        ),
        "tags": ["gluten_free", "high_protein"],
        "use_high_protein_from_neighborhood": True,
    },
    {
        "case_key": "no_dairy_avgolemono",
        "title": "Avgolemono Soup",
        "canonical_id": 30,
        "request": (
            "Higher-protein avgolemono without dairy: about 30% protein. "
            "Egg-lemon chicken soup identity. No dairy."
        ),
        "tags": ["no_dairy", "high_protein"],
        "use_high_protein_from_neighborhood": True,
    },
    {
        "case_key": "vegan_baba_ganoush",
        "title": "Baba Ganoush",
        "canonical_id": 36,
        "request": (
            "Higher-protein vegan baba ganoush: about 20% protein. "
            "No animal products. Keep eggplant-tahini identity."
        ),
        "tags": ["vegan", "high_protein"],
        "use_high_protein_from_neighborhood": True,
    },
    {
        "case_key": "vegetarian_enchiladas",
        "title": "Beef Enchiladas",
        "canonical_id": 52,
        "request": (
            "Vegetarian enchiladas: about 28% protein, 35% carbs, 37% fat. "
            "No meat. Keep enchilada identity (tortilla, sauce, cheese ok)."
        ),
        "tags": ["vegetarian", "high_protein"],
        "use_high_protein_from_neighborhood": True,
    },
    {
        "case_key": "no_beef_bourguignon",
        "title": "Beef Bourguignon",
        "canonical_id": 51,
        "request": (
            "Higher-protein bourguignon-style stew without beef: about 32% protein. "
            "Use poultry or lamb. No beef. Keep wine-braise identity."
        ),
        "tags": ["no_beef", "high_protein"],
        "use_high_protein_from_neighborhood": True,
    },
]


# Suite D — cookability under constraint. Slide-ready hooks explain the story.
COOKABILITY_CASES: list[dict[str, Any]] = [
    {
        "case_key": "hp_carbonara_cookable",
        "title": "Carbonara",
        "canonical_id": None,
        "presentation_hook": (
            "Slide: 'Would you cook this?' High-protein carbonara — frontier models "
            "often dump soft dairy or absurd seasoning; agent must keep pasta/egg/cheese "
            "with kitchen-scale grams."
        ),
        "request": (
            "Higher-protein carbonara: about 28% protein, 40% carbs, 32% fat "
            "(calorie shares). Keep it something I'd actually cook tonight — "
            "realistic spice amounts, keep pasta, egg, and hard cheese."
        ),
        "tags": ["high_protein"],
        "box": {
            "protein_min": 0.26,
            "protein_max": 0.30,
            "carb_min": 0.38,
            "carb_max": 0.42,
            "fat_min": 0.30,
            "fat_max": 0.34,
        },
        "agent_mode_name": "creative",
        "identity_staples": ["pasta", "egg", "cheese"],
    },
    {
        "case_key": "hp_focaccia_cookable",
        "title": "Focaccia",
        "canonical_id": 187,
        "presentation_hook": (
            "Slide: Coffee-in-bread / 70g yeast dumps. High-protein focaccia must stay "
            "olive-oil bread with cookable dough quantities."
        ),
        "request": (
            "Higher-protein focaccia hitting the macro box. Keep olive-oil bread identity. "
            "Something I'd bake tonight — no weird add-ins, realistic yeast/salt/herb grams."
        ),
        "tags": ["high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative_example",
        "identity_staples": ["flour", "oil", "yeast"],
    },
    {
        "case_key": "hp_avgolemono_cookable",
        "title": "Avgolemono Soup",
        "canonical_id": 30,
        "presentation_hook": (
            "Slide: Herb powder by the cup. High-protein avgolemono must stay egg-lemon "
            "soup with spoonable seasoning, not 100g dill."
        ),
        "request": (
            "Higher-protein avgolemono (egg-lemon chicken soup) hitting the macro box. "
            "Keep it cookable — normal herb/spice amounts, keep egg and lemon character."
        ),
        "tags": ["high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative_example",
        "identity_staples": ["egg", "lemon", "chicken"],
    },
    {
        "case_key": "hp_bbq_ribs_cookable",
        "title": "BBQ Ribs",
        "canonical_id": 35,
        "presentation_hook": (
            "Slide: Rub as the main ingredient. High-protein BBQ ribs — agent should "
            "scale meat, not dump 80g onion powder."
        ),
        "request": (
            "Higher-protein BBQ ribs hitting the macro box. Keep smoked-rib identity. "
            "Cookable tonight — meat is the star; rub/spice in tablespoons not cups."
        ),
        "tags": ["high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative_example",
        "identity_staples": ["pork", "rib", "meat"],
        "require_protein_line": True,
    },
    {
        "case_key": "no_pork_ribs_cookable",
        "title": "BBQ Ribs",
        "canonical_id": 35,
        "presentation_hook": (
            "Slide: Diet swap that forgets protein. No-pork BBQ ribs must still have a "
            "real meat/protein line — not sauce + spices alone."
        ),
        "request": (
            "Higher-protein BBQ-style ribs without pork: use beef or poultry. No pork. "
            "Hit the macro box. Keep it cookable with a clear protein centerpiece."
        ),
        "tags": ["no_pork", "high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative",
        "require_protein_line": True,
        "identity_staples": ["meat", "rib", "beef", "chicken", "turkey"],
    },
    {
        "case_key": "gf_focaccia_cookable",
        "title": "Focaccia",
        "canonical_id": 187,
        "presentation_hook": (
            "Slide: Gluten-free bread missing flour. GF focaccia must still be a dough "
            "(flour alternative present), not oil + salt."
        ),
        "request": (
            "Gluten-free focaccia hitting about 18% protein / 40% carbs / 42% fat. "
            "No wheat/gluten. Keep olive-oil bread identity with a real flour base I'd bake."
        ),
        "tags": ["gluten_free"],
        "box": {
            "protein_min": 0.16,
            "protein_max": 0.20,
            "carb_min": 0.38,
            "carb_max": 0.42,
            "fat_min": 0.40,
            "fat_max": 0.44,
        },
        "agent_mode_name": "creative",
        "identity_staples": ["flour", "starch", "rice flour", "almond", "oil"],
    },
    {
        "case_key": "hp_bourguignon_cookable",
        "title": "Beef Bourguignon",
        "canonical_id": 51,
        "presentation_hook": (
            "Slide: Stew you can ladle. High-protein bourguignon — wine braise with "
            "meat + veg at stew scales, not spice as bulk."
        ),
        "request": (
            "Higher-protein beef bourguignon hitting the macro box. Keep wine-braise stew "
            "identity. Cookable — realistic salt/herb grams; beef and vegetables dominate mass."
        ),
        "tags": ["high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative_example",
        "require_protein_line": True,
        "identity_staples": ["beef", "wine", "onion"],
    },
    {
        "case_key": "no_beef_bourguignon_cookable",
        "title": "Beef Bourguignon",
        "canonical_id": 51,
        "presentation_hook": (
            "Slide: 'No beef' that forgot the protein. Swap must still put poultry/lamb "
            "on the plate at stew scale."
        ),
        "request": (
            "Higher-protein bourguignon-style stew without beef — use poultry or lamb. "
            "No beef. Hit macros. Keep wine-braise identity with a real protein centerpiece."
        ),
        "tags": ["no_beef", "high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative",
        "require_protein_line": True,
        "identity_staples": ["chicken", "turkey", "lamb", "meat", "wine"],
    },
]


# Suite E — taste preference × macros. One crisp taste ask per slide + protein box.
TASTE_MACRO_CASES: list[dict[str, Any]] = [
    {
        "case_key": "focaccia_lighter",
        "title": "Focaccia",
        "canonical_id": 187,
        "presentation_hook": (
            "Slide: Taste = lighter/less oily + still hit protein. Can the system "
            "honor 'less rich' without wrecking bread identity or the macro box?"
        ),
        "taste_preference": "lighter and less oily / less rich mouthfeel",
        "request": (
            "Higher-protein focaccia hitting the macro box, but make it lighter and less "
            "oily than a typical bakery focaccia — less rich mouthfeel while staying "
            "recognizable olive-oil bread."
        ),
        "tags": ["high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative_example",
    },
    {
        "case_key": "bbq_ribs_smokier",
        "title": "BBQ Ribs",
        "canonical_id": 35,
        "presentation_hook": (
            "Slide: Taste = smokier BBQ + high protein. Preference must show up as "
            "smoke/char profile — not a cup of paprika."
        ),
        "taste_preference": "smokier / more smoked-char flavor",
        "request": (
            "Higher-protein BBQ ribs hitting the macro box. Make them smokier — more "
            "smoked-char flavor — while keeping cookable rub amounts and rib identity."
        ),
        "tags": ["high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative_example",
    },
    {
        "case_key": "avgolemono_brighter",
        "title": "Avgolemono Soup",
        "canonical_id": 30,
        "presentation_hook": (
            "Slide: Taste = brighter lemon + high protein. Classic 'more lemon/herb' "
            "ask without turning the pot into dried dill."
        ),
        "taste_preference": "brighter, more lemon-forward",
        "request": (
            "Higher-protein avgolemono hitting the macro box. Make it brighter and more "
            "lemon-forward while keeping egg-lemon soup identity and normal herb amounts."
        ),
        "tags": ["high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative_example",
    },
    {
        "case_key": "bourguignon_weeknight",
        "title": "Beef Bourguignon",
        "canonical_id": 51,
        "presentation_hook": (
            "Slide: Taste/lifestyle = weeknight-simpler stew + protein target. "
            "Fewer specialty extras; still wine-braise and on-box."
        ),
        "taste_preference": "weeknight-simpler / pantry-leaner",
        "request": (
            "Higher-protein beef bourguignon hitting the macro box. Make it weeknight-"
            "simpler and pantry-leaner — fewer specialty extras — while keeping wine-braise "
            "stew identity."
        ),
        "tags": ["high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative_example",
    },
    {
        "case_key": "grape_leaves_less_oily",
        "title": "Stuffed Grape Leaves",
        "canonical_id": 449,
        "presentation_hook": (
            "Slide: Taste = less oily, brighter herb + protein. Common healthy-taste "
            "ask on a rich stuffed dish."
        ),
        "taste_preference": "less oily, brighter herb",
        "request": (
            "Higher-protein stuffed grape leaves hitting the macro box. Make them less "
            "oily and brighter/herb-forward while keeping dolma identity."
        ),
        "tags": ["high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative_example",
    },
    {
        "case_key": "al_pastor_milder",
        "title": "Al Pastor",
        "canonical_id": 10,
        "presentation_hook": (
            "Slide: Taste = milder heat, keep pineapple-chili. Preference without "
            "erasing identity or missing protein."
        ),
        "taste_preference": "milder heat; keep chili-pineapple",
        "request": (
            "Higher-protein al pastor hitting the macro box. Make the heat milder for "
            "family dinner, but keep the chili-pineapple profile recognizable."
        ),
        "tags": ["high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative_example",
    },
    {
        "case_key": "carbonara_lighter",
        "title": "Carbonara",
        "canonical_id": None,
        "presentation_hook": (
            "Slide: Taste = lighter carbonara + protein — NOT yogurt/ricotta hacks. "
            "Tests clash gates vs soft-dairy protein cheats."
        ),
        "taste_preference": "lighter, less heavy; not cream-sauce-like",
        "request": (
            "Higher-protein carbonara: about 28% protein, 40% carbs, 32% fat. Make it "
            "feel lighter and less heavy — not cream-sauce-like — while keeping pasta, "
            "egg, and hard cheese. No yogurt/ricotta protein hacks."
        ),
        "tags": ["high_protein"],
        "box": {
            "protein_min": 0.26,
            "protein_max": 0.30,
            "carb_min": 0.38,
            "carb_max": 0.42,
            "fat_min": 0.30,
            "fat_max": 0.34,
        },
        "agent_mode_name": "creative",
    },
    {
        "case_key": "fried_rice_garlicky",
        "title": "Fried Rice",
        "canonical_id": 193,
        "presentation_hook": (
            "Slide: Taste = more garlicky/aromatic + protein. Everyday preference "
            "language users actually type."
        ),
        "taste_preference": "more garlicky / aromatic",
        "request": (
            "Higher-protein fried rice hitting the macro box. Make it more garlicky and "
            "aromatic while keeping fried-rice identity and cookable seasoning amounts."
        ),
        "tags": ["high_protein"],
        "use_high_protein_from_neighborhood": True,
        "agent_mode_name": "creative_example",
    },
]


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def _slug(title: str) -> str:
    import re

    s = re.sub(r"[^a-z0-9]+", "_", (title or "").lower()).strip("_")
    return (s[:48] or "dish")


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _user_request_high_protein(title: str, box: dict[str, float]) -> str:
    p = 100.0 * 0.5 * (box["protein_min"] + box["protein_max"])
    c = 100.0 * 0.5 * (box["carb_min"] + box["carb_max"])
    f = 100.0 * 0.5 * (box["fat_min"] + box["fat_max"])
    return (
        f"Higher-protein {title}: about {p:.0f}% protein, {c:.0f}% carbs, {f:.0f}% fat "
        f"(calorie shares). Keep the dish recognizable but boost protein relative to a "
        f"typical neighborhood version."
    )


_SEASONING_TOKENS = (
    "powder",
    "spice",
    "spices",
    "seasoning",
    "cumin",
    "paprika",
    "chili powder",
    "onion powder",
    "garlic powder",
    "yeast",
    "extract",
    "dill weed",
    "oregano",
    "thyme, dried",
    "basil, dried",
    "pepper, red or cayenne",
    "curry powder",
    "mustard powder",
)
_PROTEIN_TOKENS = (
    "chicken",
    "turkey",
    "beef",
    "pork",
    "lamb",
    "fish",
    "salmon",
    "tuna",
    "shrimp",
    "tofu",
    "tempeh",
    "egg",
    "meat",
    "rib",
    "steak",
    "thigh",
    "breast",
    "bean",
    "lentil",
    "chickpea",
)


def _cookability_metrics(
    payload: dict[str, Any],
    *,
    case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic kitchen-plausibility checks for Suite D (and as soft context elsewhere)."""
    ings = list(
        (payload.get("chosen_recipe") or {}).get("ingredients")
        or (payload.get("problem") or {}).get("chosen_recipe", {}).get("ingredients")
        or []
    )
    total = 0.0
    nonsense: list[dict[str, Any]] = []
    token_staples: list[dict[str, Any]] = []
    for row in ings:
        label = str(row.get("label") or row.get("name") or "").strip()
        try:
            grams = float(row.get("grams") or 0.0)
        except (TypeError, ValueError):
            grams = 0.0
        if grams < 0:
            grams = 0.0
        total += grams
        low = label.lower()
        is_seasoning = any(tok in low for tok in _SEASONING_TOKENS)
        if is_seasoning and (grams > 25.0 or (total > 0 and grams / max(total, 1e-9) > 0.05 and grams > 15.0)):
            nonsense.append({"label": label, "grams": grams})
        if 0 < grams < 1.0 and not is_seasoning:
            # Tiny non-seasoning lines look like optimizer tokens / dropped staples.
            token_staples.append({"label": label, "grams": grams})

    protein_lines = []
    for row in ings:
        label = str(row.get("label") or row.get("name") or "").lower()
        try:
            grams = float(row.get("grams") or 0.0)
        except (TypeError, ValueError):
            grams = 0.0
        if grams >= 20.0 and any(tok in label for tok in _PROTEIN_TOKENS):
            protein_lines.append({"label": row.get("label") or row.get("name"), "grams": grams})

    require_protein = bool((case or {}).get("require_protein_line"))
    missing_protein = require_protein and not protein_lines

    # Identity staples: at least one token match with >= 15g when listed on the case.
    staple_misses: list[str] = []
    for staple in (case or {}).get("identity_staples") or []:
        tok = str(staple).lower()
        hit = False
        for row in ings:
            label = str(row.get("label") or row.get("name") or "").lower()
            try:
                grams = float(row.get("grams") or 0.0)
            except (TypeError, ValueError):
                grams = 0.0
            if tok in label and grams >= 15.0:
                hit = True
                break
        if not hit:
            staple_misses.append(str(staple))

    cookability_fail = bool(nonsense) or missing_protein or bool(staple_misses)
    # Soft score: 0 = clean, higher = worse (for lower-better dims).
    cookability_badness = (
        float(len(nonsense))
        + (2.0 if missing_protein else 0.0)
        + 0.5 * float(len(staple_misses))
        + 0.25 * float(len(token_staples))
    )
    return {
        "nonsense_seasoning_flag": bool(nonsense),
        "nonsense_seasonings": nonsense,
        "token_staple_flag": bool(token_staples),
        "token_staples": token_staples[:8],
        "missing_protein_under_diet": missing_protein,
        "protein_lines": protein_lines[:6],
        "identity_staple_misses": staple_misses,
        "cookability_fail": cookability_fail,
        "cookability_badness": cookability_badness,
    }


def _taste_adherence_score(feval: dict[str, Any] | None) -> tuple[float | None, str | None]:
    """Map judge taste_preference_met → numeric (higher better)."""
    if not isinstance(feval, dict):
        return None, None
    raw = feval.get("taste_preference_met")
    if raw is None:
        return None, None
    s = str(raw).strip().lower()
    if s in {"yes", "true", "met"}:
        return 1.0, s
    if s in {"partially", "partial", "mostly"}:
        return 0.5, s
    if s in {"no", "false", "unmet"}:
        return 0.0, s
    return None, s


def _extract_needle_metrics(
    payload: dict[str, Any],
    *,
    case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from recipe_opt_agent.score_display import extract_ratio_and_nutrient

    display = payload.get("display_scores") or {}
    feval = payload.get("final_evaluation") or {}
    opt = payload.get("opt") or {}

    def _val(card: Any) -> float | None:
        if isinstance(card, dict):
            v = card.get("value")
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        return None

    # Prefer authoritative extraction: pasta∶egg ratio when samples exist, else
    # neighborhood mass-share fidelity (share_losses_sum). Avoids treating empty
    # ratio_samples + ratio_surrogate=0 as a real win.
    ratio, ratio_src, nutrient, nutrient_src = extract_ratio_and_nutrient(payload)
    if nutrient is None:
        nutrient = _val(display.get("nutrient_loss"))
    if nutrient is None and opt.get("nutrient_slack") is not None:
        try:
            nutrient = float(opt["nutrient_slack"])
            nutrient_src = nutrient_src or "opt_nutrient_slack"
        except (TypeError, ValueError):
            nutrient = None
    holistic = _val(display.get("holistic_0_10"))
    if holistic is None and feval.get("overall_score_0_10") is not None:
        try:
            holistic = float(feval["overall_score_0_10"])
        except (TypeError, ValueError):
            holistic = None

    pfc = opt.get("pfc_after") or {}
    cook = _cookability_metrics(payload, case=case)
    taste_score, taste_raw = _taste_adherence_score(feval if isinstance(feval, dict) else None)
    return {
        "ratio_loss": ratio,
        "ratio_loss_source": ratio_src,
        "nutrient_loss": nutrient,
        "nutrient_loss_source": nutrient_src,
        "holistic_0_10": holistic,
        "pfc_after": pfc,
        "dietary_violation_flag": bool(feval.get("dietary_violation_flag")),
        "odd_ingredients": list(feval.get("odd_ingredients") or []),
        "n_odd_ingredients": len(feval.get("odd_ingredients") or []),
        "strengths": feval.get("strengths"),
        "concerns": feval.get("concerns"),
        "judge_summary": feval.get("summary_markdown"),
        "taste_preference_met": taste_raw,
        "taste_adherence": taste_score,
        "labels": [
            str(i.get("label") or i.get("name") or "")
            for i in (payload.get("chosen_recipe") or {}).get("ingredients") or []
        ],
        **cook,
    }


def _in_box(pfc: dict[str, Any] | None, box: dict[str, float]) -> bool:
    if not pfc or not box:
        return False
    try:
        p, c, f = float(pfc["protein"]), float(pfc["carbs"]), float(pfc["fat"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        box["protein_min"] - 1e-6 <= p <= box["protein_max"] + 1e-6
        and box["carb_min"] - 1e-6 <= c <= box["carb_max"] + 1e-6
        and box["fat_min"] - 1e-6 <= f <= box["fat_max"] + 1e-6
    )


def _missing_high_hit_count(payload: dict[str, Any], *, min_hits: int = 8) -> int:
    from recipe_opt_agent.edit_grounding import missing_high_hit_basis_nodes

    foodon = payload.get("foodon_basis_report") or (payload.get("problem") or {}).get(
        "foodon_basis_report"
    )
    return len(missing_high_hit_basis_nodes(foodon, min_hits=min_hits))


def decide_winner(
    agent: dict[str, Any],
    competitor: dict[str, Any],
    *,
    box: dict[str, float],
    holistic_margin: float = 1.0,
    suite: str | None = None,
) -> dict[str, Any]:
    """Win rules:

    Default / A/B/C: ≥2 of {macro, ratio, safety, identity} OR holistic edge.
    D: ≥2 of {macro, ratio, cookability, identity}; cookability veto if only
       competitor fails kitchen-plausibility checks.
    E: ≥2 of {macro, ratio, taste_adherence, cookability} (identity demoted —
       taste preference is the user-facing dim).
    """

    def _lower_better(a: float | None, b: float | None) -> str | None:
        if a is None and b is None:
            return None
        if a is None:
            return "competitor"
        if b is None:
            return "agent"
        if abs(a - b) < 1e-12:
            return "tie"
        return "agent" if a < b else "competitor"

    def _higher_better(a: float | None, b: float | None) -> str | None:
        if a is None and b is None:
            return None
        if a is None:
            return "competitor"
        if b is None:
            return "agent"
        if abs(a - b) < 1e-12:
            return "tie"
        return "agent" if a > b else "competitor"

    agent_in = _in_box(agent.get("pfc_after"), box)
    comp_in = _in_box(competitor.get("pfc_after"), box)
    if agent_in != comp_in:
        macro = "agent" if agent_in else "competitor"
    else:
        macro = _lower_better(agent.get("nutrient_loss"), competitor.get("nutrient_loss"))

    ratio = _lower_better(agent.get("ratio_loss"), competitor.get("ratio_loss"))

    a_vio = bool(agent.get("dietary_violation_flag"))
    c_vio = bool(competitor.get("dietary_violation_flag"))
    if a_vio != c_vio:
        safety = "agent" if not a_vio else "competitor"
    else:
        safety = "tie"

    a_id = float(agent.get("n_odd_ingredients") or 0) + float(agent.get("n_missing_high_hit") or 0)
    c_id = float(competitor.get("n_odd_ingredients") or 0) + float(
        competitor.get("n_missing_high_hit") or 0
    )
    if abs(a_id - c_id) < 1e-9:
        identity = "tie"
    else:
        identity = "agent" if a_id < c_id else "competitor"

    a_cook_fail = bool(agent.get("cookability_fail"))
    c_cook_fail = bool(competitor.get("cookability_fail"))
    if a_cook_fail != c_cook_fail:
        cookability = "agent" if not a_cook_fail else "competitor"
    else:
        cookability = _lower_better(
            agent.get("cookability_badness"), competitor.get("cookability_badness")
        ) or "tie"

    taste = _higher_better(agent.get("taste_adherence"), competitor.get("taste_adherence"))

    suite_u = (suite or "").upper()
    if suite_u == "D":
        dimensions = {
            "macro_nutrient": macro,
            "ratio_loss": ratio,
            "cookability": cookability,
            "identity": identity,
        }
        dim_n = 4
        dim_label = "macro/ratio/cookability/identity"
    elif suite_u == "E":
        dimensions = {
            "macro_nutrient": macro,
            "ratio_loss": ratio,
            "taste_adherence": taste or "tie",
            "cookability": cookability,
        }
        dim_n = 4
        dim_label = "macro/ratio/taste/cookability"
    else:
        dimensions = {
            "macro_nutrient": macro,
            "ratio_loss": ratio,
            "safety_dietary": safety,
            "identity": identity,
        }
        dim_n = 4
        dim_label = "macro/ratio/safety/identity"

    agent_dims = sum(1 for v in dimensions.values() if v == "agent")
    comp_dims = sum(1 for v in dimensions.values() if v == "competitor")

    holistic = _higher_better(agent.get("holistic_0_10"), competitor.get("holistic_0_10"))
    hol_a = agent.get("holistic_0_10")
    hol_c = competitor.get("holistic_0_10")
    hol_gap = None
    if hol_a is not None and hol_c is not None:
        hol_gap = float(hol_a) - float(hol_c)

    winner = "tie"
    reason = "tied on structured dimensions"

    # Suite D hard trust veto: only competitor is uncookable → agent wins.
    if suite_u == "D" and c_cook_fail and not a_cook_fail:
        winner = "agent"
        reason = "competitor cookability_fail (kitchen-implausible) while agent is cookable"
    elif agent_dims >= 2 and agent_dims > comp_dims:
        winner = "agent"
        reason = f"agent better on {agent_dims}/{dim_n} structured dims ({dim_label})"
    elif comp_dims >= 2 and comp_dims > agent_dims:
        winner = "competitor"
        reason = f"competitor better on {comp_dims}/{dim_n} structured dims"
    elif hol_gap is not None and hol_gap >= holistic_margin and not a_vio:
        winner = "agent"
        reason = f"holistic margin {hol_gap:.1f} ≥ {holistic_margin} without dietary fail"
    elif hol_gap is not None and hol_gap <= -holistic_margin and not c_vio:
        winner = "competitor"
        reason = f"holistic margin {hol_gap:.1f} ≤ -{holistic_margin}"
    elif agent_dims > comp_dims:
        winner = "agent"
        reason = "narrow structured edge"
    elif comp_dims > agent_dims:
        winner = "competitor"
        reason = "narrow structured edge"

    return {
        "winner": winner,
        "reason": reason,
        "dimensions": dimensions,
        "agent_dim_wins": agent_dims,
        "competitor_dim_wins": comp_dims,
        "holistic_gap": hol_gap,
        "agent_in_box": agent_in,
        "competitor_in_box": comp_in,
        "ratio_loss_contextualizer": {
            "agent": agent.get("ratio_loss"),
            "competitor": competitor.get("ratio_loss"),
            "winner": ratio,
            "note": (
                "Ratio loss = pasta∶egg surrogate when neighborhood ratio_samples exist; "
                "otherwise sum of FoodOn mass-share fidelity losses. Explicit structured "
                "win dimension alongside nutrient/macro fit."
            ),
        },
        "cookability_contextualizer": {
            "agent_fail": a_cook_fail,
            "competitor_fail": c_cook_fail,
            "agent_badness": agent.get("cookability_badness"),
            "competitor_badness": competitor.get("cookability_badness"),
            "winner": cookability,
            "note": (
                "Cookability flags nonsense seasoning masses, missing protein centerpieces "
                "on swap cases, and missing identity staples."
            ),
        },
        "taste_contextualizer": {
            "agent": agent.get("taste_adherence"),
            "competitor": competitor.get("taste_adherence"),
            "agent_raw": agent.get("taste_preference_met"),
            "competitor_raw": competitor.get("taste_preference_met"),
            "winner": taste,
            "note": "Judge taste_preference_met mapped to yes=1 / partially=0.5 / no=0.",
        },
    }


def _build_case_catalog(
    *,
    suites: tuple[str, ...],
    n_dishes_a: int,
    n_dishes_c: int,
    min_neighborhood: int,
) -> list[dict[str, Any]]:
    from canonical_optimization import fetch_top_canonical_dishes
    from recipe_opt_agent.macro_target_suggestions import suggest_high_protein_targets_for_canonical

    dishes = fetch_top_canonical_dishes(limit=max(n_dishes_a, n_dishes_c, 20), min_neighborhood=min_neighborhood)
    rows = dishes.to_dict(orient="records")
    cases: list[dict[str, Any]] = []

    if "A" in suites:
        for row in rows[:n_dishes_a]:
            cid = int(row["canonical_recipe_id"])
            title = str(row.get("title") or f"canonical_{cid}")
            hp = suggest_high_protein_targets_for_canonical(cid, pad_pct=2)
            if hp.get("error"):
                continue
            box = hp["box"]
            cases.append(
                {
                    "case_id": f"A__{_slug(title)}__{cid}",
                    "suite": "A",
                    "suite_name": "macro_precision",
                    "title": title,
                    "canonical_id": cid,
                    "agent_mode_name": "creative_example",
                    "user_request": _user_request_high_protein(title, box),
                    "box": box,
                    "target_midpoint": hp["midpoint"],
                    "neighborhood_mean_pfc": hp.get("neighborhood_mean_pfc"),
                    "tags": ["high_protein"],
                }
            )

    if "B" in suites:
        for dc in DIETARY_CASES:
            cid = dc.get("canonical_id")
            title = dc["title"]
            box = dc.get("box")
            mid = None
            mean_pfc = None
            if dc.get("use_high_protein_from_neighborhood") and cid is not None:
                hp = suggest_high_protein_targets_for_canonical(int(cid), pad_pct=2)
                if not hp.get("error"):
                    box = hp["box"]
                    mid = hp["midpoint"]
                    mean_pfc = hp.get("neighborhood_mean_pfc")
                    # Rewrite request mid numbers lightly if neighborhood box used
                    req = _user_request_high_protein(title, box)
                    # Keep dietary clause from original
                    diet_bits = []
                    if "vegetarian" in dc["tags"]:
                        diet_bits.append("Vegetarian / no meat.")
                    if "vegan" in dc["tags"]:
                        diet_bits.append("Vegan / no animal products.")
                    if "no_pork" in dc["tags"]:
                        diet_bits.append("No pork.")
                    if "no_beef" in dc["tags"]:
                        diet_bits.append("No beef.")
                    if "no_dairy" in dc["tags"]:
                        diet_bits.append("No dairy.")
                    if "gluten_free" in dc["tags"]:
                        diet_bits.append("Gluten-free.")
                    req = f"{req} {' '.join(diet_bits)}"
                else:
                    req = dc["request"]
            else:
                req = dc["request"]
            if box is None:
                continue
            key = dc["case_key"]
            cases.append(
                {
                    "case_id": f"B__{key}",
                    "suite": "B",
                    "suite_name": "dietary_constraints",
                    "title": title,
                    "canonical_id": cid,
                    "agent_mode_name": "creative",
                    "user_request": req,
                    "box": box,
                    "target_midpoint": mid,
                    "neighborhood_mean_pfc": mean_pfc,
                    "tags": list(dc["tags"]),
                }
            )

    if "C" in suites:
        # Prefer dishes beyond the first few so A and C overlap less, but allow overlap.
        pool = rows[: max(n_dishes_c + 4, n_dishes_c)]
        picked = 0
        for row in pool:
            if picked >= n_dishes_c:
                break
            cid = int(row["canonical_recipe_id"])
            title = str(row.get("title") or f"canonical_{cid}")
            # Skip if already in A with same id to reduce cost — optional: allow
            hp = suggest_high_protein_targets_for_canonical(cid, pad_pct=2)
            if hp.get("error"):
                continue
            box = hp["box"]
            cases.append(
                {
                    "case_id": f"C__{_slug(title)}__{cid}",
                    "suite": "C",
                    "suite_name": "identity_under_stretch",
                    "title": title,
                    "canonical_id": cid,
                    "agent_mode_name": "creative_example",
                    "user_request": _user_request_high_protein(title, box)
                    + " Prefer traditional staples; only stretch protein with foods that appear in related recipes.",
                    "box": box,
                    "target_midpoint": hp["midpoint"],
                    "neighborhood_mean_pfc": hp.get("neighborhood_mean_pfc"),
                    "tags": ["high_protein"],
                }
            )
            picked += 1

    def _expand_static_case(dc: dict[str, Any], *, suite: str, suite_name: str) -> dict[str, Any] | None:
        cid = dc.get("canonical_id")
        title = dc["title"]
        box = dc.get("box")
        mid = None
        mean_pfc = None
        req = dc["request"]
        if dc.get("use_high_protein_from_neighborhood") and cid is not None:
            hp = suggest_high_protein_targets_for_canonical(int(cid), pad_pct=2)
            if not hp.get("error"):
                box = hp["box"]
                mid = hp["midpoint"]
                mean_pfc = hp.get("neighborhood_mean_pfc")
                # Keep the authored request (taste/cookability wording); only refresh
                # mid-box numbers into a short prefix when useful.
                p = 100.0 * 0.5 * (box["protein_min"] + box["protein_max"])
                if "hitting the macro box" in req.lower() or "about" not in req.lower()[:80]:
                    req = (
                        f"{req} Target about {p:.0f}% protein calories "
                        f"(±2% box from neighborhood stretch)."
                    )
        if box is None:
            return None
        out = {
            "case_id": f"{suite}__{dc['case_key']}",
            "suite": suite,
            "suite_name": suite_name,
            "title": title,
            "canonical_id": cid,
            "agent_mode_name": dc.get("agent_mode_name") or "creative",
            "user_request": req,
            "box": box,
            "target_midpoint": mid,
            "neighborhood_mean_pfc": mean_pfc,
            "tags": list(dc.get("tags") or []),
            "presentation_hook": dc.get("presentation_hook"),
            "require_protein_line": bool(dc.get("require_protein_line")),
            "identity_staples": list(dc.get("identity_staples") or []),
        }
        if dc.get("taste_preference"):
            out["taste_preference"] = dc["taste_preference"]
        return out

    if "D" in suites:
        for dc in COOKABILITY_CASES:
            row = _expand_static_case(
                dc, suite="D", suite_name="cookability_under_constraint"
            )
            if row:
                cases.append(row)

    if "E" in suites:
        for dc in TASTE_MACRO_CASES:
            row = _expand_static_case(
                dc, suite="E", suite_name="taste_preference_with_macros"
            )
            if row:
                cases.append(row)

    return cases


def _prepare_agent_problem(case: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from recipe_opt_agent.creative_loader import load_creative_problem
    from recipe_opt_agent.example_recipe import (
        attach_example_recipe_to_problem,
        pick_example_recipe_near_targets,
    )
    from canonical_optimization import CanonicalNeighborhood

    box = case["box"]
    cid = case.get("canonical_id")
    kwargs = dict(
        protein_min=box["protein_min"],
        protein_max=box["protein_max"],
        carb_min=box["carb_min"],
        carb_max=box["carb_max"],
        fat_min=box["fat_min"],
        fat_max=box["fat_max"],
    )
    problem = load_creative_problem(
        user_request=case["user_request"],
        canonical_id=int(cid) if cid is not None else None,
        offline=False,
        **kwargs,
    )
    mode = "creative"
    if case.get("agent_mode_name") == "creative_example" and cid is not None:
        mid = case.get("target_midpoint") or {
            "protein": 0.5 * (box["protein_min"] + box["protein_max"]),
            "carbs": 0.5 * (box["carb_min"] + box["carb_max"]),
            "fat": 0.5 * (box["fat_min"] + box["fat_max"]),
        }
        nb = CanonicalNeighborhood.build(int(cid), fast=True, use_cache=True)
        example = pick_example_recipe_near_targets(
            lines_df=nb.lines_df,
            recipe_ids=list(nb.recipe_ids),
            query=case["title"],
            target_mid=mid,
            target_box=box,
        )
        problem = attach_example_recipe_to_problem(problem, example)
        mode = "creative"
    return problem, mode


def _neighborhood_marginal_nodes(problem: dict[str, Any]) -> list[str]:
    """FoodOn nodes that have neighborhood share samples (not carbonara defaults)."""
    samples = problem.get("basis_samples") or {}
    preferred = list(problem.get("marginal_nodes") or [])
    if preferred:
        hit = [n for n in preferred if n in samples and len(samples.get(n) or []) > 0]
        if hit:
            return hit
    # Fall back to every basis node that has empirical samples.
    return [str(k) for k, v in samples.items() if v is not None and len(v) > 0]


def _optimize_problem(problem: dict[str, Any], box: dict[str, float]) -> dict[str, Any]:
    from weighted_empirical_opt import optimize_weighted_empirical_obj, term_losses

    x0 = np.asarray(problem["x0"], dtype=float)
    M = np.asarray(problem["M"], dtype=float)
    basis_samples = {
        str(k): np.asarray(v, dtype=float) for k, v in (problem.get("basis_samples") or {}).items()
    }
    ratio_samples = np.asarray(problem.get("ratio_samples") or [], dtype=float)
    marginal = _neighborhood_marginal_nodes(problem)
    ingredient_basis = list(problem.get("ingredient_basis") or [])
    weights_raw = problem.get("basis_sample_weights") or {}
    basis_sample_weights = {
        str(k): np.asarray(v, dtype=float) for k, v in weights_raw.items()
    } if weights_raw else None
    opt = optimize_weighted_empirical_obj(
        x0,
        M,
        marginal_nodes=marginal,
        basis_samples=basis_samples,
        ratio_samples=ratio_samples,
        ingredient_basis=ingredient_basis,
        kcal_target=float(problem.get("kcal_target") or 0.0) or None,
        protein_frac_min=box["protein_min"],
        protein_frac_max=box["protein_max"],
        carb_frac_min=box["carb_min"],
        carb_frac_max=box["carb_max"],
        fat_frac_min=box["fat_min"],
        fat_frac_max=box["fat_max"],
        total_mass=float(problem.get("total_mass") or float(x0.sum())),
        basis_sample_weights=basis_sample_weights,
        nutrition_slack_weight=1.0,
    )
    x_opt = np.asarray(opt["x_opt"], dtype=float)
    tl = term_losses(
        x_opt,
        marginal_nodes=marginal,
        basis_samples=basis_samples,
        ratio_samples=ratio_samples,
        total_mass=float(x_opt.sum()),
        ingredient_basis=ingredient_basis,
        basis_sample_weights=basis_sample_weights,
    )
    from weighted_empirical_opt import pfc_fractions_from_portions

    p, c, f = pfc_fractions_from_portions(x_opt, M)
    # Update chosen recipe grams to x_opt
    chosen = dict(problem.get("chosen_recipe") or {})
    ings = list(chosen.get("ingredients") or [])
    for i, row in enumerate(ings):
        if i < len(x_opt):
            row = dict(row)
            row["grams"] = float(x_opt[i])
            ings[i] = row
    chosen["ingredients"] = ings
    problem = dict(problem)
    problem["chosen_recipe"] = chosen
    problem["x_opt"] = x_opt.tolist()
    opt_pub = {
        "status": opt.get("status"),
        "objective": float(opt.get("objective") or 0.0),
        "nutrient_slack": float(opt.get("nutrient_slack") or 0.0),
        "feasible": bool(opt.get("feasible")),
        "x_opt": x_opt.tolist(),
        "term_losses": {k: float(v) for k, v in tl.items()},
        "pfc_after": {"protein": float(p), "carbs": float(c), "fat": float(f)},
    }
    return {"problem": problem, "opt": opt_pub, "chosen_recipe": chosen}


def run_competitor(
    case: dict[str, Any],
    *,
    model: str,
    case_dir: Path,
) -> dict[str, Any]:
    """One-shot GPT-5.5 draft → ground → optimize → judge."""
    from recipe_opt_agent.creative_loader import load_creative_problem
    from recipe_opt_agent.final_evaluator import dietary_precheck, evaluate_final_recipe
    from recipe_opt_agent.grounding import ground_draft_to_problem
    from recipe_opt_agent.llm import llm_draft_recipe
    from recipe_opt_agent.requirement_tags import RequirementTag, deduce_requirement_tags
    from recipe_opt_agent.score_display import build_display_scores

    box = case["box"]
    cid = case.get("canonical_id")
    t0 = time.perf_counter()
    stub = load_creative_problem(
        user_request=case["user_request"],
        canonical_id=int(cid) if cid is not None else None,
        offline=False,
        protein_min=box["protein_min"],
        protein_max=box["protein_max"],
        carb_min=box["carb_min"],
        carb_max=box["carb_max"],
        fat_min=box["fat_min"],
        fat_max=box["fat_max"],
    )
    # gpt-5.5 rejects non-default temperature; omit it for that family.
    draft_temp: float | None = None if str(model).startswith("gpt-5") else 0.2
    draft, draft_trace = llm_draft_recipe(
        case["user_request"],
        macro_box=box,
        example_recipe=None,
        model=model,
        temperature=draft_temp,
    )
    tags = deduce_requirement_tags(
        case["user_request"],
        draft_tags=draft.get("requirement_tags"),
        force_llm=False,
    )
    # Ensure explicit dietary tags from case catalog
    have = {t.tag_id for t in tags}
    for tid in case.get("tags") or []:
        if tid in have:
            continue
        kind = "macro_intent" if tid == "high_protein" else "dietary_restriction"
        tags.append(RequirementTag(tid, kind, "require", source_text=case["user_request"]))

    ctx = stub.get("retrieval_context") or {}
    nb_cat = list(ctx.get("fdc_catalog") or [])
    # Keep FoodOn neighborhood geometry on the retrieval context so grounding
    # can remap ingredient_basis onto neighborhood nodes.
    if stub.get("basis_nodes") and not ctx.get("basis_nodes"):
        ctx["basis_nodes"] = list(stub["basis_nodes"])
    if stub.get("rollup_chains") and not ctx.get("rollup_chains"):
        ctx["rollup_chains"] = stub["rollup_chains"]
    if stub.get("fdc_basis") and not ctx.get("fdc_basis"):
        ctx["fdc_basis"] = stub["fdc_basis"]
    problem, report, chosen = ground_draft_to_problem(
        draft,
        requirement_tags=tags,
        neighborhood_catalog=nb_cat,
        broader_catalog=nb_cat,
        basis_samples=stub.get("basis_samples"),
        ratio_samples=stub.get("ratio_samples"),
        retrieval_context=ctx,
        offline=False,
    )
    # Overlay canonical neighborhood fidelity geometry. Grounding alone may leave
    # carbonara-default marginals that miss FoodOn share samples.
    if stub.get("basis_samples"):
        problem["basis_samples"] = stub["basis_samples"]
    if stub.get("basis_sample_weights"):
        problem["basis_sample_weights"] = stub["basis_sample_weights"]
    if stub.get("ratio_samples") is not None:
        problem["ratio_samples"] = stub["ratio_samples"]
    if stub.get("marginal_nodes"):
        problem["marginal_nodes"] = list(stub["marginal_nodes"])
    if stub.get("foodon_basis_report") and not problem.get("foodon_basis_report"):
        problem["foodon_basis_report"] = stub["foodon_basis_report"]
    if stub.get("neighborhood_hull_context") and not problem.get("neighborhood_hull_context"):
        problem["neighborhood_hull_context"] = stub["neighborhood_hull_context"]
    scored = _optimize_problem(problem, box)
    problem = scored["problem"]
    opt = scored["opt"]
    chosen = scored["chosen_recipe"]

    payload = {
        "system": "competitor",
        "model": model,
        "title": case["title"],
        "user_request": case["user_request"],
        "chosen_recipe": chosen,
        "opt": opt,
        "problem": problem,
        "grounding_report": report.to_dict(),
        "requirement_tags": [t.to_dict() for t in tags],
        "foodon_basis_report": problem.get("foodon_basis_report"),
        "taste_preference": case.get("taste_preference"),
    }
    payload["display_scores"] = build_display_scores(payload)
    # Attach dietary precheck into a mini final_evaluation even before LLM judge
    pre = dietary_precheck(chosen.get("ingredients") or [], tags)
    state = {
        "title": case["title"],
        "user_request": case["user_request"],
        "requirement_tags": [t.to_dict() for t in tags],
        "problem": problem,
        "config": {
            "protein_min": box["protein_min"],
            "protein_max": box["protein_max"],
            "carb_min": box["carb_min"],
            "carb_max": box["carb_max"],
            "fat_min": box["fat_min"],
            "fat_max": box["fat_max"],
        },
        "identity_roles": [],
        "taste_preference": case.get("taste_preference"),
    }
    try:
        feval = evaluate_final_recipe(state, payload, model=DEFAULT_JUDGE_MODEL)
    except Exception as exc:
        feval = {
            "overall_score_0_10": None,
            "dietary_violation_flag": pre.get("dietary_violation_flag"),
            "odd_ingredients": [],
            "summary_markdown": f"Judge failed: {exc}",
            "error": str(exc),
        }
    payload["final_evaluation"] = feval
    # Refresh holistic onto display
    display = dict(payload.get("display_scores") or {})
    if feval.get("overall_score_0_10") is not None:
        display["holistic_0_10"] = {
            "value": float(feval["overall_score_0_10"]),
            "band": "info",
            "label": "Holistic",
            "source": f"{DEFAULT_JUDGE_MODEL}_final_evaluator",
        }
        payload["display_scores"] = display

    metrics = _extract_needle_metrics(payload, case=case)
    metrics["n_missing_high_hit"] = _missing_high_hit_count(payload)
    metrics["in_box"] = _in_box(metrics.get("pfc_after"), box)
    metrics["elapsed_s"] = round(time.perf_counter() - t0, 2)
    metrics["system"] = "competitor"
    metrics["model"] = model

    out = {
        "draft": draft,
        "draft_trace": {k: v for k, v in draft_trace.items() if k != "messages"}
        if isinstance(draft_trace, dict)
        else draft_trace,
        "payload": payload,
        "metrics": metrics,
    }
    _atomic_write_json(case_dir / "competitor.json", out)
    return out


def run_agent_arm(
    case: dict[str, Any],
    *,
    max_iterations: int,
    case_dir: Path,
    suite_recorder: Any,
) -> dict[str, Any]:
    from recipe_opt_agent.config import AgentConfig
    from recipe_opt_agent.eval_artifacts import run_agent_with_artifacts

    box = case["box"]
    t0 = time.perf_counter()
    problem, agent_mode = _prepare_agent_problem(case)
    if case.get("taste_preference"):
        problem["taste_preference"] = case["taste_preference"]
    cfg = AgentConfig(
        protein_min=box["protein_min"],
        protein_max=box["protein_max"],
        carb_min=box["carb_min"],
        carb_max=box["carb_max"],
        fat_min=box["fat_min"],
        fat_max=box["fat_max"],
        max_iterations=max_iterations,
        F_accept=1.0,
        F_max=1.5,
        agent_mode=agent_mode,
    )
    result, rec = run_agent_with_artifacts(
        problem=problem,
        case_name=f"{case['case_id']}__agent",
        agent_mode=agent_mode,
        suite=suite_recorder,
        taste_text=case["user_request"],
        title=case["title"],
        user_request=case["user_request"],
        canonical_id=case.get("canonical_id"),
        config=cfg,
        extra_tags=["agent_vs_gpt55", case["suite"], "agent"],
        metadata={
            "case_id": case["case_id"],
            "suite": case["suite"],
            "macro_targets": box,
            "arm": "agent",
            "taste_preference": case.get("taste_preference"),
            "presentation_hook": case.get("presentation_hook"),
        },
    )
    metrics = _extract_needle_metrics(result, case=case)
    metrics["n_missing_high_hit"] = _missing_high_hit_count(result)
    metrics["in_box"] = _in_box(metrics.get("pfc_after"), box)
    metrics["elapsed_s"] = round(time.perf_counter() - t0, 2)
    metrics["system"] = "agent"
    metrics["status"] = result.get("status")
    metrics["run_dir"] = str(rec.run_dir)

    out = {
        "run_dir": str(rec.run_dir),
        "run_id": rec.run_id,
        "metrics": metrics,
        "result_keys": sorted(result.keys()),
        "final_evaluation": result.get("final_evaluation"),
        "display_scores": result.get("display_scores"),
        "chosen_recipe": result.get("chosen_recipe"),
        "opt": result.get("opt"),
        "foodon_basis_report": result.get("foodon_basis_report"),
        "identity_roles": result.get("identity_roles"),
        "grounding_report": (result.get("problem") or {}).get("grounding_report"),
    }
    _atomic_write_json(case_dir / "agent.json", out)
    # Also keep a compact result pointer
    _atomic_write_json(
        case_dir / "agent_metrics.json",
        {"metrics": metrics, "run_dir": str(rec.run_dir)},
    )
    return out


def _load_checkpoint(suite_dir: Path) -> dict[str, Any]:
    path = suite_dir / "checkpoint.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"completed_case_ids": [], "failed_case_ids": [], "comparisons": []}


def _save_checkpoint(suite_dir: Path, ckpt: dict[str, Any]) -> None:
    ckpt = dict(ckpt)
    ckpt["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(suite_dir / "checkpoint.json", ckpt)


def _aggregate(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    by_suite: dict[str, dict[str, Any]] = {}
    overall = {"agent": 0, "competitor": 0, "tie": 0, "n": 0, "errors": 0}
    for row in comparisons:
        suite = row.get("suite") or "?"
        by_suite.setdefault(
            suite, {"agent": 0, "competitor": 0, "tie": 0, "n": 0, "errors": 0}
        )
        if row.get("error"):
            by_suite[suite]["errors"] += 1
            overall["errors"] += 1
            continue
        w = (row.get("winner") or {}).get("winner") or "tie"
        by_suite[suite][w] = by_suite[suite].get(w, 0) + 1
        by_suite[suite]["n"] += 1
        overall[w] = overall.get(w, 0) + 1
        overall["n"] += 1

        # Mean needles
        for arm, key in (("agent", "agent_metrics"), ("competitor", "competitor_metrics")):
            m = row.get(key) or {}
            for metric in ("ratio_loss", "nutrient_loss", "holistic_0_10"):
                bucket = f"mean_{arm}_{metric}"
                if m.get(metric) is None:
                    continue
                by_suite[suite].setdefault(bucket + "_sum", 0.0)
                by_suite[suite].setdefault(bucket + "_n", 0)
                by_suite[suite][bucket + "_sum"] += float(m[metric])
                by_suite[suite][bucket + "_n"] += 1
                overall.setdefault(bucket + "_sum", 0.0)
                overall.setdefault(bucket + "_n", 0)
                overall[bucket + "_sum"] += float(m[metric])
                overall[bucket + "_n"] += 1

    def _finalize(block: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in block.items() if not k.endswith("_sum") and not k.endswith("_n")}
        for k, v in list(block.items()):
            if k.endswith("_sum"):
                base = k[: -len("_sum")]
                n = block.get(base + "_n") or 0
                out[base] = (v / n) if n else None
        return out

    return {
        "overall": _finalize(overall),
        "by_suite": {k: _finalize(v) for k, v in by_suite.items()},
    }


def run_suite(
    *,
    suites: tuple[str, ...] = ("A", "B", "C"),
    n_dishes_a: int = 12,
    n_dishes_c: int = 8,
    max_iterations: int = 2,
    min_neighborhood: int = 10,
    competitor_model: str = DEFAULT_COMPETITOR_MODEL,
    resume_dir: Path | None = None,
    name: str = SUITE_NAME,
) -> Path:
    from recipe_opt_agent.eval_artifacts import DEFAULT_EVAL_ROOT, EvalSuiteRecorder

    class FixedDirSuite(EvalSuiteRecorder):
        """EvalSuiteRecorder bound to an explicit suite directory (for resume)."""

        def __init__(self, path: Path, *, name: str):
            sid = path.name
            prefix = f"{name}_"
            suite_id = sid[len(prefix) :] if sid.startswith(prefix) else sid
            super().__init__(name=name, suite_id=suite_id, root=path.parent)
            self._fixed_dir = path

        @property
        def suite_dir(self) -> Path:  # type: ignore[override]
            return self._fixed_dir

    if resume_dir is not None:
        active_suite_dir = Path(resume_dir)
        if not active_suite_dir.exists():
            raise FileNotFoundError(active_suite_dir)
        recorder_suite = FixedDirSuite(active_suite_dir, name=name)
        (active_suite_dir / "runs").mkdir(exist_ok=True)
        ckpt = _load_checkpoint(active_suite_dir)
        catalog_path = active_suite_dir / "case_catalog.json"
        if catalog_path.exists():
            cases = json.loads(catalog_path.read_text())
        else:
            cases = _build_case_catalog(
                suites=suites,
                n_dishes_a=n_dishes_a,
                n_dishes_c=n_dishes_c,
                min_neighborhood=min_neighborhood,
            )
            _atomic_write_json(catalog_path, cases)
    else:
        recorder_suite = EvalSuiteRecorder(name=name, root=DEFAULT_EVAL_ROOT)
        active_suite_dir = recorder_suite.start()
        cases = _build_case_catalog(
            suites=suites,
            n_dishes_a=n_dishes_a,
            n_dishes_c=n_dishes_c,
            min_neighborhood=min_neighborhood,
        )
        _atomic_write_json(active_suite_dir / "case_catalog.json", cases)
        _atomic_write_json(
            active_suite_dir / "suite_meta.json",
            {
                "name": name,
                "suite_id": recorder_suite.suite_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "dir": str(active_suite_dir),
                "suites": list(suites),
                "competitor_model": competitor_model,
                "judge_model": DEFAULT_JUDGE_MODEL,
                "max_iterations": max_iterations,
                "n_cases": len(cases),
                "win_rule": (
                    "A/B/C: ≥2 of {macro, ratio, safety, identity} OR holistic≥1; "
                    "D: ≥2 of {macro, ratio, cookability, identity} OR competitor-only "
                    "cookability_fail veto; E: ≥2 of {macro, ratio, taste, cookability}; "
                    "ratio_loss always an explicit contextualizer"
                ),
            },
        )
        ckpt = {"completed_case_ids": [], "failed_case_ids": [], "comparisons": []}
        _save_checkpoint(active_suite_dir, ckpt)

    completed = set(ckpt.get("completed_case_ids") or [])
    failed_ids = list(ckpt.get("failed_case_ids") or [])
    comparisons = [
        c
        for c in (ckpt.get("comparisons") or [])
        if c.get("case_id") in completed and not c.get("error")
    ]
    print(f"Suite dir: {active_suite_dir}", flush=True)
    print(f"Cases: {len(cases)} · already done: {len(completed)}", flush=True)

    for i, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        if case_id in completed:
            print(f"\n[{i}/{len(cases)}] SKIP {case_id} (checkpoint)", flush=True)
            continue

        print(
            f"\n[{i}/{len(cases)}] {case_id} · {case['title']} · suite {case['suite']}",
            flush=True,
        )
        case_dir = active_suite_dir / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(case_dir / "case.json", case)

        row: dict[str, Any] = {
            "case_id": case_id,
            "suite": case["suite"],
            "title": case["title"],
            "canonical_id": case.get("canonical_id"),
            "box": case["box"],
            "user_request": case["user_request"],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            print("  → agent …", flush=True)
            agent_out = run_agent_arm(
                case,
                max_iterations=max_iterations,
                case_dir=case_dir,
                suite_recorder=recorder_suite,
            )
            print(
                f"     agent hol={agent_out['metrics'].get('holistic_0_10')} "
                f"ratio={agent_out['metrics'].get('ratio_loss')} "
                f"nut={agent_out['metrics'].get('nutrient_loss')} "
                f"({agent_out['metrics'].get('elapsed_s')}s)",
                flush=True,
            )

            print(f"  → competitor ({competitor_model}) …", flush=True)
            comp_out = run_competitor(case, model=competitor_model, case_dir=case_dir)
            print(
                f"     competitor hol={comp_out['metrics'].get('holistic_0_10')} "
                f"ratio={comp_out['metrics'].get('ratio_loss')} "
                f"nut={comp_out['metrics'].get('nutrient_loss')} "
                f"({comp_out['metrics'].get('elapsed_s')}s)",
                flush=True,
            )

            winner = decide_winner(
                agent_out["metrics"],
                comp_out["metrics"],
                box=case["box"],
                suite=case.get("suite"),
            )
            print(
                f"  → winner={winner['winner']} ({winner['reason']}) "
                f"dims={winner['dimensions']}",
                flush=True,
            )

            row.update(
                {
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "agent_metrics": agent_out["metrics"],
                    "competitor_metrics": comp_out["metrics"],
                    "winner": winner,
                    "agent_run_dir": agent_out.get("run_dir"),
                    "error": None,
                }
            )
            _atomic_write_json(case_dir / "comparison.json", row)
            completed.add(case_id)
            failed_ids = [x for x in failed_ids if x != case_id]
            comparisons = [c for c in comparisons if c.get("case_id") != case_id]
            comparisons.append(row)
            _append_jsonl(active_suite_dir / "progress.jsonl", row)
            ckpt = {
                "completed_case_ids": sorted(completed),
                "failed_case_ids": failed_ids,
                "comparisons": comparisons,
            }
            _save_checkpoint(active_suite_dir, ckpt)
            _atomic_write_json(
                active_suite_dir / "summary_partial.json",
                {
                    "n_completed": len(completed),
                    "n_total": len(cases),
                    "aggregates": _aggregate(comparisons),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  FAILED: {exc}", flush=True)
            row.update(
                {
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                    "traceback": tb,
                }
            )
            _atomic_write_json(case_dir / "comparison.json", row)
            _append_jsonl(active_suite_dir / "progress.jsonl", row)
            if case_id not in failed_ids:
                failed_ids.append(case_id)
            comparisons = [c for c in comparisons if c.get("case_id") != case_id]
            comparisons.append(row)
            ckpt = {
                "completed_case_ids": sorted(completed),
                "failed_case_ids": failed_ids,
                "comparisons": comparisons,
            }
            _save_checkpoint(active_suite_dir, ckpt)

    summary = {
        "suite_dir": str(active_suite_dir),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(cases),
        "n_completed": len(completed),
        "n_failed": len(failed_ids),
        "aggregates": _aggregate(comparisons),
        "comparisons": comparisons,
        "win_rule": (
            "A/B/C: ≥2 of {macro, ratio, safety, identity} OR holistic≥1; "
            "D: ≥2 of {macro, ratio, cookability, identity} OR competitor cookability veto; "
            "E: ≥2 of {macro, ratio, taste, cookability}"
        ),
    }
    _atomic_write_json(active_suite_dir / "summary.json", summary)
    print("\n=== AGGREGATES ===", flush=True)
    print(json.dumps(summary["aggregates"], indent=2), flush=True)
    print(f"Wrote {active_suite_dir / 'summary.json'}", flush=True)
    return active_suite_dir



def main() -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suites",
        type=str,
        default="A,B,C",
        help="Comma-separated subset of A,B,C,D,E",
    )
    p.add_argument("--n-dishes-a", type=int, default=12)
    p.add_argument("--n-dishes-c", type=int, default=8)
    p.add_argument("--max-iterations", type=int, default=2)
    p.add_argument("--min-neighborhood", type=int, default=10)
    p.add_argument("--competitor-model", type=str, default=DEFAULT_COMPETITOR_MODEL)
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to existing suite dir to resume",
    )
    p.add_argument("--name", type=str, default=SUITE_NAME)
    args = p.parse_args()
    suites = tuple(s.strip().upper() for s in args.suites.split(",") if s.strip())
    resume = Path(args.resume) if args.resume else None
    run_suite(
        suites=suites,
        n_dishes_a=args.n_dishes_a,
        n_dishes_c=args.n_dishes_c,
        max_iterations=args.max_iterations,
        min_neighborhood=args.min_neighborhood,
        competitor_model=args.competitor_model,
        resume_dir=resume,
        name=args.name,
    )


if __name__ == "__main__":
    main()
