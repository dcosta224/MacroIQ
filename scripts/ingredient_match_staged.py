"""Staged hybrid ingredient matching: lexical base identity + semantic prep ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from progress_utils import iter_progress, progress_enabled_for_count

# ---------------------------------------------------------------------------
# Text / token utilities
# ---------------------------------------------------------------------------

STOPWORDS = frozenset(
    """
    a an and as at by for from in into of on or per the to with without
    optional desired taste garnish serving
    """.split()
)

CONTAINER_WORDS = frozenset(
    """
    pkg pkg. package can cans jar jars bottle bottles box boxes bag bags
    carton stick sticks slice slices piece pieces wedge wedges
    """.split()
)

DEFAULT_PREP_TERMS = frozenset(
    """
    raw unseasoned boneless skinless generic uncooked fresh plain natural unprepared
    """.split()
)

SPECIAL_FLAVOR_TERMS = frozenset(
    """
    spicy thai bbq teriyaki breaded fried seasoned restaurant brand smoked
    honey garlic buffalo ranch cajun lemon pepper mesquite
    """.split()
)

PHYSICAL_FORM_TERMS = frozenset(
    """
    boneless skinless chopped ground sliced diced minced grated shredded
    cubed whole fillet breast thigh drumstick wing
    """.split()
)

COOKING_STATE_TERMS = frozenset(
    """
    raw cooked grilled fried roasted boiled baked steamed poached braised
    """.split()
)

PACKAGING_TERMS = frozenset(
    """
    canned frozen dried dehydrated powdered instant smoked cured
    """.split()
)

PREP_HINTS = PHYSICAL_FORM_TERMS | COOKING_STATE_TERMS | PACKAGING_TERMS | SPECIAL_FLAVOR_TERMS

IRREGULAR_PLURALS = {
    "leaves": "leaf",
    "tomatoes": "tomato",
    "potatoes": "potato",
    "cherries": "cherry",
    "berries": "berry",
}


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def simple_lemma(token: str) -> str:
    t = token.casefold()
    if t in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[t]
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es") and not t.endswith("ses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokenize(text: Any, *, drop_stop: bool = True) -> set[str]:
    raw = normalize_text(text).split()
    out: set[str] = set()
    for tok in raw:
        if drop_stop and tok in STOPWORDS:
            continue
        if tok in CONTAINER_WORDS:
            continue
        if len(tok) <= 1 and not tok.isdigit():
            continue
        out.add(simple_lemma(tok))
    return out


def signal_tokens(text: Any) -> frozenset[str]:
    """Content tokens for candidate dedup (stopwords / conjunctions excluded).

    Unlike ``tokenize()``, keeps words like ``piece`` that matter in product names
    (e.g. ``stems and pieces`` vs ``stems`` alone).
    """
    raw = normalize_text(text).split()
    out: set[str] = set()
    for tok in raw:
        if tok in STOPWORDS:
            continue
        if len(tok) <= 1 and not tok.isdigit():
            continue
        out.add(simple_lemma(tok))
    return frozenset(out)


def dedupe_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    description_key: str = "description",
    score_key: str = "retrieval_score",
    preserve_fdc_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Keep one candidate per identical signal-token set (highest score wins).

    Candidates like ``MUSHROOMS STEMS & PIECES`` vs ``MUSHROOMS STEMS AND PIECES``
    collapse to a single row; ``red wine`` vs ``white wine`` stay distinct.
    """
    if len(rows) <= 1:
        return rows

    preserve = preserve_fdc_ids or set()
    ranked = sorted(
        rows,
        key=lambda r: (
            -float(r.get(score_key) or 0.0),
            -float(r.get("staged_final_score") or 0.0),
            int(r.get("fdc_id") or 0),
        ),
    )
    seen: set[frozenset[str]] = set()
    kept: list[dict[str, Any]] = []
    for row in ranked:
        fid = row.get("fdc_id")
        if fid is not None and int(fid) in preserve:
            kept.append(row)
            continue
        sig = signal_tokens(str(row.get(description_key) or ""))
        if sig in seen:
            continue
        seen.add(sig)
        kept.append(row)

    return sorted(
        kept,
        key=lambda r: (
            -float(r.get(score_key) or 0.0),
            -float(r.get("staged_final_score") or 0.0),
            int(r.get("fdc_id") or 0),
        ),
    )


def classify_modifier_tokens(tokens: set[str]) -> dict[str, set[str]]:
    return {
        "physical_form": tokens & PHYSICAL_FORM_TERMS,
        "cooking_state": tokens & COOKING_STATE_TERMS,
        "flavor_style": tokens & SPECIAL_FLAVOR_TERMS,
        "packaging": tokens & PACKAGING_TERMS,
        "default_like": tokens & DEFAULT_PREP_TERMS,
    }


# ---------------------------------------------------------------------------
# Config (tunable from notebook)
# ---------------------------------------------------------------------------


@dataclass
class StagedMatchConfig:
    # Stage 1 — parsed name (lexical tokens + name embedding)
    base_name_lexical_weight: float = 0.80
    base_name_semantic_weight: float = 0.20
    # Stage 1 — dequantified / full-text fallback (text + dequant embedding)
    base_dequant_lexical_weight: float = 0.80
    base_dequant_semantic_weight: float = 0.20
    # Blend name channel vs dequant channel into base identity score.
    # Dequant-driven: the full dequantified text is the primary signal, the
    # parsed name only a secondary assist (was 0.70/0.30 name-dominant).
    base_name_channel_weight: float = 0.30
    base_dequant_channel_weight: float = 0.70
    # Legacy aliases (used when name/dequant weights left at defaults)
    base_lexical_weight: float = 0.80
    base_semantic_weight: float = 0.20
    stage1_top_k: int = 40

    # Stage 2 prep / version (semantic + lexical prep; rules/default fixed)
    prep_semantic_weight: float = 0.40
    prep_lexical_weight: float = 0.30
    prep_rule_weight: float = 0.20
    prep_default_bonus_weight: float = 0.10

    # Final blend
    final_base_weight: float = 0.65
    final_prep_weight: float = 0.25
    final_default_weight: float = 0.10
    unsupported_flavor_penalty: float = 0.25
    extra_modifier_penalty: float = 0.08

    # Quality tiers
    score_high: float = 0.75
    score_medium: float = 0.55
    score_low: float = 0.40
    margin_min: float = 0.08

    def with_lexical_semantic_grid(
        self,
        *,
        name_semantic: float,
        dequant_semantic: float,
        prep_semantic: float,
        prep_core_share: float = 0.70,
    ) -> StagedMatchConfig:
        """Copy config with lex/sem weights derived from semantic fractions (0–1)."""
        import dataclasses

        name_sem = max(0.0, min(1.0, name_semantic))
        dequant_sem = max(0.0, min(1.0, dequant_semantic))
        prep_sem = max(0.0, min(1.0, prep_semantic))
        prep_lex = 1.0 - prep_sem
        core = prep_core_share
        rest = 1.0 - core
        rule_default = self.prep_rule_weight + self.prep_default_bonus_weight
        if rule_default > 0:
            rule_frac = self.prep_rule_weight / rule_default
            default_frac = self.prep_default_bonus_weight / rule_default
        else:
            rule_frac, default_frac = 0.5, 0.5
        return dataclasses.replace(
            self,
            base_name_lexical_weight=1.0 - name_sem,
            base_name_semantic_weight=name_sem,
            base_dequant_lexical_weight=1.0 - dequant_sem,
            base_dequant_semantic_weight=dequant_sem,
            prep_semantic_weight=core * prep_sem,
            prep_lexical_weight=core * prep_lex,
            prep_rule_weight=rest * rule_frac,
            prep_default_bonus_weight=rest * default_frac,
        )

    def hp_fields(self) -> dict[str, float]:
        return {
            "base_name_lexical_weight": self.base_name_lexical_weight,
            "base_name_semantic_weight": self.base_name_semantic_weight,
            "base_dequant_lexical_weight": self.base_dequant_lexical_weight,
            "base_dequant_semantic_weight": self.base_dequant_semantic_weight,
            "base_name_channel_weight": self.base_name_channel_weight,
            "base_dequant_channel_weight": self.base_dequant_channel_weight,
            "prep_semantic_weight": self.prep_semantic_weight,
            "prep_lexical_weight": self.prep_lexical_weight,
            "prep_rule_weight": self.prep_rule_weight,
            "prep_default_bonus_weight": self.prep_default_bonus_weight,
        }


def config_slug(config: StagedMatchConfig) -> str:
    """Filesystem-safe id from lex/sem HP knobs."""
    return (
        f"nS{config.base_name_semantic_weight:.2f}"
        f"_dS{config.base_dequant_semantic_weight:.2f}"
        f"_pS{config.prep_semantic_weight / max(config.prep_semantic_weight + config.prep_lexical_weight, 1e-9):.2f}"
    )


# ---------------------------------------------------------------------------
# Food index
# ---------------------------------------------------------------------------


@dataclass
class FoodCandidate:
    fdc_id: int
    data_type: str
    description: str
    prefix: str
    base_tokens: set[str]
    modifier_tokens: set[str]
    modifiers_by_cat: dict[str, set[str]]
    name_embedding: np.ndarray | None = None
    prep_embedding: np.ndarray | None = None
    dequant_embedding: np.ndarray | None = None


@dataclass
class StagedFoodIndex:
    candidates: list[FoodCandidate]
    token_index: dict[str, set[int]]  # lemma → food idx; all tokens from full description
    name_matrix: np.ndarray | None = None
    dequant_matrix: np.ndarray | None = None
    config: StagedMatchConfig = field(default_factory=StagedMatchConfig)

    @classmethod
    def from_catalog(
        cls,
        food_df: pd.DataFrame,
        *,
        name_embeddings: np.ndarray | None = None,
        prep_embeddings: np.ndarray | None = None,
        dequant_embeddings: np.ndarray | None = None,
        desc_embeddings: np.ndarray | None = None,
        config: StagedMatchConfig | None = None,
        show_progress: bool | None = None,
    ) -> StagedFoodIndex:
        """Build index. Pass three embedding matrices (name, prep, dequant) from cache."""
        config = config or StagedMatchConfig()
        if desc_embeddings is not None and name_embeddings is None:
            name_embeddings = dequant_embeddings = desc_embeddings
        records: list[FoodCandidate] = []
        token_index: dict[str, set[int]] = {}

        n_food = len(food_df)
        if show_progress is None:
            show_progress = progress_enabled_for_count(n_food, threshold=1000)
        row_iter: Iterable[Any] = food_df.itertuples(index=False)
        row_iter = iter_progress(
            row_iter,
            total=n_food,
            desc="Building food index",
            enabled=show_progress,
        )
        for i, row in enumerate(row_iter):
            desc = str(row.description) if row.description is not None else ""
            # Full description for identity/lexical (no comma truncation).
            content_tokens = tokenize(desc)
            raw_tokens = tokenize(desc, drop_stop=False)
            mods_by_cat = classify_modifier_tokens(raw_tokens)
            modifier_tokens = set().union(*(mods_by_cat.values()))
            rec = FoodCandidate(
                fdc_id=int(row.fdc_id),
                data_type=str(row.data_type),
                description=desc,
                prefix=desc,
                base_tokens=content_tokens,
                modifier_tokens=modifier_tokens,
                modifiers_by_cat=mods_by_cat,
                name_embedding=name_embeddings[i] if name_embeddings is not None else None,
                prep_embedding=prep_embeddings[i] if prep_embeddings is not None else None,
                dequant_embedding=dequant_embeddings[i] if dequant_embeddings is not None else None,
            )
            records.append(rec)
            idx = len(records) - 1
            for tok in content_tokens:
                token_index.setdefault(tok, set()).add(idx)

        name_matrix = None
        dequant_matrix = None
        if name_embeddings is not None:
            name_matrix = np.asarray(name_embeddings, dtype=np.float32)
        if dequant_embeddings is not None:
            dequant_matrix = np.asarray(dequant_embeddings, dtype=np.float32)

        return cls(
            candidates=records,
            token_index=token_index,
            name_matrix=name_matrix,
            dequant_matrix=dequant_matrix,
            config=config,
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


# Lexical scoring uses an F-beta over token overlap with beta < 1, which weights
# precision more than recall. This stops single-token queries like "sugar" from
# tying every entry that merely contains the word (recall=1.0); a clean "sugar"
# entry (high precision) now beats "sugar free syrup" / "onion barbecue sauce".
LEXICAL_PRECISION_BETA = 0.5
LEXICAL_FSCORE_WEIGHT = 0.70
LEXICAL_FUZZY_WEIGHT = 0.30


def _fbeta(precision: float, recall: float, beta: float = LEXICAL_PRECISION_BETA) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1.0 + b2) * precision * recall / denom if denom > 0 else 0.0


def _lexical_overlap_score(
    query_tokens: set[str],
    food_tokens: set[str],
    query_text: str,
    food_text: str,
) -> float:
    """Precision-weighted F-beta over token overlap, blended with a fuzzy ratio."""
    if not query_tokens:
        return 0.0
    overlap = query_tokens & food_tokens
    recall = len(overlap) / len(query_tokens)
    precision = len(overlap) / len(food_tokens) if food_tokens else 0.0
    fscore = _fbeta(precision, recall)
    fuzzy = fuzz.token_set_ratio(query_text, food_text) / 100.0
    return LEXICAL_FSCORE_WEIGHT * fscore + LEXICAL_FUZZY_WEIGHT * fuzzy


def lexical_base_score(query_tokens: set[str], food: FoodCandidate) -> float:
    food_tokens = food.base_tokens | tokenize(food.prefix)
    return _lexical_overlap_score(
        query_tokens,
        food_tokens,
        " ".join(sorted(query_tokens)),
        food.prefix or food.description,
    )


def _sim01(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return max(0.0, min(1.0, _cosine(a, b)))


def lexical_dequant_score(query_dequant: str, food: FoodCandidate) -> float:
    query_tokens = tokenize(query_dequant)
    food_tokens = tokenize(food.description)
    return _lexical_overlap_score(
        query_tokens, food_tokens, query_dequant, food.description
    )


def base_score_breakdown(
    query_name: str,
    query_tokens: set[str],
    query_dequant: str,
    query_name_emb: np.ndarray | None,
    query_dequant_emb: np.ndarray | None,
    food: FoodCandidate,
    config: StagedMatchConfig,
) -> dict[str, float]:
    name_lex = lexical_base_score(query_tokens, food)
    name_sem = _sim01(query_name_emb, food.name_embedding)
    dequant_lex = lexical_dequant_score(query_dequant, food)
    dequant_sem = _sim01(query_dequant_emb, food.dequant_embedding)
    name_channel = (
        config.base_name_lexical_weight * name_lex + config.base_name_semantic_weight * name_sem
    )
    dequant_channel = (
        config.base_dequant_lexical_weight * dequant_lex
        + config.base_dequant_semantic_weight * dequant_sem
    )
    wn = config.base_name_channel_weight
    wd = config.base_dequant_channel_weight
    denom = wn + wd
    if denom <= 0:
        combined = max(name_channel, dequant_channel)
    else:
        combined = (wn * name_channel + wd * dequant_channel) / denom
    return {
        "name_lex": name_lex,
        "name_sem": name_sem,
        "dequant_lex": dequant_lex,
        "dequant_sem": dequant_sem,
        "name_channel": name_channel,
        "dequant_channel": dequant_channel,
        "base": combined,
    }


def base_score(
    query_name: str,
    query_tokens: set[str],
    query_dequant: str,
    query_name_emb: np.ndarray | None,
    query_dequant_emb: np.ndarray | None,
    food: FoodCandidate,
    config: StagedMatchConfig,
) -> float:
    return base_score_breakdown(
        query_name,
        query_tokens,
        query_dequant,
        query_name_emb,
        query_dequant_emb,
        food,
        config,
    )["base"]


def prep_lexical_score(query_prep_tokens: set[str], food: FoodCandidate) -> float:
    if not query_prep_tokens:
        return 0.5
    overlap = query_prep_tokens & food.modifier_tokens
    if not overlap:
        return 0.0
    return len(overlap) / len(query_prep_tokens)


def prep_semantic_score(query_prep_emb: np.ndarray | None, food: FoodCandidate) -> float:
    if query_prep_emb is None or food.prep_embedding is None:
        return 0.5
    return _sim01(query_prep_emb, food.prep_embedding) or 0.5


def modifier_rule_score(
    query_mods: dict[str, set[str]],
    food: FoodCandidate,
) -> float:
    if not any(query_mods.values()):
        return 0.5
    score_parts: list[float] = []
    for cat in ("physical_form", "cooking_state", "packaging", "flavor_style"):
        q = query_mods.get(cat, set())
        f = food.modifiers_by_cat.get(cat, set())
        if not q:
            continue
        if q & f:
            score_parts.append(1.0)
        else:
            score_parts.append(0.0)
    return sum(score_parts) / len(score_parts) if score_parts else 0.5


def default_bonus(query_mods: dict[str, set[str]], food: FoodCandidate) -> float:
    food_default = food.modifiers_by_cat.get("default_like", set()) | (
        food.modifier_tokens & DEFAULT_PREP_TERMS
    )
    food_special = food.modifiers_by_cat.get("flavor_style", set())
    query_special = query_mods.get("flavor_style", set())
    query_has_special = bool(query_special)

    if not query_has_special and food_default and not food_special:
        return 1.0
    if not query_has_special and not food.modifier_tokens:
        return 0.9
    if query_has_special and query_special & food.modifier_tokens:
        return 0.5
    return 0.2


def unsupported_modifier_penalty(
    query_mods: dict[str, set[str]],
    food: FoodCandidate,
    config: StagedMatchConfig,
) -> float:
    penalty = 0.0
    query_flavor = query_mods.get("flavor_style", set())
    food_flavor = food.modifiers_by_cat.get("flavor_style", set())
    unsupported_flavor = food_flavor - query_flavor
    if unsupported_flavor and not query_flavor:
        penalty += config.unsupported_flavor_penalty * len(unsupported_flavor)

    query_all = set().union(*query_mods.values())
    extra = food.modifier_tokens - query_all - food.base_tokens
    if extra and query_all:
        penalty += config.extra_modifier_penalty * min(3, len(extra))
    return penalty


def prep_state_score(
    query_prep_tokens: set[str],
    query_prep_emb: np.ndarray | None,
    query_mods: dict[str, set[str]],
    food: FoodCandidate,
    config: StagedMatchConfig,
) -> tuple[float, float, float, dict[str, float]]:
    """Returns (prep_score, default_bonus, penalty, component breakdown)."""
    sem = prep_semantic_score(query_prep_emb, food)
    lex = prep_lexical_score(query_prep_tokens, food)
    rules = modifier_rule_score(query_mods, food)
    default_b = default_bonus(query_mods, food)
    prep = (
        config.prep_semantic_weight * sem
        + config.prep_lexical_weight * lex
        + config.prep_rule_weight * rules
        + config.prep_default_bonus_weight * default_b
    )
    penalty = unsupported_modifier_penalty(query_mods, food, config)
    return prep, default_b, penalty, {
        "prep_sem": sem,
        "prep_lex": lex,
        "prep_rules": rules,
        "prep_default": default_b,
        "prep": prep,
    }


def classify_fallback(
    base_s: float,
    prep_s: float,
    query_prep_tokens: set[str],
    food: FoodCandidate,
) -> str:
    if base_s < 0.35:
        return "weak_base"
    if query_prep_tokens and prep_s >= 0.5:
        return "exact_prep"
    if not query_prep_tokens and (food.modifier_tokens & DEFAULT_PREP_TERMS or not food.modifier_tokens):
        return "neutral_default"
    if query_prep_tokens and prep_s < 0.35:
        return "prep_fallback"
    if food.modifiers_by_cat.get("flavor_style") and not query_prep_tokens:
        return "unsupported_flavor_last_resort"
    return "base_only"


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


@dataclass
class QueryRow:
    recipe_id: int
    ingredient_idx: int
    ingredient: str
    name: str
    preparation: str
    size: str
    quantity: Any
    dequant_text: str
    name_tokens: set[str]
    prep_tokens: set[str]
    query_mods: dict[str, set[str]]
    name_embedding: np.ndarray | None = None
    prep_embedding: np.ndarray | None = None
    dequant_embedding: np.ndarray | None = None


def query_from_parsed_row(
    row: pd.Series,
    name_emb: np.ndarray,
    prep_emb: np.ndarray,
    dequant_emb: np.ndarray,
) -> QueryRow:
    name = normalize_text(row.get("name") or row.get("ingredient"))
    prep = normalize_text(row.get("preparation"))
    dequant = normalize_text(row.get("dequantified") or name)
    name_tokens = tokenize(name)
    prep_tokens = tokenize(prep, drop_stop=False) & PREP_HINTS
    if not prep_tokens:
        prep_tokens = tokenize(prep, drop_stop=False)
    return QueryRow(
        recipe_id=int(row["recipe_id"]),
        ingredient_idx=int(row["ingredient_idx"]),
        ingredient=str(row["ingredient"]),
        name=name,
        preparation=prep,
        dequant_text=dequant,
        size=normalize_text(row.get("size")),
        quantity=row.get("quantity"),
        name_tokens=name_tokens,
        prep_tokens=prep_tokens,
        query_mods=classify_modifier_tokens(prep_tokens | name_tokens),
        name_embedding=name_emb,
        prep_embedding=prep_emb,
        dequant_embedding=dequant_emb,
    )


def _query_lexical_tokens(query: QueryRow) -> set[str]:
    """Tokens for stage-1 lexical overlap (parsed ingredient name only)."""
    tokens = set(query.name_tokens)
    if query.name:
        tokens |= tokenize(query.name)
    return tokens


def has_lexical_overlap(query_tokens: set[str], food: FoodCandidate) -> bool:
    if not query_tokens:
        return False
    food_name_tokens = food.base_tokens | tokenize(food.prefix)
    return bool(query_tokens & food_name_tokens)


def _lexical_overlap_idxs(query: QueryRow, index: StagedFoodIndex) -> list[int]:
    """All catalog rows with ≥1 shared lemma between query name and food name tokens."""
    query_tokens = _query_lexical_tokens(query)
    if not query_tokens:
        return []

    overlap: set[int] = set()
    for tok in query_tokens:
        overlap |= index.token_index.get(tok, set())

    if overlap:
        return sorted(overlap)

    # Fallback: explicit check when token index misses (rare)
    return [
        i
        for i, food in enumerate(index.candidates)
        if has_lexical_overlap(query_tokens, food)
    ]


def _dequant_semantic_similarity(
    query: QueryRow,
    index: StagedFoodIndex,
    food_idxs: list[int] | np.ndarray,
) -> np.ndarray:
    """Cosine similarity between query and food dequantified-text embeddings."""
    idxs = np.asarray(food_idxs, dtype=np.int64)
    if idxs.size == 0:
        return np.array([], dtype=np.float32)

    if query.dequant_embedding is None or index.dequant_matrix is None:
        return np.zeros(idxs.shape[0], dtype=np.float32)

    sims = index.dequant_matrix[idxs] @ np.asarray(query.dequant_embedding, dtype=np.float32)
    return np.clip(sims, 0.0, 1.0)


def _stage1_candidate_idxs(query: QueryRow, index: StagedFoodIndex) -> list[int]:
    """
    Stage 1 retrieval: all foods with any lexical name overlap, then top-K by dequant embedding similarity.
    """
    config = index.config
    lexical_idxs = _lexical_overlap_idxs(query, index)

    if not lexical_idxs:
        if len(index.candidates) <= 5000:
            return list(range(len(index.candidates)))
        return []

    if index.dequant_matrix is None or query.dequant_embedding is None:
        lexical_set = set(lexical_idxs)
        counts: dict[int, int] = {}
        for tok in _query_lexical_tokens(query):
            for idx in index.token_index.get(tok, ()):
                if idx in lexical_set:
                    counts[idx] = counts.get(idx, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        return [idx for idx, _ in ranked[: config.stage1_top_k]]

    sims = _dequant_semantic_similarity(query, index, lexical_idxs)
    order = np.argsort(-sims)
    top_k = min(config.stage1_top_k, len(lexical_idxs))
    return [lexical_idxs[i] for i in order[:top_k]]


def score_food_candidate(
    query: QueryRow,
    food_idx: int,
    index: StagedFoodIndex,
) -> dict[str, Any]:
    """Full staged score breakdown for one query × food pair."""
    config = index.config
    food = index.candidates[food_idx]
    base_s = base_score(
        query.name,
        query.name_tokens,
        query.dequant_text,
        query.name_embedding,
        query.dequant_embedding,
        food,
        config,
    )
    prep_s, default_b, penalty, prep_parts = prep_state_score(
        query.prep_tokens,
        query.prep_embedding,
        query.query_mods,
        food,
        config,
    )
    final = (
        config.final_base_weight * base_s
        + config.final_prep_weight * prep_s
        + config.final_default_weight * default_b
        - penalty
    )
    final = max(0.0, min(1.0, final))
    reason = classify_fallback(base_s, prep_s, query.prep_tokens, food)
    identity_parts = base_score_breakdown(
        query.name,
        query.name_tokens,
        query.dequant_text,
        query.name_embedding,
        query.dequant_embedding,
        food,
        config,
    )
    return {
        "fdc_id": food.fdc_id,
        "description": food.description,
        "match_score": round(final, 4),
        "base_score": round(base_s, 4),
        "prep_score": round(prep_s, 4),
        "default_bonus": round(default_b, 4),
        "modifier_penalty": round(penalty, 4),
        "fallback_reason": reason,
        "name_lex_score": round(identity_parts["name_lex"], 4),
        "name_sem_score": round(identity_parts["name_sem"], 4),
        "dequant_lex_score": round(identity_parts["dequant_lex"], 4),
        "dequant_sem_score": round(identity_parts["dequant_sem"], 4),
        "name_channel_score": round(identity_parts["name_channel"], 4),
        "dequant_channel_score": round(identity_parts["dequant_channel"], 4),
        "prep_sem_score": round(prep_parts["prep_sem"], 4),
        "prep_lex_score": round(prep_parts["prep_lex"], 4),
        "prep_rules_score": round(prep_parts["prep_rules"], 4),
        "prep_default_component": round(prep_parts["prep_default"], 4),
    }


def match_query_top_k(
    query: QueryRow,
    index: StagedFoodIndex,
    *,
    top_k: int = 20,
    score_all_stage1: bool = True,
) -> pd.DataFrame:
    """
    Return top-k food candidates with full score components.

    Scores stage-1 pool (name lexical filter → dequant semantic top-K), then ranks by final blended score.
    """
    stage1_idxs = _stage1_candidate_idxs(query, index)
    if not stage1_idxs:
        return pd.DataFrame()

    if score_all_stage1:
        pool = stage1_idxs
    else:
        stage1_scored = [
            (idx, base_score(
                query.name,
                query.name_tokens,
                query.dequant_text,
                query.name_embedding,
                query.dequant_embedding,
                index.candidates[idx],
                index.config,
            ))
            for idx in stage1_idxs
        ]
        stage1_scored.sort(key=lambda x: -x[1])
        pool = [idx for idx, _ in stage1_scored[: min(15, len(stage1_scored))]]

    rows = [score_food_candidate(query, idx, index) for idx in pool]
    df = pd.DataFrame(rows).sort_values("match_score", ascending=False).head(top_k)
    df.insert(0, "rank", range(1, len(df) + 1))
    margins = df["match_score"].diff(-1).round(4)
    df["match_margin"] = margins
    return df.reset_index(drop=True)


def match_query(
    query: QueryRow,
    index: StagedFoodIndex,
) -> dict[str, Any]:
    config = index.config
    base: dict[str, Any] = {
        "match_query": query.name or query.ingredient,
        "matched_fdc_id": None,
        "matched_description": None,
        "match_stage": "unresolved",
        "match_score": 0.0,
        "match_margin": 0.0,
        "match_quality": "unresolved",
        "base_score": 0.0,
        "prep_score": 0.0,
        "default_bonus": 0.0,
        "modifier_penalty": 0.0,
        "fallback_reason": None,
        "n_candidates": 0,
        "n_lexical_overlap": 0,
    }

    if not query.name_tokens and not query.name:
        return base

    n_lexical = len(_lexical_overlap_idxs(query, index))
    stage1_idxs = _stage1_candidate_idxs(query, index)
    if not stage1_idxs:
        return base

    stage1_scored: list[tuple[int, float]] = []
    for idx in stage1_idxs:
        food = index.candidates[idx]
        s = base_score(
            query.name,
            query.name_tokens,
            query.dequant_text,
            query.name_embedding,
            query.dequant_embedding,
            food,
            config,
        )
        stage1_scored.append((idx, s))
    stage1_scored.sort(key=lambda x: -x[1])
    if not stage1_scored:
        return base

    # Stage 2: prep / version on top base candidates
    top_base = stage1_scored[: min(15, len(stage1_scored))]
    final_scored: list[tuple[int, float, float, float, float, str]] = []
    for idx, base_s in top_base:
        food = index.candidates[idx]
        prep_s, default_b, penalty, _prep_parts = prep_state_score(
            query.prep_tokens,
            query.prep_embedding,
            query.query_mods,
            food,
            config,
        )
        final = (
            config.final_base_weight * base_s
            + config.final_prep_weight * prep_s
            + config.final_default_weight * default_b
            - penalty
        )
        final = max(0.0, min(1.0, final))
        reason = classify_fallback(base_s, prep_s, query.prep_tokens, food)
        final_scored.append((idx, final, base_s, prep_s, penalty, reason))

    final_scored.sort(key=lambda x: -x[1])
    best_idx, best_score, best_base, best_prep, best_pen, best_reason = final_scored[0]
    second_score = final_scored[1][1] if len(final_scored) > 1 else 0.0
    margin = best_score - second_score
    food = index.candidates[best_idx]
    identity_parts = base_score_breakdown(
        query.name,
        query.name_tokens,
        query.dequant_text,
        query.name_embedding,
        query.dequant_embedding,
        food,
        config,
    )
    _, _, _, prep_parts = prep_state_score(
        query.prep_tokens,
        query.prep_embedding,
        query.query_mods,
        food,
        config,
    )

    if best_score >= config.score_high and margin >= config.margin_min:
        quality = "high"
    elif best_score >= config.score_medium:
        quality = "medium"
    elif best_score >= config.score_low:
        quality = "low"
    else:
        quality = "unresolved"

    stage = "staged_hybrid" if quality != "unresolved" else "unresolved"
    if best_reason in ("exact_prep", "neutral_default") and quality != "unresolved":
        stage = "exact" if best_score >= 0.85 else stage

    return {
        **base,
        "matched_fdc_id": food.fdc_id,
        "matched_description": food.description,
        "match_stage": stage,
        "match_score": round(best_score, 4),
        "match_margin": round(margin, 4),
        "match_quality": quality,
        "base_score": round(best_base, 4),
        "prep_score": round(best_prep, 4),
        "default_bonus": round(
            default_bonus(query.query_mods, food),
            4,
        ),
        "modifier_penalty": round(best_pen, 4),
        "fallback_reason": best_reason if best_reason not in ("exact_prep",) else None,
        "n_candidates": len(stage1_idxs),
        "n_lexical_overlap": n_lexical,
        "name_lex_score": round(identity_parts["name_lex"], 4),
        "name_sem_score": round(identity_parts["name_sem"], 4),
        "dequant_lex_score": round(identity_parts["dequant_lex"], 4),
        "dequant_sem_score": round(identity_parts["dequant_sem"], 4),
        "name_channel_score": round(identity_parts["name_channel"], 4),
        "dequant_channel_score": round(identity_parts["dequant_channel"], 4),
        "prep_sem_score": round(prep_parts["prep_sem"], 4),
        "prep_lex_score": round(prep_parts["prep_lex"], 4),
        "prep_rules_score": round(prep_parts["prep_rules"], 4),
    }


# ---------------------------------------------------------------------------
# LLM-judge candidate retrieval (union of full-text lexical + global semantic)
# ---------------------------------------------------------------------------


@dataclass
class LLMRetrievalConfig:
    """Cutoffs for assembling the candidate set handed to the LLM judge."""

    lexical_min_token_overlap: int = 1
    lexical_score_floor: float = 0.45
    lexical_top_k: int = 12
    semantic_top_k: int = 15
    semantic_score_floor: float = 0.55
    semantic_floor_cap: int = 50  # safety cap on floor-qualified semantic hits
    max_candidates: int = 15  # hard cap of rows sent to the LLM prompt
    description_max_chars: int = 120
    top10_size: int = 10
    # Final ordering blends the two channels, favoring semantic slightly.
    semantic_blend_weight: float = 0.55
    lexical_blend_weight: float = 0.45


def _fulltext_query_tokens(query: QueryRow) -> set[str]:
    """Tokens from the full dequantified ingredient text (plus parsed name)."""
    tokens = tokenize(query.dequant_text)
    tokens |= set(query.name_tokens)
    return tokens


def batched_dequant_similarities(
    index: StagedFoodIndex,
    query_embeddings: np.ndarray,
) -> np.ndarray:
    """Cosine sims for many queries at once: (n_food, n_queries), clipped to [0,1].

    One BLAS matrix-matrix product instead of per-query matrix-vector calls.
    Embeddings are L2-normalized at cache time, so dot product == cosine.
    """
    if index.dequant_matrix is None or query_embeddings is None or len(query_embeddings) == 0:
        n_food = len(index.candidates)
        n_q = 0 if query_embeddings is None else len(query_embeddings)
        return np.zeros((n_food, n_q), dtype=np.float32)
    q = np.asarray(query_embeddings, dtype=np.float32)
    sims = index.dequant_matrix @ q.T
    return np.clip(sims, 0.0, 1.0)


def retrieve_llm_candidates(
    query: QueryRow,
    index: StagedFoodIndex,
    retr_config: LLMRetrievalConfig | None = None,
    *,
    staged_top1_fdc_id: int | None = None,
    precomputed_sims: np.ndarray | None = None,
    allowed_fdc_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Union retrieval for the LLM judge: full-text lexical + global semantic.

    Returns one row per union candidate with lexical/semantic/staged scores,
    sorted by retrieval rank. Columns include flags marking which rows go into
    the LLM prompt (`in_llm_prompt`) and the staged top-1 (`is_staged_top1`).

    Pass `precomputed_sims` (a per-food cosine vector for this query, already in
    [0,1]) to reuse a batched matmul instead of recomputing per query.
    """
    rc = retr_config or LLMRetrievalConfig()
    n_food = len(index.candidates)
    if n_food == 0:
        return pd.DataFrame()

    allowed_idxs: set[int] | None = None
    if allowed_fdc_ids is not None:
        allowed_idxs = {
            idx
            for idx, food in enumerate(index.candidates)
            if food.fdc_id in allowed_fdc_ids
        }

    # --- Semantic channel: global cosine over all foods (normalized embeddings).
    if precomputed_sims is not None:
        sims = np.asarray(precomputed_sims, dtype=np.float32)
    elif index.dequant_matrix is not None and query.dequant_embedding is not None:
        sims = index.dequant_matrix @ np.asarray(query.dequant_embedding, dtype=np.float32)
        sims = np.clip(sims, 0.0, 1.0)
    else:
        sims = np.zeros(n_food, dtype=np.float32)

    semantic_idxs: set[int] = set()
    if sims.any():
        candidate_idxs = (
            sorted(allowed_idxs)
            if allowed_idxs is not None
            else list(range(n_food))
        )
        if candidate_idxs:
            if allowed_idxs is not None:
                sub_sims = sims[candidate_idxs]
                k = min(rc.semantic_top_k, len(candidate_idxs))
                if k > 0:
                    top_local = (
                        np.argpartition(-sub_sims, k - 1)[:k]
                        if k < len(candidate_idxs)
                        else np.arange(len(candidate_idxs))
                    )
                    semantic_idxs.update(int(candidate_idxs[i]) for i in top_local)
                floor_hits = np.flatnonzero(sub_sims >= rc.semantic_score_floor)
                if floor_hits.size > rc.semantic_floor_cap:
                    floor_hits = floor_hits[np.argsort(-sub_sims[floor_hits])[: rc.semantic_floor_cap]]
                semantic_idxs.update(int(candidate_idxs[i]) for i in floor_hits)
            else:
                k = min(rc.semantic_top_k, n_food)
                top_sem = np.argpartition(-sims, k - 1)[:k] if k < n_food else np.arange(n_food)
                semantic_idxs.update(int(i) for i in top_sem)
                floor_hits = np.flatnonzero(sims >= rc.semantic_score_floor)
                if floor_hits.size > rc.semantic_floor_cap:
                    floor_hits = floor_hits[np.argsort(-sims[floor_hits])[: rc.semantic_floor_cap]]
                semantic_idxs.update(int(i) for i in floor_hits)

    # --- Lexical channel: full-text token overlap, scored by dequant lexical sim.
    fulltext_tokens = _fulltext_query_tokens(query)
    pool: set[int] = set()
    for tok in fulltext_tokens:
        hits = index.token_index.get(tok, set())
        if allowed_idxs is not None:
            hits = hits & allowed_idxs
        pool |= hits

    lex_scores: dict[int, float] = {}
    for idx in pool:
        lex_scores[idx] = lexical_dequant_score(query.dequant_text, index.candidates[idx])

    lexical_idxs: set[int] = set()
    if lex_scores:
        ranked_lex = sorted(lex_scores.items(), key=lambda x: -x[1])
        lexical_idxs.update(idx for idx, _ in ranked_lex[: rc.lexical_top_k])
        lexical_idxs.update(
            idx for idx, score in lex_scores.items() if score >= rc.lexical_score_floor
        )

    # --- Union + include staged top-1 when it satisfies portion-capable filter (if any).
    union: set[int] = semantic_idxs | lexical_idxs
    staged_top1_idx: int | None = None
    if staged_top1_fdc_id is not None:
        for idx, food in enumerate(index.candidates):
            if food.fdc_id == staged_top1_fdc_id:
                staged_top1_idx = idx
                break
        if staged_top1_idx is not None:
            if allowed_fdc_ids is None or int(staged_top1_fdc_id) in allowed_fdc_ids:
                union.add(staged_top1_idx)

    if not union:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for idx in union:
        food = index.candidates[idx]
        lex = lex_scores.get(idx)
        if lex is None:
            lex = lexical_dequant_score(query.dequant_text, food)
        sem = float(sims[idx])
        staged = score_food_candidate(query, idx, index)
        rows.append(
            {
                "fdc_id": food.fdc_id,
                "data_type": food.data_type,
                "description": food.description,
                "lexical_dequant": round(lex, 4),
                "dequant_sem": round(sem, 4),
                "retrieval_score": round(
                    rc.semantic_blend_weight * sem + rc.lexical_blend_weight * lex, 4
                ),
                "staged_final_score": staged["match_score"],
                "staged_base_score": staged["base_score"],
                "staged_prep_score": staged["prep_score"],
                "is_staged_top1": idx == staged_top1_idx,
            }
        )

    rows = dedupe_candidate_rows(rows)

    df = pd.DataFrame(rows).sort_values(
        ["retrieval_score", "staged_final_score", "fdc_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df["in_llm_prompt"] = df["rank"] <= rc.max_candidates
    df.attrs["n_lexical_pool"] = len(lexical_idxs)
    df.attrs["n_semantic_pool"] = len(semantic_idxs)
    df.attrs["n_union"] = len(union)
    df.attrs["n_union_before_dedup"] = len(union)
    df.attrs["n_after_signal_dedup"] = len(rows)
    return df


def match_ingredients_staged(
    parsed_ingredients: pd.DataFrame,
    name_embeddings: np.ndarray,
    prep_embeddings: np.ndarray,
    dequant_embeddings: np.ndarray,
    food_index: StagedFoodIndex,
    *,
    show_progress: bool | None = None,
    progress_desc: str = "Staged matching",
    progress_position: int | None = None,
    progress_leave: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(parsed_ingredients)
    if show_progress is None:
        show_progress = progress_enabled_for_count(n)
    it = iter_progress(
        range(n),
        total=n,
        desc=progress_desc,
        enabled=show_progress,
        position=progress_position,
        leave=progress_leave,
    )

    for i in it:
        row = parsed_ingredients.iloc[i]
        q = query_from_parsed_row(
            row, name_embeddings[i], prep_embeddings[i], dequant_embeddings[i]
        )
        rows.append(match_query(q, food_index))

    match_df = pd.DataFrame(rows)
    return pd.concat([parsed_ingredients.reset_index(drop=True), match_df], axis=1)


def load_or_build_food_embeddings(
    descriptions: list[str],
    cache_path: Path,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    force: bool = False,
) -> np.ndarray:
    """Deprecated: use ingredient_query_cache.load_or_build_food_artifacts."""
    cache_path = Path(cache_path)
    if not force and cache_path.is_file():
        emb = np.load(cache_path)
        if len(emb) == len(descriptions):
            print(f"Loaded food embeddings → {cache_path}")
            return emb
    print(f"Embedding {len(descriptions):,} food descriptions → {cache_path}")
    emb = embed_food_descriptions(descriptions, model_name=model_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    return emb


def embed_food_descriptions(
    descriptions: list[str],
    *,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(
        descriptions,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(descriptions) > 500,
    )
    return np.asarray(emb, dtype=np.float32)
