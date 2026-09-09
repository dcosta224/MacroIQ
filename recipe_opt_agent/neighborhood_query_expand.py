"""Expand the working recipe neighborhood using LLM-chosen search queries.

Used when the agent considers OOD adjustments (e.g. adding chicken to carbonara):
the LLM emits culinary search phrases + ``dish_structure``; we retrieve related
recipes by title / ingredient-list overlap and optional embedding cosine, then
*verify* gram-share structure (anchor dominant vs stretch) before harvesting
FoodOn share samples into ``basis_samples``. Wrong-dominance recipes (meat with
rice as a side when the dish is rice-forward) are rejected or down-weighted so
ratio loss is not poisoned.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def _tokenize(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 2}


def normalize_dish_structure(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize LLM / fallback dish_structure; return None if unusable."""
    if not isinstance(raw, dict):
        return None
    anchors = [
        str(a).strip()
        for a in (raw.get("anchor_ingredients") or raw.get("anchors") or [])
        if str(a).strip()
    ]
    stretch = str(raw.get("stretch_ingredient") or raw.get("stretch") or "").strip()
    role = str(raw.get("stretch_role") or raw.get("role") or "accent").strip().lower()
    if role not in {"accent", "co_main"}:
        role = "accent"
    if not anchors and not stretch:
        return None
    return {
        "anchor_ingredients": anchors,
        "stretch_ingredient": stretch,
        "stretch_role": role,
    }


def fallback_dish_structure(
    *,
    stretch_ingredient: str | None,
    identity_roles: list[str] | None = None,
    current_labels: list[str] | None = None,
    stretch_role: str = "accent",
) -> dict[str, Any] | None:
    """Derive structure when the LLM omits dish_structure."""
    stretch = str(stretch_ingredient or "").strip()
    anchors: list[str] = []
    for role in identity_roles or []:
        r = str(role).strip()
        if r and r.lower() not in {stretch.lower()}:
            anchors.append(r)
    if not anchors:
        # Prefer carb/base-looking labels from the current recipe.
        base_hints = {
            "pasta", "spaghetti", "noodle", "noodles", "rice", "risotto",
            "bread", "potato", "potatoes", "tortilla", "dough", "flour",
        }
        for lab in current_labels or []:
            toks = _tokenize(lab)
            if toks & base_hints:
                anchors.append(str(lab))
        if not anchors and current_labels:
            anchors = [str(current_labels[0])]
    if not stretch and not anchors:
        return None
    return normalize_dish_structure(
        {
            "anchor_ingredients": anchors[:4],
            "stretch_ingredient": stretch,
            "stretch_role": stretch_role,
        }
    )


# Culinary modifiers that never identify the *stretch* ingredient in a query.
_QUERY_MODIFIER_STOPWORDS = {
    "creamy", "style", "styled", "recipe", "recipes", "with", "seared", "grilled",
    "baked", "fried", "roasted", "fresh", "classic", "easy", "homemade", "quick",
    "best", "simple", "traditional", "italian", "dish", "dishes", "sauce",
}


def derive_focus_terms(
    queries: list[str],
    current_labels: list[str] | None = None,
    *,
    extra_texts: list[str] | None = None,
) -> list[str]:
    """Tokens in the queries that name the *new* ingredient being considered.

    Filters out tokens already present in the current recipe (pasta, egg, …) and
    generic culinary modifiers, leaving e.g. {"chicken", "breast", "poultry"} for
    "creamy spaghetti with seared chicken breast" on a carbonara.
    """
    cur: set[str] = set()
    for lab in current_labels or []:
        cur |= _tokenize(str(lab))
    for text in extra_texts or []:
        cur |= _tokenize(str(text))
    out: list[str] = []
    for q in queries or []:
        for t in sorted(_tokenize(q)):
            if t in cur or t in _QUERY_MODIFIER_STOPWORDS:
                continue
            if t not in out:
                out.append(t)
    return out


def _score_query_against_text(query: str, text: str) -> float:
    qt = _tokenize(query)
    tt = _tokenize(text)
    if not qt or not tt:
        return 0.0
    overlap = len(qt & tt) / len(qt | tt)
    # Bonus if all query tokens appear
    if qt <= tt:
        overlap += 0.25
    return float(min(1.0, overlap))


def _load_recipe_text_corpus(exclude_ids: set[int], *, limit: int = 8000) -> list[dict[str, Any]]:
    """Title + ingredient text rows from the local cap40 store (best effort)."""
    try:
        from recipe_data_access import get_store

        store = get_store()
        out: list[dict[str, Any]] = []

        feats = None
        for attr in ("recipe_nlg_features", "nlg_features", "features"):
            fn = getattr(store, attr, None)
            if callable(fn):
                try:
                    feats = fn()
                    break
                except Exception:
                    feats = None

        if feats is not None and not getattr(feats, "empty", True):
            for row in feats.itertuples(index=False):
                rid = int(getattr(row, "recipe_id"))
                if rid in exclude_ids:
                    continue
                title = str(
                    getattr(row, "title_clean", None)
                    or getattr(row, "title", None)
                    or ""
                )
                sem = str(getattr(row, "semantic_text", "") or "")
                text = f"{title} {sem}".strip()
                out.append(
                    {
                        "recipe_id": rid,
                        "title": title,
                        "text": text,
                        "labels": sorted(_tokenize(sem)),
                    }
                )
                if len(out) >= limit:
                    return out
            if out:
                return out

        rr = store.resolved_recipes()
        if rr is None or rr.empty:
            return []
        for rid, sub in rr.groupby("recipe_id"):
            if int(rid) in exclude_ids:
                continue
            labels: list[str] = []
            if "fdc_description" in sub.columns:
                labels = [str(x) for x in sub["fdc_description"].dropna().tolist()]
            out.append(
                {
                    "recipe_id": int(rid),
                    "title": "",
                    "text": " ".join(labels),
                    "labels": [x.lower() for x in labels],
                }
            )
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def _embed_queries_and_corpus(
    queries: list[str], corpus_texts: list[str]
) -> tuple[Any | None, Any | None]:
    """Optional MiniLM embeddings via the process-wide singleton."""
    try:
        from recipe_opt_agent.embedding_model import encode_texts

        q = encode_texts(queries)
        c = encode_texts(corpus_texts, batch_size=64)
        if q is None or c is None:
            return None, None
        return q, c
    except Exception:
        return None, None


def _load_recipe_lines_for_ids(recipe_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    """resolved_recipes rows keyed by recipe_id (grams + fdc_description + foodon_id)."""
    if not recipe_ids:
        return {}
    try:
        from recipe_data_access import get_store

        store = get_store()
        rr = store.resolved_recipes([int(r) for r in recipe_ids])
        if rr is None or rr.empty:
            return {}
        rr = rr[rr["gram_weight"].notna()].copy()
        if rr.empty:
            return {}
        if "fdc_id" in rr.columns:
            rr = rr[rr["fdc_id"].notna()].copy()
            rr["fdc_id"] = rr["fdc_id"].astype(int)
            try:
                foodon = store.food_4macro_foodon(rr["fdc_id"].unique().tolist())
                keep = [c for c in ("fdc_id", "foodon_id") if c in foodon.columns]
                foodon = foodon[keep].dropna(subset=["foodon_id"]).drop_duplicates("fdc_id")
                foodon["fdc_id"] = foodon["fdc_id"].astype(int)
                rr = rr.merge(foodon, on="fdc_id", how="left")
            except Exception:
                pass
        out: dict[int, list[dict[str, Any]]] = {}
        for rid, sub in rr.groupby("recipe_id"):
            rows: list[dict[str, Any]] = []
            for row in sub.itertuples(index=False):
                rows.append(
                    {
                        "gram_weight": float(getattr(row, "gram_weight", 0.0) or 0.0),
                        "fdc_description": str(
                            getattr(row, "fdc_description", None)
                            or getattr(row, "description", None)
                            or ""
                        ),
                        "foodon_id": (
                            str(getattr(row, "foodon_id"))
                            if getattr(row, "foodon_id", None) is not None
                            and str(getattr(row, "foodon_id")) not in {"", "nan", "None"}
                            else None
                        ),
                    }
                )
            out[int(rid)] = rows
        return out
    except Exception:
        return {}


def _term_match_tokens(terms: list[str]) -> set[str]:
    toks: set[str] = set()
    for t in terms:
        toks |= _tokenize(t)
    # Drop ultra-generic prep words so they don't dominate matching.
    toks -= {
        "raw", "fresh", "cooked", "skinless", "boneless", "dried", "grated",
        "whole", "table", "part", "skim", "piece", "pieces",
    }
    return toks


def _line_matches_terms(line: dict[str, Any], terms: list[str], basis_nodes: set[str]) -> bool:
    desc_toks = _tokenize(str(line.get("fdc_description") or ""))
    term_toks = _term_match_tokens(terms)
    if term_toks and (desc_toks & term_toks):
        return True
    fid = line.get("foodon_id")
    if fid and basis_nodes and str(fid) in basis_nodes:
        return True
    return False


def _resolve_structure_basis_nodes(
    terms: list[str],
    problem: dict[str, Any] | None,
) -> set[str]:
    """Best-effort FoodOn basis/leaf ids for structure terms."""
    if not terms:
        return set()
    nodes: set[str] = set()
    try:
        from recipe_opt_agent.ood_foodon import annotate_candidate_foodon

        for term in terms:
            stub = annotate_candidate_foodon(
                {"label": term, "meta": {}, "branch": "structure"},
                problem or {},
            )
            meta = stub.get("meta") or {}
            for key in ("basis_node", "foodon_leaf_id", "foodon_id"):
                if meta.get(key):
                    nodes.add(str(meta[key]))
    except Exception:
        pass
    return nodes


def compute_recipe_structure_shares(
    lines: list[dict[str, Any]],
    *,
    anchor_terms: list[str],
    stretch_terms: list[str],
    anchor_nodes: set[str] | None = None,
    stretch_nodes: set[str] | None = None,
) -> dict[str, Any]:
    """Gram shares of anchor vs stretch in one recipe's resolved lines."""
    total = sum(float(r.get("gram_weight") or 0.0) for r in lines)
    if total <= 0:
        return {
            "has_grams": False,
            "anchor_share": 0.0,
            "stretch_share": 0.0,
            "total_grams": 0.0,
        }
    a_nodes = anchor_nodes or set()
    s_nodes = stretch_nodes or set()
    a_g = 0.0
    s_g = 0.0
    for line in lines:
        g = float(line.get("gram_weight") or 0.0)
        if g <= 0:
            continue
        if _line_matches_terms(line, anchor_terms, a_nodes):
            a_g += g
        if _line_matches_terms(line, stretch_terms, s_nodes):
            s_g += g
    return {
        "has_grams": True,
        "anchor_share": a_g / total,
        "stretch_share": s_g / total,
        "total_grams": total,
        "anchor_grams": a_g,
        "stretch_grams": s_g,
    }


def classify_structure_fit(
    shares: dict[str, Any],
    *,
    stretch_role: str = "accent",
) -> dict[str, Any]:
    """Decide pass / soft / reject / context_only from gram shares + role."""
    if not shares.get("has_grams"):
        return {
            "verdict": "context_only",
            "weight_scale": 0.0,
            "reason": "no_resolvable_grams",
        }
    a = float(shares.get("anchor_share") or 0.0)
    s = float(shares.get("stretch_share") or 0.0)
    role = (stretch_role or "accent").lower()

    if role == "co_main":
        if a >= 0.12 and s >= 0.12:
            ratio = max(a, s) / max(min(a, s), 1e-9)
            if ratio <= 4.0:
                return {"verdict": "pass", "weight_scale": 1.0, "reason": "co_main_balanced"}
            # Imbalanced but both present → soft keep
            scale = float(min(1.0, 4.0 / ratio))
            return {
                "verdict": "soft",
                "weight_scale": scale,
                "reason": "co_main_imbalanced",
            }
        if a > 0 and s > 0:
            return {
                "verdict": "soft",
                "weight_scale": 0.35,
                "reason": "co_main_thin_shares",
            }
        return {
            "verdict": "reject",
            "weight_scale": 0.0,
            "reason": "co_main_missing_component",
        }

    # accent (default): anchor should dominate
    if a >= 0.20 and a >= s and s >= 0:
        return {"verdict": "pass", "weight_scale": 1.0, "reason": "accent_anchor_dominant"}
    if a > 0 and s > 0 and a < s:
        # Anchor present but stretch-primary → soft downweight (not hard reject)
        scale = float(min(1.0, max(0.15, a / max(s, 1e-9))))
        return {
            "verdict": "soft",
            "weight_scale": scale,
            "reason": "accent_anchor_present_not_dominant",
        }
    if a > 0 and a < 0.20:
        scale = float(min(1.0, max(0.2, a / 0.20)))
        return {
            "verdict": "soft",
            "weight_scale": scale,
            "reason": "accent_anchor_thin",
        }
    if s > 0 and a <= 0:
        return {
            "verdict": "reject",
            "weight_scale": 0.0,
            "reason": "accent_stretch_without_anchor",
        }
    if a > 0 and s <= 0:
        # Anchor-only (stretch token matched text but no grams) → context only
        return {
            "verdict": "context_only",
            "weight_scale": 0.0,
            "reason": "accent_no_stretch_grams",
        }
    return {
        "verdict": "reject",
        "weight_scale": 0.0,
        "reason": "accent_no_structure",
    }


def verify_shell_structure(
    shell_recipes: list[dict[str, Any]],
    dish_structure: dict[str, Any] | None,
    *,
    problem: dict[str, Any] | None = None,
    recipe_lines_by_id: dict[int, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Filter / reweight shell recipes by FoodOn gram-share structure.

    Returns:
      accepted: recipes kept for harvest (pass + soft)
      context_only: kept for co-occurrence labels but not share harvest
      rejected: dropped from shell
      meta: structural stats for LLM feedback
    """
    structure = normalize_dish_structure(dish_structure)
    if not structure or not shell_recipes:
        return {
            "accepted": list(shell_recipes),
            "context_only": [],
            "rejected": [],
            "meta": {
                "structure_applied": False,
                "n_structure_checked": 0,
                "n_structure_passed": len(shell_recipes),
                "n_structure_soft": 0,
                "n_rejected_wrong_dominance": 0,
                "n_context_only": 0,
            },
        }

    anchor_terms = list(structure["anchor_ingredients"])
    stretch_terms = [structure["stretch_ingredient"]] if structure["stretch_ingredient"] else []
    role = structure["stretch_role"]
    anchor_nodes = _resolve_structure_basis_nodes(anchor_terms, problem)
    stretch_nodes = _resolve_structure_basis_nodes(stretch_terms, problem)

    ids = [int(r["recipe_id"]) for r in shell_recipes if r.get("recipe_id") is not None]
    lines_map = recipe_lines_by_id if recipe_lines_by_id is not None else _load_recipe_lines_for_ids(ids)

    accepted: list[dict[str, Any]] = []
    context_only: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    anchor_shares: list[float] = []
    stretch_shares: list[float] = []

    for entry in shell_recipes:
        rid = int(entry["recipe_id"])
        lines = lines_map.get(rid) or []
        shares = compute_recipe_structure_shares(
            lines,
            anchor_terms=anchor_terms,
            stretch_terms=stretch_terms,
            anchor_nodes=anchor_nodes,
            stretch_nodes=stretch_nodes,
        )
        fit = classify_structure_fit(shares, stretch_role=role)
        enriched = {
            **entry,
            "structure": {
                **shares,
                **fit,
                "stretch_role": role,
            },
        }
        if shares.get("has_grams"):
            anchor_shares.append(float(shares["anchor_share"]))
            stretch_shares.append(float(shares["stretch_share"]))
        verdict = fit["verdict"]
        if verdict == "pass":
            accepted.append(enriched)
        elif verdict == "soft":
            scale = float(fit.get("weight_scale") or 0.35)
            enriched = {
                **enriched,
                "weight": float(entry.get("weight") or 0.35) * scale,
                "similarity": float(entry.get("similarity") or 0.0) * (0.5 + 0.5 * scale),
            }
            accepted.append(enriched)
        elif verdict == "context_only":
            context_only.append(enriched)
        else:
            rejected.append(enriched)

    def _iqr(vals: list[float]) -> list[float] | None:
        if not vals:
            return None
        arr = np.asarray(vals, dtype=float)
        return [
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 75)),
        ]

    meta = {
        "structure_applied": True,
        "dish_structure": structure,
        "n_structure_checked": len(shell_recipes),
        "n_structure_passed": sum(
            1 for r in accepted if (r.get("structure") or {}).get("verdict") == "pass"
        ),
        "n_structure_soft": sum(
            1 for r in accepted if (r.get("structure") or {}).get("verdict") == "soft"
        ),
        "n_rejected_wrong_dominance": len(rejected),
        "n_context_only": len(context_only),
        "anchor_share_median": float(np.median(anchor_shares)) if anchor_shares else None,
        "stretch_share_median": float(np.median(stretch_shares)) if stretch_shares else None,
        "stretch_share_iqr": _iqr(stretch_shares),
        "anchor_basis_nodes": sorted(anchor_nodes),
        "stretch_basis_nodes": sorted(stretch_nodes),
        "reject_reasons": {
            str((r.get("structure") or {}).get("reason") or "unknown"): sum(
                1
                for x in rejected
                if (x.get("structure") or {}).get("reason")
                == (r.get("structure") or {}).get("reason")
            )
            for r in rejected
        },
    }
    return {
        "accepted": accepted,
        "context_only": context_only,
        "rejected": rejected,
        "meta": meta,
    }


def expand_neighborhood_by_queries(
    problem: dict[str, Any],
    queries: list[str],
    *,
    top_k_per_query: int = 12,
    max_recipes: int = 40,
    shell_weight: float = 0.35,
    focus_terms: list[str] | None = None,
    anchor_terms: list[str] | None = None,
    dish_structure: dict[str, Any] | None = None,
    recipe_lines_by_id: dict[int, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Merge query-retrieved recipes into problem retrieval_context + light share prior.

    ``focus_terms`` name the *stretch* ingredient (e.g. chicken). When present, a
    shell recipe MUST contain at least one focus term — otherwise embedding
    similarity just returns more copies of the base dish (plain carbonaras) and
    the expansion teaches the ratio terms nothing about the proposed ingredient.
    ``anchor_terms`` (identity foods like pasta/spaghetti) boost recipes that
    combine the stretch with the dish identity — the true co-occurrence signal.
    ``dish_structure`` (LLM-declared) enables gram-share verification so
    stretch-primary dishes with the anchor as a side are not harvested into
    ratio/share loss.

    Returns `{problem, shell_recipes, meta}`.
    """
    problem = dict(problem)
    queries = [str(q).strip() for q in (queries or []) if str(q).strip()]
    structure = normalize_dish_structure(dish_structure)
    if focus_terms is None:
        current_labels = [
            str(r.get("label") or r.get("name") or "")
            for r in ((problem.get("chosen_recipe") or {}).get("ingredients") or [])
        ]
        if structure and structure.get("stretch_ingredient"):
            focus_terms = derive_focus_terms(
                [structure["stretch_ingredient"]],
                current_labels,
                extra_texts=[str(problem.get("title") or "")],
            )
        if not focus_terms:
            focus_terms = derive_focus_terms(
                queries, current_labels, extra_texts=[str(problem.get("title") or "")]
            )
    focus = {str(t).lower() for t in (focus_terms or []) if t}
    anchors = {str(t).lower() for t in (anchor_terms or []) if t}
    if structure:
        for a in structure.get("anchor_ingredients") or []:
            anchors |= _tokenize(a)
    if not anchors:
        for r in ((problem.get("chosen_recipe") or {}).get("ingredients") or []):
            anchors |= _tokenize(str(r.get("label") or r.get("name") or ""))
    meta: dict[str, Any] = {
        "queries": queries,
        "focus_terms": sorted(focus),
        "dish_structure": structure,
        "n_added": 0,
        "n_focus_matched": 0,
        "method": None,
        "activated": False,
        "structure_applied": False,
    }
    if not queries:
        return {"problem": problem, "shell_recipes": [], "meta": meta}

    ctx = dict(problem.get("retrieval_context") or {})
    exclude = {int(x) for x in (ctx.get("core_recipe_ids") or []) if str(x).isdigit()}
    for rid in problem.get("neighbor_ids") or []:
        if str(rid).isdigit():
            exclude.add(int(rid))

    corpus = _load_recipe_text_corpus(exclude)
    if not corpus:
        meta["reason"] = "no_corpus"
        return {"problem": problem, "shell_recipes": [], "meta": meta}

    texts = [r["text"] for r in corpus]
    corpus_tokens = [_tokenize(t) for t in texts]

    def _focus_gate(j: int) -> bool:
        """Recipe must mention the stretch ingredient when focus terms exist."""
        if not focus:
            return True
        return bool(focus & corpus_tokens[j])

    def _anchor_bonus(j: int) -> float:
        """Reward recipes combining the stretch with dish-identity foods."""
        if not focus or not anchors:
            return 0.0
        return 0.15 if anchors & corpus_tokens[j] else 0.0

    q_emb, c_emb = _embed_queries_and_corpus(queries, texts)
    scored: dict[int, dict[str, Any]] = {}

    if q_emb is not None and c_emb is not None:
        import numpy as np

        meta["method"] = "embedding+token"
        sims = c_emb @ q_emb.T  # (n_corpus, n_queries)
        for qi, query in enumerate(queries):
            col = sims[:, qi].copy()
            if focus:
                # Rank only within focus-matching recipes so top_k isn't wasted
                # on plain copies of the base dish.
                mask = np.array([_focus_gate(j) for j in range(len(corpus))])
                col = np.where(mask, col, -np.inf)
            idx = np.argsort(-col)[:top_k_per_query]
            for j in idx:
                if not np.isfinite(col[j]):
                    continue
                s = float(sims[j, qi])
                token_s = _score_query_against_text(query, texts[j])
                combined = 0.65 * s + 0.35 * token_s + _anchor_bonus(j)
                if combined < 0.18:
                    continue
                rid = int(corpus[j]["recipe_id"])
                prev = scored.get(rid)
                if prev is None or combined > prev["similarity"]:
                    scored[rid] = {
                        "recipe_id": rid,
                        "title": corpus[j].get("title"),
                        "similarity": combined,
                        "query": query,
                        "labels": corpus[j].get("labels") or [],
                        "weight": float(shell_weight * min(1.0, combined / 0.35)),
                        "focus_matched": bool(focus & corpus_tokens[j]) if focus else None,
                    }
    else:
        meta["method"] = "token_overlap"
        for query in queries:
            ranked = sorted(
                (
                    (_score_query_against_text(query, r["text"]) + _anchor_bonus(j), j, r)
                    for j, r in enumerate(corpus)
                    if _focus_gate(j)
                ),
                key=lambda t: -t[0],
            )[:top_k_per_query]
            for score, j, r in ranked:
                if score < 0.12:
                    continue
                rid = int(r["recipe_id"])
                prev = scored.get(rid)
                if prev is None or score > prev["similarity"]:
                    scored[rid] = {
                        "recipe_id": rid,
                        "title": r.get("title"),
                        "similarity": float(score),
                        "query": query,
                        "labels": r.get("labels") or [],
                        "weight": float(shell_weight * min(1.0, score / 0.25)),
                        "focus_matched": bool(focus & corpus_tokens[j]) if focus else None,
                    }

    shell = sorted(scored.values(), key=lambda d: -d["similarity"])[:max_recipes]
    meta["n_focus_matched"] = sum(1 for s in shell if s.get("focus_matched"))
    if not shell:
        meta["reason"] = "no_hits_matching_focus_terms" if focus else "no_hits"
        return {"problem": problem, "shell_recipes": [], "meta": meta}

    # Gram-share structure verification (LLM declares, system verifies).
    verified = verify_shell_structure(
        shell,
        structure,
        problem=problem,
        recipe_lines_by_id=recipe_lines_by_id,
    )
    accepted = verified["accepted"]
    context_only = verified["context_only"]
    meta.update(verified.get("meta") or {})
    # Co-occurrence keeps accepted + context_only; share harvest uses accepted only.
    shell_for_context = accepted + context_only
    if not shell_for_context and shell:
        # Verifier rejected everything — keep raw shell for co-occurrence only so
        # the agent still sees titles, but do not harvest shares.
        shell_for_context = [{**e, "structure": {"verdict": "context_only"}} for e in shell]
        meta["n_context_only"] = len(shell_for_context)
        meta["n_structure_passed"] = 0
        accepted = []

    # Enrich neighbor label sets for co-occurrence retrieval
    neighbor_sets = [list(s) for s in (ctx.get("neighbor_label_sets") or [])]
    catalog = list(ctx.get("fdc_catalog") or [])
    existing_desc = {str(c.get("fdc_description") or "").lower() for c in catalog}
    for entry in shell_for_context:
        labels = [str(x).lower() for x in (entry.get("labels") or []) if x]
        if labels:
            neighbor_sets.append(labels)
        for lab in labels[:20]:
            if lab and lab not in existing_desc:
                catalog.append({"fdc_id": None, "fdc_description": lab, "from_query_shell": True})
                existing_desc.add(lab)

    ctx["neighbor_label_sets"] = neighbor_sets
    ctx["fdc_catalog"] = catalog
    ctx["query_shell_recipes"] = [
        {
            "recipe_id": e["recipe_id"],
            "title": e.get("title"),
            "similarity": e["similarity"],
            "query": e.get("query"),
            "weight": e.get("weight"),
            "focus_matched": e.get("focus_matched"),
            "structure_verdict": (e.get("structure") or {}).get("verdict"),
            "anchor_share": (e.get("structure") or {}).get("anchor_share"),
            "stretch_share": (e.get("structure") or {}).get("stretch_share"),
        }
        for e in shell_for_context
    ]
    # Only structure-verified (pass/soft) recipes feed share harvesting.
    ctx["structure_verified_shell_ids"] = [
        int(e["recipe_id"]) for e in accepted if e.get("recipe_id") is not None
    ]
    ctx["dish_structure"] = structure
    ctx["neighborhood_search_queries"] = queries
    ctx["neighborhood_structure_meta"] = {
        k: verified["meta"].get(k)
        for k in (
            "structure_applied",
            "n_structure_checked",
            "n_structure_passed",
            "n_structure_soft",
            "n_rejected_wrong_dominance",
            "n_context_only",
            "anchor_share_median",
            "stretch_share_median",
            "stretch_share_iqr",
            "dish_structure",
        )
    }
    problem["retrieval_context"] = ctx

    # Harvest FoodOn share samples from structure-verified shell recipes only.
    try:
        from recipe_opt_agent.ood_foodon import ensure_ingredient_nodes_in_loss

        target = set()
        for n in problem.get("ingredient_basis") or []:
            if n:
                target.add(str(n))
        for n in ctx.get("pending_basis_nodes") or []:
            target.add(str(n))
        if target:
            problem = ensure_ingredient_nodes_in_loss(problem, min_hits=5)
            meta["basis_hit_counts"] = problem.get("basis_hit_counts")
    except Exception as exc:
        meta["basis_enrich_error"] = str(exc)

    meta.update(
        {
            "n_added": len(shell_for_context),
            "n_harvest_eligible": len(accepted),
            "activated": True,
        }
    )
    return {"problem": problem, "shell_recipes": shell_for_context, "meta": meta}
