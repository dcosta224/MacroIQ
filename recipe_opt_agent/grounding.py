"""Ground LLM recipe drafts to FDC + optimization problem (x0, M)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from recipe_opt_agent.culinary_types import (
    content_tokens,
    families_compatible,
    families_for_text,
)
from recipe_opt_agent.draft_schema import DraftIngredient, RecipeDraft, parse_draft
from recipe_opt_agent.requirement_tags import (
    RequirementTag,
    filter_ingredients_by_tags,
    ingredient_passes_tags,
    tag_violations_for_ingredient,
)

# Strict thresholds — the old 0.05 token Jaccard admitted BBQ sauce→soy sauce.
DEFAULT_NEIGHBORHOOD_MIN_SCORE = 0.35
DEFAULT_BROADER_MIN_SCORE = 0.42


@dataclass
class GroundedLine:
    name: str
    grams: float
    role: str
    fdc_id: int | None
    label: str
    status: str  # matched | substituted | unresolved | weak_match
    substitute_from: str | None = None
    notes: str = ""
    match_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundingReport:
    matched: list[dict[str, Any]] = field(default_factory=list)
    substituted: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    weak_match: list[dict[str, Any]] = field(default_factory=list)
    dequant_cache_used: bool = False
    dequant_cache_hits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "substituted": self.substituted,
            "unresolved": self.unresolved,
            "weak_match": self.weak_match,
            "dequant_cache_used": self.dequant_cache_used,
            "dequant_cache_hits": self.dequant_cache_hits,
        }


def _token_overlap(a: str, b: str) -> float:
    ta = content_tokens(a)
    tb = content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _match_score(query: str, label: str) -> float:
    """Score a catalog label against a draft ingredient name."""
    q = (query or "").lower().strip()
    lab = (label or "").lower().strip()
    if not q or not lab:
        return 0.0
    q_fam = families_for_text(q)
    lab_fam = families_for_text(lab)
    if not families_compatible(q_fam, lab_fam):
        return 0.0

    score = _token_overlap(q, lab)
    if q in lab or lab in q:
        score = max(score, 0.85)
    # Shared culinary family is strong evidence when tokens overlap at least a little
    if q_fam and (q_fam & lab_fam) and score >= 0.15:
        score = max(score, 0.55)
    # Prefer exact content-token containment of the primary noun
    q_toks = content_tokens(q)
    lab_toks = content_tokens(lab)
    if q_toks and q_toks <= lab_toks:
        score = max(score, 0.7)
    return float(score)


def _search_catalog(
    query: str,
    catalog: list[dict[str, Any]],
    tags: list[RequirementTag],
    *,
    min_score: float = DEFAULT_NEIGHBORHOOD_MIN_SCORE,
) -> tuple[dict[str, Any] | None, float]:
    best: dict[str, Any] | None = None
    best_score = -1.0
    for row in catalog:
        label = str(row.get("fdc_description") or row.get("label") or "")
        if not ingredient_passes_tags(label, tags, fdc_description=label):
            continue
        score = _match_score(query, label)
        if score > best_score and score >= min_score:
            best_score = score
            best = row
    return best, (best_score if best is not None else 0.0)


def resolve_line_to_fdc(
    line: DraftIngredient,
    *,
    neighborhood_catalog: list[dict[str, Any]],
    broader_catalog: list[dict[str, Any]] | None,
    tags: list[RequirementTag],
    neighborhood_min_score: float = DEFAULT_NEIGHBORHOOD_MIN_SCORE,
    broader_min_score: float = DEFAULT_BROADER_MIN_SCORE,
    dequant_cache: Any | None = None,
) -> GroundedLine:
    """Neighborhood FDC first, then broader catalog — with culinary-type gates.

    When ``dequant_cache`` is provided, try a dequant-cache hit first (mass stem
    match for gram drafts; exact/volume-stem for non-mass).
    """
    if not ingredient_passes_tags(line.name, tags):
        return GroundedLine(
            name=line.name,
            grams=line.grams,
            role=line.role,
            fdc_id=None,
            label=line.name,
            status="unresolved",
            notes="draft line violates requirement tags",
        )

    if dequant_cache is not None:
        try:
            hit = dequant_cache.lookup(line.name, grams=line.grams)
        except Exception:
            hit = None
        if hit is not None and hit.fdc_id is not None:
            return GroundedLine(
                name=line.name,
                grams=line.grams,
                role=line.role,
                fdc_id=int(hit.fdc_id),
                label=str(hit.description or line.name),
                status="matched",
                notes=(
                    f"dequant_cache:{hit.match_mode} key={hit.cache_key!r} "
                    f"score={hit.score:.3f}"
                ),
                match_score=float(hit.score),
            )

    hit, score = _search_catalog(
        line.name, neighborhood_catalog, tags, min_score=neighborhood_min_score
    )
    if hit:
        fid = hit.get("fdc_id")
        label = str(hit.get("fdc_description") or hit.get("label") or line.name)
        return GroundedLine(
            name=line.name,
            grams=line.grams,
            role=line.role,
            fdc_id=int(fid) if fid is not None else None,
            label=label,
            status="matched",
            notes=f"score={score:.3f}",
            match_score=score,
        )

    if broader_catalog:
        hit, score = _search_catalog(
            line.name, broader_catalog, tags, min_score=broader_min_score
        )
        if hit:
            fid = hit.get("fdc_id")
            label = str(hit.get("fdc_description") or hit.get("label") or line.name)
            return GroundedLine(
                name=line.name,
                grams=line.grams,
                role=line.role,
                fdc_id=int(fid) if fid is not None else None,
                label=label,
                status="substituted",
                substitute_from=line.name,
                notes=f"score={score:.3f}",
                match_score=score,
            )

    # Probe best weak hit for diagnostics (not used in x0)
    weak_hit, weak_score = _search_catalog(
        line.name, neighborhood_catalog or list(broader_catalog or []), tags, min_score=0.05
    )
    notes = "no culinary-compatible FDC above threshold"
    if weak_hit is not None:
        weak_label = str(weak_hit.get("fdc_description") or weak_hit.get("label") or "")
        notes = f"rejected weak candidate '{weak_label}' score={weak_score:.3f}"

    return GroundedLine(
        name=line.name,
        grams=line.grams,
        role=line.role,
        fdc_id=None,
        label=line.name,
        status="unresolved",
        notes=notes,
        match_score=weak_score if weak_hit is not None else None,
    )


def _build_problem_from_fdc_rows(
    rows: list[dict[str, Any]],
    *,
    title: str,
    ingredient_basis: list[str | None] | None = None,
    basis_samples: dict[str, list[float]] | None = None,
    ratio_samples: list[float] | None = None,
    marginal_nodes: list[str] | None = None,
    retrieval_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build x0/M from resolved FDC rows (local store by default)."""
    import pandas as pd

    from canonical_optimization import atwater_kcal
    from mvp_data import build_recipe_macro_inputs, fetch_food_nutrients_for_recipe
    from weighted_empirical_opt import MARGINAL_COLUMN_NODES

    ingredients = pd.DataFrame(
        [
            {
                "ingredient_idx": i,
                "fdc_id": int(r["fdc_id"]),
                "gram_weight": float(r["grams"]),
                "fdc_description": str(r.get("label") or r.get("name") or ""),
            }
            for i, r in enumerate(rows)
            if r.get("fdc_id") is not None
        ]
    )
    if ingredients.empty:
        raise ValueError("No resolved FDC rows for problem build")

    food_nutrients = fetch_food_nutrients_for_recipe(
        None, ingredients["fdc_id"].astype(int).tolist()
    )

    x0, M = build_recipe_macro_inputs(ingredients, food_nutrients)
    basis = ingredient_basis or [str(r.get("role") or r.get("label")) for r in rows if r.get("fdc_id")]
    if len(basis) != len(x0):
        basis = [str(r.get("role") or "ing") for r in rows if r.get("fdc_id")]

    ing_list = [
        {
            "ingredient_idx": int(row.ingredient_idx),
            "label": str(row.fdc_description),
            "fdc_id": int(row.fdc_id),
            "grams": float(row.gram_weight),
        }
        for row in ingredients.itertuples(index=False)
    ]

    samples = basis_samples or {}
    if not samples:
        for b in set(basis):
            if b:
                samples[str(b)] = [0.45, 0.5, 0.55]

    return {
        "x0": x0.tolist(),
        "M": M.tolist(),
        "ingredient_basis": basis,
        "basis_samples": samples,
        "ratio_samples": list(ratio_samples or []),
        "marginal_nodes": marginal_nodes or [nid for _, nid in MARGINAL_COLUMN_NODES],
        "kcal_target": float(atwater_kcal(x0, M)),
        "total_mass": float(x0.sum()),
        "title": title,
        "chosen_recipe": {
            "source": "creative_grounded",
            "title": title,
            "ingredients": ing_list,
        },
        "retrieval_context": retrieval_context or {},
    }


def _assign_synthetic_fdc(line: DraftIngredient, tags: list[RequirementTag]) -> GroundedLine | None:
    """Offline fallback: invent a stable synthetic fdc_id so the stub problem can build."""
    if not ingredient_passes_tags(line.name, tags):
        return None
    # Stable synthetic ids in a reserved range (not real USDA).
    key = (line.role or line.name).lower().strip() or "ingredient"
    digest = abs(hash(key)) % 900000 + 100000
    return GroundedLine(
        name=line.name,
        grams=line.grams,
        role=line.role,
        fdc_id=int(digest),
        label=line.name,
        status="substituted",
        substitute_from=line.name,
        notes="synthetic offline FDC",
    )


def _build_offline_stub_problem(
    grounded: list[GroundedLine],
    *,
    title: str,
    retrieval_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Offline/test stub when DB unavailable: simple macro matrix from roles."""
    resolved = [g for g in grounded if g.fdc_id is not None and g.grams > 0]
    if not resolved:
        # Last resort: keep any positive-gram lines with synthetic ids already attached upstream.
        raise ValueError("No grounded lines for offline stub")

    # Per-gram macros [protein, fat, carbs, energy] — coarse defaults by role/name
    def per_gram(g: GroundedLine) -> list[float]:
        name = (g.name + " " + g.label + " " + (g.role or "")).lower()
        if any(x in name for x in ("chicken", "turkey", "tofu", "protein", "egg white", "seitan")):
            return [0.31, 0.03, 0.0, 1.65]
        if any(x in name for x in ("egg",)):
            return [0.13, 0.11, 0.01, 1.55]
        if any(x in name for x in ("cheese", "parmesan", "pecorino", "ricotta")):
            return [0.25, 0.33, 0.01, 4.0]
        if any(x in name for x in ("pasta", "spaghetti", "noodle")):
            return [0.05, 0.01, 0.25, 1.31]
        if any(x in name for x in ("oil", "butter")):
            return [0.0, 1.0, 0.0, 8.84]
        if any(x in name for x in ("bacon", "pork", "guanciale", "pancetta")):
            return [0.12, 0.42, 0.0, 4.5]
        if any(x in name for x in ("mushroom",)):
            return [0.03, 0.01, 0.03, 0.22]
        if any(x in name for x in ("rice", "carb")):
            return [0.03, 0.01, 0.28, 1.3]
        return [0.1, 0.1, 0.1, 2.0]

    n = len(resolved)
    M = np.zeros((4, n), dtype=float)
    x0 = np.zeros(n, dtype=float)
    basis: list[str] = []
    for i, g in enumerate(resolved):
        M[:, i] = np.array(per_gram(g), dtype=float)
        x0[i] = float(g.grams)
        basis.append(g.role or g.label or g.name)

    samples: dict[str, list[float]] = {}
    for b in set(basis):
        samples[b] = [0.45, 0.5, 0.55]

    from weighted_empirical_opt import MARGINAL_COLUMN_NODES, atwater_kcal

    ing_list = [
        {
            "ingredient_idx": i,
            "label": g.label or g.name,
            "fdc_id": g.fdc_id,
            "grams": g.grams,
            "original_grams": g.grams,
            "quantity": None,
            "unit": None,
        }
        for i, g in enumerate(resolved)
    ]
    return {
        "x0": x0.tolist(),
        "M": M.tolist(),
        "ingredient_basis": basis,
        "basis_samples": samples,
        "ratio_samples": [],
        "marginal_nodes": [nid for _, nid in MARGINAL_COLUMN_NODES],
        "kcal_target": float(atwater_kcal(x0, M)),
        "total_mass": float(x0.sum()),
        "title": title,
        "chosen_recipe": {
            "source": "creative_grounded_offline",
            "title": title,
            "ingredients": ing_list,
        },
        "retrieval_context": retrieval_context or {},
        "grounding_offline": True,
    }


def ground_draft_to_problem(
    draft: RecipeDraft | dict[str, Any],
    *,
    requirement_tags: list[RequirementTag],
    neighborhood_catalog: list[dict[str, Any]] | None = None,
    broader_catalog: list[dict[str, Any]] | None = None,
    basis_samples: dict[str, list[float]] | None = None,
    ratio_samples: list[float] | None = None,
    retrieval_context: dict[str, Any] | None = None,
    offline: bool = False,
    use_dequant_cache: bool = True,
    dequant_cache: Any | None = None,
) -> tuple[dict[str, Any], GroundingReport, dict[str, Any]]:
    """Ground draft → problem dict + report + chosen_recipe."""
    if not isinstance(draft, RecipeDraft):
        draft = parse_draft(draft)

    nb_cat = list(neighborhood_catalog or [])
    broad = list(broader_catalog or [])
    if not broad and retrieval_context:
        broad = list(retrieval_context.get("fdc_catalog") or [])

    cache = dequant_cache
    dequant_load_error: str | None = None
    # Always prefer the dequant LLM cache when enabled — including offline creative —
    # so curated FDC judgments are used before catalog / synthetic fallbacks.
    if cache is None and use_dequant_cache:
        try:
            from eval_fdc_grounding_ui.draft_cache import get_shared_dequant_cache

            cache = get_shared_dequant_cache()
        except Exception as exc:
            cache = None
            dequant_load_error = str(exc)

    report = GroundingReport(dequant_cache_used=cache is not None)
    grounded: list[GroundedLine] = []

    for ing in draft.ingredients:
        if ing.grams <= 0:
            continue
        gl = resolve_line_to_fdc(
            ing,
            neighborhood_catalog=nb_cat,
            broader_catalog=broad,
            tags=requirement_tags,
            dequant_cache=cache,
        )
        if gl.notes.startswith("dequant_cache:"):
            report.dequant_cache_hits += 1
        # Offline / empty-catalog fallback: keep tag-compliant lines with synthetic FDC.
        if gl.status == "unresolved" and (offline or not nb_cat and not broad):
            synth = _assign_synthetic_fdc(ing, requirement_tags)
            if synth is not None:
                gl = synth
        grounded.append(gl)
        row = gl.to_dict()
        if gl.status == "matched":
            report.matched.append(row)
        elif gl.status == "substituted":
            report.substituted.append(row)
        elif gl.status == "weak_match":
            report.weak_match.append(row)
            report.unresolved.append(row)
        else:
            report.unresolved.append(row)

    if dequant_load_error and use_dequant_cache:
        # Keep a breadcrumb for UI / tools without failing the run.
        report.unresolved.append(
            {
                "name": "_dequant_cache",
                "status": "cache_load_error",
                "notes": dequant_load_error,
            }
        )

    # If still nothing resolved but we have tag-ok draft lines, force synthetic grounding.
    if not any(g.fdc_id is not None for g in grounded):
        forced: list[GroundedLine] = []
        for ing in draft.ingredients:
            if ing.grams <= 0:
                continue
            synth = _assign_synthetic_fdc(ing, requirement_tags)
            if synth is not None:
                forced.append(synth)
                report.substituted.append(synth.to_dict())
        grounded = forced

    if not grounded:
        raise ValueError(
            "Could not ground any ingredients (all lines unresolved or blocked by requirement tags)."
        )

    resolved_rows = [
        {
            "name": g.name,
            "grams": g.grams,
            "role": g.role,
            "fdc_id": g.fdc_id,
            "label": g.label,
        }
        for g in grounded
        if g.fdc_id is not None
    ]

    ctx = dict(retrieval_context or {})
    ctx.setdefault("starting_ingredients", [])
    ctx["starting_ingredients"] = [
        {
            "ingredient_idx": i,
            "label": r["label"],
            "fdc_id": r["fdc_id"],
            "grams": r["grams"],
            "original_grams": r["grams"],
            "quantity": None,
            "unit": None,
        }
        for i, r in enumerate(resolved_rows)
    ]
    ctx["starting_fdc"] = [int(r["fdc_id"]) for r in resolved_rows if r.get("fdc_id")]
    ctx["starting_labels"] = [str(r["label"]).lower() for r in resolved_rows]

    chosen_recipe = {
        "source": "creative_draft",
        "title": draft.title,
        "ingredients": ctx["starting_ingredients"],
        "draft_notes": draft.notes,
        "servings": draft.servings,
    }

    try:
        if offline:
            problem = _build_offline_stub_problem(
                grounded,
                title=draft.title,
                retrieval_context=ctx,
            )
        else:
            problem = _build_problem_from_fdc_rows(
                resolved_rows,
                title=draft.title,
                basis_samples=basis_samples,
                ratio_samples=ratio_samples,
                retrieval_context=ctx,
            )
    except Exception:
        problem = _build_offline_stub_problem(
            grounded,
            title=draft.title,
            retrieval_context=ctx,
        )
        problem["grounding_fallback"] = True

    problem["chosen_recipe"] = chosen_recipe
    problem["grounding_report"] = report.to_dict()
    problem["grounded_r0"] = ctx["starting_ingredients"]
    # Preserve FoodOn neighborhood geometry when creative mode reuses a stub problem.
    if ctx.get("rollup_chains") and not problem.get("rollup_chains"):
        problem["rollup_chains"] = ctx["rollup_chains"]
    if ctx.get("basis_nodes") and not problem.get("basis_nodes"):
        problem["basis_nodes"] = list(ctx["basis_nodes"])
    fdc_basis = ctx.get("fdc_basis") or {}
    if fdc_basis and not problem.get("ingredient_basis"):
        problem["ingredient_basis"] = [
            fdc_basis.get(str(r["fdc_id"])) for r in resolved_rows
        ]
    elif fdc_basis and problem.get("ingredient_basis"):
        # Prefer FoodOn basis nodes over role strings when available.
        basis = list(problem["ingredient_basis"])
        for i, r in enumerate(resolved_rows):
            node = fdc_basis.get(str(r["fdc_id"]))
            if node and i < len(basis):
                basis[i] = node
        problem["ingredient_basis"] = basis
    leaves = []
    for r in resolved_rows:
        leaf = None
        # Best-effort: reverse from fdc_basis is a basis node, not a leaf.
        leaves.append(leaf)
    if not problem.get("ingredient_foodon_leaves"):
        problem["ingredient_foodon_leaves"] = leaves
    from recipe_opt_agent.foodon_basis_report import attach_foodon_basis_report

    attach_foodon_basis_report(problem)
    chosen_recipe = problem.get("chosen_recipe") or chosen_recipe
    return problem, report, chosen_recipe
