"""Gate LLM recipe edits on neighborhood co-occurrence / catalog support."""

from __future__ import annotations

from typing import Any

from recipe_opt_agent.culinary_types import content_tokens, families_compatible, families_for_text
from recipe_opt_agent.grounding import _match_score


def _catalog_labels(problem: dict[str, Any] | None) -> list[str]:
    ctx = (problem or {}).get("retrieval_context") or {}
    labels: list[str] = []
    for row in ctx.get("fdc_catalog") or []:
        lab = str(row.get("fdc_description") or row.get("label") or "").strip()
        if lab:
            labels.append(lab)
    # Structure-verified / expansion harvest labels if present
    for key in ("expansion_harvest_labels", "neighborhood_ingredient_labels", "supported_labels"):
        for lab in ctx.get(key) or []:
            s = str(lab).strip()
            if s:
                labels.append(s)
    # Starting / current recipe labels also count as attested
    for row in (problem or {}).get("grounded_r0") or []:
        lab = str(row.get("label") or "").strip()
        if lab:
            labels.append(lab)
    for row in ((problem or {}).get("chosen_recipe") or {}).get("ingredients") or []:
        lab = str(row.get("label") or "").strip()
        if lab:
            labels.append(lab)
    return labels


def label_supported_by_neighborhood(
    label: str,
    *,
    problem: dict[str, Any] | None,
    min_score: float = 0.35,
) -> tuple[bool, str]:
    """True if ``label`` is attested by neighborhood catalog / harvest labels."""
    lab = (label or "").strip()
    if not lab:
        return False, "empty_label"
    catalog = _catalog_labels(problem)
    if not catalog:
        # No catalog available (offline stub) — do not block.
        return True, "no_catalog"
    best = 0.0
    best_lab = ""
    q_fam = families_for_text(lab)
    for cand in catalog:
        if not families_compatible(q_fam, families_for_text(cand)):
            continue
        score = _match_score(lab, cand)
        if score > best:
            best = score
            best_lab = cand
    if best >= min_score:
        return True, f"supported_by:{best_lab}|score={best:.3f}"
    return False, f"unsupported_best:{best_lab or 'none'}|score={best:.3f}"


def filter_candidates_by_neighborhood_support(
    candidates: list[dict[str, Any]],
    *,
    problem: dict[str, Any] | None,
    min_score: float = 0.35,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop add/swap candidates whose labels are not neighborhood-attested."""
    from recipe_opt_agent.clash_gates import is_denylist_label
    from recipe_opt_agent.culinary_types import CONFLICTING_FAMILIES

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for c in candidates:
        action = str(c.get("action") or "")
        if action not in {"add", "swap"}:
            kept.append(c)
            continue
        # OOD branch is allowed only when explicitly flagged AND later decide
        # still sees structure support; here we still require catalog support
        # unless meta.allow_ungrounded is set (tests / explicit escape hatch).
        meta = c.get("meta") or {}
        if meta.get("allow_ungrounded"):
            kept.append(c)
            continue
        label = str(c.get("label") or c.get("name") or "")
        denied, deny_detail = is_denylist_label(label)
        if denied:
            dropped.append(
                {
                    "candidate": c,
                    "reason": "denylist_label",
                    "detail": deny_detail,
                }
            )
            continue
        ok, reason = label_supported_by_neighborhood(
            label, problem=problem, min_score=min_score
        )
        if ok:
            # Extra guard: ideation "onion" must not ground to onion rings even if
            # a fuzzy catalog score somehow passes family checks.
            idea = str(meta.get("idea_ingredient") or meta.get("query") or "")
            if idea:
                q_fam = families_for_text(idea)
                l_fam = families_for_text(label)
                blocked = False
                for q in q_fam:
                    for lab in l_fam:
                        if (q, lab) in CONFLICTING_FAMILIES:
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    dropped.append(
                        {
                            "candidate": c,
                            "reason": "family_conflict_grounding",
                            "detail": f"idea={idea!r} → label={label!r}",
                        }
                    )
                    continue
            kept.append({**c, "neighborhood_support": reason})
        else:
            dropped.append(
                {
                    "candidate": c,
                    "reason": "no_neighborhood_support",
                    "detail": reason,
                }
            )
    return kept, dropped


def missing_high_hit_basis_nodes(
    foodon_basis_report: dict[str, Any] | None,
    *,
    min_hits: int = 8,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Neighborhood basis nodes with high hit counts that are absent from the recipe."""
    nodes = list((foodon_basis_report or {}).get("basis_nodes") or [])
    missing = [
        n
        for n in nodes
        if not n.get("in_current_recipe") and int(n.get("n_hits") or 0) >= min_hits
    ]
    missing.sort(key=lambda n: -int(n.get("n_hits") or 0))
    return missing[:top_k]


def identity_critical_unresolved(
    grounding_report: dict[str, Any] | None,
    identity_roles: list[str] | None,
) -> list[dict[str, Any]]:
    """Unresolved draft lines whose role looks identity-critical."""
    roles = {str(r).lower() for r in (identity_roles or [])}
    critical_role_cues = roles | {
        "protein",
        "main protein",
        "carb",
        "rice",
        "sauce",
        "bbq_sauce",
        "pasta",
        "noodle",
        "grape_leaf",
        "wrapper",
        "dough",
        "crust",
        "flour",
        "meat",
        "ground_meat",
    }
    out: list[dict[str, Any]] = []
    for row in (grounding_report or {}).get("unresolved") or []:
        role = str(row.get("role") or "").lower()
        name = str(row.get("name") or "").lower()
        if role in critical_role_cues or any(c in name for c in ("rice", "sauce", "beef", "pork", "rib", "flour", "leaf")):
            out.append(row)
    return out
