"""Augmentation candidate retrieval: co-occurrence + geom + LP shortlist."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from hull_geometry import TargetBox, ingredient_pfc_fractions, region_intersects_hull
from weighted_empirical_opt import optimize_weighted_empirical_obj

# Deny isolates / powders / supplements by substring (lowercase).
DEFAULT_DENYLIST_SUBSTR = (
    "whey",
    "isolate",
    "protein powder",
    "casein powder",
    "supplement",
    "multivitamin",
    "creatine",
)


@dataclass
class ModCandidate:
    candidate_id: str
    action: str  # add | swap | remove
    label: str
    fdc_id: int | None = None
    foodon_id: str | None = None
    foodon_path: str | None = None
    role: str | None = None
    identity_critical_target: bool = False
    cooccurrence: float = 0.0
    geom_score: float = 0.0
    L_star: float | None = None
    search_method: str = "cooccurrence+geom+lp"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _denied(label: str, denylist: tuple[str, ...] = DEFAULT_DENYLIST_SUBSTR) -> bool:
    s = label.lower()
    return any(tok in s for tok in denylist)


def cooccurrence_scores(
    neighbor_ingredient_sets: list[set[str]],
    core: set[str],
    *,
    exclude: set[str],
) -> dict[str, float]:
    """P(c | core present): among neighbors containing all core labels, fraction with c."""
    if not neighbor_ingredient_sets or not core:
        return {}
    with_core = [s for s in neighbor_ingredient_sets if core.issubset(s)]
    if not with_core:
        with_core = neighbor_ingredient_sets
    counts: dict[str, int] = {}
    for s in with_core:
        for item in s:
            if item in exclude or item in core:
                continue
            counts[item] = counts.get(item, 0) + 1
    n = len(with_core)
    return {k: v / n for k, v in counts.items()}


def geom_vertex_score(
    M: np.ndarray,
    v: np.ndarray,
    box: TargetBox,
    *,
    kcal_target: float,
    p_B_star: tuple[float, float, float] | None,
    p_T_star: tuple[float, float, float] | None,
) -> float:
    """Higher is better: prefer vertices that open T and align with corridor."""
    v = np.asarray(v, dtype=float).ravel()
    # Append a synthetic column with macros = v (per-gram fractions → fake density).
    # Use unit density so PFC of the food equals v.
    protein_g = v[0] / 4.0 if v[0] > 0 else 0.0
    carbs_g = v[1] / 4.0 if v[1] > 0 else 0.0
    fat_g = v[2] / 9.0 if v[2] > 0 else 0.0
    # Amounts are per-gram nutrient grams in M convention (nutrient g per food g).
    # M rows are already per-gram; set so Atwater fractions recover v.
    # kcal/g = 4p_g + 4c_g + 9f_g; fractions = 4p/k etc.
    # Choose p_g,c_g,f_g proportional.
    col = np.array([protein_g, fat_g, carbs_g, 0.0], dtype=float).reshape(4, 1)
    # If all zero, useless.
    if float(col[:3].sum()) <= 0:
        return -1e9
    M2 = np.hstack([M, col])
    before = region_intersects_hull(M, box, kcal_target=kcal_target)
    after = region_intersects_hull(M2, box, kcal_target=kcal_target)
    score = 0.0
    if after["lp_feasible"] and not before["lp_feasible"]:
        score += 2.0
    elif after["lp_feasible"]:
        score += 1.0
    if p_T_star is not None:
        score -= float(np.linalg.norm(v - np.asarray(p_T_star), ord=1))
    if p_B_star is not None and p_T_star is not None:
        # Prefer v beyond p_T relative to p_B
        direction = np.asarray(p_T_star) - np.asarray(p_B_star)
        if np.linalg.norm(direction) > 1e-9:
            align = float(np.dot(v - np.asarray(p_B_star), direction) / (np.linalg.norm(direction) + 1e-12))
            score += 0.1 * align
    return float(score)


def evaluate_add(
    x0: np.ndarray,
    M: np.ndarray,
    *,
    new_col: np.ndarray,
    new_basis_id: str | None,
    ingredient_basis: list[str | None],
    marginal_nodes: list[str],
    basis_samples: dict[str, np.ndarray],
    ratio_samples: np.ndarray,
    kcal_target: float,
    total_mass: float,
    box: TargetBox,
    initial_grams: float = 50.0,
) -> dict[str, Any]:
    """Append ingredient column and re-optimize; return L_star and x_opt."""
    new_col = np.asarray(new_col, dtype=float).ravel()
    if new_col.shape[0] == 3:
        # PFC fractions → per-gram macros
        p, c, f = new_col
        # Solve for grams nutrient per gram food with unit-ish kcal.
        protein_g = p / 4.0
        carbs_g = c / 4.0
        fat_g = f / 9.0
        new_col = np.array([protein_g, fat_g, carbs_g, 0.0], dtype=float)
    if new_col.shape[0] < 4:
        new_col = np.pad(new_col, (0, 4 - new_col.shape[0]))
    M2 = np.hstack([M, new_col.reshape(4, 1)])
    x0_2 = np.concatenate([np.asarray(x0, dtype=float), [initial_grams]])
    # Renormalize mass: keep total_mass, scale x0_2
    if x0_2.sum() > 0:
        x0_2 = x0_2 * (total_mass / x0_2.sum())
    basis2 = list(ingredient_basis) + [new_basis_id]
    opt = optimize_weighted_empirical_obj(
        x0_2,
        M2,
        marginal_nodes=marginal_nodes,
        basis_samples=basis_samples,
        ratio_samples=ratio_samples,
        ingredient_basis=basis2,
        kcal_target=kcal_target,
        protein_frac_min=box.protein_min,
        protein_frac_max=box.protein_max,
        carb_frac_min=box.carb_min,
        carb_frac_max=box.carb_max,
        fat_frac_min=box.fat_min,
        fat_frac_max=box.fat_max,
        total_mass=total_mass,
    )
    return {
        "feasible": bool(opt.get("feasible")),
        "L_star": float(opt["objective"]) if opt.get("feasible") else None,
        "status": opt.get("status"),
        "x_opt": opt.get("x_opt"),
        "M": M2,
        "x0": x0_2,
        "ingredient_basis": basis2,
    }


def nutrient_direction_proxy(
    v_pfc: np.ndarray,
    p_current: np.ndarray,
    box: TargetBox,
) -> float:
    """How well a candidate's PFC pushes the recipe toward the target box center.

    Positive → moving toward the box; scaled by remaining gap so an aligned
    candidate matters less when already inside the box.
    """
    v = np.asarray(v_pfc, dtype=float).ravel()[:3]
    p = np.asarray(p_current, dtype=float).ravel()[:3]
    center = np.array(
        [
            0.5 * (box.protein_min + box.protein_max),
            0.5 * (box.carb_min + box.carb_max),
            0.5 * (box.fat_min + box.fat_max),
        ],
        dtype=float,
    )
    gap = center - p
    gap_norm = float(np.linalg.norm(gap))
    if gap_norm < 1e-9:
        return 0.0
    direction = gap / gap_norm
    return float(np.dot(v - p, direction))


def share_dilution_proxy(
    x0: np.ndarray,
    *,
    total_mass: float,
    ingredient_basis: list[str | None],
    basis_samples: dict[str, np.ndarray],
    alpha: float = 0.1,
    new_basis_id: str | None = None,
    remove_idx: int | None = None,
) -> float:
    """Approximate post-edit ratio loss without an LP.

    Adds mass share ``alpha`` for a new ingredient (diluting existing shares by
    ``1 - alpha``) and/or removes line ``remove_idx`` (redistributing its mass
    proportionally). Returns mean per-ingredient Wasserstein loss on the
    resulting share vector — O(len(samples)), no solver.
    """
    x = np.asarray(x0, dtype=float).copy()
    basis = list(ingredient_basis)
    if total_mass <= 0 or x.size == 0:
        return 0.0

    if remove_idx is not None and 0 <= int(remove_idx) < x.size:
        removed = float(x[int(remove_idx)])
        x[int(remove_idx)] = 0.0
        rest = float(x.sum())
        if rest > 0 and removed > 0:
            x = x * ((rest + removed) / rest)
        basis[int(remove_idx)] = None

    shares = x / max(float(x.sum()), 1e-9)
    losses: list[float] = []
    scale = 1.0 - alpha if (new_basis_id is not None and alpha > 0) else 1.0
    for share, node_id in zip(shares, basis, strict=False):
        if node_id is None:
            continue
        samples = basis_samples.get(node_id)
        if samples is None or len(samples) == 0:
            continue
        z = float(share) * scale
        losses.append(float(np.mean(np.abs(np.asarray(samples, dtype=float) - z))))
    if new_basis_id is not None and alpha > 0:
        samples = basis_samples.get(new_basis_id)
        if samples is not None and len(samples) > 0:
            losses.append(float(np.mean(np.abs(np.asarray(samples, dtype=float) - alpha))))
    if not losses:
        return 0.0
    return float(np.mean(losses))


def retrieve_for_slot(
    slot: dict[str, Any],
    *,
    pool: list[dict[str, Any]],
    core_labels: set[str],
    neighbor_sets: list[set[str]],
    current_ingredients: list[dict[str, Any]],
    x0: np.ndarray,
    M: np.ndarray,
    box: TargetBox,
    kcal_target: float,
    total_mass: float,
    ingredient_basis: list[str | None],
    basis_samples: dict[str, np.ndarray],
    p_current: tuple[float, float, float] | None = None,
    top_k: int = 6,
    denylist: tuple[str, ...] = DEFAULT_DENYLIST_SUBSTR,
) -> list[ModCandidate]:
    """Slot-conditioned candidate shortlist with proxy cards (no LP).

    Actions generated depend on ``slot['preferred_actions']``:
    - add: catalog foods ranked by cooc + geom + nutrient/ratio proxies
    - swap: replace a current line (target_line_label or any non-critical) with a catalog food
    - remove: current lines (target_line_label first)
    """
    preferred = tuple(slot.get("preferred_actions") or ("add",))
    slot_id = str(slot.get("slot_id") or "slot")
    target_label = slot.get("target_line_label")
    out: list[ModCandidate] = []

    cooc = cooccurrence_scores(neighbor_sets, core_labels, exclude=set(core_labels))
    p_cur = np.asarray(p_current if p_current is not None else (0.0, 0.0, 0.0), dtype=float)

    def _proxy_card(item: dict[str, Any], *, alpha: float, remove_idx: int | None) -> dict[str, float]:
        pfc = item.get("pfc")
        nutrient = 0.0
        if pfc is not None:
            nutrient = nutrient_direction_proxy(np.asarray(pfc, dtype=float), p_cur, box)
        new_basis = item.get("basis_node")
        ratio = share_dilution_proxy(
            x0,
            total_mass=total_mass,
            ingredient_basis=ingredient_basis,
            basis_samples=basis_samples,
            alpha=alpha,
            new_basis_id=new_basis,
            remove_idx=remove_idx,
        )
        return {"delta_nutrient_proxy": float(nutrient), "delta_ratio_proxy": float(ratio)}

    # --- adds ---
    if "add" in preferred:
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in pool:
            label = str(item.get("label") or "")
            if not label or _denied(label, denylist) or item.get("in_basis"):
                continue
            c = float(cooc.get(label.lower(), cooc.get(label, item.get("cooccurrence", 0.0))))
            geom = 0.0
            if item.get("pfc") is not None:
                geom = geom_vertex_score(
                    M,
                    np.asarray(item["pfc"], dtype=float),
                    box,
                    kcal_target=kcal_target,
                    p_B_star=None,
                    p_T_star=None,
                )
            card = _proxy_card(item, alpha=0.1, remove_idx=None)
            score = c + 0.5 * max(geom, -2.0) + card["delta_nutrient_proxy"] - card["delta_ratio_proxy"]
            scored.append((score, {**item, "_cooc": c, "_geom": geom, "_card": card}))
        scored.sort(key=lambda t: (-t[0], str(t[1].get("label"))))
        for _, item in scored[:top_k]:
            out.append(
                ModCandidate(
                    candidate_id=f"{slot_id}::add_{item.get('fdc_id') or item.get('id')}",
                    action="add",
                    label=str(item["label"]),
                    fdc_id=item.get("fdc_id"),
                    foodon_id=item.get("foodon_id"),
                    role=item.get("role"),
                    cooccurrence=float(item.get("_cooc", 0.0)),
                    geom_score=float(item.get("_geom", 0.0)),
                    search_method="slot_retrieval",
                    meta={
                        "slot_id": slot_id,
                        "pfc": item.get("pfc"),
                        "basis_node": item.get("basis_node"),
                        "macros_per_g": item.get("macros_per_g"),
                        **item.get("_card", {}),
                    },
                )
            )

    # --- swaps ---
    if "swap" in preferred:
        swap_lines = [
            (i, row)
            for i, row in enumerate(current_ingredients)
            if target_label is None
            or str(row.get("label") or "").lower() == str(target_label).lower()
            or str(target_label).lower() in str(row.get("label") or "").lower()
        ]
        # Take best add-style replacements for each target line (cap total).
        repl_pool = [
            item
            for item in pool
            if item.get("label") and not _denied(str(item["label"]), denylist) and not item.get("in_basis")
        ]
        repl_pool.sort(
            key=lambda it: -float(cooc.get(str(it["label"]).lower(), it.get("cooccurrence", 0.0)))
        )
        n_swaps = 0
        for i, row in swap_lines[:2]:
            for item in repl_pool[:3]:
                if n_swaps >= top_k:
                    break
                card = _proxy_card(item, alpha=0.1, remove_idx=i)
                out.append(
                    ModCandidate(
                        candidate_id=f"{slot_id}::swap_{i}_{item.get('fdc_id') or item.get('id')}",
                        action="swap",
                        label=str(item["label"]),
                        fdc_id=item.get("fdc_id"),
                        foodon_id=item.get("foodon_id"),
                        role=item.get("role"),
                        cooccurrence=float(
                            cooc.get(str(item["label"]).lower(), item.get("cooccurrence", 0.0))
                        ),
                        search_method="slot_retrieval",
                        meta={
                            "slot_id": slot_id,
                            "swap_out_idx": i,
                            "swap_out_label": row.get("label"),
                            "swap_out_grams": row.get("grams"),
                            "pfc": item.get("pfc"),
                            "basis_node": item.get("basis_node"),
                            "macros_per_g": item.get("macros_per_g"),
                            **card,
                        },
                    )
                )
                n_swaps += 1

    # --- removes ---
    if "remove" in preferred:
        for i, row in enumerate(current_ingredients):
            label = str(row.get("label") or "")
            if target_label is not None and str(target_label).lower() not in label.lower():
                continue
            card = _proxy_card({"pfc": None}, alpha=0.0, remove_idx=i)
            out.append(
                ModCandidate(
                    candidate_id=f"{slot_id}::remove_{i}_{row.get('fdc_id')}",
                    action="remove",
                    label=label,
                    fdc_id=row.get("fdc_id"),
                    search_method="slot_retrieval",
                    meta={
                        "slot_id": slot_id,
                        "remove_idx": i,
                        "grams": row.get("grams"),
                        **card,
                    },
                )
            )

    return out


def rank_add_candidates(
    *,
    pool: list[dict[str, Any]],
    core_labels: set[str],
    neighbor_sets: list[set[str]],
    M: np.ndarray,
    box: TargetBox,
    kcal_target: float,
    p_B_star: tuple[float, float, float] | None = None,
    p_T_star: tuple[float, float, float] | None = None,
    top_k_cooc: int = 20,
    top_k_lp: int = 8,
    denylist: tuple[str, ...] = DEFAULT_DENYLIST_SUBSTR,
    evaluate_fn: Callable[..., dict[str, Any]] | None = None,
) -> list[ModCandidate]:
    """pool items: {id, label, foodon_id?, fdc_id?, pfc?: (3,), role?}"""
    exclude = {p["label"] for p in pool if p.get("in_basis")}
    exclude |= set(core_labels)
    cooc = cooccurrence_scores(
        neighbor_sets,
        core_labels,
        exclude=exclude,
    )
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in pool:
        label = str(item["label"])
        if _denied(label, denylist):
            continue
        if item.get("in_basis"):
            continue
        c = float(cooc.get(label, item.get("cooccurrence", 0.0)))
        scored.append((c, item))
    scored.sort(key=lambda t: (-t[0], t[1]["label"]))
    short = scored[:top_k_cooc]

    geom_ranked: list[tuple[float, float, dict[str, Any]]] = []
    for c, item in short:
        pfc = item.get("pfc")
        if pfc is None:
            geom = 0.0
        else:
            geom = geom_vertex_score(
                M, np.asarray(pfc, dtype=float), box, kcal_target=kcal_target,
                p_B_star=p_B_star, p_T_star=p_T_star,
            )
        geom_ranked.append((geom, c, item))
    geom_ranked.sort(key=lambda t: (-t[0], -t[1], t[2]["label"]))
    lp_pool = geom_ranked[:top_k_lp]

    out: list[ModCandidate] = []
    for geom, c, item in lp_pool:
        L_star = None
        if evaluate_fn is not None and item.get("pfc") is not None:
            ev = evaluate_fn(item)
            L_star = ev.get("L_star")
        out.append(
            ModCandidate(
                candidate_id=str(item["id"]),
                action="add",
                label=str(item["label"]),
                fdc_id=item.get("fdc_id"),
                foodon_id=item.get("foodon_id"),
                foodon_path=item.get("foodon_path"),
                role=item.get("role"),
                cooccurrence=float(c),
                geom_score=float(geom),
                L_star=float(L_star) if L_star is not None else None,
                meta={"raw": {k: v for k, v in item.items() if k != "pfc"}},
            )
        )
    # Prefer lower L_star when present
    out.sort(key=lambda m: (m.L_star is None, m.L_star if m.L_star is not None else 0.0, -m.geom_score))
    return out[:5]
