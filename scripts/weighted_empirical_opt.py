"""Weighted empirical recipe objective: marginal Wasserstein + ratio surrogate LP.

Ported from notebooks/agent_optimization_sandbox.ipynb for the recipe_opt_agent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from recipe_macro_optimizer import compute_macros

MIN_INGREDIENT_GRAMS = 1e-3
MAX_GRAM_SCALE = 100.0

# Default FoodOn basis nodes (carbonara sandbox).
SPAGHETTI_BASIS_NODE = "FOODON_03316817"
EGG_YOLK_BASIS_NODE = "FOODON_03542968"
BACON_BASIS_NODE = "FOODON_03317333"
PARMESAN_BASIS_NODE = "FOODON_03302978"

MARGINAL_LOSS_WEIGHT = 0.30
RATIO_LOSS_WEIGHT = 0.14
RATIO_EPSILON_GRAMS = 1.0

MARGINAL_COLUMN_NODES: list[tuple[str, str]] = [
    ("spaghetti", SPAGHETTI_BASIS_NODE),
    ("bacon", BACON_BASIS_NODE),
    ("parmesan cheese", PARMESAN_BASIS_NODE),
    ("hen egg yolk", EGG_YOLK_BASIS_NODE),
]


def atwater_fraction_constraints(
    x_var,
    M: np.ndarray,
    *,
    kcal_target: float,
    protein_frac_min: float,
    protein_frac_max: float,
    carb_frac_min: float,
    carb_frac_max: float,
    fat_frac_min: float,
    fat_frac_max: float,
) -> list:
    p_kcal = 4.0 * (M[0] @ x_var)
    f_kcal = 9.0 * (M[1] @ x_var)
    c_kcal = 4.0 * (M[2] @ x_var)
    kcal = p_kcal + f_kcal + c_kcal
    cons: list = [kcal == kcal_target]
    for macro_kcal, frac_min, frac_max in (
        (p_kcal, protein_frac_min, protein_frac_max),
        (c_kcal, carb_frac_min, carb_frac_max),
        (f_kcal, fat_frac_min, fat_frac_max),
    ):
        cons.append((1.0 - frac_min) * macro_kcal - frac_min * (kcal - macro_kcal) >= 0)
        cons.append((1.0 - frac_max) * macro_kcal - frac_max * (kcal - macro_kcal) <= 0)
    return cons


def pfc_fractions_from_portions(x: np.ndarray, M: np.ndarray) -> tuple[float, float, float]:
    macros = compute_macros(np.asarray(x, dtype=float), M)
    protein_g, fat_g, carbs_g = float(macros[0]), float(macros[1]), float(macros[2])
    pk, ck, fk = protein_g * 4.0, carbs_g * 4.0, fat_g * 9.0
    total = pk + ck + fk
    if total <= 0:
        return 0.0, 0.0, 0.0
    return pk / total, ck / total, fk / total


def atwater_kcal(x: np.ndarray, M: np.ndarray) -> float:
    macros = compute_macros(np.asarray(x, dtype=float), M)
    return 4.0 * float(macros[0]) + 9.0 * float(macros[1]) + 4.0 * float(macros[2])


def empirical_cdf_l1_loss(
    z: float, samples: np.ndarray, weights: np.ndarray | None = None
) -> float:
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return 0.0
    if weights is None:
        return float(np.mean(np.abs(samples - z)))
    w = np.asarray(weights, dtype=float)
    if w.size != samples.size or w.sum() <= 0:
        return float(np.mean(np.abs(samples - z)))
    return float(np.sum(w * np.abs(samples - z)) / w.sum())


def node_grams_sum(x: np.ndarray, node_id: str, ingredient_basis: list[str | None]) -> float:
    return float(sum(float(x[i]) for i, nid in enumerate(ingredient_basis) if nid == node_id))


def neighborhood_ratio_samples(
    basis_share_df: pd.DataFrame,
    spaghetti_node: str,
    egg_yolk_node: str,
    eps: float = RATIO_EPSILON_GRAMS,
) -> np.ndarray:
    if basis_share_df is None or basis_share_df.empty:
        return np.array([], dtype=float)
    if "node_grams" not in basis_share_df.columns:
        # Cached neighborhoods store mass shares, not absolute grams — ratio samples N/A.
        return np.array([], dtype=float)
    piv = (
        basis_share_df.loc[basis_share_df["basis_node_id"].isin({spaghetti_node, egg_yolk_node})]
        .pivot_table(
            index="recipe_nlg_id",
            columns="basis_node_id",
            values="node_grams",
            aggfunc="sum",
        )
    )
    if spaghetti_node not in piv.columns or egg_yolk_node not in piv.columns:
        return np.array([], dtype=float)
    both = piv[[spaghetti_node, egg_yolk_node]].dropna(how="any")
    g_s = both[spaghetti_node].to_numpy(dtype=float)
    g_e = np.maximum(both[egg_yolk_node].to_numpy(dtype=float), eps)
    return g_s / g_e


def recipe_ratio(
    x: np.ndarray,
    *,
    spaghetti_node: str,
    egg_yolk_node: str,
    ingredient_basis: list[str | None],
    eps: float = RATIO_EPSILON_GRAMS,
) -> float:
    g_s = node_grams_sum(x, spaghetti_node, ingredient_basis)
    g_e = max(node_grams_sum(x, egg_yolk_node, ingredient_basis), eps)
    return g_s / g_e


def ratio_surrogate_value(
    x: np.ndarray,
    *,
    ratio_samples: np.ndarray,
    spaghetti_node: str,
    egg_yolk_node: str,
    ingredient_basis: list[str | None],
    total_mass: float,
) -> float:
    ratio_samples = np.asarray(ratio_samples, dtype=float)
    if ratio_samples.size == 0 or total_mass <= 0:
        return 0.0
    g_s = node_grams_sum(x, spaghetti_node, ingredient_basis)
    g_e = node_grams_sum(x, egg_yolk_node, ingredient_basis)
    return float(np.mean(np.abs(g_s - ratio_samples * g_e)) / total_mass)


def term_losses(
    x: np.ndarray,
    *,
    marginal_nodes: list[str],
    basis_samples: dict[str, np.ndarray],
    ratio_samples: np.ndarray,
    total_mass: float,
    ingredient_basis: list[str | None],
    basis_sample_weights: dict[str, np.ndarray] | None = None,
    spaghetti_node: str = SPAGHETTI_BASIS_NODE,
    egg_yolk_node: str = EGG_YOLK_BASIS_NODE,
    marginal_weight: float = MARGINAL_LOSS_WEIGHT,
    ratio_weight: float = RATIO_LOSS_WEIGHT,
) -> dict[str, float]:
    """Per-term weighted losses (for diagnosis / IQR comparison of shares separately)."""
    x = np.asarray(x, dtype=float)
    weights_map = basis_sample_weights or {}
    out: dict[str, float] = {}
    if total_mass <= 0:
        return out
    for nid in marginal_nodes:
        samples = basis_samples.get(nid, np.array([], dtype=float))
        if samples.size == 0:
            continue
        z_i = node_grams_sum(x, nid, ingredient_basis) / total_mass
        out[nid] = marginal_weight * empirical_cdf_l1_loss(z_i, samples, weights_map.get(nid))
        out[f"{nid}__share"] = z_i
        out[f"{nid}__abs_dev_median"] = abs(z_i - float(np.median(samples)))
    # Aggregate median-centered Wasserstein (mean |z − median|) for display / eval.
    med_terms = [v for k, v in out.items() if str(k).endswith("__abs_dev_median")]
    if med_terms:
        out["mean_abs_dev_from_median"] = float(np.mean(med_terms))
    out["ratio_surrogate"] = ratio_weight * ratio_surrogate_value(
        x,
        ratio_samples=ratio_samples,
        spaghetti_node=spaghetti_node,
        egg_yolk_node=egg_yolk_node,
        ingredient_basis=ingredient_basis,
        total_mass=total_mass,
    )
    out["ratio_value"] = recipe_ratio(
        x,
        spaghetti_node=spaghetti_node,
        egg_yolk_node=egg_yolk_node,
        ingredient_basis=ingredient_basis,
    )
    return out


def weighted_empirical_obj_value(
    x: np.ndarray,
    *,
    marginal_nodes: list[str],
    basis_samples: dict[str, np.ndarray],
    ratio_samples: np.ndarray,
    total_mass: float,
    ingredient_basis: list[str | None],
    basis_sample_weights: dict[str, np.ndarray] | None = None,
    spaghetti_node: str = SPAGHETTI_BASIS_NODE,
    egg_yolk_node: str = EGG_YOLK_BASIS_NODE,
    marginal_weight: float = MARGINAL_LOSS_WEIGHT,
    ratio_weight: float = RATIO_LOSS_WEIGHT,
) -> float:
    x = np.asarray(x, dtype=float)
    weights_map = basis_sample_weights or {}
    if total_mass <= 0:
        return 0.0
    marginal = 0.0
    for nid in marginal_nodes:
        samples = basis_samples.get(nid, np.array([], dtype=float))
        if samples.size == 0:
            continue
        z_i = node_grams_sum(x, nid, ingredient_basis) / total_mass
        marginal += marginal_weight * empirical_cdf_l1_loss(z_i, samples, weights_map.get(nid))
    ratio = ratio_weight * ratio_surrogate_value(
        x,
        ratio_samples=ratio_samples,
        spaghetti_node=spaghetti_node,
        egg_yolk_node=egg_yolk_node,
        ingredient_basis=ingredient_basis,
        total_mass=total_mass,
    )
    return float(marginal + ratio)


def _solve_lp(prob, x) -> tuple[str, np.ndarray | None]:
    import cvxpy as cp

    status = "solver_failed"
    x_sol: np.ndarray | None = None
    inaccurate_sol: np.ndarray | None = None
    solvers = []
    for name in ("HIGHS", "CLARABEL", "SCS", "OSQP"):
        if hasattr(cp, name):
            solvers.append(getattr(cp, name))
    for solver in solvers:
        try:
            prob.solve(solver=solver, verbose=False)
        except Exception:
            continue
        if x.value is None:
            continue
        if prob.status == "optimal":
            return "optimal", np.asarray(x.value, dtype=float).ravel()
        if prob.status == "optimal_inaccurate" and inaccurate_sol is None:
            status = "optimal_inaccurate"
            inaccurate_sol = np.asarray(x.value, dtype=float).ravel()
    return status, inaccurate_sol if x_sol is None else x_sol


def optimize_weighted_empirical_obj(
    x0: np.ndarray,
    M: np.ndarray,
    *,
    marginal_nodes: list[str],
    basis_samples: dict[str, np.ndarray],
    ratio_samples: np.ndarray,
    ingredient_basis: list[str | None],
    kcal_target: float,
    protein_frac_min: float,
    protein_frac_max: float,
    carb_frac_min: float,
    carb_frac_max: float,
    fat_frac_min: float,
    fat_frac_max: float,
    total_mass: float,
    basis_sample_weights: dict[str, np.ndarray] | None = None,
    nutrition_slack_weight: float | None = None,
    spaghetti_node: str = SPAGHETTI_BASIS_NODE,
    egg_yolk_node: str = EGG_YOLK_BASIS_NODE,
    marginal_weight: float = MARGINAL_LOSS_WEIGHT,
    ratio_weight: float = RATIO_LOSS_WEIGHT,
    min_grams: float = MIN_INGREDIENT_GRAMS,
    max_scale: float = MAX_GRAM_SCALE,
    pfc_equality: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Minimize weighted marginal + ratio loss subject to Atwater box (or point equality)."""
    import cvxpy as cp

    x0 = np.asarray(x0, dtype=float)
    M = np.asarray(M, dtype=float)
    n = len(x0)
    inv_mass = 1.0 / total_mass

    if pfc_equality is not None:
        tp, tc, tf = pfc_equality
        p_min = p_max = float(tp)
        c_min = c_max = float(tc)
        f_min = f_max = float(tf)
    else:
        p_min, p_max = protein_frac_min, protein_frac_max
        c_min, c_max = carb_frac_min, carb_frac_max
        f_min, f_max = fat_frac_min, fat_frac_max

    x = cp.Variable(n, nonneg=True)
    cons: list = [cp.sum(x) == total_mass]
    nutrition_slack_expr = 0
    if nutrition_slack_weight is None:
        cons.extend(
            atwater_fraction_constraints(
                x,
                M,
                kcal_target=kcal_target,
                protein_frac_min=p_min,
                protein_frac_max=p_max,
                carb_frac_min=c_min,
                carb_frac_max=c_max,
                fat_frac_min=f_min,
                fat_frac_max=f_max,
            )
        )
    else:
        # Soft PFC box: fraction-valued lower/upper slacks make the nutrition
        # penalty commensurate with the dimensionless empirical ratio loss.
        p_kcal = 4.0 * (M[0] @ x)
        f_kcal = 9.0 * (M[1] @ x)
        c_kcal = 4.0 * (M[2] @ x)
        kcal = p_kcal + f_kcal + c_kcal
        p_lo, p_hi = cp.Variable(nonneg=True), cp.Variable(nonneg=True)
        c_lo, c_hi = cp.Variable(nonneg=True), cp.Variable(nonneg=True)
        f_lo, f_hi = cp.Variable(nonneg=True), cp.Variable(nonneg=True)
        cons.extend(
            [
                kcal == kcal_target,
                p_kcal >= (p_min - p_lo) * kcal_target,
                p_kcal <= (p_max + p_hi) * kcal_target,
                c_kcal >= (c_min - c_lo) * kcal_target,
                c_kcal <= (c_max + c_hi) * kcal_target,
                f_kcal >= (f_min - f_lo) * kcal_target,
                f_kcal <= (f_max + f_hi) * kcal_target,
            ]
        )
        nutrition_slack_expr = p_lo + p_hi + c_lo + c_hi + f_lo + f_hi
    for i in range(n):
        cons.append(x[i] >= min_grams)
        # New ingredients may have x0[i]==0; allow up to mass if scale*x0 is too tight.
        upper = max(max_scale * float(x0[i]), min_grams * 10.0)
        if float(x0[i]) <= 0:
            upper = max(upper, total_mass)
        cons.append(x[i] <= upper)

    weights_map = basis_sample_weights or {}
    obj_terms: list = []
    for node_id in marginal_nodes:
        samples = basis_samples.get(node_id, np.array([], dtype=float))
        if samples.size == 0:
            continue
        idxs = [i for i, mapped in enumerate(ingredient_basis) if mapped == node_id]
        if not idxs:
            continue
        z_i = inv_mass * cp.sum(x[idxs])
        node_weights = weights_map.get(node_id)
        use_weights = (
            node_weights is not None
            and np.asarray(node_weights, dtype=float).size == samples.size
            and float(np.asarray(node_weights, dtype=float).sum()) > 0
        )
        w_arr = np.asarray(node_weights, dtype=float) if use_weights else None
        node_abs: list = []
        for j, sample in enumerate(samples):
            u = cp.Variable(nonneg=True)
            cons.append(u >= sample - z_i)
            cons.append(u >= z_i - sample)
            node_abs.append(float(w_arr[j]) * u if use_weights else u)
        denom = float(w_arr.sum()) if use_weights else float(samples.size)
        obj_terms.append(marginal_weight * cp.sum(node_abs) / denom)

    ratio_samples_arr = np.asarray(ratio_samples, dtype=float)
    if ratio_samples_arr.size:
        spaghetti_idxs = [i for i, mapped in enumerate(ingredient_basis) if mapped == spaghetti_node]
        egg_idxs = [i for i, mapped in enumerate(ingredient_basis) if mapped == egg_yolk_node]
        if spaghetti_idxs and egg_idxs:
            g_s = cp.sum(x[spaghetti_idxs])
            g_e = cp.sum(x[egg_idxs])
            ratio_abs: list = []
            for rho in ratio_samples_arr:
                u = cp.Variable(nonneg=True)
                cons.append(u >= g_s - rho * g_e)
                cons.append(u >= rho * g_e - g_s)
                ratio_abs.append(u)
            obj_terms.append(ratio_weight * inv_mass * cp.sum(ratio_abs) / ratio_samples_arr.shape[0])

    ratio_expr = cp.sum(obj_terms) if obj_terms else 0
    objective = cp.Minimize(
        ratio_expr
        + (
            float(nutrition_slack_weight) * nutrition_slack_expr
            if nutrition_slack_weight is not None
            else 0
        )
    )
    prob = cp.Problem(objective, cons)
    status, x_sol = _solve_lp(prob, x)

    eval_kwargs = dict(
        marginal_nodes=marginal_nodes,
        basis_samples=basis_samples,
        ratio_samples=ratio_samples_arr,
        total_mass=total_mass,
        ingredient_basis=ingredient_basis,
        basis_sample_weights=basis_sample_weights,
        spaghetti_node=spaghetti_node,
        egg_yolk_node=egg_yolk_node,
        marginal_weight=marginal_weight,
        ratio_weight=ratio_weight,
    )
    if x_sol is None:
        return {
            "status": status,
            "x_opt": x0.copy(),
            "objective": weighted_empirical_obj_value(x0, **eval_kwargs),
            "feasible": False,
            "nutrient_slack": float("nan"),
            "term_losses": term_losses(x0, **eval_kwargs),
        }

    ratio_obj_val = weighted_empirical_obj_value(x_sol, **eval_kwargs)
    p, c, f = pfc_fractions_from_portions(x_sol, M)
    nutrient_slack = float(
        max(p_min - p, 0.0)
        + max(p - p_max, 0.0)
        + max(c_min - c, 0.0)
        + max(c - c_max, 0.0)
        + max(f_min - f, 0.0)
        + max(f - f_max, 0.0)
    )
    obj_val = ratio_obj_val + (
        float(nutrition_slack_weight) * nutrient_slack
        if nutrition_slack_weight is not None
        else 0.0
    )
    return {
        "status": status,
        "x_opt": x_sol,
        "objective": obj_val,
        "ratio_objective": ratio_obj_val,
        "nutrient_slack": nutrient_slack,
        "feasible": True,
        "macros_before": compute_macros(x0, M),
        "macros_after": compute_macros(x_sol, M),
        "term_losses": term_losses(x_sol, **eval_kwargs),
    }
