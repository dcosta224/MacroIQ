"""Conical hull geometry for Atwater PFC vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def macro_calorie_fractions_from_grams(protein_g: float, fat_g: float, carbs_g: float) -> tuple[float, float, float]:
    pk, ck, fk = protein_g * 4.0, carbs_g * 4.0, fat_g * 9.0
    total = pk + ck + fk
    if total <= 0:
        return 0.0, 0.0, 0.0
    return pk / total, ck / total, fk / total


def ingredient_pfc_fractions(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    rows = [
        macro_calorie_fractions_from_grams(float(M[0, i]), float(M[1, i]), float(M[2, i]))
        for i in range(M.shape[1])
    ]
    return np.asarray(rows, dtype=float)


@dataclass(frozen=True)
class TargetBox:
    protein_min: float
    protein_max: float
    carb_min: float
    carb_max: float
    fat_min: float
    fat_max: float

    def contains(self, p: float, c: float, f: float, *, tol: float = 1e-6) -> bool:
        return (
            self.protein_min - tol <= p <= self.protein_max + tol
            and self.carb_min - tol <= c <= self.carb_max + tol
            and self.fat_min - tol <= f <= self.fat_max + tol
        )

    def sample_points(self, n_per_axis: int = 5) -> np.ndarray:
        """Grid sample of simplex points inside the axis-aligned PFC box."""
        pts: list[list[float]] = []
        ps = np.linspace(self.protein_min, self.protein_max, n_per_axis)
        cs = np.linspace(self.carb_min, self.carb_max, n_per_axis)
        for p in ps:
            for c in cs:
                f = 1.0 - p - c
                if self.fat_min - 1e-9 <= f <= self.fat_max + 1e-9:
                    pts.append([float(p), float(c), float(f)])
        if not pts:
            # Corners of the box projected onto simplex when possible.
            for p in (self.protein_min, self.protein_max):
                for c in (self.carb_min, self.carb_max):
                    f = 1.0 - p - c
                    if f >= 0:
                        pts.append([float(p), float(c), float(f)])
        return np.asarray(pts, dtype=float) if pts else np.zeros((0, 3))


def pfc_from_mix(lam: np.ndarray, M: np.ndarray) -> tuple[float, float, float]:
    """Atwater P/C/F from a non-negative mix (scale-free)."""
    lam = np.asarray(lam, dtype=float)
    if lam.sum() <= 0:
        return 0.0, 0.0, 0.0
    w = lam / lam.sum()
    protein_g = float(M[0] @ w)
    fat_g = float(M[1] @ w)
    carbs_g = float(M[2] @ w)
    return macro_calorie_fractions_from_grams(protein_g, fat_g, carbs_g)


def iter_conical_mixes(M: np.ndarray, *, n_samples: int = 2000, seed: int = 0) -> np.ndarray:
    """Sample non-negative mixes → PFC points (vertices, edges, Dirichlet interior)."""
    rng = np.random.default_rng(seed)
    M = np.asarray(M, dtype=float)
    n = M.shape[1]
    pts: list[tuple[float, float, float]] = []
    # Vertices
    for i in range(n):
        lam = np.zeros(n)
        lam[i] = 1.0
        pts.append(pfc_from_mix(lam, M))
    # Pairwise
    for i in range(n):
        for j in range(i + 1, n):
            for a in (0.25, 0.5, 0.75):
                lam = np.zeros(n)
                lam[i] = a
                lam[j] = 1.0 - a
                pts.append(pfc_from_mix(lam, M))
    # Dirichlet interior
    remaining = max(0, n_samples - len(pts))
    if remaining and n > 0:
        mixes = rng.dirichlet(np.ones(n), size=remaining)
        for lam in mixes:
            pts.append(pfc_from_mix(lam, M))
    return np.asarray(pts, dtype=float)


def hull_delaunay_2d(samples_pfc: np.ndarray) -> Delaunay | np.ndarray | None:
    """Delaunay of hull vertices, or (2,2) segment endpoints if hull is 1-D."""
    samples_pfc = np.asarray(samples_pfc, dtype=float)
    if samples_pfc.shape[0] < 2:
        return None
    xy = samples_pfc[:, :2]
    rounded = np.round(xy, decimals=8)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    xy = xy[np.sort(idx)]
    if xy.shape[0] < 2:
        return None
    if xy.shape[0] == 2:
        return xy  # segment
    try:
        hull = ConvexHull(xy)
        verts = xy[hull.vertices]
        if verts.shape[0] < 3:
            return verts
        return Delaunay(verts)
    except Exception:
        # Collinear → return extreme points
        order = np.argsort(xy[:, 0])
        return xy[order][[0, -1]]


def point_in_hull(
    p: float,
    c: float,
    f: float,
    hull: Delaunay | np.ndarray | None,
    *,
    tol: float = 1e-4,
) -> bool:
    if hull is None:
        return False
    if abs(p + c + f - 1.0) > 1e-3:
        return False
    pt = np.array([p, c], dtype=float)
    if isinstance(hull, np.ndarray):
        # Segment (or few points): point near the convex combination of endpoints
        if hull.shape[0] == 1:
            return float(np.linalg.norm(pt - hull[0])) <= tol
        a, b = hull[0], hull[-1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-18:
            return float(np.linalg.norm(pt - a)) <= tol
        t = float(np.clip(np.dot(pt - a, ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        return float(np.linalg.norm(pt - proj)) <= tol
    s = int(hull.find_simplex(pt))
    if s >= 0:
        return True
    d = float(np.min(np.linalg.norm(hull.points - pt, axis=1)))
    return d <= tol


def check_box_feasible_lp(
    M: np.ndarray,
    *,
    kcal_target: float,
    box: TargetBox,
) -> tuple[bool, str]:
    import cvxpy as cp

    n = M.shape[1]
    if n == 0:
        return False, "no ingredients"
    x = cp.Variable(n, nonneg=True)
    p_kcal = 4.0 * (M[0] @ x)
    f_kcal = 9.0 * (M[1] @ x)
    c_kcal = 4.0 * (M[2] @ x)
    kcal = p_kcal + f_kcal + c_kcal
    cons = [kcal == kcal_target]
    for macro_kcal, frac_min, frac_max in (
        (p_kcal, box.protein_min, box.protein_max),
        (c_kcal, box.carb_min, box.carb_max),
        (f_kcal, box.fat_min, box.fat_max),
    ):
        cons.append((1.0 - frac_min) * macro_kcal - frac_min * (kcal - macro_kcal) >= 0)
        cons.append((1.0 - frac_max) * macro_kcal - frac_max * (kcal - macro_kcal) <= 0)
    prob = cp.Problem(cp.Minimize(0), cons)
    for name in ("HIGHS", "CLARABEL", "SCS", "OSQP"):
        solver = getattr(cp, name, None)
        if solver is None:
            continue
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
                return True, "feasible"
        except Exception:
            continue
    return False, "infeasible"


def _axis_gap(lo: float, hi: float, t_lo: float, t_hi: float) -> float:
    """How far [t_lo,t_hi] sits outside [lo,hi] (0 if intervals overlap)."""
    if t_hi < lo:
        return float(lo - t_hi)
    if t_lo > hi:
        return float(t_lo - hi)
    return 0.0


def distance_outside_hull(
    samples_pfc: np.ndarray,
    box: TargetBox,
    *,
    hull: Delaunay | np.ndarray | None,
) -> dict[str, object]:
    """Quantify how far the target box sits from the conical hull.

    Distances are in the (protein, carb) plane of the Atwater simplex (same units as fractions).
    """
    samples_pfc = np.asarray(samples_pfc, dtype=float)
    target_mid = np.array(
        [
            0.5 * (box.protein_min + box.protein_max),
            0.5 * (box.carb_min + box.carb_max),
            0.5 * (box.fat_min + box.fat_max),
        ],
        dtype=float,
    )
    # Renormalize mid onto simplex for distance reporting
    s = float(target_mid.sum())
    if s > 0:
        target_mid = target_mid / s

    if samples_pfc.size == 0:
        return {
            "target_mid_pfc": target_mid.tolist(),
            "min_distance_to_hull_samples": None,
            "nearest_hull_sample": None,
            "any_box_point_in_hull": False,
            "fraction_box_points_in_hull": 0.0,
            "axis_gaps": {},
            "outside_score": None,
            "interpretation": "no hull samples",
        }

    xy = samples_pfc[:, :2]
    mid_xy = target_mid[:2]
    d2 = np.sum((xy - mid_xy) ** 2, axis=1)
    i_near = int(np.argmin(d2))
    min_dist = float(np.sqrt(d2[i_near]))
    nearest = samples_pfc[i_near]

    box_pts = box.sample_points()
    if len(box_pts):
        in_flags = [point_in_hull(float(p), float(c), float(f), hull) for p, c, f in box_pts]
        frac_in = float(np.mean(in_flags))
        any_in = bool(any(in_flags))
        # Distance of box to hull: min over box pts of dist to nearest sample (if outside)
        box_xy = box_pts[:, :2]
        # pairwise min distances
        # for each box point, min dist to hull samples
        # (box_n x hull_n) — keep modest
        diffs = box_xy[:, None, :] - xy[None, :, :]
        box_min = np.sqrt(np.sum(diffs**2, axis=2)).min(axis=1)
        for i, inside in enumerate(in_flags):
            if inside:
                box_min[i] = 0.0
        dist_box_to_hull = float(box_min.min())
    else:
        frac_in = 0.0
        any_in = False
        dist_box_to_hull = min_dist

    p_lo, p_hi = float(samples_pfc[:, 0].min()), float(samples_pfc[:, 0].max())
    c_lo, c_hi = float(samples_pfc[:, 1].min()), float(samples_pfc[:, 1].max())
    f_lo, f_hi = float(samples_pfc[:, 2].min()), float(samples_pfc[:, 2].max())
    axis_gaps = {
        "protein": _axis_gap(p_lo, p_hi, box.protein_min, box.protein_max),
        "carbs": _axis_gap(c_lo, c_hi, box.carb_min, box.carb_max),
        "fat": _axis_gap(f_lo, f_hi, box.fat_min, box.fat_max),
    }
    outside_score = float(max(axis_gaps.values()) + dist_box_to_hull)

    if any_in:
        interpretation = (
            f"Target box overlaps hull geometrically "
            f"({frac_in:.0%} of sampled box points inside). Midpoint→nearest-sample distance={min_dist:.4f}."
        )
    else:
        dominant = max(axis_gaps, key=axis_gaps.get)
        interpretation = (
            f"Target box outside hull: min (P,C)-plane distance from box to hull samples = {dist_box_to_hull:.4f}; "
            f"largest axis gap is {dominant}={axis_gaps[dominant]:.4f} "
            f"(0=overlap on that axis). Outside score≈{outside_score:.4f}."
        )

    return {
        "target_mid_pfc": target_mid.tolist(),
        "min_distance_midpoint_to_hull_sample": min_dist,
        "min_distance_box_to_hull": dist_box_to_hull,
        "nearest_hull_sample": nearest.tolist(),
        "any_box_point_in_hull": any_in,
        "fraction_box_points_in_hull": frac_in,
        "hull_axis_ranges": {
            "protein": [p_lo, p_hi],
            "carbs": [c_lo, c_hi],
            "fat": [f_lo, f_hi],
        },
        "target_box": {
            "protein": [box.protein_min, box.protein_max],
            "carbs": [box.carb_min, box.carb_max],
            "fat": [box.fat_min, box.fat_max],
        },
        "axis_gaps": axis_gaps,
        "outside_score": outside_score,
        "interpretation": interpretation,
    }


def region_intersects_hull(
    M: np.ndarray,
    box: TargetBox,
    *,
    kcal_target: float,
    n_samples: int = 2000,
) -> dict[str, object]:
    """Geometric H∩T via sampled hull + LP oracle for the Atwater box."""
    samples = iter_conical_mixes(M, n_samples=n_samples)
    tri = hull_delaunay_2d(samples)
    box_pts = box.sample_points()
    geom_hits = [
        point_in_hull(float(p), float(c), float(f), tri)
        for p, c, f in box_pts
    ] if len(box_pts) else []
    geometric = any(geom_hits)

    lp_ok, lp_msg = check_box_feasible_lp(M, kcal_target=kcal_target, box=box)

    verts = ingredient_pfc_fractions(M)
    p_lo = float(samples[:, 0].min()) if len(samples) else 0.0
    p_hi = float(samples[:, 0].max()) if len(samples) else 0.0
    target_p = 0.5 * (box.protein_min + box.protein_max)
    distance = distance_outside_hull(samples, box, hull=tri)
    residual = {
        "protein": float(target_p - np.clip(target_p, p_lo, p_hi)),
        "hull_protein_range": [p_lo, p_hi],
        "target_protein_mid": target_p,
        "distance": distance,
    }

    return {
        "geometric_intersects": geometric,
        "lp_feasible": lp_ok,
        "lp_message": lp_msg,
        "intersects": bool(lp_ok or geometric),
        "residual": residual,
        "distance": distance,
        "n_samples": int(samples.shape[0]),
        "n_vertices": int(verts.shape[0]),
        "hull_ok": tri is not None,
        "ingredient_pfc_vertices": verts.tolist(),
    }


def expand_hull_with_vertex(samples_pfc: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if v.shape[0] != 3:
        raise ValueError("vertex must be length-3 PFC")
    return np.vstack([samples_pfc, v.reshape(1, 3)])
