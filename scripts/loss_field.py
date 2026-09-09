"""Loss field L(p) = min weighted empirical objective s.t. PFC(x) ≈ p."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hull_geometry import TargetBox, hull_delaunay_2d, iter_conical_mixes, point_in_hull
from weighted_empirical_opt import optimize_weighted_empirical_obj

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "scratch" / "cache" / "loss_fields"


@dataclass
class LossField:
    protein: np.ndarray  # grid coords
    carb: np.ndarray
    loss: np.ndarray  # same shape; nan outside hull / infeasible
    in_hull: np.ndarray
    meta: dict[str, Any]

    def L_star_on_box(self, box: TargetBox) -> dict[str, Any]:
        best = None
        best_p = None
        for i in range(self.protein.shape[0]):
            for j in range(self.protein.shape[1]):
                if not self.in_hull[i, j] or not np.isfinite(self.loss[i, j]):
                    continue
                p = float(self.protein[i, j])
                c = float(self.carb[i, j])
                f = 1.0 - p - c
                if not box.contains(p, c, f):
                    continue
                val = float(self.loss[i, j])
                if best is None or val < best:
                    best = val
                    best_p = (p, c, f)
        return {"L_star": best, "p_star": best_p, "feasible_on_grid": best is not None}

    def low_loss_basin(self, q: float = 0.25) -> dict[str, Any]:
        vals = self.loss[self.in_hull & np.isfinite(self.loss)]
        if vals.size == 0:
            return {"threshold": None, "points": [], "n": 0}
        thr = float(np.quantile(vals, q))
        pts = []
        for i in range(self.protein.shape[0]):
            for j in range(self.protein.shape[1]):
                if self.in_hull[i, j] and np.isfinite(self.loss[i, j]) and self.loss[i, j] <= thr:
                    p = float(self.protein[i, j])
                    c = float(self.carb[i, j])
                    pts.append((p, c, 1.0 - p - c, float(self.loss[i, j])))
        return {"threshold": thr, "points": pts, "n": len(pts)}

    def corridor(self, box: TargetBox, q: float = 0.25) -> dict[str, Any]:
        basin = self.low_loss_basin(q=q)
        box_pts = box.sample_points(n_per_axis=7)
        if not basin["points"] or len(box_pts) == 0:
            return {"p_B_star": None, "p_T_star": None, "basin": basin}
        # Closest basin point to any box point (L1 on PFC).
        best_b = None
        best_t = None
        best_d = None
        for bp in basin["points"]:
            bvec = np.array(bp[:3])
            for t in box_pts:
                d = float(np.linalg.norm(bvec - t, ord=1))
                if best_d is None or d < best_d:
                    best_d = d
                    best_b = bp[:3]
                    best_t = tuple(float(x) for x in t)
        return {
            "p_B_star": best_b,
            "p_T_star": best_t,
            "distance_l1": best_d,
            "basin_threshold": basin["threshold"],
            "basin_n": basin["n"],
        }


def _cache_key(meta: dict[str, Any]) -> str:
    blob = json.dumps(meta, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def build_loss_field(
    x0: np.ndarray,
    M: np.ndarray,
    *,
    marginal_nodes: list[str],
    basis_samples: dict[str, np.ndarray],
    ratio_samples: np.ndarray,
    ingredient_basis: list[str | None],
    kcal_target: float,
    total_mass: float,
    grid_n: int = 15,
    pfc_eps: float = 0.01,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    cache_meta: dict[str, Any] | None = None,
    warm_start: bool = True,
) -> LossField:
    """Grid L(p)=min obj s.t. PFC within eps of p, only for p in conical hull."""
    x0 = np.asarray(x0, dtype=float)
    M = np.asarray(M, dtype=float)
    samples = iter_conical_mixes(M, n_samples=2500)
    tri = hull_delaunay_2d(samples)

    p_lo, p_hi = float(samples[:, 0].min()), float(samples[:, 0].max())
    c_lo, c_hi = float(samples[:, 1].min()), float(samples[:, 1].max())
    # Pad slightly
    pad = 0.02
    p_axis = np.linspace(max(0.0, p_lo - pad), min(1.0, p_hi + pad), grid_n)
    c_axis = np.linspace(max(0.0, c_lo - pad), min(1.0, c_hi + pad), grid_n)
    P, C = np.meshgrid(p_axis, c_axis, indexing="ij")
    loss = np.full(P.shape, np.nan, dtype=float)
    in_hull = np.zeros(P.shape, dtype=bool)

    meta = {
        "grid_n": grid_n,
        "pfc_eps": pfc_eps,
        "kcal_target": kcal_target,
        "total_mass": total_mass,
        "n_ingredients": int(len(x0)),
        **(cache_meta or {}),
    }
    key = _cache_key(meta)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{key}.npz"
        if cache_path.is_file():
            data = np.load(cache_path)
            return LossField(
                protein=data["protein"],
                carb=data["carb"],
                loss=data["loss"],
                in_hull=data["in_hull"].astype(bool),
                meta=meta,
            )

    x_warm = x0.copy()
    for i in range(grid_n):
        for j in range(grid_n):
            p = float(P[i, j])
            c = float(C[i, j])
            f = 1.0 - p - c
            if f < -1e-6 or f > 1.0 + 1e-6:
                continue
            f = float(np.clip(f, 0.0, 1.0))
            if not point_in_hull(p, c, f, tri):
                continue
            in_hull[i, j] = True
            # Tight box around p
            opt = optimize_weighted_empirical_obj(
                x_warm if warm_start else x0,
                M,
                marginal_nodes=marginal_nodes,
                basis_samples=basis_samples,
                ratio_samples=ratio_samples,
                ingredient_basis=ingredient_basis,
                kcal_target=kcal_target,
                protein_frac_min=max(0.0, p - pfc_eps),
                protein_frac_max=min(1.0, p + pfc_eps),
                carb_frac_min=max(0.0, c - pfc_eps),
                carb_frac_max=min(1.0, c + pfc_eps),
                fat_frac_min=max(0.0, f - pfc_eps),
                fat_frac_max=min(1.0, f + pfc_eps),
                total_mass=total_mass,
            )
            if opt.get("feasible"):
                loss[i, j] = float(opt["objective"])
                x_warm = np.asarray(opt["x_opt"], dtype=float)

    field = LossField(protein=P, carb=C, loss=loss, in_hull=in_hull, meta=meta)
    if cache_dir is not None:
        np.savez_compressed(
            cache_dir / f"{key}.npz",
            protein=P,
            carb=C,
            loss=loss,
            in_hull=in_hull.astype(np.uint8),
        )
    return field


def summarize_loss_field(field: LossField, box: TargetBox) -> dict[str, Any]:
    star = field.L_star_on_box(box)
    corridor = field.corridor(box)
    basin = field.low_loss_basin()
    return {
        "L_star_on_target": star,
        "corridor": {
            "p_B_star": corridor.get("p_B_star"),
            "p_T_star": corridor.get("p_T_star"),
            "distance_l1": corridor.get("distance_l1"),
        },
        "basin": {
            "threshold": basin.get("threshold"),
            "n_points": basin.get("n"),
        },
        "grid_n": field.meta.get("grid_n"),
        "n_hull_cells": int(field.in_hull.sum()),
        "n_finite_loss": int(np.isfinite(field.loss).sum()),
    }
