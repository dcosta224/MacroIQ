"""Finalist pool scoring: robust-z logistic normalization, Pareto filter, weighted composite."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


METRIC_NAMES = ("nutrient_dist", "ratio_badness", "intent_gap", "churn")
DEFAULT_WEIGHTS = {
    "nutrient": 0.4,
    "ratio": 0.3,
    "intent": 0.2,
    "churn": 0.1,
}


@dataclass
class CandidateMetrics:
    candidate_id: str
    nutrient_dist: float
    ratio_badness: float
    intent_gap: float
    churn: float
    raw: dict[str, Any] = field(default_factory=dict)

    def badness_dict(self) -> dict[str, float]:
        return {
            "nutrient_dist": self.nutrient_dist,
            "ratio_badness": self.ratio_badness,
            "intent_gap": self.intent_gap,
            "churn": self.churn,
        }


@dataclass
class ScoredCandidate:
    candidate_id: str
    metrics: CandidateMetrics
    bad_norm: dict[str, float] = field(default_factory=dict)
    good: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0
    pareto_rank: int = 0
    dominated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "metrics": asdict(self.metrics),
            "bad_norm": self.bad_norm,
            "good": self.good,
            "composite": self.composite,
            "pareto_rank": self.pareto_rank,
            "dominated": self.dominated,
        }


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def robust_z_logistic_badness(
    values: list[float],
    *,
    steepness: float = 1.5,
    z0: float = 2.0,
    eps: float = 1e-9,
) -> list[float]:
    """Map raw badness values to [0,1] via robust z + logistic."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []

    def _sig(x: float) -> float:
        # Clamp to avoid OverflowError on math.exp for extreme z-scores / L_total.
        t = max(-60.0, min(60.0, float(x)))
        return float(1.0 / (1.0 + math.exp(-t)))

    if arr.size == 1:
        v = float(arr[0])
        if not math.isfinite(v):
            return [1.0]
        return [_sig(steepness * (v - z0))]

    finite_mask = np.isfinite(arr)
    med = float(np.median(arr[finite_mask])) if np.any(finite_mask) else 0.0
    mad_vals = np.where(finite_mask, arr, med)
    mad = _mad(mad_vals)
    scale = 1.4826 * mad + eps
    out: list[float] = []
    for v in arr:
        fv = float(v)
        if not math.isfinite(fv):
            out.append(1.0)
            continue
        z = (fv - med) / scale
        out.append(_sig(steepness * (z - z0)))
    return out


def normalize_metrics(
    metrics: list[CandidateMetrics],
    *,
    steepness: float = 1.5,
    z0: float = 2.0,
) -> list[tuple[CandidateMetrics, dict[str, float], dict[str, float]]]:
    """Return (metrics, bad_norm, good) per candidate."""
    if not metrics:
        return []
    keys = METRIC_NAMES
    cols = {name: [m.badness_dict()[name] for m in metrics] for name in keys}
    bad_norms = {name: robust_z_logistic_badness(cols[name], steepness=steepness, z0=z0) for name in keys}
    out: list[tuple[CandidateMetrics, dict[str, float], dict[str, float]]] = []
    for i, m in enumerate(metrics):
        bn = {name: bad_norms[name][i] for name in keys}
        good = {name: 1.0 - bn[name] for name in keys}
        out.append((m, bn, good))
    return out


def pareto_filter(metrics: list[CandidateMetrics]) -> list[int]:
    """Return indices of non-dominated candidates (all badness minimized)."""
    n = len(metrics)
    if n <= 1:
        return list(range(n))
    keep: list[int] = []
    for i in range(n):
        bi = metrics[i].badness_dict()
        dominated = False
        for j in range(n):
            if i == j:
                continue
            bj = metrics[j].badness_dict()
            if all(bj[k] <= bi[k] for k in METRIC_NAMES) and any(
                bj[k] < bi[k] for k in METRIC_NAMES
            ):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def weighted_composite(
    good: dict[str, float],
    *,
    w_nutrient: float = DEFAULT_WEIGHTS["nutrient"],
    w_ratio: float = DEFAULT_WEIGHTS["ratio"],
    w_intent: float = DEFAULT_WEIGHTS["intent"],
    w_churn: float = DEFAULT_WEIGHTS["churn"],
) -> float:
    return float(
        w_nutrient * good["nutrient_dist"]
        + w_ratio * good["ratio_badness"]
        + w_intent * good["intent_gap"]
        + w_churn * good["churn"]
    )


def score_finalist_pool(
    entries: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
) -> list[ScoredCandidate]:
    """Score and rank finalist pool entries that already carry raw metrics."""
    weights = weights or dict(DEFAULT_WEIGHTS)
    metrics_list: list[CandidateMetrics] = []
    for i, e in enumerate(entries):
        cid = str(e.get("candidate_id") or f"finalist_{i}")
        raw = e.get("metrics") or e
        metrics_list.append(
            CandidateMetrics(
                candidate_id=cid,
                nutrient_dist=float(raw.get("nutrient_dist", 0.0)),
                ratio_badness=float(raw.get("ratio_badness", raw.get("L_max_norm", 0.0))),
                intent_gap=float(raw.get("intent_gap", 0.0)),
                churn=float(raw.get("churn", 0.0)),
                raw=e,
            )
        )

    normalized = normalize_metrics(metrics_list)
    pareto_idx = set(pareto_filter(metrics_list))
    scored: list[ScoredCandidate] = []
    for i, (m, bn, good) in enumerate(normalized):
        comp = weighted_composite(
            {
                "nutrient_dist": good["nutrient_dist"],
                "ratio_badness": good["ratio_badness"],
                "intent_gap": good["intent_gap"],
                "churn": good["churn"],
            },
            w_nutrient=weights.get("nutrient", DEFAULT_WEIGHTS["nutrient"]),
            w_ratio=weights.get("ratio", DEFAULT_WEIGHTS["ratio"]),
            w_intent=weights.get("intent", DEFAULT_WEIGHTS["intent"]),
            w_churn=weights.get("churn", DEFAULT_WEIGHTS["churn"]),
        )
        scored.append(
            ScoredCandidate(
                candidate_id=m.candidate_id,
                metrics=m,
                bad_norm=bn,
                good=good,
                composite=comp,
                pareto_rank=0 if i in pareto_idx else 1,
                dominated=i not in pareto_idx,
            )
        )
    scored.sort(key=lambda s: (-s.composite, s.metrics.ratio_badness))
    return scored


def compute_churn(
    current_ingredients: list[dict[str, Any]],
    reference_ingredients: list[dict[str, Any]],
    *,
    x_opt: list[float] | None = None,
    x_ref: list[float] | None = None,
) -> float:
    """Normalized L1 on matched grams + Jaccard on ingredient lines vs grounded R₀."""
    cur_labels = {str(r.get("label") or r.get("name") or "").lower() for r in current_ingredients}
    ref_labels = {str(r.get("label") or r.get("name") or "").lower() for r in reference_ingredients}
    if not cur_labels and not ref_labels:
        jaccard_dist = 0.0
    else:
        inter = len(cur_labels & ref_labels)
        union = len(cur_labels | ref_labels)
        jaccard_dist = 1.0 - (inter / union if union else 1.0)

    gram_dist = 0.0
    if x_opt is not None and x_ref is not None:
        a = np.asarray(x_opt, dtype=float)
        b = np.asarray(x_ref, dtype=float)
        if a.size and b.size and a.size == b.size:
            denom = max(float(np.linalg.norm(b, ord=1)), 1e-9)
            gram_dist = float(np.linalg.norm(a - b, ord=1) / denom)
    return float(0.5 * jaccard_dist + 0.5 * gram_dist)


def compute_intent_gap(request: str, recipe_title: str, ingredients: list[dict[str, Any]]) -> float:
    """1 - semantic_sim(recipe, request); lexical fallback without embeddings."""
    req = request.lower()
    title = recipe_title.lower()
    labels = " ".join(str(r.get("label") or r.get("name") or "") for r in ingredients).lower()
    text = f"{title} {labels}"
    req_tokens = {t for t in req.split() if len(t) > 2}
    text_tokens = {t for t in text.split() if len(t) > 2}
    if not req_tokens:
        return 0.0
    overlap = len(req_tokens & text_tokens) / len(req_tokens)
    return float(1.0 - overlap)


def top_survivors_for_judge(
    scored: list[ScoredCandidate],
    *,
    epsilon: float = 0.03,
    max_survivors: int = 3,
) -> tuple[list[ScoredCandidate], bool]:
    """Return survivors and whether LLM judge is needed (top two within epsilon)."""
    pareto = [s for s in scored if not s.dominated]
    pool = pareto if pareto else scored
    pool = sorted(pool, key=lambda s: -s.composite)[:max_survivors]
    need_judge = len(pool) >= 2 and abs(pool[0].composite - pool[1].composite) <= epsilon
    return pool, need_judge
