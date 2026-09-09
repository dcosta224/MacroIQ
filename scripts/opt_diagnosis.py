"""Diagnosis enums, IQR zones, and three-band fidelity gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Diagnosis(str, Enum):
    OUTSIDE_HULL = "OUTSIDE_HULL"
    OPTIMIZER_INFEASIBLE = "OPTIMIZER_INFEASIBLE"
    BINDING_MACRO_DISTORTION = "BINDING_MACRO_DISTORTION"
    SINGLE_TERM_RED = "SINGLE_TERM_RED"
    MULTI_TERM_RED = "MULTI_TERM_RED"
    OK = "OK"


class FidelityBand(str, Enum):
    ACCEPT = "accept"
    MODERATE = "moderate"
    MUST_RETRY = "must_retry"


class Zone(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class TermZone:
    name: str
    value: float
    median: float
    q25: float
    q75: float
    zone: Zone
    L_norm: float


@dataclass
class RetryTrigger:
    """Why a fidelity band was forced to must_retry (or blocked accept)."""

    metric: str
    reason: str
    current_value: Any
    threshold_to_clear: Any
    clearance: str
    primary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisResult:
    diagnosis: Diagnosis
    fidelity_band: FidelityBand
    meaning: str
    terms: list[TermZone]
    n_red: int
    n_yellow: int
    L_max_norm: float
    L_total: float
    macros_feasible: bool
    hull_intersects: bool
    binding_macros: list[str] = field(default_factory=list)
    recommended_action_class: str = "accept"
    retry_triggers: list[RetryTrigger] = field(default_factory=list)
    band_thresholds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["diagnosis"] = self.diagnosis.value
        d["fidelity_band"] = self.fidelity_band.value
        d["terms"] = [
            {**asdict(t), "zone": t.zone.value}
            for t in self.terms
        ]
        d["retry_triggers"] = [t.to_dict() if isinstance(t, RetryTrigger) else t for t in self.retry_triggers]
        return d


def build_retry_triggers(
    *,
    L_max_norm: float,
    n_red: int,
    macros_feasible: bool,
    hull_intersects: bool,
    identity_critical_red: bool,
    terms: list[TermZone],
    identity_critical_names: set[str],
    F_accept: float,
    F_max: float,
    fidelity_band: FidelityBand,
) -> list[RetryTrigger]:
    """Explain which rules fired and what values would avoid must_retry / reach accept."""
    triggers: list[RetryTrigger] = []
    if not hull_intersects:
        triggers.append(
            RetryTrigger(
                metric="hull_intersects",
                reason="Target macro box does not intersect the conical ingredient hull",
                current_value=False,
                threshold_to_clear=True,
                clearance="Need H∩T nonempty (add ingredients that reach the target PFC box, or relax the box)",
                primary=True,
            )
        )
    if not macros_feasible:
        triggers.append(
            RetryTrigger(
                metric="macros_feasible",
                reason="Optimizer could not meet macro bounds with current ingredients/constraints",
                current_value=False,
                threshold_to_clear=True,
                clearance="Need a feasible LP solution inside the protein/carb/fat box",
                primary=not triggers,
            )
        )
    if identity_critical_red:
        red_id = [t.name for t in terms if t.zone == Zone.RED and t.name in identity_critical_names]
        triggers.append(
            RetryTrigger(
                metric="identity_critical_red",
                reason="An identity-critical fidelity term is in the RED zone",
                current_value=red_id,
                threshold_to_clear="0 identity-critical RED terms",
                clearance="Bring identity-critical shares back inside the IQR whisker (YELLOW/GREEN)",
                primary=not triggers,
            )
        )
    if n_red >= 2:
        red_names = [t.name for t in terms if t.zone == Zone.RED]
        triggers.append(
            RetryTrigger(
                metric="n_red",
                reason="Two or more fidelity terms are RED",
                current_value=n_red,
                threshold_to_clear={"n_red": "≤0 for accept; ≤0 and L_max_norm≤F_max to leave must_retry via this rule", "red_terms": red_names},
                clearance=f"Reduce RED count from {n_red} to 0 (must_retry also fires at n_red≥1 when L rules apply)",
                primary=not triggers,
            )
        )
    elif n_red >= 1:
        red = next(t for t in terms if t.zone == Zone.RED)
        # Whisker edge for clearance: stay within [q25-1.5IQR, q75+1.5IQR]
        iqr = max(red.q75 - red.q25, 1e-9)
        lo = red.q25 - 1.5 * iqr
        hi = red.q75 + 1.5 * iqr
        triggers.append(
            RetryTrigger(
                metric=f"term_zone:{red.name}",
                reason=f"Fidelity term '{red.name}' is RED (outside IQR whisker)",
                current_value={"value": red.value, "L_norm": red.L_norm, "zone": red.zone.value},
                threshold_to_clear={"value_in": [lo, hi], "zone": "yellow_or_green", "L_norm_for_accept": f"≤{F_accept}"},
                clearance=f"Move {red.name} into [{lo:.4g}, {hi:.4g}] to leave RED; for accept also need L_max_norm≤{F_accept} and n_red=0",
                primary=not triggers,
            )
        )
    if L_max_norm > F_max:
        worst = max(terms, key=lambda t: t.L_norm) if terms else None
        triggers.append(
            RetryTrigger(
                metric="L_max_norm",
                reason=f"Max IQR-normalized deviation exceeds F_max={F_max}",
                current_value=L_max_norm,
                threshold_to_clear={"must_retry_if_above": F_max, "moderate_if_above": F_accept, "accept_if_at_most": F_accept},
                clearance=(
                    f"Need L_max_norm ≤ {F_max} to exit must_retry via this rule"
                    + (f" (worst term: {worst.name}={worst.L_norm:.4g})" if worst else "")
                    + f"; need ≤ {F_accept} for accept"
                ),
                primary=not triggers,
            )
        )
    elif fidelity_band == FidelityBand.MUST_RETRY and L_max_norm > F_accept and n_red == 0:
        # shouldn't happen with current rules unless other triggers; keep for completeness
        pass
    elif fidelity_band == FidelityBand.MODERATE:
        triggers.append(
            RetryTrigger(
                metric="L_max_norm",
                reason=f"Moderate band: L_max_norm between F_accept={F_accept} and F_max={F_max}",
                current_value=L_max_norm,
                threshold_to_clear={"accept_if_at_most": F_accept},
                clearance=f"Need L_max_norm ≤ {F_accept} and n_red=0 for accept",
                primary=True,
            )
        )

    return triggers


def build_binding_macro_triggers(
    binding_macros: list[str],
    *,
    fidelity_band: FidelityBand,
    macros_feasible: bool,
) -> list[RetryTrigger]:
    """Explain binding macro distortion — do not treat ratio red alone as undo-LLM-intent."""
    if not binding_macros or not macros_feasible:
        return []
    if fidelity_band not in {FidelityBand.MUST_RETRY, FidelityBand.MODERATE}:
        return []
    return [
        RetryTrigger(
            metric="binding_macro_explanation",
            reason="User macro box is binding; ratio/fidelity red may reflect forced distortion, not bad LLM intent",
            current_value=list(binding_macros),
            threshold_to_clear="Continue polish/repair; save feasible macro snapshots to pool",
            clearance="Prefer add/swap repair over undoing draft intent when macros are feasible",
            primary=False,
        )
    ]


def iqr_zone(value: float, samples: np.ndarray, *, whisker: float = 1.5) -> tuple[Zone, float, float, float, float]:
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0 or not np.isfinite(value):
        return Zone.GREEN, float("nan"), float("nan"), float("nan"), 0.0
    q25 = float(np.percentile(samples, 25))
    med = float(np.median(samples))
    q75 = float(np.percentile(samples, 75))
    raw_iqr = float(q75) - float(q25)
    # Degenerate / zero-width bands must not inflate L_norm or count as red.
    if raw_iqr <= 1e-12 or raw_iqr / max(abs(med), abs(q25), abs(q75), 1e-9) <= 1e-6:
        return Zone.GREEN, med, q25, q75, 0.0
    iqr = raw_iqr
    L_norm = abs(value - med) / iqr
    lo = q25 - whisker * iqr
    hi = q75 + whisker * iqr
    if q25 <= value <= q75:
        zone = Zone.GREEN
    elif lo <= value <= hi:
        zone = Zone.YELLOW
    else:
        zone = Zone.RED
    return zone, med, q25, q75, float(L_norm)


def classify_fidelity_band(
    *,
    L_max_norm: float,
    n_red: int,
    macros_feasible: bool,
    hull_intersects: bool,
    identity_critical_red: bool,
    F_accept: float = 1.0,
    F_max: float = 1.5,
) -> FidelityBand:
    if not macros_feasible or not hull_intersects or identity_critical_red or n_red >= 2:
        return FidelityBand.MUST_RETRY
    if L_max_norm > F_max or n_red >= 1:
        return FidelityBand.MUST_RETRY
    if L_max_norm <= F_accept and n_red == 0:
        return FidelityBand.ACCEPT
    return FidelityBand.MODERATE


def diagnose_optimizer_result(
    *,
    share_after: dict[str, float],
    share_samples: dict[str, np.ndarray],
    ratio_after: float | None,
    ratio_samples: np.ndarray | None,
    objective: float,
    macros_feasible: bool,
    hull_intersects: bool,
    binding_macros: list[str] | None = None,
    identity_critical_names: set[str] | None = None,
    F_accept: float = 1.0,
    F_max: float = 1.5,
) -> DiagnosisResult:
    binding_macros = binding_macros or []
    identity_critical_names = identity_critical_names or set()
    terms: list[TermZone] = []

    for name, val in share_after.items():
        samples = share_samples.get(name, np.array([], dtype=float))
        zone, med, q25, q75, Ln = iqr_zone(float(val), samples)
        terms.append(
            TermZone(name=name, value=float(val), median=med, q25=q25, q75=q75, zone=zone, L_norm=Ln)
        )

    if ratio_after is not None and ratio_samples is not None and np.asarray(ratio_samples).size:
        zone, med, q25, q75, Ln = iqr_zone(float(ratio_after), np.asarray(ratio_samples, dtype=float))
        terms.append(
            TermZone(
                name="spaghetti_egg_ratio",
                value=float(ratio_after),
                median=med,
                q25=q25,
                q75=q75,
                zone=zone,
                L_norm=Ln,
            )
        )

    n_red = sum(1 for t in terms if t.zone == Zone.RED)
    n_yellow = sum(1 for t in terms if t.zone == Zone.YELLOW)
    L_max_norm = max((t.L_norm for t in terms), default=0.0)
    identity_critical_red = any(
        t.zone == Zone.RED and t.name in identity_critical_names for t in terms
    )

    if not hull_intersects:
        diagnosis = Diagnosis.OUTSIDE_HULL
        meaning = "Target macro box is outside the conical hull of current ingredients."
        action = "add"
    elif not macros_feasible:
        diagnosis = Diagnosis.OPTIMIZER_INFEASIBLE
        meaning = "Macros look geometrically reachable but optimizer bounds block a feasible mix."
        action = "add"
    elif binding_macros and n_red >= 1:
        diagnosis = Diagnosis.BINDING_MACRO_DISTORTION
        meaning = "Feasible at a binding macro edge, but fidelity terms are outside neighborhood norms."
        action = "add"
    elif n_red >= 2 or L_max_norm > F_max:
        diagnosis = Diagnosis.MULTI_TERM_RED
        meaning = "Multiple fidelity terms are far from neighborhood norms."
        action = "expand"
    elif n_red == 1:
        diagnosis = Diagnosis.SINGLE_TERM_RED
        meaning = "One fidelity term is outside the neighborhood whisker."
        red = next(t for t in terms if t.zone == Zone.RED)
        action = "swap" if red.name in identity_critical_names else "remove"
    else:
        diagnosis = Diagnosis.OK
        meaning = "Feasible with fidelity inside accept/moderate neighborhood bands."
        action = "accept"

    band = classify_fidelity_band(
        L_max_norm=L_max_norm,
        n_red=n_red,
        macros_feasible=macros_feasible,
        hull_intersects=hull_intersects,
        identity_critical_red=identity_critical_red,
        F_accept=F_accept,
        F_max=F_max,
    )
    if diagnosis == Diagnosis.OK and band == FidelityBand.ACCEPT:
        action = "accept"
    elif band == FidelityBand.MODERATE:
        action = "improve"
    elif band == FidelityBand.MUST_RETRY and action == "accept":
        action = "add"

    retry_triggers = build_retry_triggers(
        L_max_norm=float(L_max_norm),
        n_red=n_red,
        macros_feasible=macros_feasible,
        hull_intersects=hull_intersects,
        identity_critical_red=identity_critical_red,
        terms=terms,
        identity_critical_names=identity_critical_names,
        F_accept=F_accept,
        F_max=F_max,
        fidelity_band=band,
    )
    retry_triggers.extend(
        build_binding_macro_triggers(
            list(binding_macros),
            fidelity_band=band,
            macros_feasible=macros_feasible,
        )
    )

    # If must_retry but no structured trigger matched, note band rules.
    if band == FidelityBand.MUST_RETRY and not retry_triggers:
        retry_triggers.append(
            RetryTrigger(
                metric="fidelity_band_rules",
                reason="must_retry under classify_fidelity_band",
                current_value={"L_max_norm": L_max_norm, "n_red": n_red},
                threshold_to_clear={"F_accept": F_accept, "F_max": F_max},
                clearance="Satisfy hull∩macros, n_red=0, and L_max_norm≤F_accept for accept",
                primary=True,
            )
        )

    return DiagnosisResult(
        diagnosis=diagnosis,
        fidelity_band=band,
        meaning=meaning,
        terms=terms,
        n_red=n_red,
        n_yellow=n_yellow,
        L_max_norm=float(L_max_norm),
        L_total=float(objective),
        macros_feasible=macros_feasible,
        hull_intersects=hull_intersects,
        binding_macros=list(binding_macros),
        recommended_action_class=action,
        retry_triggers=retry_triggers,
        band_thresholds={"F_accept": float(F_accept), "F_max": float(F_max)},
    )
