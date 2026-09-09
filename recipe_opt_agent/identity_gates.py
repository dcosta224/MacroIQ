"""Post-diagnose gates: missing neighborhood identity / bad grounding → must_retry."""

from __future__ import annotations

from typing import Any

from opt_diagnosis import DiagnosisResult, FidelityBand, RetryTrigger
from recipe_opt_agent.edit_grounding import (
    identity_critical_unresolved,
    missing_high_hit_basis_nodes,
)


def apply_identity_grounding_gates(
    diag: DiagnosisResult,
    *,
    problem: dict[str, Any] | None = None,
    grounding_report: dict[str, Any] | None = None,
    identity_roles: list[str] | None = None,
    nutrient_slack: float | None = None,
    ratio_loss: float | None = None,
    min_basis_hits: int = 8,
    nutrient_slack_accept_max: float = 0.08,
    ratio_loss_accept_max: float = 0.08,
) -> DiagnosisResult:
    """Escalate fidelity_band when grounding/identity evidence says accept is unsafe."""
    problem = problem or {}
    report = grounding_report or problem.get("grounding_report") or {}
    foodon = problem.get("foodon_basis_report") or {}
    triggers = list(diag.retry_triggers or [])
    escalate = False

    missing = missing_high_hit_basis_nodes(foodon, min_hits=min_basis_hits)
    if missing:
        escalate = True
        top = missing[0]
        triggers.append(
            RetryTrigger(
                metric="missing_neighborhood_basis",
                reason=(
                    f"High-hit neighborhood ingredient '{top.get('label')}' "
                    f"(hits={top.get('n_hits')}) missing from recipe"
                ),
                current_value=[m.get("label") for m in missing],
                threshold_to_clear=f"include basis with ≥{min_basis_hits} neighborhood hits",
                clearance="add_or_restore_attested_identity_ingredient",
                primary=True,
            )
        )

    crit_unresolved = identity_critical_unresolved(report, identity_roles)
    if crit_unresolved:
        escalate = True
        triggers.append(
            RetryTrigger(
                metric="identity_critical_unresolved",
                reason="Identity-critical draft lines failed FDC grounding",
                current_value=[u.get("name") for u in crit_unresolved],
                threshold_to_clear="resolve identity lines to neighborhood-attested FDC",
                clearance="re_ground_or_restore_canonical_role",
                primary=True,
            )
        )

    slack = nutrient_slack
    if slack is None:
        try:
            slack = float((problem.get("opt") or {}).get("nutrient_slack"))
        except (TypeError, ValueError):
            slack = None
    if slack is not None and slack > nutrient_slack_accept_max and diag.fidelity_band == FidelityBand.ACCEPT:
        escalate = True
        triggers.append(
            RetryTrigger(
                metric="nutrient_slack",
                reason="Nutrient slack too high to accept despite fidelity band",
                current_value=slack,
                threshold_to_clear=nutrient_slack_accept_max,
                clearance="improve_macro_fit_or_ingredient_set",
                primary=False,
            )
        )

    if ratio_loss is not None and ratio_loss > ratio_loss_accept_max and diag.fidelity_band == FidelityBand.ACCEPT:
        escalate = True
        triggers.append(
            RetryTrigger(
                metric="ratio_loss",
                reason="Ratio loss too high to accept",
                current_value=ratio_loss,
                threshold_to_clear=ratio_loss_accept_max,
                clearance="restore_neighborhood_share_structure",
                primary=False,
            )
        )

    if not escalate:
        return diag

    # Promote band: accept → must_retry; moderate stays moderate unless missing identity
    new_band = diag.fidelity_band
    if diag.fidelity_band == FidelityBand.ACCEPT:
        new_band = FidelityBand.MUST_RETRY if (missing or crit_unresolved) else FidelityBand.MODERATE
    elif diag.fidelity_band == FidelityBand.MODERATE and (missing or crit_unresolved):
        new_band = FidelityBand.MUST_RETRY

    action = "add" if (missing or crit_unresolved) else "improve"
    meaning = diag.meaning
    if missing or crit_unresolved:
        meaning = (
            (meaning + " ") if meaning else ""
        ) + "Identity/grounding gate blocked accept: restore neighborhood-attested ingredients."

    return DiagnosisResult(
        diagnosis=diag.diagnosis,
        fidelity_band=new_band,
        meaning=meaning.strip(),
        terms=diag.terms,
        n_red=diag.n_red,
        n_yellow=diag.n_yellow,
        L_max_norm=diag.L_max_norm,
        L_total=diag.L_total,
        macros_feasible=diag.macros_feasible,
        hull_intersects=diag.hull_intersects,
        binding_macros=list(diag.binding_macros or []),
        recommended_action_class=action,
        retry_triggers=triggers,
        band_thresholds=dict(diag.band_thresholds or {}),
    )
