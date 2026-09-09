"""Tests for opt_diagnosis bands and IQR zones."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from opt_diagnosis import (  # noqa: E402
    Diagnosis,
    FidelityBand,
    classify_fidelity_band,
    diagnose_optimizer_result,
    iqr_zone,
    Zone,
)


def test_iqr_zone_green_yellow_red():
    samples = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    z, *_ = iqr_zone(0.3, samples)
    assert z == Zone.GREEN
    z, *_ = iqr_zone(0.55, samples)
    assert z in (Zone.YELLOW, Zone.RED)
    z, *_ = iqr_zone(2.0, samples)
    assert z == Zone.RED


def test_iqr_zone_degenerate_zero_width_not_penalized():
    # All samples identical → zero-width band; far-off values must not go red.
    samples = np.array([0.35, 0.35, 0.35, 0.35, 0.35])
    z, med, q25, q75, ln = iqr_zone(0.10, samples)
    assert z == Zone.GREEN
    assert ln == 0.0
    assert q25 == q75
    z2, *_rest = iqr_zone(0.90, samples)
    assert z2 == Zone.GREEN


def test_classify_bands():
    assert classify_fidelity_band(L_max_norm=0.5, n_red=0, macros_feasible=True, hull_intersects=True, identity_critical_red=False) == FidelityBand.ACCEPT
    assert classify_fidelity_band(L_max_norm=1.2, n_red=0, macros_feasible=True, hull_intersects=True, identity_critical_red=False) == FidelityBand.MODERATE
    assert classify_fidelity_band(L_max_norm=2.0, n_red=0, macros_feasible=True, hull_intersects=True, identity_critical_red=False) == FidelityBand.MUST_RETRY
    assert classify_fidelity_band(L_max_norm=0.1, n_red=0, macros_feasible=False, hull_intersects=True, identity_critical_red=False) == FidelityBand.MUST_RETRY


def test_diagnose_outside_hull():
    samples = {"pasta": np.array([0.4, 0.5, 0.6])}
    d = diagnose_optimizer_result(
        share_after={"pasta": 0.5},
        share_samples=samples,
        ratio_after=None,
        ratio_samples=None,
        objective=0.1,
        macros_feasible=True,
        hull_intersects=False,
    )
    assert d.diagnosis == Diagnosis.OUTSIDE_HULL
    assert d.fidelity_band == FidelityBand.MUST_RETRY
    assert d.retry_triggers
    assert d.retry_triggers[0].metric == "hull_intersects"
    assert d.retry_triggers[0].threshold_to_clear is True


def test_retry_trigger_lmax():
    samples = {"pasta": np.array([0.45, 0.5, 0.55])}
    d = diagnose_optimizer_result(
        share_after={"pasta": 0.9},
        share_samples=samples,
        ratio_after=None,
        ratio_samples=None,
        objective=1.0,
        macros_feasible=True,
        hull_intersects=True,
        F_accept=1.0,
        F_max=1.5,
    )
    assert d.fidelity_band == FidelityBand.MUST_RETRY
    metrics = {t.metric for t in d.retry_triggers}
    assert any(m.startswith("term_zone:") or m == "L_max_norm" or m == "n_red" for m in metrics)
    assert "F_accept" in d.band_thresholds
