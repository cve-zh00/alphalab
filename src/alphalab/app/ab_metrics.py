"""Metric helpers for the A/B experiment runner.

Split out of :mod:`alphalab.app.ab_experiment` to keep each module small and
single-purpose. These are pure numeric helpers plus the thin certification and
calibration adapters the runner needs when turning an arm's per-decision
outcomes into a frozen :class:`~alphalab.domain.experiment.ArmResult`.
"""

from __future__ import annotations

import numpy as np

from alphalab.artifact.models import Verdict as ArtifactVerdict
from alphalab.domain.experiment import DecisionOutcome
from alphalab.domain.journal.calibration import (
    MIN_SAMPLE_SIZE,
    CalibrationReport,
    calibration_report,
)
from alphalab.domain.stats.certification import CertificationResult, certify_strategy

PERIODS_PER_YEAR = 252


def position_sign(verdict: ArtifactVerdict) -> int:
    """Map a directional verdict to a position sign (BUY +1, SELL -1, HOLD 0)."""
    if verdict is ArtifactVerdict.BUY:
        return 1
    if verdict is ArtifactVerdict.SELL:
        return -1
    return 0


def annualised_sharpe(
    series: tuple[float, ...], *, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Annualised Sharpe of a traded series; 0.0 for a degenerate series."""
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std == 0.0:
        return 0.0
    return (float(np.mean(arr)) / std) * float(np.sqrt(periods_per_year))


def max_drawdown(series: tuple[float, ...]) -> float:
    """Worst peak-to-trough drawdown of the compounded series (<= 0)."""
    if not series:
        return 0.0
    equity = np.cumprod(1.0 + np.asarray(series, dtype=np.float64))
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def certify_series(series: tuple[float, ...]) -> CertificationResult:
    """Certify an arm's traded series as an a-priori (no-selection) test.

    A degenerate/flat series is nudged by a sub-nanometre ramp so
    ``certify_strategy`` is defined; a genuinely edgeless arm simply fails to
    certify, which is the honest outcome rather than an exception.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 2 or float(np.std(arr, ddof=1)) == 0.0:
        arr = arr + np.linspace(0.0, 1e-9, arr.size)
    return certify_strategy(
        arr,
        arr.reshape(-1, 1),
        n_trials=2,
        trial_sharpe_std=0.0,
        rng=np.random.default_rng(0),
        n_resamples=2_000,
        n_strips=4,
    )


def calibrate(outcomes: tuple[DecisionOutcome, ...]) -> CalibrationReport:
    """Build a calibration report from decisions that carried a probability.

    When fewer than ``MIN_SAMPLE_SIZE`` probabilities were emitted, the series
    is padded with neutral priors so the report is well-formed but flagged
    inconclusive — honest, not flattering.
    """
    probs = [o.predicted_prob for o in outcomes if o.predicted_prob is not None]
    wins = [bool(o.was_winner) for o in outcomes if o.predicted_prob is not None]
    if len(probs) < MIN_SAMPLE_SIZE:
        pad = MIN_SAMPLE_SIZE - len(probs)
        probs = [*probs, *([0.5] * pad)]
        wins = [*wins, *([False] * pad)]
    return calibration_report([float(p) for p in probs], wins)
