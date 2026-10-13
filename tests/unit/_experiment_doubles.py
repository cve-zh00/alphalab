"""Deterministic factories for the experiment comparison tests.

These build fully-formed :class:`ArmResult` objects (with a real
``CertificationResult`` and ``CalibrationReport``) from a bare return series so
the comparison tests can focus on the verdict logic with hand-chosen inputs and
zero LLM/pipeline dependency.
"""

from __future__ import annotations

import numpy as np

from alphalab.domain.experiment.arms import ArmResult, DecisionOutcome, ExperimentArm
from alphalab.domain.journal.calibration import CalibrationReport, calibration_report
from alphalab.domain.stats.certification import CertificationResult, certify_strategy


def _certify(series: tuple[float, ...]) -> CertificationResult:
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 2 or float(np.std(arr, ddof=1)) == 0.0:
        arr = arr + np.linspace(0.0, 1e-9, arr.size)
    return certify_strategy(
        arr,
        arr.reshape(-1, 1),
        n_trials=2,
        trial_sharpe_std=0.0,
        rng=np.random.default_rng(0),
        n_resamples=1_000,
        n_strips=4,
    )


def _calibration() -> CalibrationReport:
    # A well-formed, inconclusive calibration report (neutral priors).
    return calibration_report([0.5] * 30, [False] * 30)


def _annualised_sharpe(series: tuple[float, ...]) -> float:
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std == 0.0:
        return 0.0
    return (float(np.mean(arr)) / std) * float(np.sqrt(252))


def _max_drawdown(series: tuple[float, ...]) -> float:
    if not series:
        return 0.0
    equity = np.cumprod(1.0 + np.asarray(series, dtype=np.float64))
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def make_arm_result(
    name: str,
    series: tuple[float, ...],
    *,
    n_proposed: int | None = None,
    n_executed: int | None = None,
    max_drawdown: float | None = None,
) -> ArmResult:
    """Build an ArmResult from a return series with sensible derived metrics."""
    n = len(series)
    executed = n_executed if n_executed is not None else n
    proposed = n_proposed if n_proposed is not None else executed
    winners = sum(1 for r in series if r > 0.0)
    hit_rate = winners / executed if executed else 0.0
    return ArmResult(
        arm=ExperimentArm(name=name, label=f"arm {name}", family=f"fam_{name}"),
        sharpe_gross=_annualised_sharpe(series),
        sharpe_net=_annualised_sharpe(series),
        max_drawdown=max_drawdown if max_drawdown is not None else _max_drawdown(series),
        hit_rate=hit_rate,
        n_proposed=proposed,
        n_executed=executed,
        executed_returns=series,
        certification=_certify(series),
        calibration=_calibration(),
    )


def veto_outcome(realized_return: float) -> DecisionOutcome:
    """A proposed-but-vetoed decision carrying the return it avoided."""
    return DecisionOutcome(proposed=True, executed=False, realized_return=realized_return)


def executed_outcome(realized_return: float) -> DecisionOutcome:
    """An executed decision carrying its realised return."""
    return DecisionOutcome(proposed=True, executed=True, realized_return=realized_return)
