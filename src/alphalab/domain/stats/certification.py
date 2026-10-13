"""Certification verdict and the evaporation curve report artifact.

This module ties the individual statistical gates together into a single,
immutable certification verdict and an immutable evaporation-curve record.

A strategy is ``certified`` only when **all three** gates pass at once:

* Deflated Sharpe Ratio ``DSR > 0.95``;
* Probability of Backtest Overfitting ``PBO < 0.5``;
* the bootstrap Sharpe confidence interval **excludes zero**.

SCOPE / LIMITATION: these gates correct for **multiple selection** (having
tried many strategies). They do **not** detect **look-ahead leakage**
(contamination of the information set). An oracle with deliberate leakage can
post an astronomically high Sharpe and sail through all three gates. The gates
are complementary to, not a substitute for, point-in-time discipline. See the
package docstring for the full caveat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from alphalab.domain.stats._moments import (
    kurtosis_non_central,
    sharpe_ratio,
    skewness,
)
from alphalab.domain.stats.autocorrelation import effective_sample_size
from alphalab.domain.stats.bootstrap import BootstrapCI, bootstrap_sharpe_ci
from alphalab.domain.stats.deflated_sharpe import deflated_sharpe_ratio
from alphalab.domain.stats.pbo import probability_of_backtest_overfitting

__all__ = [
    "DSR_THRESHOLD",
    "PBO_THRESHOLD",
    "CertificationResult",
    "EvaporationCurve",
    "certify_strategy",
]

DSR_THRESHOLD: float = 0.95
PBO_THRESHOLD: float = 0.5


@dataclass(frozen=True, slots=True)
class EvaporationCurve:
    """Immutable record of Sharpe decay across the research lifecycle.

    ``live_sharpe`` is intentionally optional: it is filled in Phase 5 once a
    strategy trades live. The decay fields are ``None`` until the downstream
    stage is available.
    """

    design_sharpe: float
    eval_sharpe: float
    live_sharpe: float | None = None

    @property
    def design_to_eval_decay(self) -> float:
        """Return the absolute Sharpe drop from design to evaluation."""
        return self.design_sharpe - self.eval_sharpe

    @property
    def eval_to_live_decay(self) -> float | None:
        """Return the eval-to-live Sharpe drop, or ``None`` if not yet live."""
        if self.live_sharpe is None:
            return None
        return self.eval_sharpe - self.live_sharpe


@dataclass(frozen=True, slots=True)
class CertificationResult:
    """Immutable certification verdict with all gate values and reasons."""

    dsr: float
    pbo: float
    bootstrap_ci: BootstrapCI
    t_eff: float
    observed_sharpe: float
    n_trials: int
    certified: bool
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)


def _gate_reasons(
    dsr: float,
    pbo: float,
    bootstrap_ci: BootstrapCI,
) -> tuple[str, ...]:
    """Return human-readable reasons for every gate that failed."""
    reasons: list[str] = []
    if not dsr > DSR_THRESHOLD:
        reasons.append(f"DSR {dsr:.4f} <= threshold {DSR_THRESHOLD:.2f}")
    if not pbo < PBO_THRESHOLD:
        reasons.append(f"PBO {pbo:.4f} >= threshold {PBO_THRESHOLD:.2f}")
    if not bootstrap_ci.excludes_zero:
        reasons.append(
            f"bootstrap CI [{bootstrap_ci.lower:.4f}, {bootstrap_ci.upper:.4f}] includes zero"
        )
    return tuple(reasons)


def certify_strategy(
    returns: npt.NDArray[np.float64],
    configuration_matrix: npt.NDArray[np.float64],
    *,
    n_trials: int,
    trial_sharpe_std: float,
    rng: np.random.Generator,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    n_strips: int = 16,
) -> CertificationResult:
    """Run all three gates and return an immutable certification verdict.

    Args:
        returns: Per-period returns of the selected strategy.
        configuration_matrix: ``T x N`` matrix of per-period returns across the
            tried configurations, consumed by the PBO/CSCV gate.
        n_trials: Number of configurations tried (``N``). The trial ledger is
            owned by another module; this function **receives** the count.
        trial_sharpe_std: ``sqrt(Var(SR))`` across the tried configurations.
        rng: Explicit seeded generator for the bootstrap (reproducibility).
        confidence: Bootstrap confidence level (default 0.95).
        n_resamples: Bootstrap resamples (default 10 000).
        n_strips: CSCV strip count ``S`` (default 16, even).

    Returns:
        A frozen :class:`CertificationResult`. ``certified`` is ``True`` only
        when DSR, PBO and the bootstrap CI all pass simultaneously.
    """
    t_eff = effective_sample_size(returns)
    observed_sharpe = sharpe_ratio(returns)
    dsr = deflated_sharpe_ratio(
        observed_sharpe,
        trial_sharpe_std=trial_sharpe_std,
        n_trials=n_trials,
        n_obs=t_eff,
        skew=skewness(returns),
        kurtosis=kurtosis_non_central(returns),
    )
    pbo = probability_of_backtest_overfitting(configuration_matrix, n_strips=n_strips)
    bootstrap_ci = bootstrap_sharpe_ci(
        returns,
        rng=rng,
        confidence=confidence,
        n_resamples=n_resamples,
    )
    reasons = _gate_reasons(dsr, pbo, bootstrap_ci)
    return CertificationResult(
        dsr=dsr,
        pbo=pbo,
        bootstrap_ci=bootstrap_ci,
        t_eff=t_eff,
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        certified=not reasons,
        rejection_reasons=reasons,
    )
