"""Certification verdict and the evaporation curve report artifact.

This module ties the individual statistical gates together into a single,
immutable certification verdict and an immutable evaporation-curve record.

A strategy is ``certified`` only when **all three** gates pass at once:

* Deflated Sharpe Ratio ``DSR > 0.95``;
* the **hardened** Probability of Backtest Overfitting gate (see
  :mod:`alphalab.domain.stats.pbo_gate`): the PBO point estimate, its bootstrap
  CI upper bound and its p-value against a simulated null must all clear
  calibrated bars. This replaces the old naive ``PBO < 0.5`` test, whose
  threshold coincided with the null expectation of ≈0.5 and so passed roughly
  half of all pure-noise searches;
* the bootstrap Sharpe confidence interval **excludes zero**.

SCOPE / LIMITATION: these gates correct for **multiple selection** (having
tried many strategies). They do **not** detect **look-ahead leakage**
(contamination of the information set). An oracle with deliberate leakage can
post an astronomically high Sharpe and sail through all three gates. The gates
are complementary to, not a substitute for, point-in-time discipline. See the
package docstring and :mod:`alphalab.domain.stats.pbo_gate` for the full
caveat, including that in our own acid test the **DSR** was the only gate to
catch the best-of-100 noise search.
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
from alphalab.domain.stats.pbo_gate import (
    DEFAULT_PBO_BOOTSTRAP,
    DEFAULT_PBO_NULL_SIMS,
    PBO_POINT_THRESHOLD,
    PBO_PVALUE_THRESHOLD,
    PBOGateResult,
    evaluate_pbo_gate,
)

__all__ = [
    "DSR_THRESHOLD",
    "PBO_POINT_THRESHOLD",
    "PBO_PVALUE_THRESHOLD",
    "PBO_THRESHOLD",
    "CertificationResult",
    "EvaporationCurve",
    "certify_strategy",
]

DSR_THRESHOLD: float = 0.95

# DEPRECATED naive PBO bar. Kept only for backward reference and to document
# the audit finding: 0.5 is the *null expectation* of the PBO, so a gate at
# 0.5 is a coin flip. The live gate is the hardened one in ``pbo_gate.py``
# (point threshold ``PBO_POINT_THRESHOLD`` plus a p-value and a bootstrap CI).
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
    """Immutable certification verdict with all gate values and reasons.

    ``pbo`` is retained as the scalar point estimate for backward
    compatibility; ``pbo_gate`` carries the full hardened verdict (bootstrap
    CI, null p-value and the pass/fail decision).
    """

    dsr: float
    pbo: float
    bootstrap_ci: BootstrapCI
    t_eff: float
    observed_sharpe: float
    n_trials: int
    certified: bool
    pbo_gate: PBOGateResult
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)


def _gate_reasons(
    dsr: float,
    pbo_gate: PBOGateResult,
    bootstrap_ci: BootstrapCI,
) -> tuple[str, ...]:
    """Return human-readable reasons for every gate that failed."""
    reasons: list[str] = []
    if not dsr > DSR_THRESHOLD:
        reasons.append(f"DSR {dsr:.4f} <= threshold {DSR_THRESHOLD:.2f}")
    pbo_reason = pbo_gate.reason()
    if pbo_reason is not None:
        reasons.append(pbo_reason)
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
    n_pbo_bootstrap: int = DEFAULT_PBO_BOOTSTRAP,
    n_pbo_null_sims: int = DEFAULT_PBO_NULL_SIMS,
) -> CertificationResult:
    """Run all three gates and return an immutable certification verdict.

    Args:
        returns: Per-period returns of the selected strategy.
        configuration_matrix: ``T x N`` matrix of per-period returns across the
            tried configurations, consumed by the hardened PBO/CSCV gate.
        n_trials: Number of configurations tried (``N``). The trial ledger is
            owned by another module; this function **receives** the count.
        trial_sharpe_std: ``sqrt(Var(SR))`` across the tried configurations.
        rng: Explicit seeded generator; **spawned** internally into independent
            child streams for the Sharpe bootstrap and the PBO gate so the whole
            verdict is reproducible from one seed.
        confidence: Bootstrap confidence level (default 0.95).
        n_resamples: Sharpe bootstrap resamples (default 10 000).
        n_strips: CSCV strip count ``S`` (default 16, even).
        n_pbo_bootstrap: Bootstrap resamples for the PBO CI (default 2 000).
        n_pbo_null_sims: Monte Carlo draws for the PBO null (default 200). Lower
            it in tests to keep the simulation cheap.

    Returns:
        A frozen :class:`CertificationResult`. ``certified`` is ``True`` only
        when the DSR, the **hardened** PBO gate and the bootstrap CI all pass
        simultaneously.
    """
    sharpe_rng, pbo_rng = rng.spawn(2)
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
    pbo_gate = evaluate_pbo_gate(
        configuration_matrix,
        rng=pbo_rng,
        n_strips=n_strips,
        n_bootstrap=n_pbo_bootstrap,
        n_null_sims=n_pbo_null_sims,
    )
    bootstrap_ci = bootstrap_sharpe_ci(
        returns,
        rng=sharpe_rng,
        confidence=confidence,
        n_resamples=n_resamples,
    )
    reasons = _gate_reasons(dsr, pbo_gate, bootstrap_ci)
    return CertificationResult(
        dsr=dsr,
        pbo=pbo_gate.pbo,
        bootstrap_ci=bootstrap_ci,
        t_eff=t_eff,
        observed_sharpe=observed_sharpe,
        n_trials=n_trials,
        certified=not reasons,
        pbo_gate=pbo_gate,
        rejection_reasons=reasons,
    )
