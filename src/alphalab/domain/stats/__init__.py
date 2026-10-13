"""Statistical certification gates for strategy selection.

This package implements the statistical gates that guard against **backtest
overfitting through multiple selection**: Deflated Sharpe Ratio (DSR),
Probability of Backtest Overfitting (PBO via CSCV), a bootstrap confidence
interval for the Sharpe ratio, an autocorrelation-adjusted effective sample
size, and a combined certification verdict.

CRITICAL SCOPE LIMITATION — READ BEFORE RELYING ON THESE GATES
--------------------------------------------------------------
These gates correct for **multiple selection** (the statistical cost of having
tried many strategy configurations). They do **NOT** correct for **look-ahead
leakage** (contamination of the information set with future data).

The two problems are complementary, not substitutes. In the reference study
(arXiv:2608.27734) an oracle built with deliberate leakage reached a Sharpe of
34.7 and passed DSR = 1.00 and a clean PBO without difficulty — because its
edge was real within the (contaminated) data it was given; nothing here
inspects whether that data was legitimately available at decision time.

Leakage must be prevented upstream by strict point-in-time discipline
(see ``alphalab.domain.pit``). Passing every gate in this package is necessary
but **not sufficient** for a trustworthy strategy.

Sharpe convention
-----------------
Every Sharpe ratio in this package is **per-period (non-annualized)**:
``mean(returns) / std(returns, ddof=1)``. The DSR sanity targets (N = 100,
raw Sharpe 1.0 -> DSR ~ 0.6; Sharpe ~ 1.8 -> DSR ~ 0.95) assume this
convention together with the effective sample length ``T_eff``.
"""

from alphalab.domain.stats._moments import (
    kurtosis_non_central,
    sharpe_ratio,
    skewness,
)
from alphalab.domain.stats.autocorrelation import (
    autocorrelation,
    effective_sample_size,
)
from alphalab.domain.stats.bootstrap import BootstrapCI, bootstrap_sharpe_ci
from alphalab.domain.stats.certification import (
    DSR_THRESHOLD,
    PBO_THRESHOLD,
    CertificationResult,
    EvaporationCurve,
    certify_strategy,
)
from alphalab.domain.stats.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from alphalab.domain.stats.pbo import (
    partition_logits,
    probability_of_backtest_overfitting,
)
from alphalab.domain.stats.pbo_gate import (
    PBO_POINT_THRESHOLD,
    PBO_PVALUE_THRESHOLD,
    PBOGateResult,
    PBONullStats,
    evaluate_pbo_gate,
    pbo_null_distribution,
)
from alphalab.domain.stats.synthetic import gbm_log_returns, gbm_return_matrix

__all__ = [
    "DSR_THRESHOLD",
    "PBO_POINT_THRESHOLD",
    "PBO_PVALUE_THRESHOLD",
    "PBO_THRESHOLD",
    "BootstrapCI",
    "CertificationResult",
    "EvaporationCurve",
    "PBOGateResult",
    "PBONullStats",
    "autocorrelation",
    "bootstrap_sharpe_ci",
    "certify_strategy",
    "deflated_sharpe_ratio",
    "effective_sample_size",
    "evaluate_pbo_gate",
    "expected_max_sharpe",
    "gbm_log_returns",
    "gbm_return_matrix",
    "kurtosis_non_central",
    "partition_logits",
    "pbo_null_distribution",
    "probability_of_backtest_overfitting",
    "sharpe_ratio",
    "skewness",
]
