"""Deflated Sharpe Ratio (DSR) and the expected-maximum Sharpe under H0.

Reference: Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting and Non-Normality" (Journal of
Portfolio Management, 2014).

The DSR corrects a Sharpe ratio for **multiple selection**: having tried many
strategy configurations inflates the maximum observed Sharpe even under the
null hypothesis of zero true skill. The expected maximum Sharpe of ``N`` i.i.d.
trials is

    SR0 = std(SR_trials) * [ (1 - gamma) * Phi^-1(1 - 1/N)
                             + gamma      * Phi^-1(1 - 1/(N*e)) ]

with ``gamma`` the Euler-Mascheroni constant. The DSR itself is

    DSR = Phi[ (SR_hat - SR0) * sqrt(T - 1)
               / sqrt(1 - skew*SR_hat + ((kurt - 1)/4)*SR_hat^2) ]

where ``skew`` and ``kurt`` are the skewness and the **non-central** kurtosis
of the strategy returns, and ``T`` is the effective sample length.

Convention: ``SR_hat`` and the trial Sharpe ratios are **per-period**
(non-annualized). Feeding an annualized Sharpe would break the ``sqrt(T-1)``
scaling and the sanity targets documented in the tests.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["deflated_sharpe_ratio", "expected_max_sharpe"]

_GAMMA: float = float(np.euler_gamma)


def expected_max_sharpe(trial_sharpe_std: float, n_trials: int) -> float:
    """Return ``SR0``, the expected maximum Sharpe of ``n_trials`` under H0.

    Args:
        trial_sharpe_std: Standard deviation of the Sharpe ratios across the
            tried configurations (``sqrt(Var(SR))``).
        n_trials: Number of independent trials/configurations tested (``N``).

    Raises:
        ValueError: If ``n_trials < 2`` or ``trial_sharpe_std < 0``.
    """
    if n_trials < 2:
        raise ValueError("n_trials must be at least 2")
    if trial_sharpe_std < 0.0:
        raise ValueError("trial_sharpe_std must be non-negative")
    n = float(n_trials)
    first = stats.norm.ppf(1.0 - 1.0 / n)
    second = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    expected_max_z = (1.0 - _GAMMA) * first + _GAMMA * second
    return trial_sharpe_std * float(expected_max_z)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    trial_sharpe_std: float,
    n_trials: int,
    n_obs: float,
    skew: float,
    kurtosis: float,
) -> float:
    """Return the Deflated Sharpe Ratio in ``[0, 1]``.

    Args:
        observed_sharpe: Per-period Sharpe of the selected strategy (``SR_hat``).
        trial_sharpe_std: ``sqrt(Var(SR))`` across the tried configurations.
        n_trials: Number of configurations tried (``N``).
        n_obs: Effective number of observations (``T``); pass the
            autocorrelation-adjusted ``T_eff`` here, not the raw count.
        skew: Skewness of the strategy returns.
        kurtosis: **Non-central** kurtosis of the strategy returns (3 for a
            normal distribution). Use
            :func:`alphalab.domain.stats._moments.kurtosis_non_central`.

    Raises:
        ValueError: If ``n_obs <= 1`` or the variance denominator is not
            strictly positive.
    """
    if n_obs <= 1.0:
        raise ValueError("n_obs (T) must be greater than 1")
    sr0 = expected_max_sharpe(trial_sharpe_std, n_trials)
    variance = 1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2
    if variance <= 0.0:
        raise ValueError("non-positive variance in DSR denominator")
    statistic = (observed_sharpe - sr0) * np.sqrt(n_obs - 1.0) / np.sqrt(variance)
    return float(stats.norm.cdf(statistic))
