"""Autocorrelation-adjusted effective sample size (``T_eff``).

Autocorrelated returns carry less independent information than their raw
observation count suggests, so using the raw ``T`` in the Deflated Sharpe
Ratio understates the uncertainty. We down-weight ``T`` with a Newey-West /
Lo-style correction.

Formula (Lo, "The Statistics of Sharpe Ratios", Financial Analysts Journal,
2002). The variance of the time-aggregated return inflates by

    eta = 1 + 2 * sum_{k=1}^{q} (1 - k/(q+1)) * rho_k

where ``rho_k`` is the lag-``k`` sample autocorrelation and ``q`` is the
Newey-West bandwidth (Bartlett kernel). The effective sample size is

    T_eff = T / eta

Clamped to the range ``[1, T]``: positive autocorrelation (``eta > 1``)
reduces ``T_eff`` below ``T``; no autocorrelation gives ``T_eff ~ T``.
Negative-autocorrelation cases that would push ``eta < 1`` are clamped so
``T_eff`` never exceeds the raw count (conservative choice).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["autocorrelation", "effective_sample_size"]


def autocorrelation(returns: npt.NDArray[np.float64], lag: int) -> float:
    """Return the biased sample autocorrelation of ``returns`` at ``lag``.

    Uses the biased estimator (divides by ``T``) so the implied spectral
    density stays non-negative, which keeps the Newey-West adjustment stable.

    Raises:
        ValueError: If ``lag`` is negative or not smaller than the series
            length.
    """
    arr = np.asarray(returns, dtype=np.float64)
    n = arr.size
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if lag >= n:
        raise ValueError("lag must be smaller than the series length")
    centered = arr - arr.mean()
    denom = float(np.dot(centered, centered))
    if denom == 0.0:
        return 0.0
    numer = float(np.dot(centered[: n - lag], centered[lag:]))
    return numer / denom


def effective_sample_size(
    returns: npt.NDArray[np.float64],
    *,
    max_lag: int | None = None,
) -> float:
    """Return the autocorrelation-adjusted effective sample size ``T_eff``.

    Args:
        returns: One-dimensional array of per-period returns.
        max_lag: Newey-West bandwidth ``q``. Defaults to
            ``floor(4 * (T/100)^(2/9))`` (the common Newey-West rule), capped
            at ``T - 1``.

    Returns:
        ``T_eff`` in the range ``[1, T]``.

    Raises:
        ValueError: If ``returns`` has fewer than two observations.
    """
    arr = np.asarray(returns, dtype=np.float64)
    n = arr.size
    if n < 2:
        raise ValueError("returns must contain at least two observations")
    q = _default_bandwidth(n) if max_lag is None else max_lag
    q = max(0, min(q, n - 1))
    eta = 1.0
    for k in range(1, q + 1):
        weight = 1.0 - k / (q + 1.0)
        eta += 2.0 * weight * autocorrelation(arr, k)
    eta = max(eta, 1.0)  # never inflate T_eff above the raw count
    t_eff = n / eta
    return float(min(max(t_eff, 1.0), float(n)))


def _default_bandwidth(n: int) -> int:
    """Return the Newey-West automatic bandwidth for a series of length ``n``."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
