"""Pure helpers for Sharpe ratios and return moments.

All functions here operate on a one-dimensional array of **per-period**
returns and are deterministic. The Sharpe ratio convention used across the
whole :mod:`alphalab.domain.stats` package is **per-period (not annualized)**:
``SR = mean(returns) / std(returns)`` with the sample standard deviation
(``ddof=1``). The Deflated Sharpe Ratio machinery in
:mod:`alphalab.domain.stats.deflated_sharpe` assumes this same convention, so
callers must stay consistent and never feed an annualized Sharpe into it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats

__all__ = ["kurtosis_non_central", "sharpe_ratio", "skewness"]


def sharpe_ratio(returns: npt.NDArray[np.float64]) -> float:
    """Return the per-period Sharpe ratio of ``returns``.

    The ratio is ``mean / std`` using the sample standard deviation
    (``ddof=1``). No risk-free rate is subtracted; pass excess returns if a
    risk-free adjustment is required.

    Raises:
        ValueError: If ``returns`` has fewer than two observations or a zero
            standard deviation (Sharpe undefined).
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if arr.size < 2:
        raise ValueError("returns must contain at least two observations")
    std = float(np.std(arr, ddof=1))
    if std == 0.0:
        raise ValueError("Sharpe ratio undefined: zero standard deviation")
    return float(np.mean(arr)) / std


def skewness(returns: npt.NDArray[np.float64]) -> float:
    """Return the sample skewness (third standardized moment) of ``returns``."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 2:
        raise ValueError("returns must contain at least two observations")
    return float(stats.skew(arr, bias=True))


def kurtosis_non_central(returns: npt.NDArray[np.float64]) -> float:
    """Return the **non-central** (non-excess) kurtosis of ``returns``.

    This uses ``scipy.stats.kurtosis`` with ``fisher=False`` so that a normal
    distribution yields ``3.0`` rather than ``0.0``. The Deflated Sharpe Ratio
    denominator requires this non-central (Pearson) convention; using the
    default excess kurtosis would silently bias the DSR. This is a classic
    source of error, hence the explicit, dedicated helper.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 2:
        raise ValueError("returns must contain at least two observations")
    return float(stats.kurtosis(arr, fisher=False, bias=True))
