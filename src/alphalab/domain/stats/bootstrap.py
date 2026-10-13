"""Bootstrap confidence intervals for the Sharpe ratio.

Two methods are offered:

* ``"percentile"`` — the empirical quantiles of the bootstrap distribution.
  Simple but biased when the statistic's sampling distribution is skewed.
* ``"bca"`` (default) — bias-corrected and accelerated (Efron, 1987). It
  adjusts the percentile endpoints for both median bias (``z0``) and skew
  (acceleration ``a`` from a jackknife), giving substantially better coverage
  for the Sharpe ratio, whose distribution is skewed under non-normal returns.

BCa is the default because the certification gate asks whether the interval
**excludes zero**, and percentile intervals can be materially miscentred for
skewed return series. Reproducibility is guaranteed by requiring an explicit
``numpy.random.Generator`` seeded by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats

from alphalab.domain.stats._moments import sharpe_ratio

__all__ = ["BootstrapCI", "bootstrap_sharpe_ci"]

BootstrapMethod = Literal["percentile", "bca"]


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Immutable bootstrap confidence interval result."""

    point: float
    lower: float
    upper: float
    confidence: float
    method: BootstrapMethod

    @property
    def excludes_zero(self) -> bool:
        """Return ``True`` if the interval lies strictly on one side of zero."""
        return self.lower > 0.0 or self.upper < 0.0


def _bootstrap_distribution(
    returns: npt.NDArray[np.float64],
    n_resamples: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Return the bootstrap distribution of the Sharpe ratio."""
    n = returns.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples = returns[idx]
    mean = samples.mean(axis=1)
    std = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0.0, mean / std, 0.0)
    return np.asarray(sharpe, dtype=np.float64)


def _bca_endpoints(
    returns: npt.NDArray[np.float64],
    boot: npt.NDArray[np.float64],
    point: float,
    alpha: float,
) -> tuple[float, float]:
    """Return the lower/upper percentile fractions adjusted by BCa."""
    proportion = float((boot < point).mean())
    proportion = min(max(proportion, 1e-9), 1.0 - 1e-9)
    z0 = float(stats.norm.ppf(proportion))

    jackknife = _jackknife_sharpe(returns)
    centered = jackknife.mean() - jackknife
    numer = float((centered**3).sum())
    denom = 6.0 * float((centered**2).sum()) ** 1.5
    accel = numer / denom if denom != 0.0 else 0.0

    z_lo = stats.norm.ppf(alpha)
    z_hi = stats.norm.ppf(1.0 - alpha)
    lower = stats.norm.cdf(z0 + (z0 + z_lo) / (1.0 - accel * (z0 + z_lo)))
    upper = stats.norm.cdf(z0 + (z0 + z_hi) / (1.0 - accel * (z0 + z_hi)))
    return float(lower), float(upper)


def _jackknife_sharpe(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return leave-one-out Sharpe ratios used for the BCa acceleration."""
    n = returns.size
    total = returns.sum()
    total_sq = (returns**2).sum()
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        mean = (total - returns[i]) / (n - 1)
        var = (total_sq - returns[i] ** 2) / (n - 1) - mean**2
        var = var * (n - 1) / (n - 2)  # convert to sample variance (ddof=1)
        out[i] = mean / np.sqrt(var) if var > 0.0 else 0.0
    return out


def bootstrap_sharpe_ci(
    returns: npt.NDArray[np.float64],
    *,
    rng: np.random.Generator,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    method: BootstrapMethod = "bca",
) -> BootstrapCI:
    """Return a bootstrap confidence interval for the Sharpe ratio.

    Args:
        returns: One-dimensional array of per-period returns.
        rng: Explicit seeded generator (reproducibility is mandatory).
        confidence: Two-sided confidence level in ``(0, 1)`` (default 0.95).
        n_resamples: Number of bootstrap resamples (default 10 000).
        method: ``"bca"`` (default) or ``"percentile"``.

    Raises:
        ValueError: On fewer than three observations (BCa needs a jackknife),
            or a confidence level outside ``(0, 1)``.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if arr.size < 3:
        raise ValueError("need at least three observations for a bootstrap CI")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    point = sharpe_ratio(arr)
    boot = _bootstrap_distribution(arr, n_resamples, rng)
    alpha = (1.0 - confidence) / 2.0

    if method == "percentile":
        lo_frac, hi_frac = alpha, 1.0 - alpha
    else:
        lo_frac, hi_frac = _bca_endpoints(arr, boot, point, alpha)

    lower = float(np.quantile(boot, lo_frac))
    upper = float(np.quantile(boot, hi_frac))
    return BootstrapCI(
        point=point,
        lower=lower,
        upper=upper,
        confidence=confidence,
        method=method,
    )
