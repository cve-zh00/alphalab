"""Paired bootstrap test for the difference in Sharpe between two arms.

Why paired, and why bootstrap
------------------------------
The three arms are run over the **same decisions in the same order**, so their
per-period return series are *paired*: period ``t`` in arm B and period ``t``
in arm C are the same calendar step. The honest question "is B better than C?"
is therefore a question about the paired difference ``d_t = r^B_t - r^C_t``,
not about two independent samples. A paired bootstrap resamples the *indices*
once and applies them to both arms, preserving the pairing and the
cross-arm correlation; this is strictly more powerful than an unpaired test
and avoids falsely attributing common market moves to the arm.

We bootstrap the **difference in Sharpe** (not the difference in mean return)
because Sharpe is the metric the whole project certifies on, and a difference
in raw return can be pure volatility. The statistic is
``SR(arm_a) - SR(arm_b)`` recomputed on each resample; the CI is the empirical
quantile interval of that bootstrap distribution.

A difference is declared **significant** only when the CI *excludes zero*. The
Sharpe convention here is per-period (``mean/std``, ddof=1), matching
``alphalab.domain.stats``; annualisation is a monotone positive scaling and
does not change whether the interval excludes zero.

Multiple comparisons
--------------------
Comparing three arms means three pairwise tests (A-B, A-C, B-C). Running three
tests at a nominal ``alpha`` inflates the family-wise type-I error to roughly
``1 - (1-alpha)^3``. We apply a **Bonferroni** correction: each pairwise test
uses ``alpha / n_comparisons``, which bounds the family-wise error at ``alpha``
regardless of the (unknown) correlation between tests. Bonferroni is chosen
over Holm/Šidák deliberately: it is the most conservative, needs no ordering,
and this experiment would rather miss a real difference than manufacture a
false one — the entire project exists to *not* fool itself. The resulting
per-comparison confidence is ``1 - alpha/n_comparisons`` and is applied as the
bootstrap CI level. Each pairwise comparison is itself a trial and is recorded
in the ledger by the runner.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.stats._moments import sharpe_ratio

#: Number of pairwise comparisons among three arms (A-B, A-C, B-C).
N_PAIRWISE_COMPARISONS = 3


class PairedDifference(BaseModel):
    """Immutable result of one paired-bootstrap Sharpe-difference test.

    ``delta`` is ``SR(arm_a) - SR(arm_b)`` on the observed (paired) data; a
    positive ``delta`` means ``arm_a`` has the higher Sharpe. The interval is
    the bootstrap CI of ``delta`` at the Bonferroni-corrected confidence level.

    Attributes:
        name_a: Identifier of the first arm.
        name_b: Identifier of the second arm.
        delta: Observed per-period Sharpe difference ``SR(a) - SR(b)``.
        ci_low: Lower bound of the bootstrap CI of the difference.
        ci_high: Upper bound of the bootstrap CI of the difference.
        confidence: Per-comparison (Bonferroni-corrected) confidence level.
        n_comparisons: Family size used for the Bonferroni correction.
    """

    model_config = ConfigDict(frozen=True)

    name_a: str
    name_b: str
    delta: float
    ci_low: float
    ci_high: float
    confidence: float = Field(gt=0.0, lt=1.0)
    n_comparisons: int = Field(ge=1)

    @property
    def excludes_zero(self) -> bool:
        """True iff the CI lies strictly on one side of zero (a real diff)."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def winner(self) -> str | None:
        """Return the higher-Sharpe arm's name, or ``None`` if indistinguishable."""
        if not self.excludes_zero:
            return None
        return self.name_a if self.delta > 0.0 else self.name_b


def _paired_sharpe_delta_distribution(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    *,
    rng: np.random.Generator,
    n_resamples: int,
) -> npt.NDArray[np.float64]:
    """Bootstrap distribution of ``SR(a) - SR(b)`` under shared index resampling."""
    n = a.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    sa = a[idx]
    sb = b[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_a = sa.mean(axis=1)
        std_a = sa.std(axis=1, ddof=1)
        mean_b = sb.mean(axis=1)
        std_b = sb.std(axis=1, ddof=1)
        sharpe_a = np.where(std_a > 0.0, mean_a / std_a, 0.0)
        sharpe_b = np.where(std_b > 0.0, mean_b / std_b, 0.0)
    return np.asarray(sharpe_a - sharpe_b, dtype=np.float64)


def _observed_sharpe(returns: npt.NDArray[np.float64]) -> float:
    """Per-period Sharpe, returning 0.0 for a degenerate (zero-std) series."""
    if returns.size < 2:
        return 0.0
    if float(np.std(returns, ddof=1)) == 0.0:
        return 0.0
    return sharpe_ratio(returns)


def paired_sharpe_difference(
    returns_a: tuple[float, ...],
    returns_b: tuple[float, ...],
    *,
    name_a: str,
    name_b: str,
    rng: np.random.Generator,
    alpha: float = 0.05,
    n_comparisons: int = N_PAIRWISE_COMPARISONS,
    n_resamples: int = 10_000,
) -> PairedDifference:
    """Paired bootstrap CI for the Sharpe difference between two arms.

    Args:
        returns_a: Per-period return series of the first arm.
        returns_b: Per-period return series of the second arm; must be the
            same length as ``returns_a`` (paired by period).
        name_a: Identifier for the first arm.
        name_b: Identifier for the second arm.
        rng: Explicit seeded generator (reproducibility is mandatory).
        alpha: Family-wise significance level before correction.
        n_comparisons: Bonferroni family size (3 for the full experiment).
        n_resamples: Number of bootstrap resamples.

    Raises:
        ValueError: On mismatched lengths, fewer than two observations, or an
            ``alpha`` / ``n_comparisons`` outside their valid ranges.
    """
    a = np.asarray(returns_a, dtype=np.float64)
    b = np.asarray(returns_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired difference requires equal-length return series")
    if a.size < 2:
        raise ValueError("need at least two paired observations")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be >= 1")

    confidence = 1.0 - alpha / n_comparisons
    delta = _observed_sharpe(a) - _observed_sharpe(b)
    boot = _paired_sharpe_delta_distribution(a, b, rng=rng, n_resamples=n_resamples)
    tail = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(boot, tail))
    ci_high = float(np.quantile(boot, 1.0 - tail))
    return PairedDifference(
        name_a=name_a,
        name_b=name_b,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n_comparisons=n_comparisons,
    )
