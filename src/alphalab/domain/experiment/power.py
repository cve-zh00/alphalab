"""Approximate power analysis for the Sharpe-difference test.

The most dishonest thing this experiment could do is report "the arms are
indistinguishable" when the real reason is that the sample is too small to
distinguish *anything*. "Indistinguishable" and "underpowered" are different
claims and the verdict must separate them.

Minimum detectable effect
--------------------------
For a per-period Sharpe ``SR`` estimated from ``n`` observations, the classic
Lo (2002) asymptotic standard error (ignoring higher moments) is
``se(SR) ~= sqrt((1 + SR^2 / 2) / n)``. For a two-sided test at confidence
``c`` with power ``1 - beta``, the minimum detectable difference in Sharpe is

    ``MDE ~= (z_{1-alpha/2} + z_{1-beta}) * se_diff``

where ``se_diff`` combines the two arms' standard errors. Because the arms are
*paired* and positively correlated, the true ``se_diff`` is smaller than the
independent combination; using the independent (``sqrt(2)``) combination is
therefore a **conservative upper bound** on the MDE — we would rather overstate
how large an effect we can miss than understate it. All of this is asymptotic
and ignores skew/kurtosis, so it is reported as "approximate" and used only to
decide whether a null result is *conclusive* (sample large enough to have
caught a materially-sized effect) or *inconclusive* (underpowered).
"""

from __future__ import annotations

import math

from scipy import stats

#: A Sharpe-difference the project considers "material" if it existed: a
#: per-period Sharpe gap of 0.05 annualises (x sqrt(252)) to ~0.79, roughly the
#: entire SPY buy-and-hold Sharpe. If the sample cannot even detect a gap this
#: large, a null result says nothing.
MATERIAL_PERIOD_SHARPE_GAP = 0.05


def _z(p: float) -> float:
    """Standard-normal quantile."""
    return float(stats.norm.ppf(p))


def sharpe_standard_error(observed_sharpe: float, n: int) -> float:
    """Lo (2002) asymptotic SE of a per-period Sharpe estimate.

    Raises:
        ValueError: If ``n`` is below two observations.
    """
    if n < 2:
        raise ValueError("need at least two observations for a Sharpe SE")
    return math.sqrt((1.0 + 0.5 * observed_sharpe**2) / n)


def minimum_detectable_sharpe_difference(
    observed_sharpe: float,
    n: int,
    *,
    confidence: float,
    power: float = 0.80,
) -> float:
    """Conservative minimum detectable per-period Sharpe difference.

    Uses the independent (unpaired) combination of standard errors, which is a
    conservative upper bound for the paired design.

    Args:
        observed_sharpe: A representative per-period Sharpe (e.g. the baseline
            arm's) used to scale the standard error.
        n: Number of per-period observations per arm.
        confidence: Per-comparison (Bonferroni-corrected) confidence level.
        power: Desired statistical power ``1 - beta`` (default 0.80).

    Raises:
        ValueError: On an out-of-range ``confidence``/``power`` or ``n < 2``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    alpha = 1.0 - confidence
    se = sharpe_standard_error(observed_sharpe, n)
    se_diff = math.sqrt(2.0) * se
    return (_z(1.0 - alpha / 2.0) + _z(power)) * se_diff


def is_conclusive(
    observed_sharpe: float,
    n: int,
    *,
    confidence: float,
    power: float = 0.80,
    material_gap: float = MATERIAL_PERIOD_SHARPE_GAP,
) -> bool:
    """True iff the sample could detect a *material* Sharpe gap.

    A null result is only "conclusive" (truly indistinguishable rather than
    merely underpowered) when the minimum detectable effect is no larger than
    the gap the project deems material.
    """
    mde = minimum_detectable_sharpe_difference(
        observed_sharpe, n, confidence=confidence, power=power
    )
    return mde <= material_gap
