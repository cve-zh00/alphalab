"""Hardened PBO gate: calibrated threshold, bootstrap CI and null p-value.

WHY THIS MODULE EXISTS — the naive ``PBO < 0.5`` gate is a coin flip
---------------------------------------------------------------------
The Probability of Backtest Overfitting (PBO, see :mod:`alphalab.domain.stats.pbo`)
is the fraction of symmetric CSCV partitions whose in-sample winner lands below
the out-of-sample median. Under the null hypothesis of **no skill**, the
in-sample winner's out-of-sample rank is uniformly distributed, so each
partition's "below median" indicator is a fair coin and the **expected PBO is
≈0.5**. A gate that certifies whenever ``PBO < 0.5`` therefore uses the null
expectation itself as the bar: edgeless noise falls on either side of 0.5 by
chance, so roughly half of pure-noise searches sail through.

This was confirmed empirically. Monte Carlo of the PBO under the null
(independent driftless GBM columns, ``mu=0``) gives, for the acid-test shape
``T=504, N=100, S=16`` over ``n_sims=200`` with ``seed=999``:

    mean   = 0.4810
    std    = 0.1462
    p5     = 0.2326
    p25    = 0.3807
    p50    = 0.4662
    p95    = 0.7401

The distribution is centred almost exactly on 0.5 with a wide spread
(``sd≈0.15``). That spread is the whole problem: a single edgeless search can
easily post a PBO of 0.38 or 0.42 and clear the 0.5 bar.

The audit scenarios (all built to have NO edge except D), ``seed=2024``,
``S=16``, make it concrete (``pbo`` point, 95% bootstrap CI, p-value =
``P(PBO_null <= pbo)``):

    A) 100 independent GBM columns (noise)      PBO=0.5453  CI=[0.537,0.554]  p=0.665
    B) 100 variations of one family (noise)     PBO=0.3809  CI=[0.373,0.389]  p=0.255
    C) deliberate IS/OOS reversal               PBO=0.2723  CI=[0.265,0.280]  p=0.065
    D) one genuinely dominant config (signal)   PBO=0.0000  CI=[0.000,0.000]  p=0.000

Under the old ``PBO < 0.5`` gate, B, C and D PASS and only A (barely) fails —
i.e. **two pure-noise searches (B, C) certify**. Under the hardened gate below,
only D passes.

WHAT THE HARDENED GATE REQUIRES
-------------------------------
A configuration must clear **all** of:

1. ``pbo <= PBO_POINT_THRESHOLD`` (0.25) — a calibrated bar well below the null
   median of ≈0.5, not the naive 0.5. 0.25 sits near the null's 5th percentile,
   so clearing it is already unlikely under pure noise.
2. ``p_value <= PBO_PVALUE_THRESHOLD`` (0.05) — the observed PBO must be
   *significantly* low versus the simulated null, not merely on the low side of
   a coin flip. ``p_value = P(PBO_null <= pbo)``; a low PBO is good, so a small
   p-value is the pass condition.
3. ``ci_upper <= PBO_POINT_THRESHOLD`` — the **upper** end of the bootstrap CI
   must also clear the bar, so a point estimate that is low only by luck of the
   partition draw cannot certify.

HONEST SCOPE — read before trusting this gate (do NOT overstate its power)
--------------------------------------------------------------------------
* **What PBO detects:** persistence of the in-sample→out-of-sample ranking
  *between the configurations of one search*. It answers "does the config I
  picked because it won in-sample keep winning out-of-sample?".
* **What PBO does NOT detect:**
  - **Leakage / look-ahead.** Exactly as documented in ``certification.py`` and
    the package docstring: a leaked oracle has a *real* edge within its
    contaminated data, so its PBO is 0 and it passes. Only point-in-time
    structure (``alphalab.domain.pit``) stops leakage.
  - **Independent configurations.** PBO discriminates *poorly* when the N
    configurations are mutually independent rather than variations of a common
    family. Scenario A (independent noise) sits right at the null, so its only
    signal is the p-value; scenario B (one weakly-shared family) is a cleaner
    rejection. If a search explores unrelated ideas rather than a tuning grid,
    read the PBO with suspicion.
* **The honest headline from our own acid test:** on the best-of-100 noise
  search (``seed=123``), the **DSR was the only gate that caught the overfit**
  (DSR≈0.55 rejected). The old PBO (0.4239) and the bootstrap CI both passed.
  The hardened gate here now *also* rejects that case (0.4239 > 0.25, p=0.365),
  but the credit for the original catch belongs to the DSR. Do not read a power
  into the PBO that the DSR actually supplied.

Everything is pure and explicitly seeded; the null simulation and the bootstrap
both take a caller-supplied :class:`numpy.random.Generator`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from alphalab.domain.stats.pbo import partition_logits
from alphalab.domain.stats.synthetic import gbm_return_matrix

__all__ = [
    "DEFAULT_PBO_BOOTSTRAP",
    "DEFAULT_PBO_NULL_SIMS",
    "PBO_CI_CONFIDENCE",
    "PBO_POINT_THRESHOLD",
    "PBO_PVALUE_THRESHOLD",
    "PBOGateResult",
    "PBONullStats",
    "evaluate_pbo_gate",
    "pbo_null_distribution",
]

# --- Calibrated gate constants (criterion documented in the module docstring) ---

# Point bar for the PBO, deliberately far below the null median of ≈0.5. 0.25
# is near the 5th percentile of the simulated null, so clearing it is already
# improbable under pure noise.
PBO_POINT_THRESHOLD: float = 0.25

# Significance level for the p-value against the simulated null. The observed
# PBO must be low enough that fewer than 5% of edgeless searches reach it.
PBO_PVALUE_THRESHOLD: float = 0.05

# Two-sided confidence level for the bootstrap CI over the PBO.
PBO_CI_CONFIDENCE: float = 0.95

# Default resample / simulation budgets. Both are overridable so tests (and
# small strip counts) stay fast; the vectorized CSCV makes the full-S null
# cheap enough for production use.
DEFAULT_PBO_BOOTSTRAP: int = 2_000
DEFAULT_PBO_NULL_SIMS: int = 200


@dataclass(frozen=True, slots=True)
class PBONullStats:
    """Immutable summary of the simulated PBO null distribution."""

    mean: float
    std: float
    p05: float
    p25: float
    p50: float
    p95: float
    n_sims: int


@dataclass(frozen=True, slots=True)
class PBOGateResult:
    """Immutable verdict of the hardened PBO gate.

    ``passed`` is ``True`` only when the point estimate, the bootstrap CI upper
    bound and the null p-value all clear their calibrated thresholds. For a
    single configuration (``N == 1``) there is no selection, so the result is
    the no-selection sentinel ``pbo = 0.0, p_value = 0.0, CI = [0, 0]`` and the
    gate passes on PBO (the other certification gates still apply).
    """

    pbo: float
    ci_lower: float
    ci_upper: float
    p_value: float
    null_mean: float
    null_std: float
    n_configs: int
    n_strips: int
    passed: bool

    def reason(self) -> str | None:
        """Return a numeric human-readable rejection reason, or ``None``."""
        if self.passed:
            return None
        return (
            f"PBO {self.pbo:.4f} (95% CI upper {self.ci_upper:.4f}, "
            f"p={self.p_value:.4f} vs null mean {self.null_mean:.4f}) fails "
            f"hardened gate [point<={PBO_POINT_THRESHOLD:.2f}, "
            f"p<={PBO_PVALUE_THRESHOLD:.2f}]"
        )


def _simulate_null_pbos(
    *,
    n_periods: int,
    n_configs: int,
    n_strips: int,
    rng: np.random.Generator,
    n_sims: int,
    sigma: float,
) -> npt.NDArray[np.float64]:
    """Return the raw PBO values of ``n_sims`` edgeless matrices."""
    values = np.empty(n_sims, dtype=np.float64)
    for i in range(n_sims):
        matrix = gbm_return_matrix(
            rng=rng, n_periods=n_periods, n_configs=n_configs, mu=0.0, sigma=sigma
        )
        logits = partition_logits(matrix, n_strips=n_strips)
        values[i] = float((logits < 0.0).mean()) if logits.size else 0.5
    return values


def _null_stats(values: npt.NDArray[np.float64]) -> PBONullStats:
    """Summarise raw null PBO draws into an immutable :class:`PBONullStats`."""
    n_sims = int(values.size)
    pcts = np.percentile(values, [5, 25, 50, 95])
    return PBONullStats(
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if n_sims > 1 else 0.0,
        p05=float(pcts[0]),
        p25=float(pcts[1]),
        p50=float(pcts[2]),
        p95=float(pcts[3]),
        n_sims=n_sims,
    )


def pbo_null_distribution(
    *,
    n_periods: int,
    n_configs: int,
    n_strips: int,
    rng: np.random.Generator,
    n_sims: int = DEFAULT_PBO_NULL_SIMS,
    sigma: float = 0.01,
) -> PBONullStats:
    """Simulate the PBO null distribution for a given problem shape.

    Draws ``n_sims`` edgeless matrices (independent driftless GBM columns,
    ``mu = 0``) of shape ``(n_periods, n_configs)`` and returns the summary of
    their PBO values. This is the reference against which an observed PBO is
    scored for significance.

    Args:
        n_periods: Rows ``T`` of each simulated matrix.
        n_configs: Columns ``N`` (``>= 2``; selection is undefined for one).
        n_strips: CSCV strip count ``S`` (even, ``2 <= S <= T``).
        rng: Explicit seeded generator (reproducibility is mandatory).
        n_sims: Number of Monte Carlo draws (default 200).
        sigma: Per-period volatility of the edgeless columns.

    Raises:
        ValueError: If ``n_configs < 2`` or ``n_sims < 1``.
    """
    if n_configs < 2:
        raise ValueError("null distribution needs at least two configurations")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    values = _simulate_null_pbos(
        n_periods=n_periods,
        n_configs=n_configs,
        n_strips=n_strips,
        rng=rng,
        n_sims=n_sims,
        sigma=sigma,
    )
    return _null_stats(values)


def _bootstrap_pbo_ci(
    logits: npt.NDArray[np.float64],
    *,
    rng: np.random.Generator,
    n_boot: int,
    confidence: float,
) -> tuple[float, float]:
    """Return the percentile bootstrap CI of ``(logits < 0).mean()``.

    Resamples the per-partition logits with replacement. This captures the
    sampling noise *within the observed partition set*; it is deliberately one
    of three checks, because on its own it is tight (the 12 870 partitions are
    highly correlated) and must not be trusted as the sole PBO uncertainty.
    """
    n_partitions = logits.size
    idx = rng.integers(0, n_partitions, size=(n_boot, n_partitions))
    fractions = (logits[idx] < 0.0).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(fractions, alpha))
    upper = float(np.quantile(fractions, 1.0 - alpha))
    return lower, upper


def evaluate_pbo_gate(
    configuration_matrix: npt.NDArray[np.float64],
    *,
    rng: np.random.Generator,
    n_strips: int = 16,
    n_bootstrap: int = DEFAULT_PBO_BOOTSTRAP,
    n_null_sims: int = DEFAULT_PBO_NULL_SIMS,
    point_threshold: float = PBO_POINT_THRESHOLD,
    pvalue_threshold: float = PBO_PVALUE_THRESHOLD,
    confidence: float = PBO_CI_CONFIDENCE,
) -> PBOGateResult:
    """Evaluate the hardened PBO gate on a ``T x N`` configuration matrix.

    Computes the PBO point estimate, a bootstrap CI over it, and a p-value
    against a freshly simulated null of the same shape, then applies the
    calibrated multi-criterion decision (see the module docstring).

    The ``N == 1`` branch is preserved explicitly: with a single configuration
    there is no selection, so there is no selection-overfitting to measure. The
    result is the no-selection sentinel (PBO 0, p 0, CI [0, 0]) and the gate
    passes on PBO — this is what keeps a passive benchmark (acid-test criterion
    1) certifiable.

    Args:
        configuration_matrix: ``T x N`` per-period returns across tried configs.
        rng: Explicit seeded generator; split internally for the bootstrap and
            the null simulation so both are reproducible.
        n_strips: CSCV strip count ``S`` (even, default 16).
        n_bootstrap: Bootstrap resamples over the partition logits.
        n_null_sims: Monte Carlo draws for the null distribution.
        point_threshold: Calibrated PBO bar (default :data:`PBO_POINT_THRESHOLD`).
        pvalue_threshold: Significance level (default :data:`PBO_PVALUE_THRESHOLD`).
        confidence: Bootstrap CI confidence (default :data:`PBO_CI_CONFIDENCE`).

    Raises:
        ValueError: On a non-2D matrix or zero configurations.
    """
    matrix = np.asarray(configuration_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("configuration_matrix must be a 2D (T x N) matrix")
    n_periods, n_configs = matrix.shape
    if n_configs < 1:
        raise ValueError("need at least one configuration")
    if n_configs == 1:
        # No selection => no selection-overfitting. Preserve the N=1 sentinel so
        # a passive benchmark keeps certifying (see pbo.py's N=1 branch).
        return PBOGateResult(
            pbo=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=0.0,
            null_mean=0.5,
            null_std=0.0,
            n_configs=1,
            n_strips=n_strips,
            passed=True,
        )

    # Derive independent child generators so bootstrap and null never share a
    # stream, while the whole evaluation stays reproducible from ``rng``.
    boot_rng, null_rng = rng.spawn(2)

    logits = partition_logits(matrix, n_strips=n_strips)
    if logits.size == 0:
        raise ValueError("no valid partitions produced a ranking")
    pbo = float((logits < 0.0).mean())

    ci_lower, ci_upper = _bootstrap_pbo_ci(
        logits, rng=boot_rng, n_boot=n_bootstrap, confidence=confidence
    )
    null_values = _simulate_null_pbos(
        n_periods=n_periods,
        n_configs=n_configs,
        n_strips=n_strips,
        rng=null_rng,
        n_sims=n_null_sims,
        sigma=0.01,
    )
    null = _null_stats(null_values)
    # A low PBO is the good outcome, so significance is the left-tail mass of
    # the null at or below the observed value: P(PBO_null <= observed).
    p_value = float((null_values <= pbo).mean())

    passed = pbo <= point_threshold and ci_upper <= point_threshold and p_value <= pvalue_threshold
    return PBOGateResult(
        pbo=pbo,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        null_mean=null.mean,
        null_std=null.std,
        n_configs=n_configs,
        n_strips=n_strips,
        passed=passed,
    )
