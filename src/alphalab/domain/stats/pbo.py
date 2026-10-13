"""Probability of Backtest Overfitting (PBO) via CSCV.

Reference: Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest
Overfitting" (Journal of Computational Finance, 2017).

Combinatorially Symmetric Cross-Validation (CSCV) splits a ``T x N`` matrix of
per-period returns (``T`` periods, ``N`` strategy configurations) into ``S``
contiguous strips. For every way of choosing ``S/2`` strips as the in-sample
(IS) set (the remaining ``S/2`` form the out-of-sample, OOS, set):

1. Pick the configuration with the best IS performance.
2. Measure its **relative rank** among all configurations OOS.
3. Compute the logit of that rank; a logit ``< 0`` means the IS winner landed
   below the OOS median — an overfit selection.

PBO is the fraction of partitions whose logit is below zero. A value below
``0.5`` is the certification threshold.

Performance: the number of partitions is ``C(S, S/2)``; with the default
``S = 16`` that is 12 870. The ranking is computed **fully vectorized** over
both partitions and configurations — per-strip sufficient statistics are
pooled by a single matrix product against the partition membership mask, so no
Python loop runs over the 12 870 partitions. Tests may still lower ``S`` to
keep the combinatorics cheap.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import numpy.typing as npt

__all__ = ["partition_logits", "probability_of_backtest_overfitting"]


def _strip_bounds(n_periods: int, n_strips: int) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` index bounds for ``n_strips`` contiguous strips."""
    edges = np.linspace(0, n_periods, n_strips + 1, dtype=np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_strips)]


def probability_of_backtest_overfitting(
    returns: npt.NDArray[np.float64],
    *,
    n_strips: int = 16,
) -> float:
    """Return the PBO of a ``T x N`` return matrix via CSCV.

    Args:
        returns: Matrix with shape ``(T, N)`` — ``T`` periods, ``N`` configs.
        n_strips: Even number of strips ``S`` (default 16). Must be even and
            at least 2, and no greater than ``T``.

    Returns:
        PBO in ``[0, 1]``: the fraction of symmetric partitions where the best
        IS configuration ranks below the OOS median. For a single
        configuration (``N == 1``) there is no selection, so PBO is ``0.0`` by
        definition (see the inline note).

    Raises:
        ValueError: On a non-2D matrix, zero configurations, an odd or
            out-of-range ``n_strips`` (only checked when ``N >= 2``).
    """
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("returns must be a 2D (T x N) matrix")
    _, n_configs = matrix.shape
    if n_configs < 1:
        raise ValueError("need at least one configuration")
    if n_configs == 1:
        # Degenerate single-configuration case. CSCV measures overfitting *from
        # selection*: it asks whether the best IS configuration keeps its rank
        # OOS. With exactly one configuration there is nothing to select
        # between, so selection-driven overfitting is structurally impossible
        # and PBO is 0 by definition. This is an a-priori hypothesis (e.g. a
        # passive benchmark), not the survivor of a search, so returning 0.0 is
        # the honest answer rather than an error or a meaningless logit. This
        # is a deliberate decision, documented here so N=1 cannot certify by
        # accident: the DSR and bootstrap-CI gates still have to pass on their
        # own merits.
        return 0.0

    logits = partition_logits(matrix, n_strips=n_strips)
    if logits.size == 0:
        raise ValueError("no valid partitions produced a ranking")
    return float((logits < 0.0).mean())


def partition_logits(
    returns: npt.NDArray[np.float64],
    *,
    n_strips: int = 16,
) -> npt.NDArray[np.float64]:
    """Return the finite per-partition OOS-rank logits of a ``T x N`` matrix.

    This is the raw material the PBO is built from: one logit per symmetric
    CSCV partition (``C(S, S/2)`` of them). A logit below zero marks a
    partition where the best in-sample configuration landed below the
    out-of-sample median — an overfit selection. The public
    :func:`probability_of_backtest_overfitting` is simply
    ``(logits < 0).mean()``.

    Exposing the array lets the hardened gate bootstrap a confidence interval
    over the PBO and compute a p-value against the null **without** re-running
    the expensive CSCV loop (see :mod:`alphalab.domain.stats.pbo_gate`).

    Args:
        returns: Matrix with shape ``(T, N)`` — ``T`` periods, ``N >= 2``
            configs. (``N == 1`` has no selection; use the public PBO function,
            which short-circuits to ``0.0``.)
        n_strips: Even number of strips ``S`` (default 16), ``2 <= S <= T``.

    Returns:
        A 1-D array of finite logits (partitions whose rank fell exactly on a
        boundary are dropped, matching the public PBO fraction).

    Raises:
        ValueError: On a non-2D matrix, fewer than two configurations, or an
            odd/out-of-range ``n_strips``.
    """
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("returns must be a 2D (T x N) matrix")
    n_periods, n_configs = matrix.shape
    if n_configs < 2:
        raise ValueError("partition logits need at least two configurations")
    if n_strips < 2 or n_strips % 2 != 0:
        raise ValueError("n_strips must be an even integer >= 2")
    if n_strips > n_periods:
        raise ValueError("n_strips cannot exceed the number of periods")

    bounds = _strip_bounds(n_periods, n_strips)
    half = n_strips // 2

    # Per-strip sufficient statistics per configuration, so a partition's
    # pooled Sharpe is a plain sum over its strips — no per-partition
    # ``concatenate`` and no Python loop over the C(S, S/2) partitions.
    strip_n = np.empty(n_strips, dtype=np.float64)
    strip_sum = np.empty((n_strips, n_configs), dtype=np.float64)
    strip_sumsq = np.empty((n_strips, n_configs), dtype=np.float64)
    for s, (start, stop) in enumerate(bounds):
        block = matrix[start:stop]
        strip_n[s] = float(block.shape[0])
        strip_sum[s] = block.sum(axis=0)
        strip_sumsq[s] = (block**2).sum(axis=0)

    # Boolean membership mask for every symmetric partition: (P, S), with the
    # chosen half marked in-sample and its complement out-of-sample.
    combos = np.fromiter(
        (idx for combo in combinations(range(n_strips), half) for idx in combo),
        dtype=np.int64,
    ).reshape(-1, half)
    n_partitions = combos.shape[0]
    is_mask = np.zeros((n_partitions, n_strips), dtype=np.float64)
    rows = np.arange(n_partitions)[:, None]
    is_mask[rows, combos] = 1.0
    oos_mask = 1.0 - is_mask

    is_perf = _pooled_sharpe(is_mask, strip_n, strip_sum, strip_sumsq)
    oos_perf = _pooled_sharpe(oos_mask, strip_n, strip_sum, strip_sumsq)

    best_is = np.argmax(is_perf, axis=1)
    winner_oos = oos_perf[rows[:, 0], best_is]
    # Fraction of configs the IS winner beats OOS, strictly less than it.
    rank = (oos_perf < winner_oos[:, None]).sum(axis=1) / (n_configs - 1)
    omega = (rank * n_configs + 0.5) / (n_configs + 1.0)

    finite = (omega > 0.0) & (omega < 1.0)
    omega = omega[finite]
    logits = np.log(omega / (1.0 - omega))
    return np.asarray(logits, dtype=np.float64)


def _pooled_sharpe(
    mask: npt.NDArray[np.float64],
    strip_n: npt.NDArray[np.float64],
    strip_sum: npt.NDArray[np.float64],
    strip_sumsq: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return per-partition, per-config pooled Sharpe (mean / std, ddof=1).

    ``mask`` is ``(P, S)``; the strip statistics are ``(S,)`` and ``(S, N)``.
    Pooling strips by summing their sufficient statistics reproduces exactly
    the Sharpe of the concatenated block, including unequal strip lengths and
    the zero-variance -> 0 Sharpe convention.
    """
    counts = mask @ strip_n  # (P,)
    total = mask @ strip_sum  # (P, N)
    total_sq = mask @ strip_sumsq  # (P, N)
    mean = total / counts[:, None]
    # Sample variance (ddof=1): (sum_sq - n*mean^2) / (n - 1).
    var = (total_sq - counts[:, None] * mean**2) / (counts[:, None] - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(var > 0.0, mean / np.sqrt(var), 0.0)
    return np.asarray(sharpe, dtype=np.float64)
