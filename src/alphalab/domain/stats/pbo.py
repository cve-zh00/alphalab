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
``S = 16`` that is 12 870. Rankings are computed vectorized over
configurations. Tests may lower ``S`` to keep the combinatorics cheap.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import numpy.typing as npt

__all__ = ["probability_of_backtest_overfitting"]


def _strip_bounds(n_periods: int, n_strips: int) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` index bounds for ``n_strips`` contiguous strips."""
    edges = np.linspace(0, n_periods, n_strips + 1, dtype=np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_strips)]


def _performance(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a per-configuration performance score for a period block.

    Uses the per-period Sharpe ratio (mean / std, ``ddof=1``) across the rows
    (periods) of ``block`` for every column (configuration). Degenerate
    zero-variance columns score ``0``.
    """
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(std > 0.0, mean / std, 0.0)
    return np.asarray(score, dtype=np.float64)


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
    n_periods, n_configs = matrix.shape
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
    if n_strips < 2 or n_strips % 2 != 0:
        raise ValueError("n_strips must be an even integer >= 2")
    if n_strips > n_periods:
        raise ValueError("n_strips cannot exceed the number of periods")

    bounds = _strip_bounds(n_periods, n_strips)
    strips = [matrix[start:stop] for start, stop in bounds]
    half = n_strips // 2
    all_indices = range(n_strips)

    below_median = 0
    total = 0
    for is_combo in combinations(all_indices, half):
        logit = _partition_logit(strips, set(is_combo), n_strips, n_configs)
        if logit is None:
            continue
        total += 1
        if logit < 0.0:
            below_median += 1
    if total == 0:
        raise ValueError("no valid partitions produced a ranking")
    return below_median / total


def _partition_logit(
    strips: list[npt.NDArray[np.float64]],
    is_indices: set[int],
    n_strips: int,
    n_configs: int,
) -> float | None:
    """Return the logit of the IS winner's OOS rank for one partition.

    Returns ``None`` when the partition cannot produce a finite logit (rank at
    the boundary), so the caller can skip it without biasing the fraction.
    """
    is_blocks = [strips[i] for i in range(n_strips) if i in is_indices]
    oos_blocks = [strips[i] for i in range(n_strips) if i not in is_indices]
    is_perf = _performance(np.concatenate(is_blocks, axis=0))
    oos_perf = _performance(np.concatenate(oos_blocks, axis=0))

    best_is = int(np.argmax(is_perf))
    # Relative rank in (0, 1): fraction of configs the IS winner beats OOS.
    rank = float((oos_perf < oos_perf[best_is]).sum()) / (n_configs - 1)
    # Map rank to the open interval to keep the logit finite.
    omega = (rank * n_configs + 0.5) / (n_configs + 1.0)
    if omega <= 0.0 or omega >= 1.0:
        return None
    return math.log(omega / (1.0 - omega))
