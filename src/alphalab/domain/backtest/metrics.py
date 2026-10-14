"""Performance statistics for a return stream.

Kept separate from the engine so the same definitions apply to a strategy, to a
buy-and-hold benchmark, and to any future return stream — making comparisons
strictly homogeneous rather than each path computing its own Sharpe.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alphalab.domain.backtest.types import PerformanceMetrics

__all__ = ["TRADING_DAYS_PER_YEAR", "max_drawdown", "metrics_from_returns"]

#: Conventional count of US equity trading sessions in a calendar year.
TRADING_DAYS_PER_YEAR = 252


def metrics_from_returns(
    returns: Sequence[float],
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceMetrics:
    """Summarise a stream of simple per-period returns.

    An empty or constant stream yields a zero Sharpe rather than a division by
    zero: a strategy that never moves has no risk-adjusted return to report,
    and silently emitting ``inf`` or ``nan`` would poison downstream gates.
    """
    if not returns:
        return _empty_metrics()

    arr = np.asarray(returns, dtype=float)
    n = int(arr.size)
    growth = np.cumprod(1.0 + arr)
    total_return = float(growth[-1] - 1.0)

    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    mean = float(arr.mean())
    sharpe = 0.0 if std == 0.0 else mean / std * float(np.sqrt(periods_per_year))

    years = n / periods_per_year
    cagr = _cagr(total_return, years)
    vol = std * float(np.sqrt(periods_per_year))
    wins = int(np.count_nonzero(arr > 0.0))

    return PerformanceMetrics(
        sharpe_annualized=sharpe,
        total_return=total_return,
        cagr=cagr,
        volatility_annualized=vol,
        max_drawdown=max_drawdown(growth.tolist()),
        hit_rate=wins / n,
        n_periods=n,
    )


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Largest peak-to-trough decline, as a non-positive fraction."""
    if not equity_curve:
        return 0.0
    arr = np.asarray(equity_curve, dtype=float)
    running_peak = np.maximum.accumulate(arr)
    drawdowns = arr / running_peak - 1.0
    return float(min(0.0, drawdowns.min()))


def _cagr(total_return: float, years: float) -> float:
    """Compound annual growth rate, guarding total wipeout and short spans."""
    if years <= 0.0:
        return 0.0
    final = 1.0 + total_return
    if final <= 0.0:
        return -1.0
    return float(final ** (1.0 / years) - 1.0)


def _empty_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        sharpe_annualized=0.0,
        total_return=0.0,
        cagr=0.0,
        volatility_annualized=0.0,
        max_drawdown=0.0,
        hit_rate=0.0,
        n_periods=0,
    )
