"""Benchmark-relative performance — the bias-neutral way to read a backtest.

Why this module is necessary
----------------------------
An absolute Sharpe computed on a hand-assembled universe is not interpretable.
Measured on this project's 48-name fixture, simply equal-weighting the universe
beat SPY by **+7.47% a year** with no signal whatsoever. That excess is not
skill; it is hindsight baked into which names were chosen in the first place —
a bias no statistical gate detects, because the returns are perfectly real.

Comparing a strategy against the *same universe equal-weighted* cancels that
shared composition. What survives is attributable to the signal: the timing and
concentration decisions, not the roster. Any absolute figure on a curated
universe should be read as an upper bound, and the relative figure as the
defensible estimate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.backtest.metrics import TRADING_DAYS_PER_YEAR

__all__ = ["RelativePerformance", "relative_performance"]


class RelativePerformance(BaseModel):
    """A strategy's performance net of its benchmark's."""

    model_config = ConfigDict(frozen=True)

    information_ratio: float
    mean_excess_annualized: float
    tracking_error_annualized: float = Field(ge=0.0)
    n_periods: int = Field(ge=0)

    @property
    def beats_benchmark(self) -> bool:
        """Whether mean excess return is positive.

        Deliberately not a significance test: use the project's statistical
        gates for that. This is a direction flag only.
        """
        return self.mean_excess_annualized > 0.0


def relative_performance(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> RelativePerformance:
    """Compare two aligned return streams period by period.

    The streams are aligned on their common tail, so a strategy that starts
    later than its benchmark is compared only over dates both cover rather than
    being credited with the benchmark's earlier history.

    Raises:
        ValueError: if the streams share fewer than two periods, where a
            tracking error cannot be estimated.
    """
    n = min(len(strategy_returns), len(benchmark_returns))
    if n < 2:
        raise ValueError("need at least two overlapping periods to compare")

    strategy = np.asarray(strategy_returns[-n:], dtype=float)
    benchmark = np.asarray(benchmark_returns[-n:], dtype=float)
    excess = strategy - benchmark

    tracking = float(excess.std(ddof=1))
    mean = float(excess.mean())
    scale = float(np.sqrt(periods_per_year))
    ratio = 0.0 if tracking == 0.0 else mean / tracking * scale

    return RelativePerformance(
        information_ratio=ratio,
        mean_excess_annualized=mean * periods_per_year,
        tracking_error_annualized=tracking * scale,
        n_periods=n,
    )
