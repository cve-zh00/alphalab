"""Buy-and-hold benchmark, computed by the same engine as any strategy.

Running the benchmark through :func:`~alphalab.domain.backtest.engine.run_backtest`
rather than a separate code path means its Sharpe, drawdown and cost treatment
come from identical machinery. A benchmark computed by different code is not a
fair comparison, it is a second opinion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from alphalab.domain.backtest.context import SignalContext
from alphalab.domain.backtest.costs_model import TurnoverCostModel
from alphalab.domain.backtest.engine import BacktestError, run_backtest
from alphalab.domain.backtest.metrics import TRADING_DAYS_PER_YEAR
from alphalab.domain.backtest.types import BacktestResult, PriceBar
from alphalab.domain.types import Ticker

__all__ = ["buy_and_hold"]


def buy_and_hold(
    *,
    ticker: Ticker,
    bars: Sequence[PriceBar],
    cost_model: TurnoverCostModel,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BacktestResult:
    """Hold ``ticker`` from its first tradable bar to the end of the series.

    A single rebalance at the first available date buys the name; no further
    rebalance means no further turnover, so the only cost charged is entry.
    """
    if not bars:
        raise BacktestError(f"no bars supplied for {ticker}")

    ordered = sorted(bars, key=lambda bar: bar.on)

    def always_hold(context: SignalContext) -> Mapping[Ticker, float]:
        del context  # the decision is unconditional
        return {ticker: 1.0}

    return run_backtest(
        prices={ticker: ordered},
        rebalance_dates=[ordered[0].on],
        signal=always_hold,
        cost_model=cost_model,
        top_n=1,
        periods_per_year=periods_per_year,
    )
