"""Cross-sectional backtest engine with structural anti-look-ahead.

The engine is deliberately ignorant of *why* a ticker is attractive: it accepts
a :class:`~alphalab.domain.backtest.context.SignalFunction` and does nothing
but schedule, size, charge and account. One engine therefore serves the real
strategy, a synthetic positive control, and a buy-and-hold benchmark, so their
numbers are produced by identical machinery and remain comparable.

Timing contract (the part that matters)
---------------------------------------
For a rebalance date ``t`` the engine builds the signal context from bars dated
``<= t`` and fills the resulting trades at the first bar **strictly after**
``t``. The future is unreachable twice over: it is absent from the context, and
the fill price is one bar removed from the information set. See
:mod:`alphalab.domain.backtest.context`.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np

from alphalab.domain.backtest.context import SignalContext, SignalFunction
from alphalab.domain.backtest.costs_model import TurnoverCostModel
from alphalab.domain.backtest.metrics import TRADING_DAYS_PER_YEAR, metrics_from_returns
from alphalab.domain.backtest.types import BacktestResult, PriceBar, RebalanceRecord
from alphalab.domain.types import Ticker

__all__ = ["BacktestError", "run_backtest"]


class BacktestError(ValueError):
    """Raised when the backtest cannot be run as specified."""


class _Panel:
    """Forward-filled price matrix aligned to a single calendar.

    Valuation uses the forward-filled price so a temporary data gap does not
    silently zero a holding. Selection instead requires an *exact* bar on the
    execution date, so the engine never initiates a position on a day the name
    did not actually trade.

    Known limitation, stated rather than hidden: a name that stops trading
    permanently keeps its last price here, which flatters results. Only a
    universe carrying delisted names with real terminal values removes that
    bias.
    """

    __slots__ = ("calendar", "exact", "filled", "tickers")

    def __init__(self, prices: Mapping[Ticker, Sequence[PriceBar]]) -> None:
        self.tickers = tuple(sorted(prices))
        dates = {bar.on for bars in prices.values() for bar in bars}
        self.calendar = tuple(sorted(dates))
        index = {d: i for i, d in enumerate(self.calendar)}
        n = len(self.calendar)
        self.filled = {}
        self.exact = {}
        for ticker in self.tickers:
            row = np.full(n, np.nan)
            for bar in prices[ticker]:
                row[index[bar.on]] = bar.close
            self.exact[ticker] = row.copy()
            self.filled[ticker] = _forward_fill(row)

    def price(self, ticker: Ticker, j: int) -> float:
        """Forward-filled close, ``nan`` before the first observation."""
        return float(self.filled[ticker][j])

    def tradable(self, ticker: Ticker, j: int) -> bool:
        """Whether ``ticker`` actually printed a price at calendar index ``j``."""
        return not bool(np.isnan(self.exact[ticker][j]))


def run_backtest(
    *,
    prices: Mapping[Ticker, Sequence[PriceBar]],
    rebalance_dates: Sequence[date],
    signal: SignalFunction,
    cost_model: TurnoverCostModel,
    top_n: int = 10,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BacktestResult:
    """Simulate an equal-weight, periodically rebalanced long-only portfolio.

    Positions are sized to equal weights among the ``top_n`` highest-scoring
    tradable names and held until the next rebalance, with weights left to
    drift in between (a real portfolio does not rebalance continuously).

    Costs are charged on realised turnover at each rebalance, including the
    initial purchase, and are reported as a drag separate from gross returns so
    neither figure can be mistaken for the other.
    """
    if top_n <= 0:
        raise BacktestError("top_n must be positive")
    if not prices:
        raise BacktestError("prices is empty; nothing to simulate")

    sorted_prices = {t: _sorted_bars(b) for t, b in prices.items() if b}
    if not sorted_prices:
        raise BacktestError("every price series is empty")

    panel = _Panel(sorted_prices)
    schedule = _schedule(panel.calendar, rebalance_dates)
    if not schedule:
        raise BacktestError("no rebalance date precedes a tradable bar")

    return _simulate(
        panel=panel,
        sorted_prices=sorted_prices,
        schedule=schedule,
        signal=signal,
        cost_model=cost_model,
        top_n=top_n,
        periods_per_year=periods_per_year,
    )


def _simulate(
    *,
    panel: _Panel,
    sorted_prices: Mapping[Ticker, tuple[PriceBar, ...]],
    schedule: dict[int, date],
    signal: SignalFunction,
    cost_model: TurnoverCostModel,
    top_n: int,
    periods_per_year: int,
) -> BacktestResult:
    """Walk the calendar once, marking to market and rebalancing on schedule."""
    start = min(schedule)
    shares: dict[Ticker, float] = {}
    equity = 1.0
    gross_returns: list[float] = []
    net_returns: list[float] = []
    curve: list[float] = []
    out_dates: list[date] = []
    rebalances: list[RebalanceRecord] = []
    total_cost = 0.0

    for j in range(start, len(panel.calendar)):
        value = _value(shares, panel, j)
        gross = (value / equity - 1.0) if (shares and equity > 0.0) else 0.0
        equity = value if shares else equity

        cost_fraction = 0.0
        if j in schedule:
            equity, shares, record, cost_fraction = _rebalance(
                panel=panel,
                sorted_prices=sorted_prices,
                j=j,
                signal_date=schedule[j],
                signal=signal,
                cost_model=cost_model,
                top_n=top_n,
                equity=equity,
                shares=shares,
            )
            rebalances.append(record)
            total_cost += cost_fraction

        gross_returns.append(gross)
        net_returns.append((1.0 + gross) * (1.0 - cost_fraction) - 1.0)
        curve.append(equity)
        out_dates.append(panel.calendar[j])

    return BacktestResult(
        gross=metrics_from_returns(gross_returns, periods_per_year=periods_per_year),
        net=metrics_from_returns(net_returns, periods_per_year=periods_per_year),
        gross_returns=tuple(gross_returns),
        net_returns=tuple(net_returns),
        equity_curve=tuple(curve),
        dates=tuple(out_dates),
        rebalances=tuple(rebalances),
        total_cost_fraction=total_cost,
    )


def _rebalance(
    *,
    panel: _Panel,
    sorted_prices: Mapping[Ticker, tuple[PriceBar, ...]],
    j: int,
    signal_date: date,
    signal: SignalFunction,
    cost_model: TurnoverCostModel,
    top_n: int,
    equity: float,
    shares: Mapping[Ticker, float],
) -> tuple[float, dict[Ticker, float], RebalanceRecord, float]:
    """Score, select, charge costs and re-allocate at calendar index ``j``."""
    context = SignalContext(as_of=signal_date, prices=sorted_prices)
    scores = signal(context)
    selected = _select(scores, panel, j, top_n)

    target = {t: 1.0 / len(selected) for t in selected} if selected else {}
    current = _weights(shares, panel, j, equity)
    turnover = _turnover(current, target)
    cost_fraction = cost_model.cost_fraction(turnover=turnover)

    after = equity * (1.0 - cost_fraction)
    new_shares = {
        t: after * w / panel.price(t, j) for t, w in target.items() if panel.price(t, j) > 0.0
    }
    record = RebalanceRecord(
        signal_date=signal_date,
        execution_date=panel.calendar[j],
        selected=selected,
        turnover=turnover,
        cost_fraction=cost_fraction,
    )
    return after, new_shares, record, cost_fraction


def _select(
    scores: Mapping[Ticker, float], panel: _Panel, j: int, top_n: int
) -> tuple[Ticker, ...]:
    """Highest-scoring names that actually traded at index ``j``.

    Ties break on ticker symbol so the selection is deterministic.
    """
    eligible = [
        (score, ticker)
        for ticker, score in scores.items()
        if ticker in panel.exact and panel.tradable(ticker, j) and not _is_nan(score)
    ]
    eligible.sort(key=lambda pair: (-pair[0], pair[1]))
    return tuple(ticker for _, ticker in eligible[:top_n])


def _value(shares: Mapping[Ticker, float], panel: _Panel, j: int) -> float:
    """Mark holdings to market, ignoring names with no price yet."""
    total = 0.0
    for ticker, qty in shares.items():
        price = panel.price(ticker, j)
        if not _is_nan(price):
            total += qty * price
    return total


def _weights(
    shares: Mapping[Ticker, float], panel: _Panel, j: int, equity: float
) -> dict[Ticker, float]:
    if equity <= 0.0:
        return {}
    out: dict[Ticker, float] = {}
    for ticker, qty in shares.items():
        price = panel.price(ticker, j)
        if not _is_nan(price):
            out[ticker] = qty * price / equity
    return out


def _turnover(current: Mapping[Ticker, float], target: Mapping[Ticker, float]) -> float:
    """Total traded notional as a fraction of equity (buys plus sells)."""
    names = set(current) | set(target)
    return sum(abs(target.get(t, 0.0) - current.get(t, 0.0)) for t in names)


def _schedule(calendar: Sequence[date], rebalance_dates: Sequence[date]) -> dict[int, date]:
    """Map execution index to signal date, enforcing the one-bar delay.

    A rebalance decided on ``t`` executes at the first calendar bar strictly
    after ``t``. Dates with no subsequent bar are dropped: there is no way to
    act on them without inventing a fill.
    """
    out: dict[int, date] = {}
    for t in sorted(set(rebalance_dates)):
        idx = bisect_right(list(calendar), t)
        if idx < len(calendar) and idx not in out:
            out[idx] = t
    return out


def _sorted_bars(bars: Sequence[PriceBar]) -> tuple[PriceBar, ...]:
    return tuple(sorted(bars, key=lambda bar: bar.on))


def _forward_fill(row: np.ndarray) -> np.ndarray:
    """Carry the last observed value forward over ``nan`` gaps."""
    idx = np.where(~np.isnan(row), np.arange(row.size), 0)
    np.maximum.accumulate(idx, out=idx)
    filled: np.ndarray = row[idx]
    filled[np.isnan(row) & (np.arange(row.size) < _first_valid(row))] = np.nan
    return filled


def _first_valid(row: np.ndarray) -> int:
    valid = np.where(~np.isnan(row))[0]
    return int(valid[0]) if valid.size else row.size


def _is_nan(value: float) -> bool:
    return bool(np.isnan(value))
