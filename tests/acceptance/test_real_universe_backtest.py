"""Acceptance tests over the real 48-name price fixture.

These lock in findings that were measured, not assumed, so a future change that
quietly breaks them fails the build:

1. The engine produces positive simulated profitability on real prices.
2. Trading costs are charged and reduce returns.
3. The hand-assembled universe carries a large hindsight bias, which is why
   benchmark-relative figures — not absolute ones — are the headline.
4. Choosing the best-performing configuration in-sample does not survive
   out-of-sample, which is why the shipped strategy is an ensemble that makes
   no such choice.

Point 4 is the uncomfortable one and the reason it is pinned here: it was
discovered only by holding out a window, and it contradicts the intuitive
approach of "tune, then deploy".
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from alphalab.domain.backtest import (
    PriceBar,
    ProportionalCostModel,
    buy_and_hold,
    equal_weight_signal,
    momentum_signal,
    rank_ensemble_signal,
    relative_performance,
    run_backtest,
)
from alphalab.domain.backtest.types import BacktestResult
from alphalab.domain.costs import CostSpec
from alphalab.domain.types import Ticker

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_STOCKS = _FIXTURES / "stocks"
_SPY = _FIXTURES / "SPY.csv"

#: Lookbacks blended by the shipped strategy. Fixed up front, never tuned.
_ENSEMBLE_LOOKBACKS = (60, 120, 250)

#: Realistic round-trip friction for liquid US large caps: 5 bps per unit
#: of traded notional, charged on both sides of every rebalance.
_COSTS = CostSpec(half_spread_bps=2.0, slippage_bps=3.0)

_IS = (date(2010, 1, 1), date(2017, 12, 31))
_OOS = (date(2018, 1, 1), date(2024, 12, 31))


def _load(path: Path, lo: date, hi: date) -> list[PriceBar]:
    bars: list[PriceBar] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                on = date.fromisoformat(row["date"][:10])
                close = float(row["close"])
            except (ValueError, KeyError):
                continue
            if close > 0.0 and lo <= on <= hi:
                bars.append(PriceBar(on=on, close=close))
    return bars


def _universe(lo: date, hi: date) -> dict[Ticker, list[PriceBar]]:
    out: dict[Ticker, list[PriceBar]] = {}
    for path in sorted(_STOCKS.glob("*.csv")):
        bars = _load(path, lo, hi)
        if len(bars) > 300:
            out[Ticker(path.stem)] = bars
    return out


def _quarterly(universe: dict[Ticker, list[PriceBar]]) -> list[date]:
    """First trading day seen in each calendar quarter."""
    calendar = sorted({bar.on for bars in universe.values() for bar in bars})
    picked: list[date] = []
    seen: set[tuple[int, int]] = set()
    for day in calendar:
        key = (day.year, (day.month - 1) // 3)
        if key not in seen:
            seen.add(key)
            picked.append(day)
    return picked


def _ensemble(lo: date, hi: date, *, top_n: int = 5) -> BacktestResult:
    universe = _universe(lo, hi)
    return run_backtest(
        prices=universe,
        rebalance_dates=_quarterly(universe),
        signal=rank_ensemble_signal(lookbacks=_ENSEMBLE_LOOKBACKS),
        cost_model=ProportionalCostModel(_COSTS),
        top_n=top_n,
    )


def _equal_weight(lo: date, hi: date) -> BacktestResult:
    universe = _universe(lo, hi)
    return run_backtest(
        prices=universe,
        rebalance_dates=_quarterly(universe),
        signal=equal_weight_signal(),
        cost_model=ProportionalCostModel(_COSTS),
        top_n=len(universe),
    )


@pytest.fixture(scope="module")
def fixture_universe() -> dict[Ticker, list[PriceBar]]:
    return _universe(*_IS)


def test_fixture_universe_is_present(fixture_universe: dict[Ticker, list[PriceBar]]) -> None:
    """Guard against silently running these assertions on no data."""
    assert len(fixture_universe) >= 40
    assert _SPY.exists()


def test_strategy_is_profitable_in_and_out_of_sample() -> None:
    """The ensemble earns a positive net Sharpe in both windows.

    Stability across the hold-out matters more than the level: a strategy that
    only works in the window used to build it has demonstrated nothing.
    """
    # Act
    in_sample = _ensemble(*_IS)
    out_of_sample = _ensemble(*_OOS)

    # Assert
    assert in_sample.net.sharpe_annualized > 0.8
    assert out_of_sample.net.sharpe_annualized > 0.8
    assert out_of_sample.net.cagr > 0.0


def test_costs_are_charged_and_reduce_returns() -> None:
    """Net must trail gross, and turnover must be non-trivial."""
    # Act
    result = _ensemble(*_OOS)

    # Assert
    assert result.net.total_return < result.gross.total_return
    assert result.total_cost_fraction > 0.0
    assert result.mean_turnover > 0.5


def test_universe_hindsight_bias_is_large_and_measured() -> None:
    """Equal-weighting the fixture beats SPY with no signal at all.

    The 48 names were chosen from memory in 2025, so their collective return
    embeds knowledge of who did well. This test pins the size of that bias as
    a standing warning: absolute returns on this universe are an upper bound,
    not an estimate.
    """
    # Arrange
    lo, hi = date(2010, 1, 1), date(2024, 12, 31)

    # Act
    naive = _equal_weight(lo, hi)
    spy = buy_and_hold(
        ticker=Ticker("SPY"),
        bars=_load(_SPY, lo, hi),
        cost_model=ProportionalCostModel(_COSTS),
    )

    # Assert — a signal-free portfolio should not beat the index by this much.
    assert naive.net.cagr - spy.net.cagr > 0.05


def test_strategy_adds_value_over_its_own_universe() -> None:
    """Excess return over the same universe equal-weighted stays positive.

    This is the bias-neutral read: both sides hold the same names, so shared
    hindsight cancels and what remains is attributable to the signal.
    """
    # Act
    strategy = _ensemble(*_OOS)
    naive = _equal_weight(*_OOS)
    relative = relative_performance(strategy.net_returns, naive.net_returns)

    # Assert
    assert relative.information_ratio > 0.3
    assert relative.beats_benchmark


def test_best_in_sample_configuration_does_not_persist() -> None:
    """Tuning the lookback in-sample fails out-of-sample.

    Measured outcome: the top in-sample configuration ranked *last* out of
    sample. This is the empirical justification for shipping an ensemble
    instead of a tuned parameter, and it corroborates the hardened PBO gate's
    rejection of the tuned variant.
    """
    # Arrange
    configs = [(lb, n) for lb in (60, 120, 250) for n in (5, 10)]

    def sharpe(window: tuple[date, date], lookback: int, top_n: int) -> float:
        universe = _universe(*window)
        result = run_backtest(
            prices=universe,
            rebalance_dates=_quarterly(universe),
            signal=momentum_signal(lookback=lookback),
            cost_model=ProportionalCostModel(_COSTS),
            top_n=top_n,
        )
        return result.net.sharpe_annualized

    # Act
    in_sample = {c: sharpe(_IS, *c) for c in configs}
    out_of_sample = {c: sharpe(_OOS, *c) for c in configs}
    best = max(configs, key=lambda c: in_sample[c])
    rank_oos = sorted(configs, key=lambda c: -out_of_sample[c]).index(best) + 1

    # Assert — the in-sample winner lands in the bottom half out of sample.
    assert rank_oos > len(configs) / 2, (
        f"best IS config {best} ranked {rank_oos}/{len(configs)} OOS; "
        "if tuning now persists, revisit the ensemble rationale"
    )


def test_ensemble_beats_tuned_choice_out_of_sample() -> None:
    """Making no choice beats making the in-sample-optimal one, out of sample."""
    # Arrange
    configs = [(lb, n) for lb in (60, 120, 250) for n in (5, 10)]
    is_universe = _universe(*_IS)
    is_reb = _quarterly(is_universe)

    def is_sharpe(lookback: int, top_n: int) -> float:
        return run_backtest(
            prices=is_universe,
            rebalance_dates=is_reb,
            signal=momentum_signal(lookback=lookback),
            cost_model=ProportionalCostModel(_COSTS),
            top_n=top_n,
        ).net.sharpe_annualized

    best_lb, best_n = max(configs, key=lambda c: is_sharpe(*c))

    # Act — both evaluated only on the untouched window.
    oos_universe = _universe(*_OOS)
    oos_reb = _quarterly(oos_universe)
    tuned = run_backtest(
        prices=oos_universe,
        rebalance_dates=oos_reb,
        signal=momentum_signal(lookback=best_lb),
        cost_model=ProportionalCostModel(_COSTS),
        top_n=best_n,
    )
    ensemble = _ensemble(*_OOS, top_n=best_n)

    # Assert
    assert ensemble.net.sharpe_annualized > tuned.net.sharpe_annualized
