"""Backtest engine tests, built around controls that can actually fail.

The two controls below are the reason this suite exists. Without a positive
control, a flat result cannot be distinguished from a broken engine; without a
negative control, an engine that manufactures profit from noise looks healthy.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta

import numpy as np
import pytest

from alphalab.domain.backtest import (
    BacktestError,
    PriceBar,
    ProportionalCostModel,
    SignalContext,
    buy_and_hold,
    momentum_signal,
    run_backtest,
)
from alphalab.domain.costs import CostSpec
from alphalab.domain.types import Ticker

_START = date(2010, 1, 4)


def _free() -> ProportionalCostModel:
    """A frictionless cost model, for isolating signal quality from costs."""
    return ProportionalCostModel(CostSpec(half_spread_bps=0.0, slippage_bps=0.0))


def _calendar(n: int) -> list[date]:
    return [_START + timedelta(days=i) for i in range(n)]


def _series(closes: list[float], days: list[date]) -> list[PriceBar]:
    return [PriceBar(on=d, close=c) for d, c in zip(days, closes, strict=True)]


def _drifting_universe(
    *, n_assets: int, n_days: int, seed: int, drift_spread: float
) -> dict[Ticker, list[PriceBar]]:
    """Assets whose expected returns differ persistently.

    Drift is constant per asset for the whole history, so it is *inferable from
    the past* — which is what makes this a fair positive control rather than a
    disguised oracle. A trailing-return signal should concentrate in the
    high-drift names using only information it legitimately has.
    """
    rng = np.random.default_rng(seed)
    days = _calendar(n_days)
    drifts = np.linspace(-drift_spread, drift_spread, n_assets)
    universe: dict[Ticker, list[PriceBar]] = {}
    for i in range(n_assets):
        shocks = rng.normal(drifts[i], 0.01, size=n_days - 1)
        closes = [100.0]
        for shock in shocks:
            closes.append(closes[-1] * (1.0 + float(shock)))
        universe[Ticker(f"A{i:02d}")] = _series(closes, days)
    return universe


def _monthly(days: list[date], step: int = 21) -> list[date]:
    return days[::step]


def test_positive_control_engine_detects_real_edge() -> None:
    """With persistent cross-sectional drift, momentum must earn a positive Sharpe.

    This proves the engine can *find* profitability when it genuinely exists.
    Any later negative result on real data is then a statement about the data,
    not about broken plumbing.
    """
    # Arrange
    universe = _drifting_universe(n_assets=20, n_days=1500, seed=7, drift_spread=0.0008)
    days = _calendar(1500)

    # Act
    result = run_backtest(
        prices=universe,
        rebalance_dates=_monthly(days),
        signal=momentum_signal(lookback=120),
        cost_model=_free(),
        top_n=5,
    )

    # Assert
    assert result.gross.sharpe_annualized > 1.0
    assert result.gross.total_return > 0.0


def test_negative_control_no_edge_yields_near_zero_sharpe() -> None:
    """On driftless noise the engine must not manufacture an edge."""
    # Arrange — identical distribution for every asset: nothing to select on.
    universe = _drifting_universe(n_assets=20, n_days=1500, seed=11, drift_spread=0.0)
    days = _calendar(1500)

    # Act
    result = run_backtest(
        prices=universe,
        rebalance_dates=_monthly(days),
        signal=momentum_signal(lookback=120),
        cost_model=_free(),
        top_n=5,
    )

    # Assert — a wide but finite band; noise should not look like skill.
    assert abs(result.gross.sharpe_annualized) < 1.0


def _oracle(universe: Mapping[Ticker, list[PriceBar]], *, offset: int) -> object:
    """A signal that cheats by reading raw prices smuggled in via closure.

    ``offset`` chooses which future bar it peeks at, relative to the decision
    date, letting a test target the bar the engine fills at versus the bar
    after it.
    """
    lookup = {t: {b.on: i for i, b in enumerate(bars)} for t, bars in universe.items()}

    def score(context: SignalContext) -> Mapping[Ticker, float]:
        out: dict[Ticker, float] = {}
        for ticker, bars in universe.items():
            i = lookup[ticker].get(context.as_of)
            if i is None or i + offset + 1 >= len(bars):
                continue
            start, end = bars[i + offset], bars[i + offset + 1]
            out[ticker] = end.close / start.close - 1.0
        return out

    return score


def test_execution_delay_blocks_same_bar_exploitation() -> None:
    """The one-bar fill delay is what neutralises an oracle, and it is measurable.

    Both runs use the same cheating signal on the same driftless data; only the
    bar it peeks at differs. The oracle aimed at the bar the engine fills at
    gains nothing, because that move is already spent when the position opens.
    Aimed one bar later — genuinely unspent — the same oracle lights up.

    Asserting the *gap* rather than two absolute thresholds makes this robust:
    it simultaneously shows the engine blocks same-bar leakage and that the
    test is sensitive enough to detect leakage when it is present. Were the
    engine to fill at the signal bar, both runs would score alike and the gap
    would collapse.
    """
    # Arrange — pure noise, so any Sharpe must come from the peeking.
    universe = _drifting_universe(n_assets=20, n_days=900, seed=3, drift_spread=0.0)
    days = _calendar(900)
    reb = _monthly(days)

    def sharpe_of(offset: int) -> float:
        result = run_backtest(
            prices=universe,
            rebalance_dates=reb,
            signal=_oracle(universe, offset=offset),  # type: ignore[arg-type]
            cost_model=_free(),
            top_n=5,
        )
        return result.gross.sharpe_annualized

    # Act
    blocked = sharpe_of(0)  # the bar consumed by the fill
    leaked = sharpe_of(1)  # the bar after the fill

    # Assert
    assert abs(blocked) < 1.0, f"same-bar oracle extracted an edge: {blocked:.4f}"
    assert leaked > 1.5, f"test is insensitive to real leakage: {leaked:.4f}"
    assert leaked - blocked > 1.0


def test_costs_materially_reduce_returns() -> None:
    """Turnover must be charged: net has to fall below gross."""
    # Arrange — frequent rebalancing plus a wide spread.
    universe = _drifting_universe(n_assets=20, n_days=900, seed=5, drift_spread=0.0005)
    days = _calendar(900)
    pricey = ProportionalCostModel(CostSpec(half_spread_bps=25.0, slippage_bps=25.0))

    # Act
    result = run_backtest(
        prices=universe,
        rebalance_dates=days[::5],
        signal=momentum_signal(lookback=60),
        cost_model=pricey,
        top_n=5,
    )

    # Assert
    assert result.net.total_return < result.gross.total_return
    assert result.total_cost_fraction > 0.0
    assert result.mean_turnover > 0.0


def test_determinism_same_inputs_same_result() -> None:
    """Identical inputs must produce bit-identical output."""
    # Arrange
    universe = _drifting_universe(n_assets=10, n_days=600, seed=2, drift_spread=0.0005)
    days = _calendar(600)
    args = {
        "prices": universe,
        "rebalance_dates": _monthly(days),
        "signal": momentum_signal(lookback=60),
        "cost_model": _free(),
        "top_n": 3,
    }

    # Act
    first = run_backtest(**args)  # type: ignore[arg-type]
    second = run_backtest(**args)  # type: ignore[arg-type]

    # Assert
    assert first.gross_returns == second.gross_returns
    assert first.net_returns == second.net_returns


def test_rebalance_executes_strictly_after_signal_date() -> None:
    """Every fill is dated after the decision that caused it."""
    # Arrange
    universe = _drifting_universe(n_assets=5, n_days=300, seed=1, drift_spread=0.0005)
    days = _calendar(300)

    # Act
    result = run_backtest(
        prices=universe,
        rebalance_dates=_monthly(days),
        signal=momentum_signal(lookback=30),
        cost_model=_free(),
        top_n=2,
    )

    # Assert
    assert result.rebalances
    for record in result.rebalances:
        assert record.execution_date > record.signal_date


def test_buy_and_hold_matches_price_appreciation() -> None:
    """The benchmark's total return equals the instrument's, free of costs."""
    # Arrange
    days = _calendar(400)
    closes = [100.0 * (1.01**i) for i in range(400)]
    bars = _series(closes, days)

    # Act
    result = buy_and_hold(ticker=Ticker("X"), bars=bars, cost_model=_free())

    # Assert — entry is the first bar, so appreciation runs from there.
    expected = closes[-1] / closes[1] - 1.0
    assert result.gross.total_return == pytest.approx(expected, rel=1e-9)
    assert result.gross.max_drawdown == 0.0


def test_empty_prices_is_rejected() -> None:
    """A backtest with no data must fail loudly, not return a flat zero."""
    with pytest.raises(BacktestError):
        run_backtest(
            prices={},
            rebalance_dates=[_START],
            signal=momentum_signal(lookback=5),
            cost_model=_free(),
        )


def test_rebalance_after_last_bar_is_dropped() -> None:
    """A decision with no subsequent bar cannot be filled, so it is discarded."""
    # Arrange
    days = _calendar(50)
    bars = _series([100.0 + i for i in range(50)], days)

    # Act
    result = run_backtest(
        prices={Ticker("X"): bars},
        rebalance_dates=[days[10], days[-1] + timedelta(days=30)],
        signal=momentum_signal(lookback=5),
        cost_model=_free(),
        top_n=1,
    )

    # Assert
    assert len(result.rebalances) == 1
    assert result.rebalances[0].signal_date == days[10]


def test_untradable_name_is_never_selected() -> None:
    """A name with no print on the fill date cannot be bought."""
    # Arrange — B is missing exactly at the execution bar.
    days = _calendar(60)
    a = _series([100.0 + i for i in range(60)], days)
    b_days = [d for i, d in enumerate(days) if i != 11]
    b = [PriceBar(on=d, close=500.0) for d in b_days]

    # Act — signal date index 10, so execution lands on index 11.
    result = run_backtest(
        prices={Ticker("A"): a, Ticker("B"): b},
        rebalance_dates=[days[10]],
        signal=lambda ctx: dict.fromkeys(ctx.tickers(), 1.0),
        cost_model=_free(),
        top_n=2,
    )

    # Assert
    assert result.rebalances[0].selected == (Ticker("A"),)
