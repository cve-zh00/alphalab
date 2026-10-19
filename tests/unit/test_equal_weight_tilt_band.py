"""Equal-weight base with bounded tilt and rebalance tolerance band."""

from __future__ import annotations

from datetime import date

from alphalab.app.agentic_portfolio import MAX_TILT, conviction_volatility_targets
from alphalab.domain.execution import Holding, RiskLimits
from alphalab.domain.execution.planner import plan_trades
from alphalab.domain.types import Ticker

A, B, C = Ticker("AAA"), Ticker("BBB"), Ticker("CCC")


def test_tilt_is_bounded_around_equal_base() -> None:
    vols = {A: 0.10, B: 0.80, C: 0.25}
    w = conviction_volatility_targets([A, B, C], slots=14, volatilities=vols)
    base = 0.075
    for v in w.values():
        assert base * (1 - MAX_TILT) - 1e-9 <= v <= base * (1 + MAX_TILT) + 1e-9
    assert w[A] > w[B]


def _plan(current: float, target: float, band: float = 0.30):
    equity, price = 10_000.0, 100.0
    return plan_trades(
        as_of=date(2024, 6, 10),
        equity=equity,
        holdings={A: Holding(ticker=A, shares=equity * current / price, price=price)},
        targets={A: target},
        prices={A: price},
        limits=RiskLimits(max_order_weight=1.0, min_order_weight=0.0, rebalance_band=band),
    )


def test_within_band_no_trade() -> None:
    assert _plan(0.09, 0.07).intents == ()


def test_outside_band_trades() -> None:
    assert _plan(0.10, 0.06).intents


def test_exit_never_suppressed() -> None:
    assert _plan(0.05, 0.0).intents


def test_band_zero_disables() -> None:
    assert _plan(0.10, 0.08, band=0.0).intents
