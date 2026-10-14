"""Unit tests for the liquidity-only candidate screen (no network).

The screen is the single pre-filter the LLM universe passes through, so these
tests prove it keeps names on tradability alone: a name below the price floor or
the dollar-volume floor is dropped, a name that clears both survives, and the
returned metric is the median dollar volume (never a return) used only to recut
the cap.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from alphalab.app.agentic_universe import screen_liquidity_only
from alphalab.domain.backtest import PriceBar, SignalContext
from alphalab.domain.types import Ticker


def _bars(close: float, volume: float, *, n: int = 70) -> tuple[PriceBar, ...]:
    """A flat series of ``n`` daily bars at a fixed close and volume.

    Flat closes carry no return, which is the point: the screen must keep or
    drop a name on liquidity alone, with nothing a momentum signal could read.
    """
    start = date(2024, 1, 1)
    return tuple(
        PriceBar(on=start + timedelta(days=i), close=close, volume=volume) for i in range(n)
    )


def _context(prices: dict[Ticker, tuple[PriceBar, ...]]) -> SignalContext:
    return SignalContext(as_of=date(2024, 4, 1), prices=prices)


def test_keeps_liquid_name_and_reports_median_dollar_volume() -> None:
    # Arrange: $10 close, 1M shares/day -> $10M dollar volume, clears $5M floor.
    context = _context({Ticker("AAA"): _bars(close=10.0, volume=1_000_000.0)})

    # Act
    screened = screen_liquidity_only(context)

    # Assert: kept, and the metric is the median dollar volume (10 * 1M).
    assert set(screened) == {Ticker("AAA")}
    assert screened[Ticker("AAA")] == pytest.approx(10_000_000.0)


def test_drops_name_below_price_floor() -> None:
    # Arrange: $2 close is under the $5 min_price even though volume is huge.
    context = _context({Ticker("PENNY"): _bars(close=2.0, volume=100_000_000.0)})

    # Act / Assert
    assert screen_liquidity_only(context) == {}


def test_drops_name_below_dollar_volume_floor() -> None:
    # Arrange: $10 close but only 100 shares/day -> $1k, under the $5M floor.
    context = _context({Ticker("THIN"): _bars(close=10.0, volume=100.0)})

    # Act / Assert
    assert screen_liquidity_only(context) == {}


def test_drops_name_with_no_volume_data() -> None:
    # Arrange: volume absent -> untradable by assumption, never assumed liquid.
    start = date(2024, 1, 1)
    novol = tuple(PriceBar(on=start + timedelta(days=i), close=10.0) for i in range(70))
    context = _context({Ticker("NOVOL"): novol})

    # Act / Assert
    assert screen_liquidity_only(context) == {}


def test_screen_is_return_free_two_identical_liquidity_names_both_survive() -> None:
    # Arrange: one name rose, one fell, but both equally liquid. A return-based
    # filter would rank them; a liquidity-only screen keeps both.
    start = date(2024, 1, 1)
    rising = tuple(
        PriceBar(on=start + timedelta(days=i), close=6.0 + i * 0.05, volume=2_000_000.0)
        for i in range(70)
    )
    falling = tuple(
        PriceBar(on=start + timedelta(days=i), close=20.0 - i * 0.05, volume=2_000_000.0)
        for i in range(70)
    )
    context = _context({Ticker("UP"): rising, Ticker("DOWN"): falling})

    # Act
    screened = screen_liquidity_only(context)

    # Assert: both kept — the screen expresses no view on direction.
    assert set(screened) == {Ticker("UP"), Ticker("DOWN")}
