"""Tests for the point-in-time liquidity screen.

The screen is not a nicety. Without it, a mechanically sampled universe
produced an equal-weight portfolio compounding at +430% a year — bid-ask
bounce on sub-dollar stocks masquerading as return. These tests pin the
behaviour that removes it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from alphalab.domain.backtest import (
    PriceBar,
    SignalContext,
    equal_weight_signal,
    liquidity_screened,
)
from alphalab.domain.types import Ticker

_START = date(2020, 1, 1)


def _bars(*, close: float, volume: float | None, n: int = 80) -> list[PriceBar]:
    return [PriceBar(on=_START + timedelta(days=i), close=close, volume=volume) for i in range(n)]


def _context(series: dict[str, list[PriceBar]]) -> SignalContext:
    prices = {Ticker(k): v for k, v in series.items()}
    return SignalContext(as_of=_START + timedelta(days=79), prices=prices)


def _screen(context: SignalContext, **kwargs: float) -> set[Ticker]:
    signal = liquidity_screened(equal_weight_signal(), **kwargs)  # type: ignore[arg-type]
    return set(signal(context))


def test_penny_stock_is_excluded() -> None:
    """A sub-threshold price is screened out however large its volume."""
    # Arrange — huge share volume, but the price is pennies.
    context = _context(
        {
            "PENNY": _bars(close=0.02, volume=500_000_000.0),
            "BLUE": _bars(close=100.0, volume=1_000_000.0),
        }
    )

    # Act
    kept = _screen(context, min_price=5.0, min_dollar_volume=1_000_000.0)

    # Assert
    assert kept == {Ticker("BLUE")}


def test_illiquid_name_is_excluded() -> None:
    """A high price does not compensate for thin traded value."""
    # Arrange — $500 a share but only 10 shares a day changes hands.
    context = _context(
        {
            "THIN": _bars(close=500.0, volume=10.0),
            "BLUE": _bars(close=100.0, volume=1_000_000.0),
        }
    )

    # Act
    kept = _screen(context, min_price=5.0, min_dollar_volume=1_000_000.0)

    # Assert
    assert kept == {Ticker("BLUE")}


def test_missing_volume_is_excluded_not_assumed_liquid() -> None:
    """Absent volume data is treated as unproven tradability.

    Assuming liquidity when the data is silent is how untradable names enter a
    backtest and inflate it.
    """
    # Arrange
    context = _context(
        {
            "NOVOL": _bars(close=100.0, volume=None),
            "BLUE": _bars(close=100.0, volume=1_000_000.0),
        }
    )

    # Act
    kept = _screen(context, min_price=5.0, min_dollar_volume=1_000_000.0)

    # Assert
    assert kept == {Ticker("BLUE")}


def test_screen_uses_only_trailing_window() -> None:
    """Liquidity is judged on recent bars, so a name can lose eligibility.

    The window reads from a context already truncated at the decision date, so
    the judgement is point-in-time by construction.
    """
    # Arrange — liquid early, then dried up over the last 60 bars.
    bars = _bars(close=100.0, volume=10_000_000.0, n=20)
    bars += [
        PriceBar(on=_START + timedelta(days=20 + i), close=100.0, volume=1.0) for i in range(60)
    ]
    context = _context({"FADED": bars})

    # Act
    kept = _screen(context, min_price=5.0, min_dollar_volume=1_000_000.0, window=60)

    # Assert
    assert kept == set()


def test_screen_preserves_inner_scores() -> None:
    """Surviving names keep the score the wrapped signal gave them."""
    # Arrange
    context = _context({"A": _bars(close=100.0, volume=1_000_000.0)})

    def constant(_: SignalContext) -> dict[Ticker, float]:
        return {Ticker("A"): 0.42}

    # Act
    screened = liquidity_screened(
        constant,  # type: ignore[arg-type]
        min_price=5.0,
        min_dollar_volume=1_000.0,
    )(context)

    # Assert
    assert screened == {Ticker("A"): 0.42}


def test_even_length_window_uses_median_of_two_middles() -> None:
    """Median over an even-sized window averages the two central values."""
    # Arrange — two bars: $1M and $9M traded, median $5M, exactly at threshold.
    bars = [
        PriceBar(on=_START, close=10.0, volume=100_000.0),
        PriceBar(on=_START + timedelta(days=1), close=10.0, volume=900_000.0),
    ]
    context = SignalContext(as_of=_START + timedelta(days=1), prices={Ticker("M"): bars})

    # Act
    kept = _screen(context, min_price=5.0, min_dollar_volume=5_000_000.0, window=2)

    # Assert
    assert kept == {Ticker("M")}


def test_invalid_thresholds_are_rejected() -> None:
    """Nonsensical screen parameters fail at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        liquidity_screened(equal_weight_signal(), min_price=-1.0)
    with pytest.raises(ValueError, match="window"):
        liquidity_screened(equal_weight_signal(), window=0)
