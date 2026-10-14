"""Tests for price-based signal constructors."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from alphalab.domain.backtest import (
    PriceBar,
    SignalContext,
    equal_weight_signal,
    momentum_signal,
    rank_ensemble_signal,
)
from alphalab.domain.types import Ticker

_START = date(2020, 1, 1)


def _bars(closes: list[float]) -> list[PriceBar]:
    return [PriceBar(on=_START + timedelta(days=i), close=c) for i, c in enumerate(closes)]


def _context(series: dict[str, list[float]], *, as_of_index: int) -> SignalContext:
    prices = {Ticker(k): _bars(v) for k, v in series.items()}
    return SignalContext(as_of=_START + timedelta(days=as_of_index), prices=prices)


def test_momentum_scores_trailing_return() -> None:
    """The score is the return over the lookback window, not the full history."""
    # Arrange — a steady 1%/bar riser; the last 10 bars gain 1.01**10 - 1.
    context = _context({"UP": [100.0 * 1.01**i for i in range(30)]}, as_of_index=29)

    # Act
    scores = momentum_signal(lookback=10)(context)

    # Assert
    assert scores[Ticker("UP")] == pytest.approx(1.01**10 - 1.0, rel=1e-9)


def test_momentum_omits_names_with_short_history() -> None:
    """A name without a full window is skipped, not scored on a partial one."""
    # Arrange
    context = _context({"NEW": [100.0, 101.0, 102.0]}, as_of_index=2)

    # Act
    scores = momentum_signal(lookback=50)(context)

    # Assert
    assert Ticker("NEW") not in scores


def test_momentum_skip_excludes_recent_bars() -> None:
    """``skip`` shifts the window back, ignoring the latest bars."""
    # Arrange — flat for 10 bars, then a spike that skip should exclude.
    closes = [100.0] * 20 + [200.0]
    context = _context({"X": closes}, as_of_index=20)

    # Act
    skipped = momentum_signal(lookback=5, skip=1)(context)

    # Assert — measured entirely inside the flat stretch.
    assert skipped[Ticker("X")] == pytest.approx(0.0)


def test_momentum_rejects_invalid_parameters() -> None:
    """Nonsensical windows fail at construction, not silently at scoring."""
    with pytest.raises(ValueError, match="lookback"):
        momentum_signal(lookback=0)
    with pytest.raises(ValueError, match="skip"):
        momentum_signal(lookback=5, skip=-1)


def test_equal_weight_scores_everything_the_same() -> None:
    """Every visible name receives an identical score."""
    # Arrange
    context = _context({"A": [1.0, 2.0], "B": [3.0, 4.0]}, as_of_index=1)

    # Act
    scores = equal_weight_signal()(context)

    # Assert
    assert set(scores) == {Ticker("A"), Ticker("B")}
    assert len(set(scores.values())) == 1


def test_rank_ensemble_rewards_consistency_across_lookbacks() -> None:
    """A name strong over every horizon outranks one strong over only a short one.

    This is the behaviour that makes the ensemble robust: it will not crown a
    name on the strength of a single window. SPIKY below wins the 10-bar
    window outright, yet loses the blend because it is down over the two
    longer ones.
    """
    # Arrange — STEADY compounds gently; SPIKY slides for 70 bars then rallies.
    steady = [100.0 * 1.004**i for i in range(80)]
    spiky = [200.0 * 0.99**i for i in range(70)]
    spiky += [spiky[-1] * 1.02**i for i in range(1, 11)]
    context = _context({"STEADY": steady, "SPIKY": spiky}, as_of_index=79)

    # Act
    short_only = momentum_signal(lookback=10)(context)
    blended = rank_ensemble_signal(lookbacks=(10, 40, 70))(context)

    # Assert — the short window favours SPIKY, the blend does not.
    assert short_only[Ticker("SPIKY")] > short_only[Ticker("STEADY")]
    assert blended[Ticker("STEADY")] > blended[Ticker("SPIKY")]


def test_rank_ensemble_matches_single_lookback_when_given_one() -> None:
    """With one lookback the ensemble degenerates to a pure rank of it."""
    # Arrange
    series = {
        "A": [100.0 * 1.01**i for i in range(40)],
        "B": [100.0 * 1.002**i for i in range(40)],
        "C": [100.0 * 0.999**i for i in range(40)],
    }
    context = _context(series, as_of_index=39)

    # Act
    ranks = rank_ensemble_signal(lookbacks=(20,))(context)

    # Assert — top name scores 1.0, bottom 0.0.
    assert ranks[Ticker("A")] == pytest.approx(1.0)
    assert ranks[Ticker("C")] == pytest.approx(0.0)
    assert 0.0 < ranks[Ticker("B")] < 1.0


def test_rank_ensemble_requires_lookbacks() -> None:
    """An empty ensemble is a configuration error."""
    with pytest.raises(ValueError, match="lookbacks"):
        rank_ensemble_signal(lookbacks=())


def test_rank_ensemble_handles_single_eligible_name() -> None:
    """One scorable name is ranked top rather than crashing on a zero span."""
    # Arrange — only A has enough history for the lookback.
    series = {"A": [100.0 + i for i in range(40)], "B": [100.0, 101.0]}
    context = _context(series, as_of_index=39)

    # Act
    ranks = rank_ensemble_signal(lookbacks=(20,))(context)

    # Assert
    assert ranks == {Ticker("A"): 1.0}
