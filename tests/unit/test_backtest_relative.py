"""Tests for benchmark-relative performance."""

from __future__ import annotations

import pytest

from alphalab.domain.backtest import relative_performance


def test_positive_excess_gives_positive_information_ratio() -> None:
    """Beating the benchmark every period yields a positive ratio."""
    # Arrange
    strategy = [0.02, 0.01, 0.03, 0.015]
    benchmark = [0.01, 0.005, 0.02, 0.010]

    # Act
    result = relative_performance(strategy, benchmark)

    # Assert
    assert result.information_ratio > 0.0
    assert result.beats_benchmark
    assert result.mean_excess_annualized > 0.0


def test_underperformance_is_reported_as_negative() -> None:
    """Losing to the benchmark is reported, not clipped at zero."""
    # Arrange
    strategy = [0.00, -0.01, 0.01, -0.02]
    benchmark = [0.01, 0.005, 0.02, 0.010]

    # Act
    result = relative_performance(strategy, benchmark)

    # Assert
    assert result.information_ratio < 0.0
    assert not result.beats_benchmark


def test_streams_align_on_common_tail() -> None:
    """A shorter strategy history is compared only over shared periods.

    Crediting a strategy with benchmark periods it was not live for would
    fabricate track record, so the overlap is taken from the end.
    """
    # Arrange — benchmark has two extra leading periods.
    strategy = [0.01, 0.01, 0.01]
    benchmark = [0.50, -0.50, 0.00, 0.00, 0.00]

    # Act
    result = relative_performance(strategy, benchmark)

    # Assert
    assert result.n_periods == 3
    assert result.mean_excess_annualized == pytest.approx(0.01 * 252)


def test_identical_streams_have_zero_ratio() -> None:
    """No tracking error means no information ratio, not a division by zero."""
    # Arrange
    returns = [0.01, -0.02, 0.03]

    # Act
    result = relative_performance(returns, returns)

    # Assert
    assert result.information_ratio == 0.0
    assert result.tracking_error_annualized == 0.0
    assert not result.beats_benchmark


def test_too_short_overlap_is_rejected() -> None:
    """A single shared period cannot support a tracking error estimate."""
    with pytest.raises(ValueError, match="two overlapping periods"):
        relative_performance([0.01], [0.01])
