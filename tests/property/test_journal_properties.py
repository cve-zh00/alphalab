"""Property-based tests for the prediction journal (Hypothesis)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from alphalab.domain.journal import (
    Direction,
    Prediction,
    brier_score,
    calibration_report,
    expected_calibration_error,
    overconfidence,
    reliability_diagram,
    settle_prediction,
)

# Aligned (probability, outcome) sample pairs.
_pairs = st.lists(
    st.tuples(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        st.integers(min_value=0, max_value=1),
    ),
    min_size=1,
    max_size=200,
)


def _split(pairs: list[tuple[float, int]]) -> tuple[list[float], list[int]]:
    probs = [p for p, _ in pairs]
    outs = [o for _, o in pairs]
    return probs, outs


class TestCalibrationBounds:
    @given(pairs=_pairs)
    @settings(max_examples=300)
    def test_brier_in_unit_interval(self, pairs: list[tuple[float, int]]) -> None:
        # Act
        probs, outs = _split(pairs)
        score = brier_score(probs, outs)

        # Assert
        assert 0.0 <= score <= 1.0

    @given(pairs=_pairs)
    @settings(max_examples=300)
    def test_ece_in_unit_interval(self, pairs: list[tuple[float, int]]) -> None:
        # Act
        probs, outs = _split(pairs)
        ece = expected_calibration_error(probs, outs)

        # Assert
        assert 0.0 <= ece <= 1.0

    @given(pairs=_pairs)
    @settings(max_examples=300)
    def test_overconfidence_in_signed_unit_interval(self, pairs: list[tuple[float, int]]) -> None:
        # Act
        probs, outs = _split(pairs)
        oc = overconfidence(probs, outs)

        # Assert
        assert -1.0 <= oc <= 1.0


class TestReliabilityDiagramProperty:
    @given(pairs=_pairs, n_bins=st.integers(min_value=1, max_value=20))
    @settings(max_examples=200)
    def test_counts_sum_to_sample_size(self, pairs: list[tuple[float, int]], n_bins: int) -> None:
        # Act
        probs, outs = _split(pairs)
        bins = reliability_diagram(probs, outs, n_bins=n_bins)

        # Assert
        assert sum(b.count for b in bins) == len(pairs)


class TestDeterminism:
    @given(pairs=_pairs)
    @settings(max_examples=100)
    def test_report_is_deterministic(self, pairs: list[tuple[float, int]]) -> None:
        # Act
        probs, outs = _split(pairs)
        first = calibration_report(probs, outs)
        second = calibration_report(probs, outs)

        # Assert
        assert first == second


class TestImmutability:
    @given(prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=100)
    def test_prediction_is_frozen(self, prob: float) -> None:
        # Arrange
        pred = Prediction(
            recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
            ticker="AAPL",
            direction=Direction.UP,
            probability=prob,
            horizon=date(2026, 2, 1),
            entry_price=150.0,
            knowledge_date=date(2026, 1, 1),
        )

        # Act / Assert
        with pytest.raises(ValidationError):
            pred.probability = 0.0  # type: ignore[misc]

    @given(
        realized=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False),
        benchmark=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False),
        drag=st.floats(min_value=0.0, max_value=0.1, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_settlement_alpha_identities(
        self, realized: float, benchmark: float, drag: float
    ) -> None:
        # Arrange
        pred = Prediction(
            recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
            ticker="AAPL",
            direction=Direction.UP,
            probability=0.6,
            horizon=date(2026, 2, 1),
            entry_price=150.0,
            knowledge_date=date(2026, 1, 1),
        )

        # Act
        settlement = settle_prediction(
            pred,
            realized_return=realized,
            benchmark_return=benchmark,
            cost_drag=drag,
            settled_at=datetime(2026, 2, 1, tzinfo=UTC),
        )

        # Assert: net alpha never flatters gross; identities hold.
        assert settlement.alpha_gross == pytest.approx(realized - benchmark)
        assert settlement.alpha_net == pytest.approx(settlement.alpha_gross - drag)
        assert settlement.alpha_net <= settlement.alpha_gross + 1e-12
