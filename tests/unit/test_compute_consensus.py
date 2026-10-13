"""Unit tests for :mod:`alphalab.domain.compute.operators.consensus`.

Expected min/mid/max and relative span are computed by hand.
"""

from __future__ import annotations

import pytest

from alphalab.domain.compute.errors import InvalidValuationInputError
from alphalab.domain.compute.operators.consensus import consensus
from alphalab.domain.compute.provenance import leaf
from alphalab.domain.compute.thresholds import CONSENSUS_WIDE_SPAN_THRESHOLD


def _est(value: float, name: str) -> object:
    return leaf(value=value, name=name, unit="USD_per_share")


class TestConsensus:
    def test_hand_computed_point_and_range(self) -> None:
        # Arrange: values 80, 83.2, 90.
        # low=80, high=90, mid=median=83.2.
        # relative_span = (90-80)/83.2 = 10/83.2 = 0.120192...
        estimates = [
            _est(80.0, "ev_ebitda"),
            _est(83.2, "ddm"),
            _est(90.0, "pe"),
        ]

        # Act
        result = consensus(estimates=estimates, unit="USD_per_share")  # type: ignore[arg-type]

        # Assert
        assert result.low == pytest.approx(80.0)
        assert result.high == pytest.approx(90.0)
        assert result.mid == pytest.approx(83.2)
        assert result.point.value == pytest.approx(83.2)
        assert result.relative_span == pytest.approx(10.0 / 83.2)
        assert result.is_wide is False

    def test_all_estimates_are_parents_of_point(self) -> None:
        estimates = [_est(10.0, "a"), _est(12.0, "b")]
        result = consensus(estimates=estimates, unit="USD_per_share")  # type: ignore[arg-type]
        assert len(result.point.parents) == 2

    def test_wide_divergence_flagged(self) -> None:
        # Arrange: 50 and 100 => mid=75, span=50/75=0.6667 > 0.35.
        estimates = [_est(50.0, "a"), _est(100.0, "b")]

        # Act
        result = consensus(estimates=estimates, unit="USD_per_share")  # type: ignore[arg-type]

        # Assert
        assert result.relative_span > CONSENSUS_WIDE_SPAN_THRESHOLD
        assert result.is_wide is True

    def test_identical_estimates_zero_span(self) -> None:
        estimates = [_est(42.0, "a"), _est(42.0, "b")]
        result = consensus(estimates=estimates, unit="USD_per_share")  # type: ignore[arg-type]
        assert result.relative_span == pytest.approx(0.0)
        assert result.is_wide is False

    def test_empty_estimates_rejected(self) -> None:
        with pytest.raises(InvalidValuationInputError, match="non-empty"):
            consensus(estimates=[], unit="USD_per_share")

    def test_unit_mismatch_rejected(self) -> None:
        bad = [_est(10.0, "a"), leaf(value=5.0, name="b", unit="USD")]
        with pytest.raises(InvalidValuationInputError, match="unit"):
            consensus(estimates=bad, unit="USD_per_share")  # type: ignore[arg-type]
