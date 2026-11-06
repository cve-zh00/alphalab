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

    def test_family_balanced_consensus_1_intrinsic_3_multiples(self) -> None:
        # Arrange: 1 intrinsic (DCF=100) + 3 multiples (PE=140, EV/EBITDA=150, PB=160).
        # Flat median would be 0.5*(140+150)=145 (100% multiples).
        # Family-balanced: intrinsic_mid=100, multiples_mid=150 => mid=125.0.
        from alphalab.domain.compute.provenance import combine

        def _op(val: float, op: str) -> object:
            return combine(
                value=val,
                operator=op,
                version="1.0.0",
                unit="USD_per_share",
                parents=(leaf(value=val, name=op, unit="USD_per_share"),),
            )

        estimates = [
            _op(100.0, "dcf_value_per_share"),
            _op(140.0, "pe_implied_value"),
            _op(150.0, "ev_ebitda_implied_value"),
            _op(160.0, "pb_implied_value"),
        ]

        # Act
        result = consensus(estimates=estimates, unit="USD_per_share")  # type: ignore[arg-type]

        # Assert
        assert result.low == pytest.approx(100.0)
        assert result.high == pytest.approx(160.0)
        assert result.mid == pytest.approx(125.0)
        assert result.point.value == pytest.approx(125.0)

    def test_family_balanced_consensus_2_intrinsic_3_multiples(self) -> None:
        # Arrange: 2 intrinsic (DCF=100, DDM=90 -> mid=95) + 3 multiples (140, 150, 160 -> mid=150).
        # Flat median of 5 would be 140.0 (100% multiples).
        # Family-balanced: 0.5 * (95.0 + 150.0) = 122.5.
        from alphalab.domain.compute.provenance import combine

        def _op(val: float, op: str) -> object:
            return combine(
                value=val,
                operator=op,
                version="1.0.0",
                unit="USD_per_share",
                parents=(leaf(value=val, name=op, unit="USD_per_share"),),
            )

        estimates = [
            _op(100.0, "dcf_value_per_share"),
            _op(90.0, "ddm_value_per_share"),
            _op(140.0, "pe_implied_value"),
            _op(150.0, "ev_ebitda_implied_value"),
            _op(160.0, "pb_implied_value"),
        ]

        # Act
        result = consensus(estimates=estimates, unit="USD_per_share")  # type: ignore[arg-type]

        # Assert
        assert result.mid == pytest.approx(122.5)
        assert result.point.value == pytest.approx(122.5)

    def test_family_balanced_consensus_single_family_uses_within_family_median(self) -> None:
        from alphalab.domain.compute.provenance import combine

        def _op(val: float, op: str) -> object:
            return combine(
                value=val,
                operator=op,
                version="1.0.0",
                unit="USD_per_share",
                parents=(leaf(value=val, name=op, unit="USD_per_share"),),
            )

        only_mult = [
            _op(100.0, "pe_implied_value"),
            _op(120.0, "ev_ebitda_implied_value"),
            _op(180.0, "pb_implied_value"),
        ]
        result = consensus(estimates=only_mult, unit="USD_per_share")  # type: ignore[arg-type]
        assert result.mid == pytest.approx(120.0)
