"""Unit tests for the calibration metrics, with hand-verifiable values."""

from __future__ import annotations

import math

import pytest

from alphalab.domain.journal.calibration import (
    MIN_SAMPLE_SIZE,
    brier_decomposition,
    brier_score,
    calibration_report,
    expected_calibration_error,
    overconfidence,
    reliability_diagram,
)


class TestBrierScore:
    def test_perfect_predictor_is_zero(self) -> None:
        # Arrange: p=1 exactly when the event happens, p=0 when it does not.
        probs = [1.0, 0.0, 1.0, 0.0]
        outs = [1, 0, 1, 0]

        # Act
        score = brier_score(probs, outs)

        # Assert
        assert score == pytest.approx(0.0)

    def test_always_half_is_quarter(self) -> None:
        # Arrange: a 50/50 coin on any outcome mix.
        probs = [0.5] * 6
        outs = [1, 0, 1, 0, 1, 0]

        # Act / Assert: mean((0.5)^2) = 0.25.
        assert brier_score(probs, outs) == pytest.approx(0.25)

    def test_perfectly_anticorrelated_is_one(self) -> None:
        # Arrange: p=1 when the event does NOT happen and vice versa.
        probs = [1.0, 0.0, 1.0, 0.0]
        outs = [0, 1, 0, 1]

        # Act / Assert
        assert brier_score(probs, outs) == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            brier_score([0.5, 0.5], [1])

    def test_probability_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier_score([1.5], [1])

    def test_outcome_not_binary_raises(self) -> None:
        with pytest.raises(ValueError, match="0 or 1"):
            brier_score([0.5], [2])


class TestECE:
    def test_hand_computed_two_bins(self) -> None:
        # Arrange: construct bins whose accuracy and confidence are known.
        # Bin [0.0,0.1): one prediction p=0.0, outcome 0 -> acc 0, conf 0, gap 0.
        # Bin [0.9,1.0]: three predictions p=0.9, two hits -> acc 2/3, conf 0.9,
        #   gap |0.6667 - 0.9| = 0.2333.
        probs = [0.0, 0.9, 0.9, 0.9]
        outs = [0, 1, 1, 0]

        # Act
        ece = expected_calibration_error(probs, outs, n_bins=10)

        # Assert: weighted mean = (1*0 + 3*0.2333)/4 = 0.175.
        expected = (1 * 0.0 + 3 * abs((2 / 3) - 0.9)) / 4
        assert ece == pytest.approx(expected)

    def test_perfectly_calibrated_is_zero(self) -> None:
        # Arrange: in the p=0.5 bin, exactly half the outcomes are hits.
        probs = [0.5, 0.5, 0.5, 0.5]
        outs = [1, 0, 1, 0]

        # Act / Assert
        assert expected_calibration_error(probs, outs) == pytest.approx(0.0)


class TestReliabilityDiagram:
    def test_counts_sum_to_sample_size(self) -> None:
        # Arrange
        probs = [0.1, 0.2, 0.85, 0.9, 0.95]
        outs = [0, 0, 1, 1, 1]

        # Act
        bins = reliability_diagram(probs, outs, n_bins=10)

        # Assert
        assert sum(b.count for b in bins) == len(probs)

    def test_top_bin_is_closed_for_one(self) -> None:
        # Arrange: p=1.0 must land in the last bin, not overflow.
        probs = [1.0]
        outs = [1]

        # Act
        bins = reliability_diagram(probs, outs, n_bins=10)

        # Assert
        assert len(bins) == 1
        assert bins[0].upper == pytest.approx(1.0)
        assert bins[0].count == 1

    def test_gap_sign(self) -> None:
        # Arrange: confidence 0.9, accuracy 0.5 -> positive gap (overconfident).
        probs = [0.9, 0.9]
        outs = [1, 0]

        # Act
        bins = reliability_diagram(probs, outs, n_bins=10)

        # Assert
        assert bins[0].gap == pytest.approx(0.4)


class TestOverconfidence:
    def test_positive_when_overconfident(self) -> None:
        # Arrange: mean confidence 0.9, accuracy 0.25.
        probs = [0.9, 0.9, 0.9, 0.9]
        outs = [1, 0, 0, 0]

        # Act / Assert
        assert overconfidence(probs, outs) == pytest.approx(0.9 - 0.25)

    def test_negative_when_underconfident(self) -> None:
        # Arrange: mean confidence 0.2, accuracy 0.75.
        probs = [0.2, 0.2, 0.2, 0.2]
        outs = [1, 1, 1, 0]

        # Act / Assert
        assert overconfidence(probs, outs) == pytest.approx(0.2 - 0.75)


class TestBrierDecomposition:
    def test_identity_holds(self) -> None:
        # Arrange
        probs = [0.1, 0.4, 0.6, 0.9, 0.9, 0.2]
        outs = [0, 1, 0, 1, 1, 0]

        # Act
        decomp = brier_decomposition(probs, outs, n_bins=10)
        brier = brier_score(probs, outs)

        # Assert: brier = reliability - resolution + uncertainty.
        assert brier == pytest.approx(decomp.reliability - decomp.resolution + decomp.uncertainty)

    def test_uncertainty_is_base_rate_variance(self) -> None:
        # Arrange: base rate 0.5 -> uncertainty 0.25.
        probs = [0.3, 0.7]
        outs = [1, 0]

        # Act / Assert
        assert brier_decomposition(probs, outs).uncertainty == pytest.approx(0.25)


class TestCalibrationReport:
    def test_small_sample_is_not_conclusive(self) -> None:
        # Arrange: fewer than MIN_SAMPLE_SIZE predictions.
        probs = [0.6, 0.4]
        outs = [1, 0]

        # Act
        report = calibration_report(probs, outs)

        # Assert
        assert report.sample_size == 2
        assert report.conclusive is False

    def test_large_sample_is_conclusive(self) -> None:
        # Arrange
        n = MIN_SAMPLE_SIZE
        probs = [0.5] * n
        outs = [i % 2 for i in range(n)]

        # Act
        report = calibration_report(probs, outs)

        # Assert
        assert report.conclusive is True
        assert math.isclose(report.hit_rate, sum(outs) / n)


class TestWinklerAndOverconfidencePenalty:
    def test_winkler_score_penalizes_misses_outside_interval(self) -> None:
        from alphalab.domain.journal.calibration import winkler_score

        inside = winkler_score([90.0], [110.0], [100.0], alpha=0.20)
        outside = winkler_score([90.0], [110.0], [120.0], alpha=0.20)
        assert inside == pytest.approx(20.0)
        assert outside == pytest.approx(20.0 + (2.0 / 0.20) * 10.0)

    def test_overconfidence_penalty_shrinks_forecast_probability(self) -> None:
        from alphalab.domain.forecast.probability import BASE_RATE, forecast_probability

        raw = forecast_probability(
            implied_upside=0.30,
            relative_span=0.15,
            valuation_withheld=False,
        )
        shrunk = forecast_probability(
            implied_upside=0.30,
            relative_span=0.15,
            valuation_withheld=False,
            overconfidence_penalty=0.08,
        )
        assert shrunk < raw
        assert shrunk >= BASE_RATE
