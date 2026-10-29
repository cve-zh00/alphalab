"""Unit tests for calibration metrics and the anytime-valid wealth e-process."""

from __future__ import annotations

import math

import pytest

from alphalab.domain.journal.calibration import (
    WealthEProcessResult,
    brier_score,
    compute_wealth_eprocess,
    expected_calibration_error,
)


class TestWealthEProcess:
    def test_wealth_remains_at_most_one_on_zero_or_negative_excess_returns(self) -> None:
        zero_res = compute_wealth_eprocess([0.0, 0.0, 0.0, 0.0, 0.0])
        assert isinstance(zero_res, WealthEProcessResult)
        assert zero_res.wealth == pytest.approx(1.0)
        assert zero_res.log_wealth == pytest.approx(0.0)
        assert zero_res.observations == 5
        assert zero_res.reject_null_alpha_05 is False
        assert zero_res.safe_lower_bound <= 0.0

        neg_res = compute_wealth_eprocess(
            [-0.01, -0.02, -0.015, -0.005, -0.01, -0.02],
            [0.005, 0.002, 0.001, 0.003, 0.004, 0.002],
        )
        assert neg_res.wealth <= 1.0
        assert neg_res.reject_null_alpha_05 is False
        assert neg_res.safe_lower_bound <= 0.0

    def test_wealth_crosses_twenty_and_rejects_null_on_persistent_positive_excess_returns(
        self,
    ) -> None:
        # Persistent positive excess return (+8% vs +1% benchmark = +7% excess per step)
        # With bet_fraction_cap = 0.5, each step multiplies wealth by ~1.035;
        # after 100 observations, (1.035)^99 > 30.0 >= 20.0.
        strategy = [0.08] * 100
        benchmark = [0.01] * 100

        res = compute_wealth_eprocess(strategy, benchmark, bet_fraction_cap=0.5)

        assert res.observations == 100
        assert res.wealth >= 20.0
        assert res.log_wealth == pytest.approx(math.log(res.wealth))
        assert res.reject_null_alpha_05 is True
        assert res.safe_lower_bound > 0.0

    def test_empty_returns_yields_unit_wealth(self) -> None:
        res = compute_wealth_eprocess([])
        assert res.wealth == pytest.approx(1.0)
        assert res.log_wealth == pytest.approx(0.0)
        assert res.observations == 0
        assert res.reject_null_alpha_05 is False

    def test_clips_extreme_negative_excess_returns_above_minus_one(self) -> None:
        res = compute_wealth_eprocess([0.05, 0.05, -1.50], [0.0, 0.0, 0.0], bet_fraction_cap=0.5)
        assert res.wealth > 0.0
        assert math.isfinite(res.log_wealth)

    def test_validates_inputs(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            compute_wealth_eprocess([0.01, 0.02], [0.01])
        with pytest.raises(ValueError, match="bet_fraction_cap"):
            compute_wealth_eprocess([0.01], bet_fraction_cap=0.0)


class TestExistingCalibrationSanity:
    def test_brier_and_ece_remain_intact(self) -> None:
        probs = [0.0, 1.0, 0.0, 1.0]
        outs = [0, 1, 0, 1]
        assert brier_score(probs, outs) == pytest.approx(0.0)
        assert expected_calibration_error(probs, outs) == pytest.approx(0.0)
