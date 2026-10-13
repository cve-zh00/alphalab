"""Unit tests for the DSR gate and return moments.

The numeric sanity targets come from Bailey & Lopez de Prado. With the
per-period Sharpe convention, N = 100 trials, a trial-Sharpe std of 0.35 and a
short sample (T = 10) of Gaussian returns (skew 0, non-central kurtosis 3):

* a raw Sharpe of 1.0 yields DSR ~ 0.61 (insignificant);
* a raw Sharpe of ~1.76 is needed to reach DSR = 0.95.

These reproduce the documented orders of magnitude (DSR ~ 0.6 at SR 1.0, and
SR ~ 1.8 for DSR 0.95). Tolerances are deliberately loose (abs=0.03 on DSR,
abs=0.1 on the SR root) because the targets are stated as approximate orders
of magnitude, not exact constants.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from alphalab.domain.stats._moments import (
    kurtosis_non_central,
    sharpe_ratio,
    skewness,
)
from alphalab.domain.stats.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
)

# Shared calibration for the sanity scenario.
_N_TRIALS = 100
_TRIAL_SHARPE_STD = 0.35
_T_OBS = 10.0
_GAUSSIAN_SKEW = 0.0
_GAUSSIAN_KURTOSIS = 3.0  # non-central kurtosis of a normal distribution


def _dsr(observed_sharpe: float) -> float:
    """Evaluate the DSR under the shared sanity calibration."""
    return deflated_sharpe_ratio(
        observed_sharpe,
        trial_sharpe_std=_TRIAL_SHARPE_STD,
        n_trials=_N_TRIALS,
        n_obs=_T_OBS,
        skew=_GAUSSIAN_SKEW,
        kurtosis=_GAUSSIAN_KURTOSIS,
    )


class TestSharpeRatio:
    def test_known_value(self) -> None:
        # Arrange: returns with mean 1 and sample std 1 -> Sharpe 1.
        returns = np.array([0.0, 2.0, 0.0, 2.0], dtype=np.float64)

        # Act
        result = sharpe_ratio(returns)

        # Assert
        expected = float(np.mean(returns)) / float(np.std(returns, ddof=1))
        assert result == pytest.approx(expected)

    def test_zero_variance_raises(self) -> None:
        # Arrange
        returns = np.ones(5, dtype=np.float64)

        # Act / Assert
        with pytest.raises(ValueError, match="zero standard deviation"):
            sharpe_ratio(returns)

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            sharpe_ratio(np.array([1.0]))


class TestKurtosisConvention:
    def test_non_central_kurtosis_of_normal_is_three(self) -> None:
        # Arrange: a large Gaussian sample.
        rng = np.random.default_rng(12345)
        returns = rng.standard_normal(200_000)

        # Act
        result = kurtosis_non_central(returns)

        # Assert: non-central (fisher=False) kurtosis of a normal is 3, not 0.
        assert result == pytest.approx(3.0, abs=0.05)

    def test_skewness_of_normal_is_zero(self) -> None:
        rng = np.random.default_rng(999)
        returns = rng.standard_normal(200_000)
        assert skewness(returns) == pytest.approx(0.0, abs=0.03)


class TestExpectedMaxSharpe:
    def test_grows_with_n_trials(self) -> None:
        # Arrange / Act
        values = [expected_max_sharpe(_TRIAL_SHARPE_STD, n) for n in (10, 100, 1000)]

        # Assert: SR0 is strictly increasing in N.
        assert values[0] < values[1] < values[2]

    def test_invalid_n_trials_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            expected_max_sharpe(0.35, 1)


class TestDeflatedSharpeSanity:
    def test_dsr_at_sharpe_one_is_around_point_six(self) -> None:
        # Act
        dsr = _dsr(1.0)

        # Assert: insignificant despite a raw Sharpe of 1.0.
        assert dsr == pytest.approx(0.6, abs=0.03)

    def test_sharpe_needed_for_0_95_is_around_1_8(self) -> None:
        # Act: invert DSR(sr) = 0.95 for the sanity calibration.
        sr_95 = brentq(lambda sr: _dsr(sr) - 0.95, 0.5, 5.0)

        # Assert
        assert sr_95 == pytest.approx(1.8, abs=0.1)

    def test_dsr_in_unit_interval(self) -> None:
        for sr in (-2.0, 0.0, 1.0, 3.0):
            dsr = _dsr(sr)
            assert 0.0 <= dsr <= 1.0

    def test_non_central_kurtosis_changes_dsr(self) -> None:
        # Using excess kurtosis (0) instead of non-central (3) must change the
        # DSR — this is the classic error the convention guards against.
        dsr_non_central = _dsr(1.5)
        dsr_excess = deflated_sharpe_ratio(
            1.5,
            trial_sharpe_std=_TRIAL_SHARPE_STD,
            n_trials=_N_TRIALS,
            n_obs=_T_OBS,
            skew=_GAUSSIAN_SKEW,
            kurtosis=0.0,
        )
        assert dsr_non_central != pytest.approx(dsr_excess)

    def test_invalid_n_obs_raises(self) -> None:
        with pytest.raises(ValueError, match="greater than 1"):
            deflated_sharpe_ratio(
                1.0,
                trial_sharpe_std=_TRIAL_SHARPE_STD,
                n_trials=_N_TRIALS,
                n_obs=1.0,
                skew=0.0,
                kurtosis=3.0,
            )
