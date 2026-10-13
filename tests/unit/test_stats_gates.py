"""Unit tests for PBO, effective sample size, bootstrap and synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from alphalab.domain.stats.autocorrelation import (
    autocorrelation,
    effective_sample_size,
)
from alphalab.domain.stats.bootstrap import bootstrap_sharpe_ci
from alphalab.domain.stats.pbo import probability_of_backtest_overfitting
from alphalab.domain.stats.synthetic import gbm_log_returns, gbm_return_matrix


class TestPBO:
    def test_genuine_stable_signal_gives_low_pbo(self) -> None:
        # Arrange: one configuration dominates in every period (real, stable
        # edge); the rest are pure noise.
        rng = np.random.default_rng(7)
        n_periods, n_configs = 240, 10
        matrix = rng.normal(0.0, 0.01, size=(n_periods, n_configs))
        matrix[:, 0] += 0.02  # column 0 has a persistent positive mean.

        # Act
        pbo = probability_of_backtest_overfitting(matrix, n_strips=8)

        # Assert: a genuine, stable winner should rarely be overfit.
        assert pbo < 0.5

    def test_is_winner_is_oos_loser_gives_high_pbo(self) -> None:
        # Arrange: build a matrix where each strip's best configuration is the
        # worst in every other strip. We make the per-strip mean ranking of the
        # configs a cyclic permutation of the strip index, so whichever config
        # wins in-sample is systematically poor out-of-sample regardless of how
        # the symmetric CSCV partition is drawn.
        n_strips = 8
        periods_per_strip = 30
        n_configs = n_strips
        n_periods = n_strips * periods_per_strip
        rng = np.random.default_rng(11)
        matrix = rng.normal(0.0, 1e-4, size=(n_periods, n_configs))
        base = np.linspace(0.0, 0.02, n_configs)
        for s in range(n_strips):
            start = s * periods_per_strip
            stop = start + periods_per_strip
            # Rotate which config is best in this strip.
            means = np.roll(base, s)
            matrix[start:stop, :] += means

        # Act
        pbo = probability_of_backtest_overfitting(matrix, n_strips=n_strips)

        # Assert: no config has a stable edge, so IS winners lose OOS -> PBO high.
        assert pbo >= 0.5

    def test_pbo_in_unit_interval(self) -> None:
        rng = np.random.default_rng(1)
        matrix = rng.normal(0.0, 0.01, size=(120, 6))
        pbo = probability_of_backtest_overfitting(matrix, n_strips=6)
        assert 0.0 <= pbo <= 1.0

    def test_odd_strips_raises(self) -> None:
        matrix = np.random.default_rng(0).normal(size=(100, 4))
        with pytest.raises(ValueError, match="even integer"):
            probability_of_backtest_overfitting(matrix, n_strips=7)

    def test_single_configuration_raises(self) -> None:
        matrix = np.random.default_rng(0).normal(size=(100, 1))
        with pytest.raises(ValueError, match="at least two configurations"):
            probability_of_backtest_overfitting(matrix, n_strips=4)


class TestEffectiveSampleSize:
    def test_positive_autocorrelation_shrinks_t_eff(self) -> None:
        # Arrange: an AR(1) series with strong positive autocorrelation.
        rng = np.random.default_rng(42)
        n = 500
        phi = 0.7
        noise = rng.standard_normal(n)
        series = np.empty(n, dtype=np.float64)
        series[0] = noise[0]
        for i in range(1, n):
            series[i] = phi * series[i - 1] + noise[i]

        # Act
        t_eff = effective_sample_size(series)

        # Assert: positive autocorrelation reduces independent information.
        assert t_eff < n

    def test_white_noise_t_eff_close_to_t(self) -> None:
        # Arrange: i.i.d. returns have no autocorrelation.
        rng = np.random.default_rng(2024)
        n = 2000
        series = rng.standard_normal(n)

        # Act
        t_eff = effective_sample_size(series)

        # Assert: T_eff should be close to the raw count (within 15%).
        assert t_eff == pytest.approx(float(n), rel=0.15)

    def test_autocorrelation_lag_zero_is_one(self) -> None:
        series = np.random.default_rng(0).standard_normal(100)
        assert autocorrelation(series, 0) == pytest.approx(1.0)

    def test_t_eff_never_exceeds_t(self) -> None:
        rng = np.random.default_rng(5)
        series = rng.standard_normal(300)
        assert effective_sample_size(series) <= 300.0


class TestBootstrap:
    def test_reproducible_with_same_seed(self) -> None:
        # Arrange
        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)
        returns = np.random.default_rng(0).normal(0.01, 0.02, size=250)

        # Act
        ci_a = bootstrap_sharpe_ci(returns, rng=rng_a, n_resamples=2000)
        ci_b = bootstrap_sharpe_ci(returns, rng=rng_b, n_resamples=2000)

        # Assert: identical seed -> identical interval.
        assert ci_a.lower == ci_b.lower
        assert ci_a.upper == ci_b.upper

    def test_different_seeds_differ(self) -> None:
        returns = np.random.default_rng(0).normal(0.01, 0.02, size=250)
        ci_a = bootstrap_sharpe_ci(returns, rng=np.random.default_rng(1), n_resamples=2000)
        ci_b = bootstrap_sharpe_ci(returns, rng=np.random.default_rng(2), n_resamples=2000)
        assert (ci_a.lower, ci_a.upper) != (ci_b.lower, ci_b.upper)

    def test_strong_edge_ci_excludes_zero(self) -> None:
        # Arrange: a clear positive edge.
        returns = np.random.default_rng(3).normal(0.02, 0.01, size=300)

        # Act
        ci = bootstrap_sharpe_ci(returns, rng=np.random.default_rng(9))

        # Assert
        assert ci.excludes_zero
        assert ci.lower > 0.0

    def test_no_edge_ci_includes_zero(self) -> None:
        # Arrange: driftless noise.
        returns = np.random.default_rng(4).normal(0.0, 0.01, size=300)

        # Act
        ci = bootstrap_sharpe_ci(returns, rng=np.random.default_rng(9))

        # Assert
        assert not ci.excludes_zero

    def test_percentile_method_available(self) -> None:
        returns = np.random.default_rng(3).normal(0.02, 0.01, size=300)
        ci = bootstrap_sharpe_ci(returns, rng=np.random.default_rng(9), method="percentile")
        assert ci.method == "percentile"
        assert ci.lower <= ci.point <= ci.upper


class TestSynthetic:
    def test_gbm_reproducible(self) -> None:
        a = gbm_log_returns(rng=np.random.default_rng(1), n_periods=100)
        b = gbm_log_returns(rng=np.random.default_rng(1), n_periods=100)
        assert np.array_equal(a, b)

    def test_gbm_no_drift_has_near_zero_mean(self) -> None:
        # Arrange / Act: driftless GBM over many periods.
        returns = gbm_log_returns(
            rng=np.random.default_rng(10), n_periods=100_000, mu=0.0, sigma=0.01
        )

        # Assert: no edge by construction -> mean near the small -sigma^2/2 drift.
        assert abs(float(returns.mean())) < 1e-3

    def test_gbm_matrix_shape(self) -> None:
        matrix = gbm_return_matrix(rng=np.random.default_rng(0), n_periods=120, n_configs=8)
        assert matrix.shape == (120, 8)
