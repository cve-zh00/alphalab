"""Property-based tests for the statistical gates (Hypothesis).

These encode the invariants that must hold for *every* valid input, not just
hand-picked cases:

* DSR always lies in ``[0, 1]``.
* DSR is monotonically decreasing in the number of trials ``N`` — the "false
  strategy theorem": the more configurations you tried, the less significant a
  fixed Sharpe becomes.
* ``SR0`` (expected max Sharpe) grows with ``N``.
* PBO always lies in ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from alphalab.domain.stats.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from alphalab.domain.stats.pbo import probability_of_backtest_overfitting
from alphalab.domain.stats.pbo_gate import evaluate_pbo_gate
from alphalab.domain.stats.synthetic import gbm_return_matrix

_finite = {"allow_nan": False, "allow_infinity": False}


@given(
    observed_sharpe=st.floats(min_value=-5.0, max_value=5.0, **_finite),
    trial_sharpe_std=st.floats(min_value=0.0, max_value=2.0, **_finite),
    n_trials=st.integers(min_value=2, max_value=100_000),
    n_obs=st.floats(min_value=2.0, max_value=5000.0, **_finite),
    skew=st.floats(min_value=-1.0, max_value=1.0, **_finite),
)
@settings(max_examples=300, deadline=None)
def test_dsr_in_unit_interval(
    observed_sharpe: float,
    trial_sharpe_std: float,
    n_trials: int,
    n_obs: float,
    skew: float,
) -> None:
    # Arrange: non-central kurtosis >= 1 keeps the denominator well-defined.
    kurtosis = 3.0
    # Act
    dsr = deflated_sharpe_ratio(
        observed_sharpe,
        trial_sharpe_std=trial_sharpe_std,
        n_trials=n_trials,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
    )
    # Assert
    assert 0.0 <= dsr <= 1.0


@given(
    observed_sharpe=st.floats(min_value=0.5, max_value=4.0, **_finite),
    trial_sharpe_std=st.floats(min_value=0.1, max_value=1.0, **_finite),
    n_low=st.integers(min_value=2, max_value=500),
    extra=st.integers(min_value=1, max_value=10_000),
    n_obs=st.floats(min_value=5.0, max_value=2000.0, **_finite),
)
@settings(max_examples=300, deadline=None)
def test_dsr_monotonically_decreasing_in_n_trials(
    observed_sharpe: float,
    trial_sharpe_std: float,
    n_low: int,
    extra: int,
    n_obs: float,
) -> None:
    # Arrange: a larger trial count for the same Sharpe.
    n_high = n_low + extra

    def dsr_for(n: int) -> float:
        return deflated_sharpe_ratio(
            observed_sharpe,
            trial_sharpe_std=trial_sharpe_std,
            n_trials=n,
            n_obs=n_obs,
            skew=0.0,
            kurtosis=3.0,
        )

    # Act / Assert: more trials never increases DSR (false-strategy theorem).
    assert dsr_for(n_high) <= dsr_for(n_low) + 1e-12


@given(
    trial_sharpe_std=st.floats(min_value=0.1, max_value=2.0, **_finite),
    n_low=st.integers(min_value=2, max_value=500),
    extra=st.integers(min_value=1, max_value=10_000),
)
@settings(max_examples=300, deadline=None)
def test_sr0_grows_with_n_trials(trial_sharpe_std: float, n_low: int, extra: int) -> None:
    # Act
    low = expected_max_sharpe(trial_sharpe_std, n_low)
    high = expected_max_sharpe(trial_sharpe_std, n_low + extra)
    # Assert
    assert high >= low - 1e-12


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=50, deadline=None)
def test_pbo_in_unit_interval(seed: int) -> None:
    # Arrange: a random (edgeless) configuration matrix.
    rng = np.random.default_rng(seed)
    matrix = gbm_return_matrix(rng=rng, n_periods=96, n_configs=6, sigma=0.02)
    # Act
    pbo = probability_of_backtest_overfitting(matrix, n_strips=6)
    # Assert
    assert 0.0 <= pbo <= 1.0


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=40, deadline=None)
def test_pbo_gate_point_and_ci_in_unit_interval(seed: int) -> None:
    # Arrange: a random edgeless configuration matrix.
    rng = np.random.default_rng(seed)
    matrix = gbm_return_matrix(rng=rng, n_periods=96, n_configs=6, sigma=0.02)
    # Act: run the hardened gate with cheap budgets.
    result = evaluate_pbo_gate(
        matrix,
        rng=np.random.default_rng(seed + 1),
        n_strips=6,
        n_bootstrap=200,
        n_null_sims=20,
    )
    # Assert: the point, both CI endpoints and the p-value stay in [0, 1].
    assert 0.0 <= result.pbo <= 1.0
    assert 0.0 <= result.ci_lower <= 1.0
    assert 0.0 <= result.ci_upper <= 1.0
    assert 0.0 <= result.p_value <= 1.0


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=40, deadline=None)
def test_pbo_gate_ci_contains_point_estimate(seed: int) -> None:
    # Arrange
    rng = np.random.default_rng(seed)
    matrix = gbm_return_matrix(rng=rng, n_periods=96, n_configs=6, sigma=0.02)
    # Act
    result = evaluate_pbo_gate(
        matrix,
        rng=np.random.default_rng(seed + 1),
        n_strips=6,
        n_bootstrap=200,
        n_null_sims=20,
    )
    # Assert: the percentile bootstrap CI brackets its own point estimate
    # (small tolerance for the discrete quantile grid).
    assert result.ci_lower <= result.pbo + 1e-9
    assert result.pbo <= result.ci_upper + 1e-9
