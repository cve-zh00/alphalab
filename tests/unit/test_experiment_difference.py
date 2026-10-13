"""Unit tests for the paired-bootstrap Sharpe-difference test.

AAA structure throughout. The statistic and the Bonferroni bookkeeping are
checked against hand-computed values; the stochastic bootstrap is pinned with
an explicit seed so every assertion is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphalab.domain.experiment.difference import (
    N_PAIRWISE_COMPARISONS,
    PairedDifference,
    paired_sharpe_difference,
)


def test_bonferroni_confidence_is_corrected_by_family_size() -> None:
    # Arrange
    rng = np.random.default_rng(0)
    a = tuple(float(x) for x in rng.normal(0.001, 0.01, size=200))
    b = tuple(float(x) for x in rng.normal(0.001, 0.01, size=200))

    # Act
    diff = paired_sharpe_difference(
        a, b, name_a="B", name_b="A", rng=np.random.default_rng(1), alpha=0.05
    )

    # Assert — 1 - 0.05/3 = 0.98333...
    assert diff.n_comparisons == N_PAIRWISE_COMPARISONS
    assert diff.confidence == pytest.approx(1.0 - 0.05 / 3.0)


def test_delta_sign_matches_higher_sharpe_arm() -> None:
    # Arrange — arm A has a clearly higher mean, same vol structure.
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.01, size=300)
    high = tuple(float(x + 0.004) for x in noise)
    low = tuple(float(x) for x in noise)

    # Act — name_a is the challenger; put the HIGH arm as challenger.
    diff = paired_sharpe_difference(
        high, low, name_a="HIGH", name_b="LOW", rng=np.random.default_rng(2)
    )

    # Assert — challenger (HIGH) has higher Sharpe => positive delta.
    assert diff.delta > 0.0
    assert diff.winner == "HIGH"
    assert diff.excludes_zero


def test_identical_series_delta_is_zero_and_ci_brackets_zero() -> None:
    # Arrange — two identical series: no difference exists.
    rng = np.random.default_rng(5)
    series = tuple(float(x) for x in rng.normal(0.0005, 0.01, size=250))

    # Act
    diff = paired_sharpe_difference(
        series, series, name_a="X", name_b="Y", rng=np.random.default_rng(3)
    )

    # Assert — paired resampling cancels exactly: delta == 0, CI == [0, 0].
    assert diff.delta == pytest.approx(0.0, abs=1e-12)
    assert diff.ci_low == pytest.approx(0.0, abs=1e-12)
    assert diff.ci_high == pytest.approx(0.0, abs=1e-12)
    assert not diff.excludes_zero
    assert diff.winner is None


def test_mismatched_lengths_raise() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="equal-length"):
        paired_sharpe_difference(
            (0.1, 0.2, 0.3), (0.1, 0.2), name_a="A", name_b="B", rng=np.random.default_rng(0)
        )


def test_too_few_observations_raise() -> None:
    with pytest.raises(ValueError, match="at least two"):
        paired_sharpe_difference(
            (0.1,), (0.2,), name_a="A", name_b="B", rng=np.random.default_rng(0)
        )


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 2.0])
def test_out_of_range_alpha_raises(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        paired_sharpe_difference(
            (0.1, 0.2, 0.3),
            (0.1, 0.2, 0.3),
            name_a="A",
            name_b="B",
            rng=np.random.default_rng(0),
            alpha=alpha,
        )


def test_invalid_n_comparisons_raises() -> None:
    with pytest.raises(ValueError, match="n_comparisons"):
        paired_sharpe_difference(
            (0.1, 0.2, 0.3),
            (0.1, 0.2, 0.3),
            name_a="A",
            name_b="B",
            rng=np.random.default_rng(0),
            n_comparisons=0,
        )


def test_result_is_frozen() -> None:
    # Arrange
    diff = paired_sharpe_difference(
        (0.1, 0.2, 0.3),
        (0.1, 0.2, 0.3),
        name_a="A",
        name_b="B",
        rng=np.random.default_rng(0),
    )

    # Act / Assert — pydantic frozen model rejects mutation.
    with pytest.raises(Exception, match=r"frozen|Instance is frozen"):
        diff.delta = 1.0  # type: ignore[misc]
    assert isinstance(diff, PairedDifference)


def test_deterministic_with_fixed_seed() -> None:
    # Arrange
    rng = np.random.default_rng(11)
    a = tuple(float(x) for x in rng.normal(0.001, 0.01, size=120))
    b = tuple(float(x) for x in rng.normal(0.0005, 0.01, size=120))

    # Act — same seed twice.
    d1 = paired_sharpe_difference(a, b, name_a="A", name_b="B", rng=np.random.default_rng(99))
    d2 = paired_sharpe_difference(a, b, name_a="A", name_b="B", rng=np.random.default_rng(99))

    # Assert
    assert d1.ci_low == d2.ci_low
    assert d1.ci_high == d2.ci_high
    assert d1.delta == d2.delta
