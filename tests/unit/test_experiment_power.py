"""Unit tests for the approximate power analysis.

The standard-error formula and the MDE scaling are checked against values
computed by hand from the Lo (2002) asymptotic SE.
"""

from __future__ import annotations

import math

import pytest
from scipy import stats

from alphalab.domain.experiment.power import (
    MATERIAL_PERIOD_SHARPE_GAP,
    is_conclusive,
    minimum_detectable_sharpe_difference,
    sharpe_standard_error,
)


def test_sharpe_standard_error_matches_closed_form() -> None:
    # Arrange — SR=0.1, n=100 => sqrt((1 + 0.005)/100) = sqrt(0.01005).
    expected = math.sqrt((1.0 + 0.5 * 0.1**2) / 100)

    # Act
    se = sharpe_standard_error(0.1, 100)

    # Assert
    assert se == pytest.approx(expected)
    assert se == pytest.approx(math.sqrt(0.01005))


def test_sharpe_standard_error_shrinks_with_n() -> None:
    # Arrange / Act
    small = sharpe_standard_error(0.1, 50)
    large = sharpe_standard_error(0.1, 5000)

    # Assert — more data, tighter estimate.
    assert large < small


def test_sharpe_standard_error_requires_two_observations() -> None:
    with pytest.raises(ValueError, match="at least two"):
        sharpe_standard_error(0.1, 1)


def test_mde_matches_closed_form() -> None:
    # Arrange — reproduce the formula exactly.
    sr, n, conf, power = 0.05, 500, 1.0 - 0.05 / 3.0, 0.80
    alpha = 1.0 - conf
    se = math.sqrt((1.0 + 0.5 * sr**2) / n)
    expected = (stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power)) * math.sqrt(2.0) * se

    # Act
    mde = minimum_detectable_sharpe_difference(sr, n, confidence=conf, power=power)

    # Assert
    assert mde == pytest.approx(expected)


def test_mde_shrinks_with_sample_size() -> None:
    # Arrange / Act — a larger sample can detect a smaller effect.
    conf = 1.0 - 0.05 / 3.0
    small_n = minimum_detectable_sharpe_difference(0.05, 50, confidence=conf)
    large_n = minimum_detectable_sharpe_difference(0.05, 50_000, confidence=conf)

    # Assert
    assert large_n < small_n


def test_tiny_sample_is_not_conclusive() -> None:
    # Arrange — 10 observations cannot detect a material Sharpe gap.
    conf = 1.0 - 0.05 / 3.0

    # Act
    conclusive = is_conclusive(0.05, 10, confidence=conf)
    mde = minimum_detectable_sharpe_difference(0.05, 10, confidence=conf)

    # Assert
    assert conclusive is False
    assert mde > MATERIAL_PERIOD_SHARPE_GAP


def test_huge_sample_is_conclusive() -> None:
    # Arrange — a very large sample can detect a material gap.
    conf = 1.0 - 0.05 / 3.0

    # Act
    conclusive = is_conclusive(0.05, 200_000, confidence=conf)
    mde = minimum_detectable_sharpe_difference(0.05, 200_000, confidence=conf)

    # Assert
    assert conclusive is True
    assert mde <= MATERIAL_PERIOD_SHARPE_GAP


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5])
def test_bad_confidence_raises(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        minimum_detectable_sharpe_difference(0.05, 100, confidence=confidence)


@pytest.mark.parametrize("power", [0.0, 1.0, 1.5])
def test_bad_power_raises(power: float) -> None:
    with pytest.raises(ValueError, match="power"):
        minimum_detectable_sharpe_difference(0.05, 100, confidence=0.95, power=power)
