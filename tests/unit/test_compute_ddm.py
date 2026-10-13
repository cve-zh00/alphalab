"""Unit tests for :mod:`alphalab.domain.compute.operators.ddm`.

Expected values are computed by hand.
"""

from __future__ import annotations

import pytest

from alphalab.domain.compute.errors import (
    GrowthExceedsDiscountError,
    InvalidValuationInputError,
)
from alphalab.domain.compute.operators.ddm import ddm_value_per_share


class TestDdmValuePerShare:
    def test_hand_computed_value(self) -> None:
        # Arrange: D0=4, r=0.09, g=0.04.
        # D1 = 4 * 1.04 = 4.16 ; P = 4.16 / (0.09-0.04) = 4.16/0.05 = 83.2.
        node = ddm_value_per_share(
            dividend_per_share=4.0,
            required_return=0.09,
            dividend_growth=0.04,
        )

        # Assert
        assert node.value == pytest.approx(83.2)
        assert node.unit == "USD_per_share"
        assert node.operator == "ddm_value_per_share"

    def test_zero_dividend_yields_zero(self) -> None:
        node = ddm_value_per_share(
            dividend_per_share=0.0,
            required_return=0.08,
            dividend_growth=0.02,
        )
        assert node.value == pytest.approx(0.0)

    def test_growth_equals_return_raises(self) -> None:
        with pytest.raises(GrowthExceedsDiscountError):
            ddm_value_per_share(
                dividend_per_share=4.0,
                required_return=0.05,
                dividend_growth=0.05,
            )

    def test_growth_exceeds_return_raises(self) -> None:
        with pytest.raises(GrowthExceedsDiscountError):
            ddm_value_per_share(
                dividend_per_share=4.0,
                required_return=0.05,
                dividend_growth=0.09,
            )

    def test_negative_dividend_rejected(self) -> None:
        with pytest.raises(InvalidValuationInputError, match="dividend_per_share"):
            ddm_value_per_share(
                dividend_per_share=-1.0,
                required_return=0.09,
                dividend_growth=0.04,
            )
