"""Unit tests for :mod:`alphalab.domain.costs.components`.

Each test follows the Arrange / Act / Assert structure with numbers that can
be checked by hand.
"""

from __future__ import annotations

import math

import pytest

from alphalab.domain.costs.components import (
    borrow_cost,
    commission_cost,
    market_impact_bps,
    market_impact_cost,
    slippage_cost,
    spread_cost,
)
from alphalab.domain.types import Side


class TestSpreadCost:
    def test_half_spread_on_notional(self) -> None:
        # Arrange: 10 bps half-spread on 100_000 notional.
        notional = 100_000.0
        half_spread_bps = 10.0

        # Act
        result = spread_cost(notional, half_spread_bps)

        # Assert: 100_000 * 10 * 1e-4 = 100.0
        assert result == pytest.approx(100.0)

    def test_zero_spread_is_free(self) -> None:
        # Arrange / Act / Assert
        assert spread_cost(50_000.0, 0.0) == 0.0

    def test_negative_notional_raises(self) -> None:
        # Arrange / Act / Assert
        with pytest.raises(ValueError, match="notional"):
            spread_cost(-1.0, 10.0)


class TestSlippageCost:
    def test_base_slippage(self) -> None:
        # Arrange: 5 bps slippage on 200_000 notional.
        # Act
        result = slippage_cost(200_000.0, 5.0)

        # Assert: 200_000 * 5 * 1e-4 = 100.0
        assert result == pytest.approx(100.0)

    def test_negative_bps_raises(self) -> None:
        with pytest.raises(ValueError, match="slippage_bps"):
            slippage_cost(1_000.0, -1.0)


class TestMarketImpactBps:
    def test_square_root_law_value(self) -> None:
        # Arrange: eta=1, sigma=100bps, Q/ADV = 0.01, alpha=0.5.
        # Act
        result = market_impact_bps(1_000.0, 100_000.0, 100.0, eta=1.0, alpha=0.5)

        # Assert: 1 * 100 * (0.01)**0.5 = 100 * 0.1 = 10.0
        assert result == pytest.approx(10.0)

    def test_linear_alpha_one(self) -> None:
        # Arrange: alpha=1 -> impact linear in participation.
        # Act
        result = market_impact_bps(2_000.0, 100_000.0, 50.0, eta=1.0, alpha=1.0)

        # Assert: 1 * 50 * (0.02)**1 = 1.0
        assert result == pytest.approx(1.0)

    def test_doubling_more_than_doubles_total_with_sqrt(self) -> None:
        # Arrange: with alpha=0.5, impact(2Q) = sqrt(2) * impact(Q) per bps,
        # but TOTAL cost ~ Q * impact grows faster than linear.
        single = market_impact_bps(1_000.0, 100_000.0, 100.0)
        double = market_impact_bps(2_000.0, 100_000.0, 100.0)

        # Act: total impact cost proportional to Q * impact_bps.
        total_single = 1_000.0 * single
        total_double = 2_000.0 * double

        # Assert: doubling Q more than doubles the total impact cost.
        assert total_double > 2.0 * total_single

    def test_zero_adv_raises(self) -> None:
        with pytest.raises(ValueError, match="adv"):
            market_impact_bps(100.0, 0.0, 50.0)

    def test_non_positive_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            market_impact_bps(100.0, 1_000.0, 50.0, alpha=0.0)


class TestMarketImpactCost:
    def test_currency_conversion(self) -> None:
        # Arrange: price=50, Q=1000 -> notional 50_000; impact 10 bps.
        # Act
        result = market_impact_cost(50.0, 1_000.0, 100_000.0, 100.0)

        # Assert: impact_bps = 10 (see above); 50_000 * 10 * 1e-4 = 50.0
        assert result == pytest.approx(50.0)


class TestCommissionCost:
    def test_per_share_above_minimum(self) -> None:
        # Arrange: 1000 shares * 0.01 = 10 > min 5.
        # Act
        result = commission_cost(1_000.0, 50_000.0, per_share=0.01, min_commission=5.0)

        # Assert
        assert result == pytest.approx(10.0)

    def test_minimum_floor_applies(self) -> None:
        # Arrange: 100 shares * 0.01 = 1 < min 5 -> floor to 5.
        # Act
        result = commission_cost(100.0, 5_000.0, per_share=0.01, min_commission=5.0)

        # Assert
        assert result == pytest.approx(5.0)

    def test_ad_valorem_component_adds(self) -> None:
        # Arrange: floor 5 + 2 bps of 50_000 = 5 + 10 = 15.
        # Act
        result = commission_cost(
            100.0, 50_000.0, per_share=0.0, min_commission=5.0, ad_valorem_bps=2.0
        )

        # Assert
        assert result == pytest.approx(15.0)


class TestBorrowCost:
    def test_long_pays_nothing(self) -> None:
        # Arrange / Act
        result = borrow_cost(Side.BUY, 100_000.0, 30.0, 0.05)

        # Assert
        assert result == 0.0

    def test_short_prorated_by_days(self) -> None:
        # Arrange: 100_000 * 5%/yr * (73/365) = 100_000 * 0.05 * 0.2 = 1000.
        # Act
        result = borrow_cost(Side.SELL, 100_000.0, 73.0, 0.05)

        # Assert
        assert result == pytest.approx(1_000.0)

    def test_zero_holding_is_free(self) -> None:
        # Arrange / Act / Assert
        assert borrow_cost(Side.SELL, 100_000.0, 0.0, 0.05) == 0.0

    def test_non_positive_days_per_year_raises(self) -> None:
        with pytest.raises(ValueError, match="days_per_year"):
            borrow_cost(Side.SELL, 100.0, 1.0, 0.05, days_per_year=0.0)


def test_impact_bps_is_finite_for_tiny_participation() -> None:
    # Arrange / Act
    result = market_impact_bps(1e-9, 1e9, 10.0)

    # Assert
    assert math.isfinite(result)
    assert result >= 0.0
