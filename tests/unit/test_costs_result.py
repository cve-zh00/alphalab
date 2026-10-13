"""Unit tests for :mod:`alphalab.domain.costs.result`.

Covers the gross/net invariant, the per-trade breakdown composition, the
Sharpe gross-vs-net calculation and immutability of the frozen structures.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alphalab.domain.costs.result import (
    CostBreakdown,
    CostSpec,
    GrossNetResult,
    cost_breakdown_for_trade,
    sharpe_gross_net,
)
from alphalab.domain.types import Side


class TestCostBreakdown:
    def test_total_sums_components(self) -> None:
        # Arrange
        bd = CostBreakdown(spread=1.0, slippage=2.0, market_impact=3.0, commission=4.0, borrow=5.0)

        # Act / Assert
        assert bd.total == pytest.approx(15.0)

    def test_is_immutable(self) -> None:
        # Arrange
        bd = CostBreakdown(spread=1.0, slippage=0.0, market_impact=0.0, commission=0.0, borrow=0.0)

        # Act / Assert: mutation must raise.
        with pytest.raises(ValidationError):
            bd.spread = 9.0  # type: ignore[misc]

    def test_negative_component_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CostBreakdown(spread=-1.0, slippage=0.0, market_impact=0.0, commission=0.0, borrow=0.0)


class TestGrossNetResult:
    def test_carries_gross_net_and_costs(self) -> None:
        # Arrange
        bd = CostBreakdown(spread=0.0, slippage=10.0, market_impact=0.0, commission=0.0, borrow=0.0)

        # Act
        res = GrossNetResult(gross=100.0, net=90.0, costs=bd)

        # Assert: gross is only ever available next to net + breakdown.
        assert res.gross == 100.0
        assert res.net == 90.0
        assert res.total_cost == pytest.approx(10.0)

    def test_cannot_construct_without_net(self) -> None:
        # Arrange
        bd = CostBreakdown(spread=0.0, slippage=0.0, market_impact=0.0, commission=0.0, borrow=0.0)

        # Act / Assert: 'net' is required -> no gross-only object exists.
        with pytest.raises(ValidationError):
            GrossNetResult(gross=100.0, costs=bd)  # type: ignore[call-arg]

    def test_is_immutable(self) -> None:
        # Arrange
        bd = CostBreakdown(spread=0.0, slippage=0.0, market_impact=0.0, commission=0.0, borrow=0.0)
        res = GrossNetResult(gross=1.0, net=1.0, costs=bd)

        # Act / Assert
        with pytest.raises(ValidationError):
            res.gross = 2.0  # type: ignore[misc]


class TestCostSpecImmutable:
    def test_frozen(self) -> None:
        # Arrange
        spec = CostSpec(half_spread_bps=5.0)

        # Act / Assert
        with pytest.raises(ValidationError):
            spec.half_spread_bps = 9.0  # type: ignore[misc]


class TestCostBreakdownForTrade:
    def test_short_includes_borrow_long_does_not(self) -> None:
        # Arrange: identical trades differing only by side.
        spec = CostSpec(
            half_spread_bps=5.0,
            slippage_bps=2.0,
            impact_eta=1.0,
            impact_alpha=0.5,
            per_share=0.005,
            min_commission=1.0,
            borrow_annual_rate=0.10,
        )
        common = dict(price=20.0, shares=1_000.0, adv=100_000.0, sigma_bps=100.0, holding_days=30.0)

        # Act
        long_bd = cost_breakdown_for_trade(side=Side.BUY, spec=spec, **common)
        short_bd = cost_breakdown_for_trade(side=Side.SELL, spec=spec, **common)

        # Assert: borrow only for the short; all else identical.
        assert long_bd.borrow == 0.0
        assert short_bd.borrow > 0.0
        assert long_bd.spread == pytest.approx(short_bd.spread)
        assert short_bd.total > long_bd.total


class TestSharpeGrossNet:
    def test_net_sharpe_not_greater_than_gross(self) -> None:
        # Arrange: positive-drift returns with positive cost drag.
        gross = [0.01, 0.02, -0.005, 0.015, 0.008, -0.002]
        drag = [0.001] * len(gross)

        # Act
        res = sharpe_gross_net(gross, drag, periods_per_year=252)

        # Assert
        assert res.net <= res.gross
        assert res.total_cost == pytest.approx(0.006)

    def test_zero_cost_gross_equals_net(self) -> None:
        # Arrange
        gross = [0.01, -0.01, 0.02, 0.0]
        drag = [0.0, 0.0, 0.0, 0.0]

        # Act
        res = sharpe_gross_net(gross, drag)

        # Assert
        assert res.net == pytest.approx(res.gross)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            sharpe_gross_net([0.01, 0.02], [0.001])

    def test_constant_returns_zero_sharpe(self) -> None:
        # Arrange: zero variance -> Sharpe defined as 0.
        res = sharpe_gross_net([0.01, 0.01, 0.01], [0.0, 0.0, 0.0])

        # Assert
        assert res.gross == 0.0
        assert res.net == 0.0
