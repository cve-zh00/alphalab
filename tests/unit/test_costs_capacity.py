"""Unit tests for :mod:`alphalab.domain.costs.capacity`.

Covers the edge-degradation curve, the break-even capacity size, the stressed
stop cost, and the end-to-end DEMO comparing gross vs net Sharpe alongside the
capacity curve.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alphalab.domain.costs.capacity import (
    capacity_curve,
    capacity_size,
    net_edge_bps,
    stressed_stop,
)
from alphalab.domain.costs.result import CostSpec, cost_breakdown_for_trade, sharpe_gross_net
from alphalab.domain.types import Side


class TestNetEdge:
    def test_impact_erodes_gross_edge(self) -> None:
        # Arrange: gross 20 bps, impact at this size = 10 bps, no fixed cost.
        # Act
        net = net_edge_bps(20.0, 1_000.0, 100_000.0, 100.0)

        # Assert: 20 - 0 - 10 = 10 bps net.
        assert net == pytest.approx(10.0)

    def test_fixed_cost_subtracted(self) -> None:
        # Arrange / Act
        net = net_edge_bps(20.0, 1_000.0, 100_000.0, 100.0, fixed_cost_bps=5.0)

        # Assert: 20 - 5 - 10 = 5.
        assert net == pytest.approx(5.0)

    def test_net_decreases_with_size(self) -> None:
        # Arrange / Act
        small = net_edge_bps(20.0, 1_000.0, 100_000.0, 100.0)
        large = net_edge_bps(20.0, 4_000.0, 100_000.0, 100.0)

        # Assert
        assert large < small


class TestCapacitySize:
    def test_break_even_is_net_zero(self) -> None:
        # Arrange
        gross, adv, sigma = 20.0, 100_000.0, 100.0

        # Act
        q_star = capacity_size(gross, adv, sigma)
        net_at_star = net_edge_bps(gross, q_star, adv, sigma)

        # Assert: net edge is (numerically) zero at the capacity size.
        assert net_at_star == pytest.approx(0.0, abs=1e-6)

    def test_closed_form_matches_hand_calc(self) -> None:
        # Arrange: gross 20, eta 1, sigma 100 -> budget 20 bps.
        # Q* = ADV * (20 / (1*100))**(1/0.5) = ADV * 0.2**2 = ADV * 0.04.
        # Act
        q_star = capacity_size(20.0, 100_000.0, 100.0)

        # Assert
        assert q_star == pytest.approx(4_000.0)

    def test_exhausted_edge_returns_zero(self) -> None:
        # Arrange: fixed cost already exceeds gross edge.
        # Act
        q_star = capacity_size(5.0, 100_000.0, 100.0, fixed_cost_bps=10.0)

        # Assert
        assert q_star == 0.0

    def test_zero_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma_bps"):
            capacity_size(20.0, 100_000.0, 0.0)


class TestStressedStop:
    def test_reference_example(self) -> None:
        # Arrange: trigger 100, slippage 1.50/leg, fee 1.325/leg.
        # Expected: 100 + 2*1.50 + 2*1.325 = 100 + 3.00 + 2.65 = 105.65.
        # Act
        stop = stressed_stop(100.0, slippage_per_leg=1.50, taker_fee_per_leg=1.325)

        # Assert
        assert stop.slippage == pytest.approx(3.00)
        assert stop.fees == pytest.approx(2.65)
        assert stop.risk_per_unit == pytest.approx(105.65)

    def test_risk_strictly_above_trigger(self) -> None:
        # Arrange / Act
        stop = stressed_stop(50.0, slippage_per_leg=0.1, taker_fee_per_leg=0.05)

        # Assert
        assert stop.risk_per_unit > stop.nominal_distance

    def test_is_immutable(self) -> None:
        # Arrange
        stop = stressed_stop(10.0, slippage_per_leg=0.0, taker_fee_per_leg=0.0)

        # Act / Assert
        with pytest.raises(ValidationError):
            stop.risk_per_unit = 99.0  # type: ignore[misc]

    def test_negative_slippage_raises(self) -> None:
        with pytest.raises(ValueError, match="slippage_per_leg"):
            stressed_stop(10.0, slippage_per_leg=-0.1, taker_fee_per_leg=0.0)


class TestCapacityCurve:
    def test_curve_crosses_zero(self) -> None:
        # Arrange: a size grid that straddles the break-even point.
        sizes = (1_000.0, 2_000.0, 4_000.0, 8_000.0)

        # Act
        curve = capacity_curve(20.0, sizes, 100_000.0, 100.0)

        # Assert: net edge is positive for small sizes and negative for large.
        net_values = [net for _, net in curve]
        assert net_values[0] > 0.0
        assert net_values[-1] < 0.0


def test_demo_gross_vs_net_sharpe_and_capacity_curve() -> None:
    """DEMO: gross vs net Sharpe side by side + edge-degradation curve.

    Builds a small set of synthetic daily trades, derives a per-period cost
    drag from the cost model, and shows (a) gross vs net Sharpe and (b) the
    capacity curve crossing zero.
    """
    # Arrange: synthetic daily gross returns for a small-cap-style signal,
    # with realistic dispersion (wins and losses) so the Sharpe is plausible.
    gross_returns = [
        0.018,
        -0.022,
        0.031,
        -0.015,
        0.027,
        -0.019,
        0.024,
        0.009,
        -0.012,
        0.021,
        -0.008,
        0.016,
    ]

    # Per-trade cost drag derived from the cost model, expressed as a return
    # fraction of a fixed notional traded each day.
    spec = CostSpec(
        half_spread_bps=15.0,
        slippage_bps=5.0,
        impact_eta=1.0,
        impact_alpha=0.5,
        per_share=0.005,
        min_commission=1.0,
        ad_valorem_bps=1.0,
        borrow_annual_rate=0.08,
    )
    price, shares, adv, sigma_bps = 10.0, 3_000.0, 150_000.0, 120.0
    notional = price * shares
    per_trade = cost_breakdown_for_trade(
        side=Side.SELL,
        price=price,
        shares=shares,
        adv=adv,
        sigma_bps=sigma_bps,
        holding_days=5.0,
        spec=spec,
    )
    drag_fraction = per_trade.total / notional
    cost_fractions = [drag_fraction] * len(gross_returns)

    # Act: gross vs net Sharpe in one immutable result.
    sharpe = sharpe_gross_net(gross_returns, cost_fractions, periods_per_year=252)

    # Capacity curve across increasing sizes for a 60 bps gross edge.
    # Fixed cost = half-spread + slippage = 20 bps -> 40 bps impact budget.
    # Break-even Q* = adv*(40/120)**2 ~= 16_667 shares, inside the grid.
    gross_edge_bps = 60.0
    fixed_cost_bps = spec.half_spread_bps + spec.slippage_bps
    sizes = (1_000.0, 4_000.0, 8_000.0, 16_000.0, 32_000.0, 64_000.0)
    curve = capacity_curve(gross_edge_bps, sizes, adv, sigma_bps, fixed_cost_bps=fixed_cost_bps)
    q_star = capacity_size(gross_edge_bps, adv, sigma_bps, fixed_cost_bps=fixed_cost_bps)

    # Assert: net Sharpe strictly below gross, and both reported together.
    assert sharpe.net < sharpe.gross
    assert sharpe.gross > 0.0

    # Assert: the capacity curve starts positive and crosses to negative.
    net_values = [net for _, net in curve]
    assert net_values[0] > 0.0
    assert net_values[-1] < 0.0

    # Assert: the analytic break-even size lies within the sign-change bracket.
    positive_sizes = [size for size, net in curve if net > 0.0]
    negative_sizes = [size for size, net in curve if net < 0.0]
    assert max(positive_sizes) <= q_star <= min(negative_sizes)

    # Emit the demo numbers so they appear in `pytest -v -s` output.
    print(
        f"\nDEMO Sharpe gross={sharpe.gross:.4f} "
        f"net={sharpe.net:.4f} (drag/period={drag_fraction:.5f})"
    )
    print(f"DEMO capacity break-even size Q*={q_star:.1f} shares")
    for size, net in curve:
        print(f"  size={size:>8.0f}  net_edge_bps={net:+.3f}")
