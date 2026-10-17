"""Unit tests for :mod:`alphalab.domain.compute.operators.dcf`.

Expected values are computed independently by hand (see inline arithmetic),
never read back from the implementation.
"""

from __future__ import annotations

import pytest

from alphalab.domain.compute.errors import (
    GrowthExceedsDiscountError,
    InvalidValuationInputError,
)
from alphalab.domain.compute.operators.dcf import dcf_value_per_share, wacc


class TestWacc:
    def test_hand_computed_value(self) -> None:
        # Arrange: E=600, D=400 => V=1000; Re=0.10, Rd=0.05, Tc=0.21.
        # WACC = 0.6*0.10 + 0.4*0.05*(1-0.21) = 0.06 + 0.0158 = 0.0758.
        node = wacc(
            equity_value=600.0,
            debt_value=400.0,
            cost_of_equity=0.10,
            cost_of_debt=0.05,
            tax_rate=0.21,
        )

        # Assert
        assert node.value == pytest.approx(0.0758)
        assert node.unit == "ratio"
        assert node.operator == "wacc"

    def test_zero_debt_equals_cost_of_equity(self) -> None:
        node = wacc(
            equity_value=100.0,
            debt_value=0.0,
            cost_of_equity=0.12,
            cost_of_debt=0.05,
            tax_rate=0.21,
        )
        assert node.value == pytest.approx(0.12)

    def test_non_positive_equity_rejected(self) -> None:
        with pytest.raises(InvalidValuationInputError, match="equity_value"):
            wacc(
                equity_value=0.0,
                debt_value=10.0,
                cost_of_equity=0.1,
                cost_of_debt=0.05,
                tax_rate=0.2,
            )

    def test_tax_rate_out_of_range_rejected(self) -> None:
        with pytest.raises(InvalidValuationInputError, match="tax_rate"):
            wacc(
                equity_value=100.0,
                debt_value=10.0,
                cost_of_equity=0.1,
                cost_of_debt=0.05,
                tax_rate=1.0,
            )

    def test_negative_debt_rejected(self) -> None:
        with pytest.raises(InvalidValuationInputError, match="debt_value"):
            wacc(
                equity_value=100.0,
                debt_value=-1.0,
                cost_of_equity=0.1,
                cost_of_debt=0.05,
                tax_rate=0.2,
            )


class TestDcfValuePerShare:
    def test_hand_computed_value(self) -> None:
        # Arrange: cfs=[100,110,121], r=0.10, g=0.03, shares=50.
        # PV_explicit = 100/1.1 + 110/1.1^2 + 121/1.1^3 = 272.727272...
        # TV = 121*1.03 / (0.10-0.03) = 124.63/0.07 = 1780.428571...
        # PV_TV = 1780.428571 / 1.1^3 = 1337.662337...
        # per_share = (272.727272 + 1337.662337) / 50 = 32.207792...
        # Build an explicit discount rate of 0.10 via a debt-free WACC.
        r = wacc(
            equity_value=100.0,
            debt_value=0.0,
            cost_of_equity=0.10,
            cost_of_debt=0.0,
            tax_rate=0.0,
        )

        # Act
        node = dcf_value_per_share(
            cash_flows=[100.0, 110.0, 121.0],
            discount_rate=r,
            terminal_growth=0.03,
            shares_outstanding=50.0,
        )

        # Assert
        assert node.value == pytest.approx(32.2077922077922)
        assert node.unit == "USD_per_share"
        # The discount rate node is part of the derivation chain.
        assert r in node.parents

    def test_growth_equals_discount_raises(self) -> None:
        r = wacc(
            equity_value=100.0,
            debt_value=0.0,
            cost_of_equity=0.08,
            cost_of_debt=0.0,
            tax_rate=0.0,
        )
        with pytest.raises(GrowthExceedsDiscountError):
            dcf_value_per_share(
                cash_flows=[100.0],
                discount_rate=r,
                terminal_growth=0.08,
                shares_outstanding=10.0,
            )

    def test_growth_exceeds_discount_raises(self) -> None:
        r = wacc(
            equity_value=100.0,
            debt_value=0.0,
            cost_of_equity=0.08,
            cost_of_debt=0.0,
            tax_rate=0.0,
        )
        with pytest.raises(GrowthExceedsDiscountError):
            dcf_value_per_share(
                cash_flows=[100.0],
                discount_rate=r,
                terminal_growth=0.12,
                shares_outstanding=10.0,
            )

    def test_empty_cash_flows_rejected(self) -> None:
        r = wacc(
            equity_value=100.0,
            debt_value=0.0,
            cost_of_equity=0.10,
            cost_of_debt=0.0,
            tax_rate=0.0,
        )
        with pytest.raises(InvalidValuationInputError, match="non-empty"):
            dcf_value_per_share(
                cash_flows=[],
                discount_rate=r,
                terminal_growth=0.03,
                shares_outstanding=10.0,
            )

    def test_non_positive_shares_rejected(self) -> None:
        r = wacc(
            equity_value=100.0,
            debt_value=0.0,
            cost_of_equity=0.10,
            cost_of_debt=0.0,
            tax_rate=0.0,
        )
        with pytest.raises(InvalidValuationInputError, match="shares_outstanding"):
            dcf_value_per_share(
                cash_flows=[100.0],
                discount_rate=r,
                terminal_growth=0.03,
                shares_outstanding=0.0,
            )

    def test_fcff_net_debt_bridge_and_capm_and_reverse_dcf(self) -> None:
        from alphalab.domain.compute.operators.dcf import (
            cost_of_equity_capm,
            reverse_dcf_implied_growth,
        )
        from alphalab.domain.compute.sectors import (
            GicsSector,
            compute_capm_cost_of_equity,
            estimate_beta_from_prices,
            resolve_sector_profile,
        )
        from alphalab.domain.types import Ticker

        ke = cost_of_equity_capm(risk_free_rate=0.042, beta=1.2, equity_risk_premium=0.05)
        assert ke.value == pytest.approx(0.102)

        fcfe_node = dcf_value_per_share(
            cash_flows=[100.0, 105.0, 110.0],
            discount_rate=ke,
            terminal_growth=0.025,
            shares_outstanding=10.0,
        )
        fcff_node = dcf_value_per_share(
            cash_flows=[100.0, 105.0, 110.0],
            discount_rate=ke,
            terminal_growth=0.025,
            shares_outstanding=10.0,
            net_debt=50.0,
        )
        assert fcff_node.value == pytest.approx(fcfe_node.value - 5.0)

        g_star = reverse_dcf_implied_growth(
            base_cash_flow=100.0,
            discount_rate=0.10,
            terminal_growth=0.025,
            current_price=fcfe_node.value,
            shares_outstanding=10.0,
            projection_years=3,
        )
        assert g_star is not None
        assert -0.25 <= g_star <= 0.40

        prof = resolve_sector_profile(Ticker("NVDA"))
        assert prof.sector is GicsSector.INFORMATION_TECHNOLOGY
        assert prof.peer_pe == (22.0, 25.0, 28.0)

        prices = [100.0 + i * 0.5 for i in range(70)]
        bench = [400.0 + i * 1.5 for i in range(70)]
        beta = estimate_beta_from_prices(prices, bench, default_beta=1.1)
        assert 0.40 <= beta <= 2.20
        capm_ke = compute_capm_cost_of_equity(beta=beta)
        assert 0.065 <= capm_ke <= 0.145

