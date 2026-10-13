"""Unit tests for the number-vs-number audit checks.

Each check is tested in BOTH directions: it must flag the bad case AND leave
the good case alone (a detector that always alarms is useless). It must also
return a structured :class:`AuditFinding`, never a bare bool.
"""

from __future__ import annotations

from alphalab.domain.compute.operators.audit.checks import (
    currency_caliber,
    ev_bridge_check,
    magnitude_sanity,
    sector_sign_check,
    ttm_period_check,
)
from alphalab.domain.compute.operators.audit.findings import AuditFinding, Severity


class TestCurrencyCaliber:
    def test_mismatch_is_hard_failure(self) -> None:
        f = currency_caliber(left_unit="USD", right_unit="EUR", left_value=10.0, right_value=9.0)
        assert isinstance(f, AuditFinding)
        assert f.passed is False
        assert f.severity is Severity.HARD
        assert f.is_blocking() is True

    def test_matching_units_pass(self) -> None:
        f = currency_caliber(left_unit="USD", right_unit="USD", left_value=10.0, right_value=9.0)
        assert f.passed is True
        assert f.is_blocking() is False


class TestEvBridgeCheck:
    def test_reconciled_bridge_passes(self) -> None:
        # equity 1600 + net_debt 400 = EV 2000 exactly.
        f = ev_bridge_check(equity_value=1600.0, net_debt=400.0, enterprise_value=2000.0)
        assert f.passed is True

    def test_within_tolerance_passes(self) -> None:
        # 0.1% off on a 2000 EV is below the 0.5% tolerance.
        f = ev_bridge_check(equity_value=1600.0, net_debt=400.0, enterprise_value=2002.0)
        assert f.passed is True

    def test_large_mismatch_is_hard_failure(self) -> None:
        # equity+net_debt = 2000 but EV claimed 2500 => 20% off.
        f = ev_bridge_check(equity_value=1600.0, net_debt=400.0, enterprise_value=2500.0)
        assert f.passed is False
        assert f.severity is Severity.HARD


class TestTtmPeriodCheck:
    def test_matching_periods_pass(self) -> None:
        f = ttm_period_check(left_period="TTM", right_period="TTM")
        assert f.passed is True

    def test_mismatched_periods_hard_failure(self) -> None:
        f = ttm_period_check(left_period="TTM", right_period="FY2023")
        assert f.passed is False
        assert f.severity is Severity.HARD


class TestSectorSignCheck:
    def test_valid_signs_pass(self) -> None:
        f = sector_sign_check(ebitda=200.0, applied_multiple=10.0, margin=0.25)
        assert f.passed is True

    def test_positive_multiple_on_negative_ebitda_fails(self) -> None:
        f = sector_sign_check(ebitda=-50.0, applied_multiple=10.0, margin=0.1)
        assert f.passed is False
        assert f.severity is Severity.HARD

    def test_impossible_margin_fails(self) -> None:
        f = sector_sign_check(ebitda=200.0, applied_multiple=10.0, margin=1.5)
        assert f.passed is False
        assert f.severity is Severity.HARD


class TestMagnitudeSanity:
    def test_plausible_values_pass(self) -> None:
        f = magnitude_sanity(price_per_share=85.0, pe=18.0, ev_ebitda=10.0)
        assert f.passed is True

    def test_negative_price_fails(self) -> None:
        f = magnitude_sanity(price_per_share=-5.0, pe=18.0, ev_ebitda=10.0)
        assert f.passed is False
        assert f.severity is Severity.HARD

    def test_absurd_pe_fails(self) -> None:
        f = magnitude_sanity(price_per_share=85.0, pe=10000.0, ev_ebitda=10.0)
        assert f.passed is False
        assert f.severity is Severity.HARD

    def test_absurd_ev_ebitda_fails(self) -> None:
        f = magnitude_sanity(price_per_share=85.0, pe=18.0, ev_ebitda=500.0)
        assert f.passed is False
        assert f.severity is Severity.HARD


class TestFindingImmutability:
    def test_finding_is_frozen(self) -> None:
        import pytest
        from pydantic import ValidationError

        f = ttm_period_check(left_period="TTM", right_period="TTM")
        with pytest.raises(ValidationError):
            f.passed = False  # type: ignore[misc]
