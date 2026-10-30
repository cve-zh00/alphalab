"""Unit tests for NOFX guards: conviction slot ceiling, Dykstra projection, and OU exit."""

from __future__ import annotations

from datetime import date

import pytest

from alphalab.artifact import Verdict
from alphalab.domain.execution.nofx_guards import (
    GuardActionKind,
    PositionRiskSnapshot,
    apply_nofx_guards,
    project_feasible_weights_dykstra,
    should_drawdown_close,
)
from alphalab.domain.types import Ticker

_AS_OF = date(2025, 3, 10)


class TestSlotCeilingAndSectorCapOrdering:
    def test_slot_ceiling_orders_by_conviction_weight_and_preserves_weights(self) -> None:
        # Arrange: ZZZ and YYY have higher conviction weights than AAA and BBB
        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={
                Ticker("AAA"): 0.04,
                Ticker("BBB"): 0.05,
                Ticker("YYY"): 0.08,
                Ticker("ZZZ"): 0.10,
            },
            verdicts={
                Ticker("AAA"): Verdict.BUY,
                Ticker("BBB"): Verdict.BUY,
                Ticker("YYY"): Verdict.BUY,
                Ticker("ZZZ"): Verdict.BUY,
            },
            positions={},
            max_positions=2,
        )

        # Assert: YYY and ZZZ are kept with their conviction weights (not reset to 1/2 = 0.50)
        assert set(guarded.targets) == {Ticker("YYY"), Ticker("ZZZ")}
        assert guarded.targets[Ticker("ZZZ")] == pytest.approx(0.10)
        assert guarded.targets[Ticker("YYY")] == pytest.approx(0.08)

    def test_slot_ceiling_keeps_existing_holdings_first_then_highest_conviction(self) -> None:
        held = PositionRiskSnapshot(
            ticker=Ticker("MMM"),
            shares=10.0,
            entry_price=100.0,
            current_price=102.0,
            peak_price=103.0,
        )
        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={
                Ticker("AAA"): 0.04,
                Ticker("ZZZ"): 0.09,
            },
            verdicts={
                Ticker("MMM"): Verdict.HOLD,
                Ticker("AAA"): Verdict.BUY,
                Ticker("ZZZ"): Verdict.BUY,
            },
            positions={Ticker("MMM"): held},
            current_weights={Ticker("MMM"): 0.06},
            max_positions=2,
        )

        assert set(guarded.targets) == {Ticker("MMM"), Ticker("ZZZ")}
        assert guarded.targets[Ticker("MMM")] == pytest.approx(0.06)
        assert guarded.targets[Ticker("ZZZ")] == pytest.approx(0.09)

    def test_slot_ceiling_scales_proportionally_when_flat_and_sum_exceeds_one(self) -> None:
        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={
                Ticker("AAA"): 0.20,
                Ticker("YYY"): 0.60,
                Ticker("ZZZ"): 0.80,
            },
            verdicts={},
            positions={},
            max_positions=2,
        )

        assert set(guarded.targets) == {Ticker("YYY"), Ticker("ZZZ")}
        assert sum(guarded.targets.values()) == pytest.approx(1.0)
        assert guarded.targets[Ticker("ZZZ")] == pytest.approx(0.80 / 1.40)
        assert guarded.targets[Ticker("YYY")] == pytest.approx(0.60 / 1.40)

    def test_sector_cap_admits_highest_conviction_first_and_is_not_overwritten_by_slot_ceiling(
        self,
    ) -> None:
        # Arrange: 3 tech entries (ZZZ=0.15, YYY=0.12, AAA=0.10) + 1 fin entry (FFF=0.08)
        # With max_positions=3, slot ceiling selects ZZZ (0.15), YYY (0.12), AAA (0.10),
        # and then sector cap (0.25) keeps ZZZ at 0.15, scales YYY to 0.10, and blocks AAA!
        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={
                Ticker("AAA"): 0.10,
                Ticker("FFF"): 0.08,
                Ticker("YYY"): 0.12,
                Ticker("ZZZ"): 0.15,
            },
            verdicts={},
            positions={},
            sectors={
                Ticker("AAA"): "tech",
                Ticker("FFF"): "fin",
                Ticker("YYY"): "tech",
                Ticker("ZZZ"): "tech",
            },
            max_sector_weight=0.25,
            max_positions=3,
        )

        assert guarded.targets[Ticker("ZZZ")] == pytest.approx(0.15)
        assert guarded.targets[Ticker("YYY")] == pytest.approx(0.10)
        assert Ticker("AAA") not in guarded.targets
        tech_tickers = {Ticker("AAA"), Ticker("YYY"), Ticker("ZZZ")}
        tech_total = sum(w for t, w in guarded.targets.items() if t in tech_tickers)
        assert tech_total <= 0.25 + 1e-9
        assert any(a.kind is GuardActionKind.SECTOR_CAP_BLOCKED for a in guarded.actions)


class TestProjectFeasibleWeightsDykstra:
    def test_satisfies_box_sector_and_gross_constraints_simultaneously(self) -> None:
        weights = {
            Ticker("NVDA"): 0.18,
            Ticker("MSFT"): 0.14,
            Ticker("AAPL"): 0.12,
            Ticker("AMD"): 0.09,
            Ticker("JPM"): 0.15,
            Ticker("GS"): 0.12,
            Ticker("BAC"): 0.10,
            Ticker("XOM"): 0.14,
            Ticker("CVX"): 0.12,
            Ticker("UNH"): 0.13,
            Ticker("LLY"): 0.11,
        }
        sectors = {
            Ticker("NVDA"): "Technology",
            Ticker("MSFT"): "Technology",
            Ticker("AAPL"): "Technology",
            Ticker("AMD"): "Technology",
            Ticker("JPM"): "Financials",
            Ticker("GS"): "Financials",
            Ticker("BAC"): "Financials",
            Ticker("XOM"): "Energy",
            Ticker("CVX"): "Energy",
            Ticker("UNH"): "Health Care",
            Ticker("LLY"): "Health Care",
        }

        projected = project_feasible_weights_dykstra(
            weights,
            max_position_weight=0.10,
            sectors=sectors,
            max_sector_weight=0.25,
            max_gross_exposure=1.0,
            iterations=20,
        )

        # Box constraint: every weight in (0, 0.10]
        for w in projected.values():
            assert 0.0 < w <= 0.10 + 1e-7

        # Sector constraint: every sector sum <= 0.25
        for sec in set(sectors.values()):
            sec_sum = sum(w for t, w in projected.items() if sectors.get(t) == sec)
            assert sec_sum <= 0.25 + 1e-7

        # Gross exposure constraint: total sum <= 1.0
        assert sum(projected.values()) <= 1.0 + 1e-7

        # Relative conviction order within Technology preserved
        assert projected[Ticker("NVDA")] >= projected[Ticker("MSFT")] - 1e-7
        assert projected[Ticker("MSFT")] >= projected[Ticker("AAPL")] - 1e-7
        assert projected[Ticker("AAPL")] >= projected[Ticker("AMD")] - 1e-7

    def test_preserves_locked_tickers_and_drops_negligible_weights(self) -> None:
        weights = {
            Ticker("HELD"): 0.16,
            Ticker("NEW1"): 0.12,
            Ticker("NEW2"): 0.08,
            Ticker("TINY"): 1e-8,
        }
        sectors = {
            Ticker("HELD"): "Technology",
            Ticker("NEW1"): "Technology",
            Ticker("NEW2"): "Technology",
            Ticker("TINY"): "Utilities",
        }

        projected = project_feasible_weights_dykstra(
            weights,
            max_position_weight=0.10,
            sectors=sectors,
            max_sector_weight=0.25,
            max_gross_exposure=1.0,
            locked_tickers=frozenset({Ticker("HELD")}),
        )

        assert projected[Ticker("HELD")] == pytest.approx(0.16)
        assert Ticker("TINY") not in projected
        tech_sum = sum(w for t, w in projected.items() if sectors.get(t) == "Technology")
        assert tech_sum == pytest.approx(0.25, abs=1e-6)


class TestOrnsteinUhlenbeckConvergenceExit:
    def test_tightens_arm_and_giveback_near_or_past_fair_value(self) -> None:
        # Peak is +3.5%, current is +2.3% -> giveback = 34.29%
        # Under default rules (arm=5%, giveback=40%), this does NOT close.
        # Near fair value (fair_value_upside_pct=1.5 <= 2.0), arm tightens to 3.0%
        # and giveback tightens to 30.0%, so it closes to lock in OU convergence!
        base_snap = PositionRiskSnapshot(
            ticker=Ticker("AAPL"),
            shares=10.0,
            entry_price=100.0,
            current_price=102.3,
            peak_price=103.5,
            fair_value_upside_pct=None,
        )
        converged_snap = PositionRiskSnapshot(
            ticker=Ticker("AAPL"),
            shares=10.0,
            entry_price=100.0,
            current_price=102.3,
            peak_price=103.5,
            fair_value_upside_pct=1.5,
        )

        assert should_drawdown_close(base_snap) is False
        assert should_drawdown_close(converged_snap) is True

    def test_scales_arm_threshold_for_deep_value_convergence_in_progress(self) -> None:
        # Peak is +6.0%, current is +3.0% -> giveback = 50% (>= 40%).
        # Under default rules (arm=5%), this closes.
        # With fair_value_upside_pct=36.0 (>= 25.0), arm scales to min(10.0, 0.25 * 36.0) = 9.0%,
        # so a +6.0% peak does NOT prematurely trigger drawdown close!
        deep_value_snap = PositionRiskSnapshot(
            ticker=Ticker("NVDA"),
            shares=10.0,
            entry_price=100.0,
            current_price=103.0,
            peak_price=106.0,
            fair_value_upside_pct=36.0,
        )
        deep_value_armed_snap = PositionRiskSnapshot(
            ticker=Ticker("NVDA"),
            shares=10.0,
            entry_price=100.0,
            current_price=105.0,
            peak_price=109.5,  # +9.5% >= 9.0% scaled arm, giveback = 47.4% >= 40%
            fair_value_upside_pct=36.0,
        )

        assert should_drawdown_close(deep_value_snap) is False
        assert should_drawdown_close(deep_value_armed_snap) is True
