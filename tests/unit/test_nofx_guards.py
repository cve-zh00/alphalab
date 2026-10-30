"""Unit tests for the NOFX-inspired deterministic runtime risk guards."""

from __future__ import annotations

from datetime import date

import pytest

from alphalab.artifact import Verdict
from alphalab.domain.execution.nofx_guards import (
    GuardActionKind,
    PositionRiskSnapshot,
    apply_nofx_guards,
    should_drawdown_close,
)
from alphalab.domain.types import Ticker

_AS_OF = date(2025, 3, 10)


class TestPositionRiskSnapshot:
    def test_computes_price_pnl_peak_and_giveback(self) -> None:
        # Arrange: entry $100, peak $110 (+10%), current $105 (+5% -> 50% giveback)
        snap = PositionRiskSnapshot(
            ticker=Ticker("AAPL"),
            shares=10.0,
            entry_price=100.0,
            current_price=105.0,
            peak_price=110.0,
        )

        # Act / Assert
        assert snap.price_pnl_pct == pytest.approx(5.0)
        assert snap.peak_pnl_pct == pytest.approx(10.0)
        assert snap.profit_giveback_pct == pytest.approx(50.0)
        assert should_drawdown_close(snap) is True

    def test_drawdown_does_not_arm_below_five_percent_peak(self) -> None:
        # Arrange: peak +4% (< +5% arm threshold), current +1% (75% giveback)
        snap = PositionRiskSnapshot(
            ticker=Ticker("AAPL"),
            shares=10.0,
            entry_price=100.0,
            current_price=101.0,
            peak_price=104.0,
        )

        # Act / Assert
        assert snap.peak_pnl_pct == pytest.approx(4.0)
        assert snap.profit_giveback_pct == pytest.approx(75.0)
        assert should_drawdown_close(snap) is False

    def test_giveback_is_zero_when_at_peak(self) -> None:
        snap = PositionRiskSnapshot(
            ticker=Ticker("AAPL"),
            shares=10.0,
            entry_price=100.0,
            current_price=112.0,
            peak_price=110.0,
        )
        assert snap.effective_peak_price == pytest.approx(112.0)
        assert snap.profit_giveback_pct == 0.0


class TestApplyNofxGuards:
    def test_stop_loss_forces_exit_despite_buy_verdict(self) -> None:
        # Arrange: AAPL is down -9.5% (below -8% floor), even though model says BUY
        pos = PositionRiskSnapshot(
            ticker=Ticker("AAPL"),
            shares=10.0,
            entry_price=100.0,
            current_price=90.5,
            peak_price=100.0,
        )

        # Act
        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={Ticker("AAPL"): 0.05, Ticker("MSFT"): 0.05},
            verdicts={Ticker("AAPL"): Verdict.BUY, Ticker("MSFT"): Verdict.BUY},
            positions={Ticker("AAPL"): pos},
            max_positions=20,
        )

        # Assert: AAPL removed from targets; MSFT kept; stop-loss action recorded
        assert Ticker("AAPL") not in guarded.targets
        assert guarded.targets == {Ticker("MSFT"): pytest.approx(0.05)}
        assert len(guarded.actions) == 1
        assert guarded.actions[0].ticker == Ticker("AAPL")
        assert guarded.actions[0].kind is GuardActionKind.STOP_LOSS_EXIT

    def test_peak_drawdown_forces_exit_when_profit_given_back(self) -> None:
        # Arrange: NVDA peaked +10%, now +5.5% -> 45% giveback (>= 40% threshold)
        pos = PositionRiskSnapshot(
            ticker=Ticker("NVDA"),
            shares=5.0,
            entry_price=100.0,
            current_price=105.5,
            peak_price=110.0,
        )

        # Act
        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={Ticker("NVDA"): 0.05},
            verdicts={Ticker("NVDA"): Verdict.BUY},
            positions={Ticker("NVDA"): pos},
            max_positions=20,
        )

        # Assert
        assert Ticker("NVDA") not in guarded.targets
        assert guarded.actions[0].kind is GuardActionKind.DRAWDOWN_EXIT

    def test_sell_verdict_exits_held_position(self) -> None:
        pos = PositionRiskSnapshot(
            ticker=Ticker("AAPL"),
            shares=10.0,
            entry_price=100.0,
            current_price=102.0,
            peak_price=103.0,
        )

        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={},
            verdicts={Ticker("AAPL"): Verdict.SELL},
            positions={Ticker("AAPL"): pos},
            max_positions=20,
        )

        assert guarded.targets == {}
        assert guarded.actions[0].kind is GuardActionKind.SELL_VERDICT_EXIT

    def test_hold_verdict_retains_existing_healthy_position(self) -> None:
        pos = PositionRiskSnapshot(
            ticker=Ticker("AAPL"),
            shares=10.0,
            entry_price=100.0,
            current_price=102.0,
            peak_price=103.0,
        )

        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={Ticker("MSFT"): 0.05},
            verdicts={Ticker("AAPL"): Verdict.HOLD, Ticker("MSFT"): Verdict.BUY},
            positions={Ticker("AAPL"): pos},
            max_positions=20,
        )

        assert guarded.targets == {
            Ticker("AAPL"): pytest.approx(0.05),
            Ticker("MSFT"): pytest.approx(0.05),
        }
        assert any(a.kind is GuardActionKind.HOLD_RETAINED for a in guarded.actions)

    def test_reentry_cooldown_blocks_recently_closed_ticker(self) -> None:
        # Arrange: AMD was closed 1 day ago (cooldown is 3 days)
        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={Ticker("AMD"): 0.05, Ticker("MSFT"): 0.05},
            verdicts={Ticker("AMD"): Verdict.BUY, Ticker("MSFT"): Verdict.BUY},
            positions={},
            recent_sell_dates={Ticker("AMD"): date(2025, 3, 9)},
            max_positions=20,
        )

        # Assert: AMD blocked by cooldown; MSFT allowed
        assert guarded.targets == {Ticker("MSFT"): pytest.approx(0.05)}
        assert len(guarded.actions) == 1
        assert guarded.actions[0].ticker == Ticker("AMD")
        assert guarded.actions[0].kind is GuardActionKind.COOLDOWN_BLOCKED

    def test_reentry_allowed_once_cooldown_expires(self) -> None:
        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={Ticker("AMD"): 0.05},
            verdicts={Ticker("AMD"): Verdict.BUY},
            positions={},
            recent_sell_dates={Ticker("AMD"): date(2025, 3, 7)},  # 3 days ago
            max_positions=20,
        )
        assert guarded.targets == {Ticker("AMD"): pytest.approx(0.05)}
        assert guarded.actions == ()

    def test_safe_mode_blocks_new_buys_but_preserves_healthy_holdings_and_stops(self) -> None:
        # Arrange: AAPL is healthy (+1%), TSLA hit stop-loss (-10%), MSFT is a new BUY
        aapl = PositionRiskSnapshot(
            ticker=Ticker("AAPL"),
            shares=10.0,
            entry_price=100.0,
            current_price=101.0,
            peak_price=102.0,
        )
        tsla = PositionRiskSnapshot(
            ticker=Ticker("TSLA"),
            shares=5.0,
            entry_price=200.0,
            current_price=180.0,
            peak_price=200.0,
        )

        # Act: actionable=False triggers Safe Mode
        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={Ticker("MSFT"): 0.05},
            verdicts={Ticker("MSFT"): Verdict.BUY},
            positions={Ticker("AAPL"): aapl, Ticker("TSLA"): tsla},
            actionable=False,
            safe_mode_reason="only 1/10 candidates evaluated",
            max_positions=20,
        )

        # Assert: Safe Mode active; MSFT new entry blocked; TSLA stop-loss exited; AAPL kept
        assert guarded.safe_mode is True
        assert guarded.safe_mode_reason == "only 1/10 candidates evaluated"
        assert guarded.targets == {Ticker("AAPL"): pytest.approx(0.05)}
        kinds = {a.ticker: a.kind for a in guarded.actions}
        assert kinds[Ticker("TSLA")] is GuardActionKind.STOP_LOSS_EXIT
        assert kinds[Ticker("MSFT")] is GuardActionKind.SAFE_MODE_BLOCKED

    def test_slot_ceiling_prioritizes_existing_positions(self) -> None:
        aapl = PositionRiskSnapshot(
            ticker=Ticker("AAPL"),
            shares=10.0,
            entry_price=100.0,
            current_price=101.0,
            peak_price=102.0,
        )
        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={Ticker("BBB"): 0.5, Ticker("CCC"): 0.5},
            verdicts={
                Ticker("AAPL"): Verdict.HOLD,
                Ticker("BBB"): Verdict.BUY,
                Ticker("CCC"): Verdict.BUY,
            },
            positions={Ticker("AAPL"): aapl},
            max_positions=2,
        )
        assert len(guarded.targets) == 2
        assert Ticker("AAPL") in guarded.targets
        assert Ticker("BBB") in guarded.targets

    def test_rejects_non_positive_max_positions(self) -> None:
        with pytest.raises(ValueError, match="max_positions"):
            apply_nofx_guards(
                as_of=_AS_OF,
                proposed_targets={},
                verdicts={},
                positions={},
                max_positions=0,
            )

    def test_research_failure_retains_open_position_by_default(self) -> None:
        aapl = PositionRiskSnapshot(
            ticker=Ticker("AAPL"),
            shares=10.0,
            entry_price=100.0,
            current_price=102.0,
            peak_price=103.0,
        )
        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={},
            verdicts={Ticker("AAPL"): None},
            positions={Ticker("AAPL"): aapl},
            current_weights={Ticker("AAPL"): 0.075},
            max_positions=20,
        )
        assert guarded.targets == {Ticker("AAPL"): pytest.approx(0.075)}
        assert guarded.actions[0].kind is GuardActionKind.RESEARCH_FAILED_RETAINED

    def test_sell_verdict_requires_consecutive_confirmations(self) -> None:
        from alphalab.domain.execution.nofx_guards import AutopilotGuardConfig

        aapl = PositionRiskSnapshot(
            ticker=Ticker("AAPL"),
            shares=10.0,
            entry_price=100.0,
            current_price=102.0,
            peak_price=103.0,
        )
        cfg = AutopilotGuardConfig(sell_confirmations_required=2)

        # First SELL verdict -> retained with SELL_PENDING_CONFIRMATION
        first = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={},
            verdicts={Ticker("AAPL"): Verdict.SELL},
            positions={Ticker("AAPL"): aapl},
            prior_sell_counts={},
            max_positions=20,
            config=cfg,
        )
        assert Ticker("AAPL") in first.targets
        assert first.actions[0].kind is GuardActionKind.SELL_PENDING_CONFIRMATION

        # Second consecutive SELL verdict -> exited with SELL_VERDICT_EXIT
        second = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={},
            verdicts={Ticker("AAPL"): Verdict.SELL},
            positions={Ticker("AAPL"): aapl},
            prior_sell_counts={Ticker("AAPL"): 1},
            max_positions=20,
            config=cfg,
        )
        assert Ticker("AAPL") not in second.targets
        assert second.actions[0].kind is GuardActionKind.SELL_VERDICT_EXIT

    def test_volatility_scaled_stop_loss_clamps_and_triggers(self) -> None:
        from alphalab.domain.execution.nofx_guards import effective_stop_loss_pct

        # Low-vol REIT (14% annualized -> ~0.88% daily -> clamped to -6.0% floor)
        low_vol = PositionRiskSnapshot(
            ticker=Ticker("REG"),
            shares=10.0,
            entry_price=100.0,
            current_price=93.8,  # -6.2% triggers -6.0% stop!
            peak_price=100.0,
            annualized_volatility=0.14,
        )
        assert effective_stop_loss_pct(low_vol) == pytest.approx(-6.0)

        # High-vol Tech (68% annualized -> ~4.28% daily -> clamped to -12.0% floor)
        high_vol = PositionRiskSnapshot(
            ticker=Ticker("IT"),
            shares=10.0,
            entry_price=100.0,
            current_price=90.5,  # -9.5% survives -12.0% stop!
            peak_price=100.0,
            annualized_volatility=0.68,
        )
        assert effective_stop_loss_pct(high_vol) == pytest.approx(-12.0)

        guarded = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={Ticker("REG"): 0.05, Ticker("IT"): 0.05},
            verdicts={Ticker("REG"): Verdict.BUY, Ticker("IT"): Verdict.BUY},
            positions={Ticker("REG"): low_vol, Ticker("IT"): high_vol},
            max_positions=20,
        )
        assert Ticker("REG") not in guarded.targets
        assert Ticker("IT") in guarded.targets
        assert guarded.actions[0].ticker == Ticker("REG")
        assert guarded.actions[0].kind is GuardActionKind.STOP_LOSS_EXIT


class TestSectorCap:
    def _run(self, **kw: object):
        t = Ticker
        return apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={t("AAA"): 0.10, t("BBB"): 0.10, t("CCC"): 0.10, t("DDD"): 0.10},
            verdicts={},
            positions=kw.pop("positions", {}),  # type: ignore[arg-type]
            sectors={t("AAA"): "tech", t("BBB"): "tech", t("CCC"): "tech", t("DDD"): "fin"},
            max_sector_weight=0.25,
            max_positions=10,
            **kw,  # type: ignore[arg-type]
        )

    def test_new_entries_scaled_and_blocked(self) -> None:
        res = self._run()
        assert res.targets[Ticker("AAA")] == pytest.approx(0.10)
        assert res.targets[Ticker("BBB")] == pytest.approx(0.10)
        assert res.targets[Ticker("CCC")] == pytest.approx(0.05)
        assert res.targets[Ticker("DDD")] == pytest.approx(0.10)
        assert [a.kind for a in res.actions] == [GuardActionKind.SECTOR_CAP_BLOCKED]

    def test_held_positions_never_sold_for_cap(self) -> None:
        pos = {
            Ticker("AAA"): PositionRiskSnapshot(
                ticker=Ticker("AAA"),
                shares=1.0,
                entry_price=100.0,
                current_price=100.0,
                peak_price=100.0,
            )
        }
        res = self._run(positions=pos, current_weights={Ticker("AAA"): 0.30})
        assert Ticker("AAA") in res.targets
        assert Ticker("BBB") not in res.targets
        assert Ticker("CCC") not in res.targets

    def test_cash_headroom_bounds_new_entries_without_trimming_held(self) -> None:
        pos = {
            Ticker("HELD"): PositionRiskSnapshot(
                ticker=Ticker("HELD"),
                shares=10.0,
                entry_price=100.0,
                current_price=100.0,
                peak_price=100.0,
            )
        }
        res = apply_nofx_guards(
            as_of=_AS_OF,
            proposed_targets={
                Ticker("NEW1"): 0.15,
                Ticker("NEW2"): 0.12,
                Ticker("NEW3"): 0.10,
            },
            verdicts={Ticker("HELD"): Verdict.HOLD},
            positions=pos,
            current_weights={Ticker("HELD"): 0.80},
            max_positions=10,
        )
        # HELD keeps 0.80; remaining 0.20 headroom admits NEW1 (0.15) and scales NEW2 (0.05)
        assert res.targets[Ticker("HELD")] == pytest.approx(0.80)
        assert res.targets[Ticker("NEW1")] == pytest.approx(0.15)
        assert res.targets[Ticker("NEW2")] == pytest.approx(0.05)
        assert Ticker("NEW3") not in res.targets
        assert sum(res.targets.values()) == pytest.approx(1.0)
