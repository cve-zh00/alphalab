"""Unit tests for paper rendering and the shadow/reconciliation value types."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from alphalab.adapters.alpaca_broker import (
    AccountSnapshot,
    OrderOutcome,
    OrderStatus,
)
from alphalab.adapters.execution_log import ExecutionEvent, ExecutionEventKind
from alphalab.adapters.kill_switch import KillSwitchState
from alphalab.app.paper_models import Reconciliation, ReconciliationRow, ShadowBenchmark
from alphalab.app.paper_runner import CycleResult
from alphalab.cli.paper_render import render_log, render_run, render_status
from alphalab.domain.execution import OrderIntent, RejectedOrder, TradePlan
from alphalab.domain.types import Side, Ticker
from alphalab.report.theme import Theme

_PLAIN = Theme(enabled=False)


def _account(*, trading_blocked: bool = False, account_blocked: bool = False) -> AccountSnapshot:
    return AccountSnapshot(
        equity=100_000.0,
        buying_power=200_000.0,
        trading_blocked=trading_blocked,
        account_blocked=account_blocked,
    )


def _result(
    *,
    plan: TradePlan,
    submitted: bool = False,
    outcomes: tuple[OrderOutcome, ...] = (),
    shadow: ShadowBenchmark | None = None,
    reconciliation: Reconciliation | None = None,
    halted_reason: str | None = None,
    account: AccountSnapshot | None = None,
    market_open: bool = True,
) -> CycleResult:
    return CycleResult(
        cycle_id="c1",
        as_of=date(2024, 1, 2),
        account=account or _account(),
        market_open=market_open,
        submitted=submitted,
        plan=plan,
        outcomes=outcomes,
        reconciliation=reconciliation or Reconciliation(rows=()),
        shadow=shadow,
        halted_reason=halted_reason,
        skipped_reason=None,
    )


def _empty_plan() -> TradePlan:
    return TradePlan(as_of=date(2024, 1, 2), intents=(), rejected=(), equity=100_000.0)


def _plan_with_intent() -> TradePlan:
    return TradePlan(
        as_of=date(2024, 1, 2),
        intents=(
            OrderIntent(ticker=Ticker("AAA"), side=Side.BUY, shares=100.0, reference_price=10.0),
        ),
        rejected=(RejectedOrder(ticker=Ticker("ZZZ"), reason="dust"),),
        equity=100_000.0,
    )


# -- ShadowBenchmark / Reconciliation -------------------------------------


def test_shadow_benchmark_math() -> None:
    shadow = ShadowBenchmark(
        symbol=Ticker("SPY"), initial_capital=1000.0, entry_price=100.0, latest_price=110.0
    )
    assert shadow.shares == 10.0
    assert shadow.current_value == pytest.approx(1100.0)
    assert shadow.total_return == pytest.approx(0.1)


def test_shadow_benchmark_zero_entry_is_safe() -> None:
    shadow = ShadowBenchmark(
        symbol=Ticker("SPY"), initial_capital=1000.0, entry_price=0.0, latest_price=110.0
    )
    # A zero entry price buys no shares, so the mark-to-market value is zero.
    assert shadow.shares == 0.0
    assert shadow.current_value == 0.0


def test_reconciliation_alignment() -> None:
    aligned = Reconciliation(
        rows=(ReconciliationRow(ticker=Ticker("A"), target_weight=0.2, actual_weight=0.202),)
    )
    drifted = Reconciliation(
        rows=(ReconciliationRow(ticker=Ticker("A"), target_weight=0.2, actual_weight=0.3),)
    )
    assert aligned.is_aligned
    assert not drifted.is_aligned
    assert drifted.max_abs_deviation == pytest.approx(0.1)


def test_empty_reconciliation_is_aligned() -> None:
    recon = Reconciliation(rows=())
    assert recon.max_abs_deviation == 0.0
    assert recon.is_aligned


# -- render_status --------------------------------------------------------


def test_render_status_flags_blocked_account() -> None:
    result = _result(
        plan=_empty_plan(),
        account=_account(trading_blocked=True, account_blocked=True),
    )
    state = KillSwitchState(halt_file_present=False, env_var_engaged=False, halt_file_path="HALT")
    text = render_status(result, state, _PLAIN)
    assert "trading_blocked is set" in text
    assert "account_blocked is set" in text
    assert "(no positions and no targets)" in text


def test_render_status_shows_shadow_and_engaged_switch() -> None:
    shadow = ShadowBenchmark(
        symbol=Ticker("SPY"), initial_capital=100_000.0, entry_price=400.0, latest_price=440.0
    )
    recon = Reconciliation(
        rows=(ReconciliationRow(ticker=Ticker("AAA"), target_weight=0.2, actual_weight=0.2),)
    )
    result = _result(plan=_empty_plan(), shadow=shadow, reconciliation=recon)
    state = KillSwitchState(halt_file_present=True, env_var_engaged=False, halt_file_path="HALT")
    text = render_status(result, state, _PLAIN)
    assert "shadow SPY buy&hold" in text
    assert "HALTED" in text
    assert "aligned" in text


# -- render_run -----------------------------------------------------------


def test_render_run_empty_plan() -> None:
    text = render_run(_result(plan=_empty_plan()), _PLAIN)
    assert "DRY RUN" in text
    assert "already matches target" in text


def test_render_run_with_intents_rejections_and_outcomes() -> None:
    plan = _plan_with_intent()
    outcome = OrderOutcome(
        intent=plan.intents[0],
        status=OrderStatus.SUBMITTED,
        client_order_id="rank_ensemble-2024-01-02-AAA-buy",
        broker_order_id="srv-1",
    )
    result = _result(plan=plan, submitted=True, outcomes=(outcome,), halted_reason=None)
    text = render_run(result, _PLAIN)
    assert "SUBMITTED" in text
    assert "BUY" in text
    assert "REJECTED BY RISK LIMITS" in text
    assert "ZZZ: dust" in text
    assert "SUBMISSION OUTCOMES" in text


def test_render_run_shows_halt_reason() -> None:
    result = _result(plan=_plan_with_intent(), halted_reason="order emission HALTED")
    text = render_run(result, _PLAIN)
    assert "order emission HALTED" in text


def test_render_run_closed_market_label() -> None:
    result = _result(plan=_empty_plan(), market_open=False)
    text = render_run(result, _PLAIN)
    assert "CLOSED" in text


# -- render_log -----------------------------------------------------------


def test_render_log_empty() -> None:
    assert "No events recorded yet" in render_log([], _PLAIN)


def test_render_log_with_events() -> None:
    event = ExecutionEvent(
        cycle_id="c1",
        recorded_at=datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
        kind=ExecutionEventKind.SUBMITTED,
        ticker="AAA",
        side="buy",
        shares=100.0,
        reference_price=10.0,
        client_order_id="rank_ensemble-2024-01-02-AAA-buy",
        broker_order_id="srv-1",
        detail="ok",
    )
    text = render_log([event], _PLAIN)
    assert "SUBMITTED" in text
    assert "AAA" in text
    assert "ok" in text
