"""Drift band in the planner and entry tracking in the autopilot runner."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from alphalab.adapters.execution_log import (
    ExecutionEvent,
    ExecutionEventKind,
    SqliteExecutionLog,
)
from alphalab.app.autopilot_runner import _latest_entries
from alphalab.domain.execution import Holding, RiskLimits
from alphalab.domain.execution.planner import plan_trades
from alphalab.domain.types import Ticker

T = Ticker("PHM")


def _plan(weight: float, target: float):
    equity = 10_000.0
    price = 100.0
    return plan_trades(
        as_of=date(2024, 6, 10),
        equity=equity,
        holdings={T: Holding(ticker=T, shares=equity * weight / price, price=price)},
        targets={T: target},
        prices={T: price},
        limits=RiskLimits(max_order_weight=1.0),
    )


def test_small_drift_above_limit_is_not_trimmed() -> None:
    plan = _plan(0.103, 0.103)
    assert plan.intents == ()


def test_drift_above_ceiling_is_clipped_to_ceiling() -> None:
    plan = _plan(0.13, 0.13)
    assert len(plan.intents) == 1
    assert plan.intents[0].side.value.lower() == "sell"
    assert abs(plan.intents[0].shares * 100.0 / 10_000.0 - 0.01) < 1e-3
    assert plan.rejected


def test_new_buy_still_capped_at_position_limit() -> None:
    plan = plan_trades(
        as_of=date(2024, 6, 10),
        equity=10_000.0,
        holdings={},
        targets={T: 0.12},
        prices={T: 100.0},
        limits=RiskLimits(max_order_weight=1.0),
    )
    assert abs(plan.intents[0].shares - 10.0) < 1e-6


def _event(kind_side: str, shares: float, price: float, day: int) -> ExecutionEvent:
    return ExecutionEvent(
        cycle_id="c",
        recorded_at=datetime(2024, 6, day, 15, tzinfo=UTC),
        kind=ExecutionEventKind.SUBMITTED,
        ticker="PHM",
        side=kind_side,
        shares=shares,
        reference_price=price,
        client_order_id=None,
        broker_order_id=None,
        detail=None,
    )


def test_partial_sell_keeps_entry(tmp_path: Path) -> None:
    log = SqliteExecutionLog(tmp_path / "e.sqlite")
    log.append(_event("buy", 10.0, 100.0, 3))
    log.append(_event("sell", 3.0, 110.0, 5))
    assert _latest_entries(log)[T] == (date(2024, 6, 3), 100.0)
    log.append(_event("sell", 7.0, 110.0, 6))
    assert T not in _latest_entries(log)
