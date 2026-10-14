"""Unit tests for the append-only execution log and the kill switch."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from alphalab.adapters.execution_log import (
    ExecutionEvent,
    ExecutionEventKind,
    SqliteExecutionLog,
)
from alphalab.adapters.kill_switch import HALT_ENV_VAR, KillSwitch


def _event(kind: ExecutionEventKind = ExecutionEventKind.INTENT) -> ExecutionEvent:
    return ExecutionEvent(
        cycle_id="rank_ensemble-2024-01-02T10:00:00+00:00",
        recorded_at=datetime(2024, 1, 2, 10, 0, tzinfo=UTC),
        kind=kind,
        ticker="AAPL",
        side="buy",
        shares=10.0,
        reference_price=100.0,
        client_order_id="rank_ensemble-2024-01-02-AAPL-buy",
        broker_order_id=None,
        detail=None,
    )


# -- execution log --------------------------------------------------------


def test_append_and_read_round_trips(tmp_path: Path) -> None:
    # Arrange
    log = SqliteExecutionLog(tmp_path / "exec.db")

    # Act
    log.append(_event())
    events = log.list_events()
    log.close()

    # Assert
    assert len(events) == 1
    assert events[0].ticker == "AAPL"
    assert events[0].kind is ExecutionEventKind.INTENT


def test_list_events_filters_by_cycle(tmp_path: Path) -> None:
    # Arrange
    log = SqliteExecutionLog(tmp_path / "exec.db")
    first = _event()
    second = ExecutionEvent(
        cycle_id="other-cycle",
        recorded_at=datetime(2024, 1, 3, tzinfo=UTC),
        kind=ExecutionEventKind.REJECTED,
        ticker="MSFT",
        side=None,
        shares=None,
        reference_price=None,
        client_order_id=None,
        broker_order_id=None,
        detail="dust",
    )

    # Act
    log.append(first)
    log.append(second)
    only_first = log.list_events(cycle_id=first.cycle_id)
    log.close()

    # Assert
    assert len(only_first) == 1
    assert only_first[0].ticker == "AAPL"


def test_update_is_rejected_by_trigger(tmp_path: Path) -> None:
    # Arrange
    db = tmp_path / "exec.db"
    log = SqliteExecutionLog(db)
    log.append(_event())
    log.close()

    # Act / Assert — a raw UPDATE outside the class must be aborted by SQLite.
    conn = sqlite3.connect(db)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE execution_events SET ticker = 'HACK'")
            conn.commit()
    finally:
        conn.close()


def test_delete_is_rejected_by_trigger(tmp_path: Path) -> None:
    # Arrange
    db = tmp_path / "exec.db"
    log = SqliteExecutionLog(db)
    log.append(_event())
    log.close()

    # Act / Assert — a raw DELETE must be aborted by SQLite.
    conn = sqlite3.connect(db)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM execution_events")
            conn.commit()
    finally:
        conn.close()


def test_history_survives_an_edit_attempt(tmp_path: Path) -> None:
    # Arrange
    db = tmp_path / "exec.db"
    log = SqliteExecutionLog(db)
    log.append(_event())
    log.close()

    # Act — try (and fail) to tamper, then re-open and read.
    conn = sqlite3.connect(db)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE execution_events SET ticker = 'HACK'")
    finally:
        conn.close()
    reopened = SqliteExecutionLog(db)
    events = reopened.list_events()
    reopened.close()

    # Assert — the original row is intact.
    assert events[0].ticker == "AAPL"


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    # Arrange / Act — opening the same DB twice must not error or duplicate schema.
    db = tmp_path / "exec.db"
    SqliteExecutionLog(db).close()
    log = SqliteExecutionLog(db)
    log.append(_event())
    count = log.count()
    log.close()

    # Assert
    assert count == 1


# -- kill switch ----------------------------------------------------------


def test_kill_switch_clear_by_default(tmp_path: Path) -> None:
    # Arrange / Act
    switch = KillSwitch(halt_path=tmp_path / "HALT", env={})

    # Assert
    assert not switch.engaged()
    assert "clear" in switch.state().reason()


def test_kill_switch_engaged_by_file(tmp_path: Path) -> None:
    # Arrange
    halt = tmp_path / "HALT"
    halt.write_text("", encoding="utf-8")

    # Act
    switch = KillSwitch(halt_path=halt, env={})

    # Assert
    assert switch.engaged()
    assert "halt file" in switch.state().reason()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "STOP"])
def test_kill_switch_engaged_by_truthy_env(tmp_path: Path, value: str) -> None:
    # Arrange / Act
    switch = KillSwitch(halt_path=tmp_path / "HALT", env={HALT_ENV_VAR: value})

    # Assert
    assert switch.engaged()


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_kill_switch_falsey_env_is_clear(tmp_path: Path, value: str) -> None:
    # Arrange / Act
    switch = KillSwitch(halt_path=tmp_path / "HALT", env={HALT_ENV_VAR: value})

    # Assert
    assert not switch.engaged()


def test_kill_switch_reads_fresh_each_time(tmp_path: Path) -> None:
    # Arrange — switch starts clear.
    halt = tmp_path / "HALT"
    switch = KillSwitch(halt_path=halt, env={})
    assert not switch.engaged()

    # Act — trip it after construction.
    halt.write_text("", encoding="utf-8")

    # Assert — the very next check sees it, nothing was cached.
    assert switch.engaged()
