"""Append-only SQLite execution log for the paper-trading loop.

Why append-only, enforced by the engine
----------------------------------------
An execution history that can be edited is not a history: if a row can be
changed after the fact, no later audit can trust any row. So immutability is
enforced the same way the trial ledger and the prediction journal enforce it —
``BEFORE UPDATE`` / ``BEFORE DELETE`` triggers that ``RAISE(ABORT)``. Even a raw
``UPDATE`` issued outside this class is rejected by SQLite itself, not merely
discouraged by a missing method.

What is recorded
----------------
Every event in one cycle, each as its own immutable row with a timestamp and a
``cycle_id`` that ties a run together:

* ``intent`` — an order the planner proposed, before any submission.
* ``submitted`` / ``duplicate`` / ``halted`` / ``failed`` — the outcome of
  trying to send it (mirroring
  :class:`~alphalab.adapters.alpaca_broker.OrderStatus`).
* ``rejected`` — an order a risk rule refused, with the reason.

Every value is bound through ``?`` placeholders; nothing is string-formatted
into SQL, per the project's anti-injection guideline. This adapter owns its own
database file and never touches the journal or ledger schemas.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

__all__ = ["SCHEMA_VERSION", "ExecutionEvent", "ExecutionEventKind", "SqliteExecutionLog"]

#: Bump when the physical schema changes; add an idempotent migration step.
SCHEMA_VERSION = 2


class ExecutionEventKind(StrEnum):
    """What kind of event a log row records."""

    INTENT = "intent"
    SUBMITTED = "submitted"
    FILLED = "filled"
    DUPLICATE = "duplicate"
    HALTED = "halted"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExecutionEvent:
    """One immutable row in the execution log."""

    cycle_id: str
    recorded_at: datetime
    kind: ExecutionEventKind
    ticker: str
    side: str | None
    shares: float | None
    reference_price: float | None
    client_order_id: str | None
    broker_order_id: str | None
    detail: str | None


_CREATE_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at TEXT    NOT NULL
)
"""

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS execution_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id        TEXT    NOT NULL,
    recorded_at     TEXT    NOT NULL,
    kind            TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    side            TEXT,
    shares          REAL,
    reference_price REAL,
    client_order_id TEXT,
    broker_order_id TEXT,
    detail          TEXT
)
"""

_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_events_cycle ON execution_events(cycle_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_recorded_at ON execution_events(recorded_at)",
    (
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_unique_fill "
        "ON execution_events(broker_order_id, kind) "
        "WHERE kind = 'filled' AND broker_order_id IS NOT NULL"
    ),
)

# Engine-level immutability: block UPDATE and DELETE on the events table.
_CREATE_TRIGGERS = (
    """
    CREATE TRIGGER IF NOT EXISTS trg_events_no_update
    BEFORE UPDATE ON execution_events
    BEGIN
        SELECT RAISE(ABORT, 'execution log is append-only: UPDATE forbidden');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_events_no_delete
    BEFORE DELETE ON execution_events
    BEGIN
        SELECT RAISE(ABORT, 'execution log is append-only: DELETE forbidden');
    END
    """,
)

_INSERT_EVENT = """
INSERT INTO execution_events (
    cycle_id, recorded_at, kind, ticker, side, shares,
    reference_price, client_order_id, broker_order_id, detail
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_COLUMNS = """
    cycle_id, recorded_at, kind, ticker, side, shares,
    reference_price, client_order_id, broker_order_id, detail
"""


class SqliteExecutionLog:
    """Append-only execution log backed by SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> SqliteExecutionLog:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- schema -----------------------------------------------------------

    def _current_version(self) -> int:
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT MAX(version) AS v FROM schema_version")
            row = cur.fetchone()
        if row is None or row["v"] is None:
            return 0
        return int(row["v"])

    def _migrate(self) -> None:
        """Create the schema if absent; idempotent on repeat runs."""
        with self._conn:
            self._conn.execute(_CREATE_SCHEMA_VERSION)
            if self._current_version() >= SCHEMA_VERSION:
                return
            self._conn.execute(_CREATE_EVENTS)
            for ddl in _CREATE_INDEXES:
                self._conn.execute(ddl)
            for ddl in _CREATE_TRIGGERS:
                self._conn.execute(ddl)
            self._conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, datetime.now().isoformat()),
            )

    # -- writes -----------------------------------------------------------

    def append(self, event: ExecutionEvent) -> None:
        """Persist one event. The only mutating operation the log exposes."""
        with self._conn:
            self._conn.execute(
                _INSERT_EVENT,
                (
                    event.cycle_id,
                    event.recorded_at.isoformat(),
                    event.kind.value,
                    event.ticker,
                    event.side,
                    event.shares,
                    event.reference_price,
                    event.client_order_id,
                    event.broker_order_id,
                    event.detail,
                ),
            )

    def append_fill_if_new(self, event: ExecutionEvent) -> bool:
        """Persist a FILLED event idempotently by ``broker_order_id``.

        Returns ``True`` when the fill was newly inserted, or ``False`` if a
        fill with the same ``broker_order_id`` was already recorded.
        """
        if event.kind is not ExecutionEventKind.FILLED:
            raise ValueError("append_fill_if_new requires ExecutionEventKind.FILLED")
        with self._conn, closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                INSERT OR IGNORE INTO execution_events (
                    cycle_id, recorded_at, kind, ticker, side, shares,
                    reference_price, client_order_id, broker_order_id, detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.cycle_id,
                    event.recorded_at.isoformat(),
                    event.kind.value,
                    event.ticker,
                    event.side,
                    event.shares,
                    event.reference_price,
                    event.client_order_id,
                    event.broker_order_id,
                    event.detail,
                ),
            )
            return bool(cur.rowcount and cur.rowcount > 0)

    # -- reads ------------------------------------------------------------

    def count(self) -> int:
        """Total number of events recorded."""
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT COUNT(*) AS n FROM execution_events")
            row = cur.fetchone()
        return int(row["n"])

    def list_events(self, *, cycle_id: str | None = None) -> list[ExecutionEvent]:
        """Return events in insertion order, optionally for one cycle."""
        with closing(self._conn.cursor()) as cur:
            if cycle_id is None:
                cur.execute(f"SELECT {_SELECT_COLUMNS} FROM execution_events ORDER BY event_id")
            else:
                cur.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM execution_events "
                    "WHERE cycle_id = ? ORDER BY event_id",
                    (cycle_id,),
                )
            rows = cur.fetchall()
        return [_row_to_event(row) for row in rows]

    def __iter__(self) -> Iterator[ExecutionEvent]:
        return iter(self.list_events())


def _row_to_event(row: sqlite3.Row) -> ExecutionEvent:
    """Rebuild an :class:`ExecutionEvent` from a database row."""
    return ExecutionEvent(
        cycle_id=row["cycle_id"],
        recorded_at=datetime.fromisoformat(row["recorded_at"]),
        kind=ExecutionEventKind(row["kind"]),
        ticker=row["ticker"],
        side=row["side"],
        shares=row["shares"],
        reference_price=row["reference_price"],
        client_order_id=row["client_order_id"],
        broker_order_id=row["broker_order_id"],
        detail=row["detail"],
    )
