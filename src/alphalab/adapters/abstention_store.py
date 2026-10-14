"""Durable record of the predictions the system *declined* to make.

Why abstentions get their own store
-----------------------------------
Calibration is scored only over claims that were actually made. But if the
abstentions were simply dropped, "the system declined 40% of the time, for
these reasons" would vanish from the record — and the surviving hit rate would
look better than the process earned. Hiding selectivity flatters the track
record exactly where honesty matters most, so abstentions are persisted as a
first-class finding, not discarded.

Why a separate database
------------------------
The journal (:mod:`alphalab.adapters.journal_sqlite`) is append-only and
guarded by immutability triggers; its schema is a fixed contract this module
must not touch. Abstentions are a different kind of record — they carry no
probability and are never settled — so they live in their own small SQLite
file. Keeping them apart means this store can evolve without any risk to the
journal's integrity guarantees.

This is an *adapter*: I/O lives here, never in the domain layer.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from types import TracebackType

from alphalab.domain.forecast import Abstention

__all__ = ["SqliteAbstentionStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS abstentions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker         TEXT NOT NULL,
    knowledge_date TEXT NOT NULL,
    reason         TEXT NOT NULL
);
"""


class SqliteAbstentionStore:
    """Append-only SQLite store for :class:`Abstention` records.

    The table is intentionally minimal: an abstention is a stated refusal, not
    a scoreable claim, so it needs only the ticker, the knowledge date it
    refers to, and the reason. Duplicates are allowed — the same name can be
    declined on different runs, and that repetition is itself signal.
    """

    __slots__ = ("_conn",)

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> SqliteAbstentionStore:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.close()

    def record(self, abstention: Abstention) -> None:
        """Persist one abstention. Repeated abstentions are retained verbatim."""
        self._conn.execute(
            "INSERT INTO abstentions (ticker, knowledge_date, reason) VALUES (?, ?, ?)",
            (
                abstention.ticker,
                abstention.knowledge_date.isoformat(),
                abstention.reason,
            ),
        )
        self._conn.commit()

    def list_abstentions(self) -> list[Abstention]:
        """Return every recorded abstention, oldest insertion first."""
        rows = self._conn.execute(
            "SELECT ticker, knowledge_date, reason FROM abstentions ORDER BY id",
        ).fetchall()
        return [self._row_to_abstention(row) for row in rows]

    def count(self) -> int:
        """Return the number of abstentions on record."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM abstentions").fetchone()
        return int(row["n"])

    def __iter__(self) -> Iterator[Abstention]:
        return iter(self.list_abstentions())

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    @staticmethod
    def _row_to_abstention(row: sqlite3.Row) -> Abstention:
        return Abstention(
            ticker=row["ticker"],
            knowledge_date=date.fromisoformat(row["knowledge_date"]),
            reason=row["reason"],
        )
