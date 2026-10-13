"""SQLite persistence for the prediction journal.

This adapter is the I/O boundary for the journal. It is deliberately kept out
of the pure domain layer (which forbids ``sqlite3``) and holds every emitted
prediction, its settlement at the horizon, and the reflection lesson.

Design guarantees:

* **Partial immutability, enforced by the engine.** Unlike the trial ledger,
  the journal is *not* append-only: a prediction transitions once from
  not-settled to settled. But a settled prediction must never be re-settled
  nor have its original ``probability`` rewritten - that would destroy the
  calibration record. We model "settled" as the *existence of a settlement
  row* rather than a mutable flag on the prediction, so:

  - the ``predictions`` row is written once and a ``BEFORE UPDATE`` trigger
    ``RAISE(ABORT)`` rejects any change to its immutable core (``probability``
    above all), and a ``BEFORE DELETE`` trigger rejects deletion;
  - ``settlements`` has ``prediction_id`` as its PRIMARY KEY, so a second
    settlement insert for the same prediction fails, and ``BEFORE UPDATE`` /
    ``BEFORE DELETE`` triggers reject re-liquidation and tampering.

* **Versioned schema.** A ``schema_version`` table records the applied
  version; :meth:`_migrate` is idempotent (``IF NOT EXISTS`` plus a version
  check) so running it twice is a no-op.
* **Parameterised queries only.** Every value is bound through ``?``
  placeholders - never string-formatted.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

from alphalab.domain.journal import (
    Direction,
    Lesson,
    OutcomeGrade,
    Prediction,
    ProcessGrade,
    Settlement,
)

__all__ = ["SCHEMA_VERSION", "SqliteJournal"]

#: Current schema version. Bump when the physical schema changes and add a
#: corresponding idempotent migration step in :meth:`SqliteJournal._migrate`.
SCHEMA_VERSION = 1

_CREATE_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at TEXT    NOT NULL
)
"""

_CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id  TEXT NOT NULL PRIMARY KEY,
    recorded_at    TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    direction      TEXT NOT NULL,
    probability    REAL NOT NULL,
    horizon        TEXT NOT NULL,
    entry_price    REAL NOT NULL,
    knowledge_date TEXT NOT NULL,
    llm_version    TEXT,
    llm_cutoff     TEXT
)
"""

_CREATE_SETTLEMENTS = """
CREATE TABLE IF NOT EXISTS settlements (
    prediction_id    TEXT NOT NULL PRIMARY KEY,
    settled_at       TEXT NOT NULL,
    realized_return  REAL NOT NULL,
    benchmark_return REAL NOT NULL,
    alpha_gross      REAL NOT NULL,
    alpha_net        REAL NOT NULL,
    outcome          INTEGER NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
)
"""

_CREATE_LESSONS = """
CREATE TABLE IF NOT EXISTS lessons (
    prediction_id TEXT NOT NULL PRIMARY KEY,
    text          TEXT NOT NULL,
    process_grade TEXT NOT NULL,
    outcome_grade TEXT NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
)
"""

_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_horizon ON predictions(horizon)",
)

# Engine-level partial immutability.
_CREATE_TRIGGERS = (
    # A prediction's immutable core - probability above all - can never change.
    """
    CREATE TRIGGER IF NOT EXISTS trg_predictions_no_update
    BEFORE UPDATE ON predictions
    BEGIN
        SELECT RAISE(ABORT,
            'prediction is immutable: its probability and core fields cannot change');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_predictions_no_delete
    BEFORE DELETE ON predictions
    BEGIN
        SELECT RAISE(ABORT, 'prediction is immutable: DELETE forbidden');
    END
    """,
    # A settlement is written once; re-liquidation and tampering are rejected.
    """
    CREATE TRIGGER IF NOT EXISTS trg_settlements_no_update
    BEFORE UPDATE ON settlements
    BEGIN
        SELECT RAISE(ABORT, 'settled prediction cannot be re-settled: UPDATE forbidden');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_settlements_no_delete
    BEFORE DELETE ON settlements
    BEGIN
        SELECT RAISE(ABORT, 'settled prediction cannot be re-settled: DELETE forbidden');
    END
    """,
)

_INSERT_PREDICTION = """
INSERT INTO predictions (
    prediction_id, recorded_at, ticker, direction, probability,
    horizon, entry_price, knowledge_date, llm_version, llm_cutoff
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_SETTLEMENT = """
INSERT INTO settlements (
    prediction_id, settled_at, realized_return, benchmark_return,
    alpha_gross, alpha_net, outcome
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_LESSON = """
INSERT INTO lessons (prediction_id, text, process_grade, outcome_grade)
VALUES (?, ?, ?, ?)
"""

_PREDICTION_COLUMNS = """
    prediction_id, recorded_at, ticker, direction, probability,
    horizon, entry_price, knowledge_date, llm_version, llm_cutoff
"""

_SETTLEMENT_COLUMNS = """
    prediction_id, settled_at, realized_return, benchmark_return,
    alpha_gross, alpha_net, outcome
"""


class SqliteJournal:
    """SQLite-backed prediction journal with partial immutability."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> SqliteJournal:
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
        """Create the schema if absent and record the version.

        Idempotent: every DDL uses ``IF NOT EXISTS`` and the version row is
        only written when the stored version is behind.
        """
        with self._conn:
            self._conn.execute(_CREATE_SCHEMA_VERSION)
            if self._current_version() >= SCHEMA_VERSION:
                return
            self._conn.execute(_CREATE_PREDICTIONS)
            self._conn.execute(_CREATE_SETTLEMENTS)
            self._conn.execute(_CREATE_LESSONS)
            for ddl in _CREATE_INDEXES:
                self._conn.execute(ddl)
            for ddl in _CREATE_TRIGGERS:
                self._conn.execute(ddl)
            self._conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, datetime.now().isoformat()),
            )

    # -- writes -----------------------------------------------------------

    def record(self, prediction: Prediction) -> str:
        """Persist a new ``prediction`` and return its id."""
        with self._conn:
            self._conn.execute(
                _INSERT_PREDICTION,
                (
                    prediction.prediction_id,
                    prediction.recorded_at.isoformat(),
                    prediction.ticker,
                    prediction.direction.value,
                    prediction.probability,
                    prediction.horizon.isoformat(),
                    prediction.entry_price,
                    prediction.knowledge_date.isoformat(),
                    prediction.llm_version,
                    prediction.llm_cutoff.isoformat()
                    if prediction.llm_cutoff is not None
                    else None,
                ),
            )
        return prediction.prediction_id

    def settle(self, settlement: Settlement) -> None:
        """Persist a ``settlement``, liquidating a prediction exactly once.

        A second attempt to settle the same prediction raises
        ``sqlite3.IntegrityError`` (PRIMARY KEY), so re-liquidation is
        impossible even via this API.
        """
        with self._conn:
            self._conn.execute(
                _INSERT_SETTLEMENT,
                (
                    settlement.prediction_id,
                    settlement.settled_at.isoformat(),
                    settlement.realized_return,
                    settlement.benchmark_return,
                    settlement.alpha_gross,
                    settlement.alpha_net,
                    1 if settlement.outcome else 0,
                ),
            )

    def add_lesson(self, lesson: Lesson) -> None:
        """Persist the reflection ``lesson`` for a settled prediction."""
        with self._conn:
            self._conn.execute(
                _INSERT_LESSON,
                (
                    lesson.prediction_id,
                    lesson.text,
                    lesson.process_grade.value,
                    lesson.outcome_grade.value,
                ),
            )

    # -- reads ------------------------------------------------------------

    def get_prediction(self, prediction_id: str) -> Prediction | None:
        """Return the prediction with ``prediction_id`` or ``None``."""
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                f"SELECT {_PREDICTION_COLUMNS} FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            )
            row = cur.fetchone()
        return _row_to_prediction(row) if row is not None else None

    def get_settlement(self, prediction_id: str) -> Settlement | None:
        """Return the settlement for ``prediction_id`` or ``None`` if unsettled."""
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                f"SELECT {_SETTLEMENT_COLUMNS} FROM settlements WHERE prediction_id = ?",
                (prediction_id,),
            )
            row = cur.fetchone()
        return _row_to_settlement(row) if row is not None else None

    def get_lesson(self, prediction_id: str) -> Lesson | None:
        """Return the lesson for ``prediction_id`` or ``None`` if none exists."""
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "SELECT prediction_id, text, process_grade, outcome_grade "
                "FROM lessons WHERE prediction_id = ?",
                (prediction_id,),
            )
            row = cur.fetchone()
        return _row_to_lesson(row) if row is not None else None

    def settled_ids(self) -> list[str]:
        """Return the ids of every prediction that has been settled."""
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT prediction_id FROM settlements")
            rows = cur.fetchall()
        return [row["prediction_id"] for row in rows]

    def list_predictions(self) -> list[Prediction]:
        """Return every prediction ordered by insertion time."""
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                f"SELECT {_PREDICTION_COLUMNS} FROM predictions ORDER BY recorded_at, prediction_id"
            )
            rows = cur.fetchall()
        return [_row_to_prediction(row) for row in rows]

    def list_settlements(self) -> list[Settlement]:
        """Return every settlement ordered by settlement time."""
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                f"SELECT {_SETTLEMENT_COLUMNS} FROM settlements ORDER BY settled_at, prediction_id"
            )
            rows = cur.fetchall()
        return [_row_to_settlement(row) for row in rows]

    def __iter__(self) -> Iterator[Prediction]:
        return iter(self.list_predictions())


def _row_to_prediction(row: sqlite3.Row) -> Prediction:
    """Rebuild a :class:`Prediction` from a database row."""
    cutoff = row["llm_cutoff"]
    return Prediction(
        prediction_id=row["prediction_id"],
        recorded_at=datetime.fromisoformat(row["recorded_at"]),
        ticker=row["ticker"],
        direction=Direction(row["direction"]),
        probability=row["probability"],
        horizon=date.fromisoformat(row["horizon"]),
        entry_price=row["entry_price"],
        knowledge_date=date.fromisoformat(row["knowledge_date"]),
        llm_version=row["llm_version"],
        llm_cutoff=date.fromisoformat(cutoff) if cutoff is not None else None,
    )


def _row_to_settlement(row: sqlite3.Row) -> Settlement:
    """Rebuild a :class:`Settlement` from a database row."""
    return Settlement(
        prediction_id=row["prediction_id"],
        settled_at=datetime.fromisoformat(row["settled_at"]),
        realized_return=row["realized_return"],
        benchmark_return=row["benchmark_return"],
        alpha_gross=row["alpha_gross"],
        alpha_net=row["alpha_net"],
        outcome=bool(row["outcome"]),
    )


def _row_to_lesson(row: sqlite3.Row) -> Lesson:
    """Rebuild a :class:`Lesson` from a database row."""
    return Lesson(
        prediction_id=row["prediction_id"],
        text=row["text"],
        process_grade=ProcessGrade(row["process_grade"]),
        outcome_grade=OutcomeGrade(row["outcome_grade"]),
    )
