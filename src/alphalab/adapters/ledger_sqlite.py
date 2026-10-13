"""Append-only SQLite persistence for the trial ledger.

This adapter is the I/O boundary for the trial ledger. It is deliberately kept
out of the pure domain layer (which forbids ``sqlite3``) and holds every
backtest trial so the Deflated Sharpe Ratio sees an honest trial count.

Design guarantees:

* **Append-only by construction.** The public API exposes ``append`` and
  reads only — no ``update`` or ``delete``. Defence in depth: SQLite
  ``BEFORE UPDATE`` / ``BEFORE DELETE`` triggers ``RAISE(ABORT, ...)`` so even
  a raw ``UPDATE``/``DELETE`` issued outside this class is rejected by the
  engine.
* **Versioned schema.** A ``schema_version`` table records the applied
  version; :meth:`_migrate` is idempotent (safe to run repeatedly) and uses
  ``CREATE TABLE IF NOT EXISTS`` plus a version check.
* **Parameterised queries only.** Every value is bound through ``?``
  placeholders — never string-formatted — per the anti-injection guideline.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from typing import Any

from alphalab.domain.ledger import TrialMetrics, TrialRecord

__all__ = ["SCHEMA_VERSION", "SqliteTrialLedger"]

#: Current schema version. Bump when the physical schema changes and add a
#: corresponding idempotent migration step in :meth:`SqliteTrialLedger._migrate`.
SCHEMA_VERSION = 1

_CREATE_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at TEXT    NOT NULL
)
"""

_CREATE_TRIALS = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id     TEXT    NOT NULL PRIMARY KEY,
    recorded_at  TEXT    NOT NULL,
    spec_hash    TEXT    NOT NULL,
    dataset_hash TEXT    NOT NULL,
    family       TEXT    NOT NULL,
    window_start TEXT    NOT NULL,
    window_end   TEXT    NOT NULL,
    params_json  TEXT    NOT NULL,
    sharpe_gross REAL    NOT NULL,
    sharpe_net   REAL    NOT NULL,
    return_gross REAL    NOT NULL,
    return_net   REAL    NOT NULL,
    max_drawdown REAL    NOT NULL,
    llm_version  TEXT,
    llm_cutoff   TEXT
)
"""

_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_trials_family ON trials(family)",
    "CREATE INDEX IF NOT EXISTS idx_trials_recorded_at ON trials(recorded_at)",
)

# Engine-level immutability: block UPDATE and DELETE on the trials table.
_CREATE_TRIGGERS = (
    """
    CREATE TRIGGER IF NOT EXISTS trg_trials_no_update
    BEFORE UPDATE ON trials
    BEGIN
        SELECT RAISE(ABORT, 'trial ledger is append-only: UPDATE forbidden');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_trials_no_delete
    BEFORE DELETE ON trials
    BEGIN
        SELECT RAISE(ABORT, 'trial ledger is append-only: DELETE forbidden');
    END
    """,
)

_INSERT_TRIAL = """
INSERT INTO trials (
    trial_id, recorded_at, spec_hash, dataset_hash, family,
    window_start, window_end, params_json,
    sharpe_gross, sharpe_net, return_gross, return_net, max_drawdown,
    llm_version, llm_cutoff
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_COLUMNS = """
    trial_id, recorded_at, spec_hash, dataset_hash, family,
    window_start, window_end, params_json,
    sharpe_gross, sharpe_net, return_gross, return_net, max_drawdown,
    llm_version, llm_cutoff
"""


class SqliteTrialLedger:
    """Append-only trial ledger backed by SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> SqliteTrialLedger:
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
        only written when the stored version is behind, so running this twice
        is a no-op on the second pass.
        """
        with self._conn:
            self._conn.execute(_CREATE_SCHEMA_VERSION)
            if self._current_version() >= SCHEMA_VERSION:
                return
            self._conn.execute(_CREATE_TRIALS)
            for ddl in _CREATE_INDEXES:
                self._conn.execute(ddl)
            for ddl in _CREATE_TRIGGERS:
                self._conn.execute(ddl)
            self._conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, datetime.now().isoformat()),
            )

    # -- writes -----------------------------------------------------------

    def append(self, record: TrialRecord) -> str:
        """Persist ``record`` and return its ``trial_id``.

        This is the only mutating operation the ledger exposes.
        """
        params_json = json.dumps(dict(record.params), sort_keys=True, separators=(",", ":"))
        with self._conn:
            self._conn.execute(
                _INSERT_TRIAL,
                (
                    record.trial_id,
                    record.recorded_at.isoformat(),
                    record.spec_hash,
                    record.dataset_hash,
                    record.family,
                    record.window_start.isoformat(),
                    record.window_end.isoformat(),
                    params_json,
                    record.metrics.sharpe_gross,
                    record.metrics.sharpe_net,
                    record.metrics.return_gross,
                    record.metrics.return_net,
                    record.metrics.max_drawdown,
                    record.llm_version,
                    record.llm_cutoff.isoformat() if record.llm_cutoff is not None else None,
                ),
            )
        return record.trial_id

    # -- counts -----------------------------------------------------------

    def trial_count(self, family: str | None = None) -> int:
        """Count trials, optionally restricted to one ``family``.

        With ``family`` this is the ``N`` that feeds the DSR's ``SR0`` for a
        strategy family. Without it, the grand total across all families.
        """
        with closing(self._conn.cursor()) as cur:
            if family is None:
                cur.execute("SELECT COUNT(*) AS n FROM trials")
            else:
                cur.execute("SELECT COUNT(*) AS n FROM trials WHERE family = ?", (family,))
            row = cur.fetchone()
        return int(row["n"])

    def distinct_spec_count(self, family: str | None = None) -> int:
        """Count distinct specifications (unique ``spec_hash`` values).

        This is deliberately *different* from :meth:`trial_count`: the DSR
        counts executions, but knowing how many unique specs produced them is
        useful signal about how much the search space was actually explored.
        """
        with closing(self._conn.cursor()) as cur:
            if family is None:
                cur.execute("SELECT COUNT(DISTINCT spec_hash) AS n FROM trials")
            else:
                cur.execute(
                    "SELECT COUNT(DISTINCT spec_hash) AS n FROM trials WHERE family = ?",
                    (family,),
                )
            row = cur.fetchone()
        return int(row["n"])

    # -- reads ------------------------------------------------------------

    def get(self, trial_id: str) -> TrialRecord | None:
        """Return the trial with ``trial_id`` or ``None`` if absent."""
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                f"SELECT {_SELECT_COLUMNS} FROM trials WHERE trial_id = ?",
                (trial_id,),
            )
            row = cur.fetchone()
        return _row_to_record(row) if row is not None else None

    def list_by_family(self, family: str) -> list[TrialRecord]:
        """Return all trials in ``family`` ordered by insertion time."""
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                f"SELECT {_SELECT_COLUMNS} FROM trials WHERE family = ? "
                "ORDER BY recorded_at, trial_id",
                (family,),
            )
            rows = cur.fetchall()
        return [_row_to_record(row) for row in rows]

    def list_by_window(self, start: date, end: date) -> list[TrialRecord]:
        """Return trials whose ``window_start`` falls within ``[start, end]``."""
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                f"SELECT {_SELECT_COLUMNS} FROM trials "
                "WHERE window_start >= ? AND window_start <= ? "
                "ORDER BY window_start, trial_id",
                (start.isoformat(), end.isoformat()),
            )
            rows = cur.fetchall()
        return [_row_to_record(row) for row in rows]

    def __iter__(self) -> Iterator[TrialRecord]:
        with closing(self._conn.cursor()) as cur:
            cur.execute(f"SELECT {_SELECT_COLUMNS} FROM trials ORDER BY recorded_at, trial_id")
            rows = cur.fetchall()
        return iter(_row_to_record(row) for row in rows)


def _row_to_record(row: sqlite3.Row) -> TrialRecord:
    """Rebuild a :class:`TrialRecord` from a database row."""
    params: Mapping[str, Any] = json.loads(row["params_json"])
    cutoff = row["llm_cutoff"]
    return TrialRecord(
        trial_id=row["trial_id"],
        recorded_at=datetime.fromisoformat(row["recorded_at"]),
        spec_hash=row["spec_hash"],
        dataset_hash=row["dataset_hash"],
        family=row["family"],
        window_start=date.fromisoformat(row["window_start"]),
        window_end=date.fromisoformat(row["window_end"]),
        params=params,
        metrics=TrialMetrics(
            sharpe_gross=row["sharpe_gross"],
            sharpe_net=row["sharpe_net"],
            return_gross=row["return_gross"],
            return_net=row["return_net"],
            max_drawdown=row["max_drawdown"],
        ),
        llm_version=row["llm_version"],
        llm_cutoff=date.fromisoformat(cutoff) if cutoff is not None else None,
    )
