"""SQLite persistence for the AlphaResearch weekly committee (`alpharesearch.db`).

Persists:
1. ``source_watermarks``: last processed commit SHA / paper ID / postmortem token
   per source so weekly runs are incremental.
2. ``proposal_decisions``: immutable audit log of every :class:`CommitteeDecision`
   evaluated (both promoted and rejected), powering the Graveyard anti-snooping
   memory (:meth:`SQLiteAlphaResearchStore.rejected_proposal_ids`) and the
   cumulative trial counter (:meth:`SQLiteAlphaResearchStore.total_trials_count`).
3. ``research_runs``: full JSON-serialized :class:`AlphaResearchRunReport` records
   queryable via ``alphalab alpharesearch history``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from alphalab.domain.alpharesearch.contracts import (
    AlphaResearchRunReport,
    CommitteeDecision,
    ProposalDecisionStatus,
)

__all__ = ["SQLiteAlphaResearchStore"]

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS source_watermarks (
    source_key TEXT PRIMARY KEY,
    watermark_token TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposal_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    as_of TEXT NOT NULL,
    title TEXT NOT NULL,
    subsystem TEXT NOT NULL,
    status TEXT NOT NULL,
    summary_reason TEXT NOT NULL,
    dsr REAL,
    pbo REAL,
    sharpe_delta REAL,
    n_trials_total INTEGER NOT NULL,
    recorded_at TEXT NOT NULL,
    decision_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_proposal_decisions_pid
    ON proposal_decisions(proposal_id);

CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY,
    as_of TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    n_signals INTEGER NOT NULL,
    n_proposals INTEGER NOT NULL,
    n_promoted INTEGER NOT NULL,
    n_rejected INTEGER NOT NULL,
    n_trials_after INTEGER NOT NULL,
    plan_output_path TEXT,
    report_json TEXT NOT NULL
);
"""


class SQLiteAlphaResearchStore:
    """SQLite store backing weekly AlphaResearch runs and anti-snooping memory."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def load_watermarks(self) -> dict[str, str]:
        """Return ``{source_key: watermark_token}`` from prior weekly runs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_key, watermark_token FROM source_watermarks"
            ).fetchall()
        return {str(r["source_key"]): str(r["watermark_token"]) for r in rows}

    def update_watermarks(self, updates: Mapping[str, str]) -> None:
        """Upsert source watermarks after a successful scan."""
        if not updates:
            return
        now_iso = datetime.now(tz=UTC).isoformat()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO source_watermarks (source_key, watermark_token, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    watermark_token = excluded.watermark_token,
                    updated_at = excluded.updated_at
                """,
                [(k, v, now_iso) for k, v in updates.items()],
            )

    def rejected_proposal_ids(self) -> frozenset[str]:
        """Return all ``proposal_id`` hashes previously rejected by the committee."""
        rejected_statuses = (
            ProposalDecisionStatus.REJECTED_BY_SKEPTIC.value,
            ProposalDecisionStatus.REJECTED_BY_EMPIRICAL_GATE.value,
        )
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT proposal_id FROM proposal_decisions WHERE status IN (?, ?)",
                rejected_statuses,
            ).fetchall()
        return frozenset(str(r["proposal_id"]) for r in rows)

    def promoted_proposal_ids(self) -> frozenset[str]:
        """Return all ``proposal_id`` hashes previously promoted to a weekly plan."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT proposal_id FROM proposal_decisions WHERE status = ?",
                (ProposalDecisionStatus.PROMOTED_TO_PLAN.value,),
            ).fetchall()
        return frozenset(str(r["proposal_id"]) for r in rows)

    def evaluated_proposal_ids(self) -> frozenset[str]:
        """Return all ``proposal_id`` hashes previously evaluated by the committee."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT proposal_id FROM proposal_decisions"
            ).fetchall()
        return frozenset(str(r["proposal_id"]) for r in rows)

    def total_trials_count(self) -> int:
        """Return cumulative number of proposals evaluated across all runs."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM proposal_decisions").fetchone()
        return int(row["cnt"] if row else 0)

    def record_run(self, report: AlphaResearchRunReport) -> None:
        """Persist a complete weekly run report and all its proposal decisions."""
        now_iso = datetime.now(tz=UTC).isoformat()
        with self._connect() as conn:
            for d in report.decisions:
                sc = d.empirical_scorecard
                conn.execute(
                    """
                    INSERT INTO proposal_decisions (
                        proposal_id, run_id, as_of, title, subsystem, status,
                        summary_reason, dsr, pbo, sharpe_delta, n_trials_total,
                        recorded_at, decision_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        d.proposal.proposal_id,
                        report.run_id,
                        report.as_of.isoformat(),
                        d.proposal.title,
                        d.proposal.subsystem.value,
                        d.status.value,
                        d.summary_reason,
                        sc.dsr,
                        sc.pbo,
                        sc.sharpe_delta,
                        sc.n_trials_total,
                        now_iso,
                        d.model_dump_json(),
                    ),
                )
            conn.execute(
                """
                INSERT OR REPLACE INTO research_runs (
                    run_id, as_of, started_at, finished_at,
                    n_signals, n_proposals, n_promoted, n_rejected,
                    n_trials_after, plan_output_path, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_id,
                    report.as_of.isoformat(),
                    report.started_at.isoformat(),
                    report.finished_at.isoformat(),
                    len(report.signals_discovered),
                    len(report.decisions),
                    len(report.promoted_decisions),
                    len(report.rejected_decisions),
                    report.n_trials_after,
                    report.plan_output_path,
                    report.model_dump_json(),
                ),
            )

    def list_runs(self, *, limit: int = 10) -> tuple[AlphaResearchRunReport, ...]:
        """Return the most recent :class:`AlphaResearchRunReport` records."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT report_json FROM research_runs
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return tuple(
            AlphaResearchRunReport.model_validate_json(str(r["report_json"])) for r in rows
        )

    def load_decision(self, proposal_id: str) -> CommitteeDecision | None:
        """Load the latest decision for ``proposal_id``, if any."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT decision_json FROM proposal_decisions
                WHERE proposal_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (proposal_id,),
            ).fetchone()
        if row is None:
            return None
        return CommitteeDecision.model_validate_json(str(row["decision_json"]))
