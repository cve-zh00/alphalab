"""Unit tests for :class:`alphalab.adapters.journal_sqlite.SqliteJournal`."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from alphalab.adapters.journal_sqlite import SCHEMA_VERSION, SqliteJournal
from alphalab.domain.journal import (
    Direction,
    OutcomeGrade,
    Prediction,
    ProcessGrade,
    Settlement,
    TemplateReflector,
)


def _prediction(*, pid: str = "p1", direction: Direction = Direction.UP) -> Prediction:
    return Prediction(
        prediction_id=pid,
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
        ticker="AAPL",
        direction=direction,
        probability=0.7,
        horizon=date(2026, 2, 1),
        entry_price=150.0,
        knowledge_date=date(2026, 1, 1),
        llm_version="frontier-2026-09",
        llm_cutoff=date(2026, 9, 1),
    )


def _settlement(pid: str = "p1", *, outcome: bool = True) -> Settlement:
    return Settlement(
        prediction_id=pid,
        settled_at=datetime(2026, 2, 1, tzinfo=UTC),
        realized_return=0.1,
        benchmark_return=0.02,
        alpha_gross=0.08,
        alpha_net=0.07,
        outcome=outcome,
    )


@pytest.fixture
def journal(tmp_path: Path) -> Iterator[SqliteJournal]:
    with SqliteJournal(tmp_path / "journal.db") as j:
        yield j


class TestRecordAndRead:
    def test_record_roundtrips(self, journal: SqliteJournal) -> None:
        # Arrange
        pred = _prediction()

        # Act
        journal.record(pred)
        loaded = journal.get_prediction("p1")

        # Assert
        assert loaded is not None
        assert loaded.probability == pytest.approx(0.7)
        assert loaded.direction is Direction.UP
        assert loaded.llm_version == "frontier-2026-09"
        assert loaded.llm_cutoff == date(2026, 9, 1)

    def test_get_missing_returns_none(self, journal: SqliteJournal) -> None:
        assert journal.get_prediction("missing") is None

    def test_settle_roundtrips(self, journal: SqliteJournal) -> None:
        # Arrange
        journal.record(_prediction())

        # Act
        journal.settle(_settlement())
        loaded = journal.get_settlement("p1")

        # Assert
        assert loaded is not None
        assert loaded.outcome is True
        assert loaded.alpha_net == pytest.approx(0.07)

    def test_lesson_roundtrips(self, journal: SqliteJournal) -> None:
        # Arrange
        pred = _prediction()
        journal.record(pred)
        settlement = _settlement()
        journal.settle(settlement)
        lesson = TemplateReflector().reflect(pred, settlement)

        # Act
        journal.add_lesson(lesson)
        loaded = journal.get_lesson("p1")

        # Assert
        assert loaded is not None
        assert loaded.process_grade in set(ProcessGrade)
        assert loaded.outcome_grade is OutcomeGrade.WIN

    def test_settled_ids_and_lists(self, journal: SqliteJournal) -> None:
        # Arrange
        journal.record(_prediction(pid="a"))
        journal.record(_prediction(pid="b"))
        journal.settle(_settlement("a"))

        # Act / Assert
        assert journal.settled_ids() == ["a"]
        assert {p.prediction_id for p in journal.list_predictions()} == {"a", "b"}
        assert [s.prediction_id for s in journal.list_settlements()] == ["a"]
        assert {p.prediction_id for p in journal} == {"a", "b"}


class TestNoReliquidation:
    """A settled prediction cannot be re-settled nor have probability mutated."""

    def test_double_settle_via_api_raises(self, journal: SqliteJournal) -> None:
        # Arrange
        journal.record(_prediction())
        journal.settle(_settlement())

        # Act / Assert: a second settlement (even with different numbers) fails.
        with pytest.raises(sqlite3.IntegrityError):
            journal.settle(_settlement(outcome=False))

    def test_raw_update_settlement_aborts(self, tmp_path: Path) -> None:
        # Arrange
        db = tmp_path / "j.db"
        with SqliteJournal(db) as j:
            j.record(_prediction())
            j.settle(_settlement())

        # Act / Assert: raw SQL cannot flip the outcome.
        raw = sqlite3.connect(str(db))
        try:
            with pytest.raises(sqlite3.IntegrityError, match="re-settled"):
                raw.execute("UPDATE settlements SET outcome = 0 WHERE prediction_id = 'p1'")
                raw.commit()
        finally:
            raw.close()

    def test_raw_delete_settlement_aborts(self, tmp_path: Path) -> None:
        # Arrange
        db = tmp_path / "j.db"
        with SqliteJournal(db) as j:
            j.record(_prediction())
            j.settle(_settlement())

        # Act / Assert
        raw = sqlite3.connect(str(db))
        try:
            with pytest.raises(sqlite3.IntegrityError, match="re-settled"):
                raw.execute("DELETE FROM settlements WHERE prediction_id = 'p1'")
                raw.commit()
        finally:
            raw.close()

    def test_raw_update_probability_aborts(self, tmp_path: Path) -> None:
        # Arrange
        db = tmp_path / "j.db"
        with SqliteJournal(db) as j:
            j.record(_prediction())

        # Act / Assert: rewriting the declared probability is rejected — that
        # would destroy the calibration record.
        raw = sqlite3.connect(str(db))
        try:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                raw.execute("UPDATE predictions SET probability = 0.1 WHERE prediction_id = 'p1'")
                raw.commit()
        finally:
            raw.close()

    def test_raw_insert_second_settlement_aborts(self, tmp_path: Path) -> None:
        # Arrange
        db = tmp_path / "j.db"
        with SqliteJournal(db) as j:
            j.record(_prediction())
            j.settle(_settlement())

        # Act / Assert: a raw duplicate INSERT trips the PRIMARY KEY.
        raw = sqlite3.connect(str(db))
        try:
            with pytest.raises(sqlite3.IntegrityError):
                raw.execute(
                    "INSERT INTO settlements (prediction_id, settled_at, realized_return, "
                    "benchmark_return, alpha_gross, alpha_net, outcome) "
                    "VALUES ('p1', '2026-03-01', 0.0, 0.0, 0.0, 0.0, 0)"
                )
                raw.commit()
        finally:
            raw.close()


class TestSchemaMigration:
    def test_schema_version_recorded(self, tmp_path: Path) -> None:
        # Arrange / Act
        db = tmp_path / "j.db"
        with SqliteJournal(db):
            pass

        raw = sqlite3.connect(str(db))
        try:
            version = raw.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        finally:
            raw.close()

        # Assert
        assert version == SCHEMA_VERSION

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        # Arrange
        db = tmp_path / "j.db"
        with SqliteJournal(db) as j:
            j.record(_prediction())

        # Act: reopen (migrate runs again).
        with SqliteJournal(db) as j:
            j.record(_prediction(pid="p2"))
            raw = sqlite3.connect(str(db))
            try:
                count = raw.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
            finally:
                raw.close()

            # Assert
            assert count == 1
            assert len(j.list_predictions()) == 2

    def test_reopen_preserves_data(self, tmp_path: Path) -> None:
        # Arrange
        db = tmp_path / "j.db"
        with SqliteJournal(db) as j:
            j.record(_prediction())
            j.settle(_settlement())

        # Act
        with SqliteJournal(db) as j:
            loaded = j.get_settlement("p1")

        # Assert
        assert loaded is not None
        assert loaded.outcome is True
