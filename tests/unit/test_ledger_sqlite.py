"""Unit tests for :class:`alphalab.adapters.ledger_sqlite.SqliteTrialLedger`."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from alphalab.adapters.ledger_sqlite import SCHEMA_VERSION, SqliteTrialLedger
from alphalab.domain.ledger import TrialMetrics, TrialRecord


def _record(
    *,
    family: str = "momentum",
    params: Mapping[str, Any] | None = None,
    window_start: date = date(2020, 1, 1),
    window_end: date = date(2021, 1, 1),
    recorded_at: datetime | None = None,
    llm_version: str | None = None,
    llm_cutoff: date | None = None,
) -> TrialRecord:
    spec = dict(params if params is not None else {"lookback": 20})
    return TrialRecord.create(
        spec=spec,
        dataset_hash="dataset-1",
        family=family,
        window_start=window_start,
        window_end=window_end,
        params=spec,
        metrics=TrialMetrics(
            sharpe_gross=1.4,
            sharpe_net=1.0,
            return_gross=0.2,
            return_net=0.15,
            max_drawdown=-0.1,
        ),
        recorded_at=recorded_at or datetime(2026, 1, 1, tzinfo=UTC),
        llm_version=llm_version,
        llm_cutoff=llm_cutoff,
    )


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[SqliteTrialLedger]:
    db = tmp_path / "ledger.db"
    with SqliteTrialLedger(db) as led:
        yield led


class TestAppendAndCount:
    def test_append_returns_trial_id(self, ledger: SqliteTrialLedger) -> None:
        # Arrange
        record = _record()

        # Act
        trial_id = ledger.append(record)

        # Assert
        assert trial_id == record.trial_id

    def test_trial_count_total(self, ledger: SqliteTrialLedger) -> None:
        # Arrange
        for _ in range(3):
            ledger.append(_record(family="a"))
        ledger.append(_record(family="b"))

        # Act / Assert
        assert ledger.trial_count() == 4

    def test_trial_count_by_family(self, ledger: SqliteTrialLedger) -> None:
        # Arrange
        for _ in range(3):
            ledger.append(_record(family="a"))
        ledger.append(_record(family="b"))

        # Act / Assert
        assert ledger.trial_count("a") == 3
        assert ledger.trial_count("b") == 1
        assert ledger.trial_count("missing") == 0


class TestReads:
    def test_get_by_id_roundtrips(self, ledger: SqliteTrialLedger) -> None:
        # Arrange
        record = _record(
            params={"lookback": 20, "z": 1.5},
            llm_version="frontier-2026-09",
            llm_cutoff=date(2026, 9, 1),
        )
        ledger.append(record)

        # Act
        loaded = ledger.get(record.trial_id)

        # Assert
        assert loaded is not None
        assert loaded.trial_id == record.trial_id
        assert loaded.family == record.family
        assert loaded.spec_hash == record.spec_hash
        assert dict(loaded.params) == dict(record.params)
        assert loaded.metrics == record.metrics
        assert loaded.llm_version == "frontier-2026-09"
        assert loaded.llm_cutoff == date(2026, 9, 1)

    def test_get_missing_returns_none(self, ledger: SqliteTrialLedger) -> None:
        # Act / Assert
        assert ledger.get("does-not-exist") is None

    def test_list_by_family(self, ledger: SqliteTrialLedger) -> None:
        # Arrange
        ledger.append(_record(family="a"))
        ledger.append(_record(family="a"))
        ledger.append(_record(family="b"))

        # Act
        rows = ledger.list_by_family("a")

        # Assert
        assert len(rows) == 2
        assert all(r.family == "a" for r in rows)

    def test_list_by_window(self, ledger: SqliteTrialLedger) -> None:
        # Arrange
        ledger.append(_record(window_start=date(2019, 1, 1), window_end=date(2019, 6, 1)))
        ledger.append(_record(window_start=date(2020, 1, 1), window_end=date(2020, 6, 1)))
        ledger.append(_record(window_start=date(2022, 1, 1), window_end=date(2022, 6, 1)))

        # Act
        rows = ledger.list_by_window(date(2019, 12, 1), date(2021, 1, 1))

        # Assert
        assert len(rows) == 1
        assert rows[0].window_start == date(2020, 1, 1)

    def test_iter_yields_all(self, ledger: SqliteTrialLedger) -> None:
        # Arrange
        for _ in range(5):
            ledger.append(_record())

        # Act
        all_rows = list(ledger)

        # Assert
        assert len(all_rows) == 5


class TestDistinctSpecCount:
    def test_distinct_specs_vs_trials(self, ledger: SqliteTrialLedger) -> None:
        # Arrange: same spec in two key orders -> same spec_hash, two trials.
        ledger.append(_record(params={"a": 1, "b": 2}))
        ledger.append(_record(params={"b": 2, "a": 1}))
        ledger.append(_record(params={"a": 9}))

        # Act / Assert
        assert ledger.trial_count() == 3
        assert ledger.distinct_spec_count() == 2

    def test_distinct_specs_by_family(self, ledger: SqliteTrialLedger) -> None:
        # Arrange
        ledger.append(_record(family="a", params={"x": 1}))
        ledger.append(_record(family="a", params={"x": 1}))
        ledger.append(_record(family="b", params={"x": 1}))

        # Act / Assert
        assert ledger.distinct_spec_count("a") == 1
        assert ledger.distinct_spec_count("b") == 1


class TestImmutabilityTriggers:
    """Defence in depth: the engine must reject raw UPDATE/DELETE."""

    def test_raw_update_aborts(self, tmp_path: Path) -> None:
        # Arrange
        db = tmp_path / "led.db"
        with SqliteTrialLedger(db) as led:
            record = _record()
            led.append(record)

        # Act / Assert: a raw connection (no API) still cannot mutate.
        raw = sqlite3.connect(str(db))
        try:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                raw.execute("UPDATE trials SET family = 'hacked' WHERE 1=1")
                raw.commit()
        finally:
            raw.close()

    def test_raw_delete_aborts(self, tmp_path: Path) -> None:
        # Arrange
        db = tmp_path / "led.db"
        with SqliteTrialLedger(db) as led:
            led.append(_record())

        # Act / Assert
        raw = sqlite3.connect(str(db))
        try:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                raw.execute("DELETE FROM trials WHERE 1=1")
                raw.commit()
        finally:
            raw.close()

    def test_row_survives_blocked_delete(self, tmp_path: Path) -> None:
        # Arrange
        db = tmp_path / "led.db"
        with SqliteTrialLedger(db) as led:
            led.append(_record())

            raw = sqlite3.connect(str(db))
            try:
                with pytest.raises(sqlite3.IntegrityError):
                    raw.execute("DELETE FROM trials")
                    raw.commit()
            finally:
                raw.close()

            # Assert: the record is still there.
            assert led.trial_count() == 1


class TestSchemaMigration:
    def test_schema_version_recorded(self, tmp_path: Path) -> None:
        # Arrange
        db = tmp_path / "led.db"

        # Act
        with SqliteTrialLedger(db):
            pass

        raw = sqlite3.connect(str(db))
        try:
            cur = raw.execute("SELECT MAX(version) FROM schema_version")
            version = cur.fetchone()[0]
        finally:
            raw.close()

        # Assert
        assert version == SCHEMA_VERSION

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        # Arrange: opening the same DB twice re-runs _migrate.
        db = tmp_path / "led.db"
        with SqliteTrialLedger(db) as led:
            led.append(_record())

        # Act: reopen (migrate runs again) and append more.
        with SqliteTrialLedger(db) as led:
            led.append(_record())

            # Assert: no duplicate schema_version rows, data preserved.
            raw = sqlite3.connect(str(db))
            try:
                count = raw.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
            finally:
                raw.close()
            assert count == 1
            assert led.trial_count() == 2

    def test_reopen_preserves_data(self, tmp_path: Path) -> None:
        # Arrange
        db = tmp_path / "led.db"
        with SqliteTrialLedger(db) as led:
            rid = led.append(_record())

        # Act
        with SqliteTrialLedger(db) as led:
            loaded = led.get(rid)

        # Assert
        assert loaded is not None
        assert loaded.trial_id == rid
