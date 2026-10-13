"""Unit tests for the DuckDB/Parquet point-in-time store adapter."""

from __future__ import annotations

import inspect
from datetime import date
from pathlib import Path

import pytest

from alphalab.adapters.pit_store import ParquetPITStore
from alphalab.domain.pit import KnowledgeDate, PITRecord
from alphalab.domain.types import Ticker


def _restated_dataset() -> list[PITRecord]:
    """One field with an original filing and a later restatement."""
    original = PITRecord(
        ticker=Ticker("ACME"),
        field="eps",
        value=1.00,
        as_of=date(2008, 6, 30),
        observed_at=date(2008, 8, 10),
        source="SEC-EDGAR",
    )
    restated = PITRecord(
        ticker=Ticker("ACME"),
        field="eps",
        value=0.40,
        as_of=date(2008, 6, 30),
        observed_at=date(2009, 2, 15),
        source="SEC-EDGAR",
    )
    return [original, restated]


@pytest.fixture
def store(tmp_path: Path) -> ParquetPITStore:
    s = ParquetPITStore(root=tmp_path)
    s.write(_restated_dataset())
    return s


class TestKnowledgeDateIsMandatory:
    def test_query_has_no_default_knowledge_date(self) -> None:
        # Arrange
        sig = inspect.signature(ParquetPITStore.query)

        # Act
        param = sig.parameters["knowledge_date"]

        # Assert — there is no way to omit the temporal filter.
        assert param.default is inspect.Parameter.empty

    def test_calling_without_knowledge_date_is_a_type_error(self, store: ParquetPITStore) -> None:
        # Act / Assert
        with pytest.raises(TypeError):
            store.query(ticker=Ticker("ACME"), field="eps")  # type: ignore[call-arg]


class TestAsFiledSemantics:
    def test_returns_original_before_restatement(self, store: ParquetPITStore) -> None:
        # Arrange — knowledge date sits between the two filings.
        kd = KnowledgeDate(date(2008, 10, 1))

        # Act
        rows = store.query(ticker=Ticker("ACME"), field="eps", knowledge_date=kd)

        # Assert — only the original value is knowable; the restatement is not.
        assert [r.value for r in rows] == [1.00]
        assert all(r.observed_at <= kd.value for r in rows)

    def test_restated_value_unreachable_at_early_knowledge_date(
        self, store: ParquetPITStore
    ) -> None:
        # Arrange
        kd = KnowledgeDate(date(2008, 10, 1))

        # Act
        rows = store.query(ticker=Ticker("ACME"), field="eps", knowledge_date=kd)

        # Assert — the corrected 0.40 value is structurally unreachable.
        assert 0.40 not in {r.value for r in rows}

    def test_returns_both_after_restatement(self, store: ParquetPITStore) -> None:
        # Arrange — later knowledge date sees the full filing history.
        kd = KnowledgeDate(date(2009, 6, 1))

        # Act
        rows = store.query(ticker=Ticker("ACME"), field="eps", knowledge_date=kd)

        # Assert
        assert sorted(r.value for r in rows) == [0.40, 1.00]


class TestQueryIsParameterised:
    def test_rejects_unknown_field_identifier_shape(self, store: ParquetPITStore) -> None:
        # Act / Assert — identifiers are allow-listed, values are bound params.
        rows = store.query(
            ticker=Ticker("ACME"),
            field="revenue",
            knowledge_date=KnowledgeDate(date(2010, 1, 1)),
        )
        assert rows == []

    def test_sql_injection_in_value_is_inert(self, store: ParquetPITStore) -> None:
        # Arrange — a hostile ticker string must be treated as data, not SQL.
        hostile = Ticker("ACME'; DROP TABLE facts;--")

        # Act
        rows = store.query(
            ticker=hostile,
            field="eps",
            knowledge_date=KnowledgeDate(date(2010, 1, 1)),
        )

        # Assert — no match, and the store is still usable afterwards.
        assert rows == []
        again = store.query(
            ticker=Ticker("ACME"),
            field="eps",
            knowledge_date=KnowledgeDate(date(2010, 1, 1)),
        )
        assert len(again) == 2
