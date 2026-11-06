"""Unit tests for the DuckDB/Parquet point-in-time store adapter."""

from __future__ import annotations

import inspect
from datetime import date
from pathlib import Path

import pytest

from alphalab.adapters.pit_store import ParquetPITStore, query_latest_cross_section
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


class TestCrossSectionQuery:
    def test_cross_section_enforces_pit_and_selects_latest_period(self, tmp_path: Path) -> None:
        sig = inspect.signature(query_latest_cross_section)
        assert sig.parameters["knowledge_date"].default is inspect.Parameter.empty

        # Empty root returns {}
        assert (
            query_latest_cross_section(
                root=tmp_path,
                fields=("eps_diluted",),
                knowledge_date=KnowledgeDate(date(2024, 6, 1)),
            )
            == {}
        )

        s_aapl = ParquetPITStore(root=tmp_path / "AAPL")
        s_aapl.write(
            [
                PITRecord(
                    ticker=Ticker("AAPL"),
                    field="eps_diluted",
                    value=6.00,
                    as_of=date(2023, 9, 30),
                    observed_at=date(2023, 11, 3),
                    source="SEC-EDGAR",
                ),
                PITRecord(
                    ticker=Ticker("AAPL"),
                    field="eps_diluted",
                    value=6.50,
                    as_of=date(2024, 9, 30),
                    observed_at=date(2024, 11, 1),
                    source="SEC-EDGAR",
                ),
            ]
        )
        before_2024_filing = query_latest_cross_section(
            root=tmp_path,
            fields=("eps_diluted",),
            knowledge_date=KnowledgeDate(date(2024, 6, 1)),
        )
        assert before_2024_filing == {"AAPL": {"eps_diluted": 6.00}}

        after_2024_filing = query_latest_cross_section(
            root=tmp_path,
            fields=("eps_diluted",),
            knowledge_date=KnowledgeDate(date(2024, 12, 1)),
        )
        assert after_2024_filing == {"AAPL": {"eps_diluted": 6.50}}
