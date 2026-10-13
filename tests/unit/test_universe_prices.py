"""Unit tests for the point-in-time universe and price series adapters."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from alphalab.adapters.prices import CsvPriceSeriesSource, PricePoint
from alphalab.adapters.universe import (
    InMemoryUniverseSource,
    UniverseMember,
)
from alphalab.domain.pit import KnowledgeDate
from alphalab.domain.types import Ticker


class TestUniverse:
    def test_includes_delisted_members(self) -> None:
        # Arrange — a listed name and one that was delisted mid-sample.
        members = [
            UniverseMember(
                ticker=Ticker("ALIVE"),
                listed_on=date(2000, 1, 1),
                delisted_on=None,
            ),
            UniverseMember(
                ticker=Ticker("DEAD"),
                listed_on=date(2000, 1, 1),
                delisted_on=date(2008, 9, 15),
            ),
        ]
        source = InMemoryUniverseSource(members)

        # Act — as known on a date before the delisting, both are members.
        constituents = source.constituents(KnowledgeDate(date(2008, 1, 1)))

        # Assert — survivorship bias avoided: DEAD is present.
        assert {m.ticker for m in constituents} == {Ticker("ALIVE"), Ticker("DEAD")}

    def test_excludes_members_delisted_before_knowledge_date(self) -> None:
        # Arrange
        source = InMemoryUniverseSource(
            [
                UniverseMember(
                    ticker=Ticker("DEAD"),
                    listed_on=date(2000, 1, 1),
                    delisted_on=date(2008, 9, 15),
                )
            ]
        )

        # Act — after the delisting it is no longer a current constituent.
        constituents = source.constituents(KnowledgeDate(date(2009, 1, 1)))

        # Assert
        assert constituents == []

    def test_excludes_members_not_yet_listed(self) -> None:
        # Arrange
        source = InMemoryUniverseSource(
            [
                UniverseMember(
                    ticker=Ticker("FUTURE"),
                    listed_on=date(2010, 1, 1),
                    delisted_on=None,
                )
            ]
        )

        # Act
        constituents = source.constituents(KnowledgeDate(date(2008, 1, 1)))

        # Assert
        assert constituents == []

    def test_member_is_frozen(self) -> None:
        # Arrange
        member = UniverseMember(
            ticker=Ticker("ALIVE"), listed_on=date(2000, 1, 1), delisted_on=None
        )

        # Act / Assert
        with pytest.raises(Exception):  # noqa: B017
            member.ticker = Ticker("OTHER")  # type: ignore[misc]


class TestPriceSeriesSource:
    def test_reads_prices_from_csv(self, tmp_path: Path) -> None:
        # Arrange
        csv = tmp_path / "SPY.csv"
        csv.write_text("date,close\n2008-09-30,115.99\n2008-10-01,113.66\n")
        source = CsvPriceSeriesSource(root=tmp_path)

        # Act
        series = source.series(Ticker("SPY"))

        # Assert
        assert series == [
            PricePoint(on=date(2008, 9, 30), close=115.99),
            PricePoint(on=date(2008, 10, 1), close=113.66),
        ]

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        # Arrange
        source = CsvPriceSeriesSource(root=tmp_path)

        # Act
        series = source.series(Ticker("NOPE"))

        # Assert
        assert series == []
