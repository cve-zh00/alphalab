"""Unit tests for the current-price lookup (no look-ahead)."""

from __future__ import annotations

from datetime import date

import pytest

from alphalab.adapters.prices import PricePoint
from alphalab.adapters.quotes import NoPriceAvailableError, current_price_at
from alphalab.domain.pit import KnowledgeDate
from alphalab.domain.types import Ticker


class _StubPriceSource:
    def __init__(self, points: list[PricePoint]) -> None:
        self._points = points
        self.requested: list[Ticker] = []

    def series(self, ticker: Ticker) -> list[PricePoint]:
        self.requested.append(ticker)
        return list(self._points)


def _series() -> list[PricePoint]:
    return [
        PricePoint(on=date(2020, 1, 2), close=10.0),
        PricePoint(on=date(2020, 1, 3), close=11.0),
        PricePoint(on=date(2020, 1, 6), close=12.0),
    ]


class TestCurrentPrice:
    def test_returns_last_close_on_or_before_knowledge_date(self) -> None:
        # Arrange
        source = _StubPriceSource(_series())

        # Act — knowledge date lands on a trading day.
        price = current_price_at(source, Ticker("ACME"), KnowledgeDate(date(2020, 1, 3)))

        # Assert
        assert price == 11.0

    def test_uses_most_recent_prior_close_on_a_non_trading_day(self) -> None:
        # Arrange
        source = _StubPriceSource(_series())

        # Act — 2020-01-04 is a Saturday; the last close is Friday the 3rd.
        price = current_price_at(source, Ticker("ACME"), KnowledgeDate(date(2020, 1, 4)))

        # Assert — never the future close on the 6th.
        assert price == 11.0

    def test_never_reads_a_future_close(self) -> None:
        # Arrange
        source = _StubPriceSource(_series())

        # Act — knowledge date is before the whole series: no price knowable.
        with pytest.raises(NoPriceAvailableError):
            current_price_at(source, Ticker("ACME"), KnowledgeDate(date(2019, 12, 31)))

    def test_empty_series_raises(self) -> None:
        # Arrange
        source = _StubPriceSource([])

        # Act / Assert
        with pytest.raises(NoPriceAvailableError):
            current_price_at(source, Ticker("ACME"), KnowledgeDate(date(2020, 1, 3)))

    def test_unsorted_series_is_handled(self) -> None:
        # Arrange — points out of order must still resolve correctly.
        source = _StubPriceSource(list(reversed(_series())))

        # Act
        price = current_price_at(source, Ticker("ACME"), KnowledgeDate(date(2020, 1, 10)))

        # Assert — the latest close at/under the knowledge date.
        assert price == 12.0
