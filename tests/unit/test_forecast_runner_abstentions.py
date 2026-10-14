"""Tests for the standalone abstention store.

Abstentions are a finding, not noise: "declined 40% of the time, for these
reasons" must survive in its own durable record so the hit rate is never
flattered by quietly dropping the names the system had no view on.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from alphalab.adapters.abstention_store import SqliteAbstentionStore
from alphalab.domain.forecast import Abstention


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SqliteAbstentionStore]:
    with SqliteAbstentionStore(tmp_path / "abstentions.db") as s:
        yield s


def _abstention(ticker: str = "AAPL", reason: str = "no directional claim") -> Abstention:
    return Abstention(ticker=ticker, knowledge_date=date(2026, 10, 13), reason=reason)


def test_recorded_abstention_round_trips(store: SqliteAbstentionStore) -> None:
    store.record(_abstention())

    stored = store.list_abstentions()
    assert len(stored) == 1
    assert stored[0].ticker == "AAPL"
    assert stored[0].knowledge_date == date(2026, 10, 13)
    assert stored[0].reason == "no directional claim"


def test_repeated_abstentions_are_all_retained(store: SqliteAbstentionStore) -> None:
    """The same name declined twice is kept twice — repetition is itself signal."""
    store.record(_abstention(ticker="MSFT", reason="internal contradiction"))
    store.record(_abstention(ticker="MSFT", reason="internal contradiction"))

    assert store.count() == 2


def test_empty_store_reports_zero(store: SqliteAbstentionStore) -> None:
    assert store.count() == 0
    assert store.list_abstentions() == []


def test_iteration_yields_recorded_abstentions(store: SqliteAbstentionStore) -> None:
    store.record(_abstention(ticker="A"))
    store.record(_abstention(ticker="B"))

    tickers = [a.ticker for a in store]
    assert tickers == ["A", "B"]


def test_records_persist_across_connections(tmp_path: Path) -> None:
    """A fresh connection to the same file sees prior abstentions (durable, not in-memory)."""
    path = tmp_path / "abstentions.db"
    with SqliteAbstentionStore(path) as writer:
        writer.record(_abstention(ticker="DUR"))

    with SqliteAbstentionStore(path) as reader:
        assert reader.count() == 1
        assert reader.list_abstentions()[0].ticker == "DUR"
