"""DEMO: "what was knowable on 2008-10-01?"

This acceptance test is the human-facing proof of the anti-leakage design. We
load a synthetic fundamental that was later restated and ask the store what
was knowable on 2008-10-01. The original as-filed value must come back and the
later restatement must be structurally unreachable from that knowledge date.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from alphalab.adapters.pit_store import ParquetPITStore
from alphalab.domain.pit import KnowledgeDate, PITRecord
from alphalab.domain.types import Ticker

pytestmark = pytest.mark.acceptance

_AS_OF = date(2008, 6, 30)
_ORIGINAL_FILING = date(2008, 8, 10)
_RESTATEMENT_FILING = date(2009, 2, 15)
_KNOWLEDGE = KnowledgeDate(date(2008, 10, 1))


def _synthetic_history() -> list[PITRecord]:
    original = PITRecord(
        ticker=Ticker("ACME"),
        field="eps",
        value=1.00,
        as_of=_AS_OF,
        observed_at=_ORIGINAL_FILING,
        source="SEC-EDGAR",
    )
    restated = PITRecord(
        ticker=Ticker("ACME"),
        field="eps",
        value=0.40,
        as_of=_AS_OF,
        observed_at=_RESTATEMENT_FILING,
        source="SEC-EDGAR",
    )
    return [original, restated]


def test_demo_what_was_known_on_2008_10_01(tmp_path: Path) -> None:
    # Arrange — persist the original filing and its later restatement.
    store = ParquetPITStore(root=tmp_path)
    store.write(_synthetic_history())

    # Act — ask what was knowable on 2008-10-01.
    knowable = store.query(ticker=Ticker("ACME"), field="eps", knowledge_date=_KNOWLEDGE)

    # Assert — only the as-filed original is returned.
    assert [r.value for r in knowable] == [1.00]

    # Assert — the restated value is unreachable from this knowledge date.
    assert 0.40 not in {r.value for r in knowable}

    # Assert — and the one record returned predates the knowledge date.
    assert knowable[0].observed_at == _ORIGINAL_FILING
    assert knowable[0].observed_at <= _KNOWLEDGE.value

    # Sanity — a later knowledge date does see the restatement, proving the
    # earlier result was a temporal cut, not missing data.
    later = store.query(
        ticker=Ticker("ACME"),
        field="eps",
        knowledge_date=KnowledgeDate(date(2009, 6, 1)),
    )
    assert sorted(r.value for r in later) == [0.40, 1.00]
