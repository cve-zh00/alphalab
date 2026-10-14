"""EDGAR facts whose period ends after the filing date must be skipped.

Regression test for a bug found only with live SEC data: Coca-Cola's
``companyfacts`` payload contains 4 facts (of roughly 30k) where ``filed``
precedes ``end`` — a period that had not closed when the filing went out.
Those violate the ``observed_at >= as_of`` invariant and crashed ingestion.

The fix skips them rather than coercing ``as_of`` to the filing date, which
would misstate the period, or relaxing the invariant, which would weaken the
anti-leakage guarantee for every other record.
"""

from __future__ import annotations

from datetime import date

from alphalab.adapters.edgar import EdgarClient, EdgarSettings, HttpResponse
from alphalab.domain.types import Ticker


class _StubTransport:
    """Returns one canned ``companyfacts`` payload regardless of URL."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        return HttpResponse(status_code=200, json=self._payload)


def _payload(facts: list[dict[str, object]]) -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                "CommonStockSharesAuthorized": {"units": {"shares": facts}},
            }
        }
    }


def _client(payload: dict[str, object]) -> EdgarClient:
    return EdgarClient(
        EdgarSettings(user_agent="alphalab-test test@example.com"),
        transport=_StubTransport(payload),
    )


def test_forward_looking_fact_is_skipped() -> None:
    """A fact filed before its period ends is dropped, not coerced."""
    # Arrange — the exact shape seen in KO: end 2012-07-27, filed 2012-04-26.
    client = _client(_payload([{"end": "2012-07-27", "filed": "2012-04-26", "val": 11200000000}]))

    # Act
    records = client.company_facts(Ticker("KO"), cik="0000021344")

    # Assert — skipped, and ingestion did not raise.
    assert records == []


def test_normal_fact_is_kept() -> None:
    """A fact filed after its period ends is retained unchanged."""
    # Arrange
    client = _client(_payload([{"end": "2012-06-30", "filed": "2012-07-26", "val": 11200000000}]))

    # Act
    records = client.company_facts(Ticker("KO"), cik="0000021344")

    # Assert
    assert len(records) == 1
    assert records[0].as_of == date(2012, 6, 30)
    assert records[0].observed_at == date(2012, 7, 26)
    assert records[0].value == 11200000000.0


def test_mixed_payload_keeps_only_valid_facts() -> None:
    """One bad fact does not discard the good ones alongside it."""
    # Arrange
    client = _client(
        _payload(
            [
                {"end": "2012-07-27", "filed": "2012-04-26", "val": 1.0},  # skip
                {"end": "2012-06-30", "filed": "2012-07-26", "val": 2.0},  # keep
                {"end": "2012-09-30", "filed": "2012-10-25", "val": 3.0},  # keep
            ]
        )
    )

    # Act
    records = client.company_facts(Ticker("KO"), cik="0000021344")

    # Assert
    assert sorted(r.value for r in records) == [2.0, 3.0]
