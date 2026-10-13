"""Unit tests for the SEC EDGAR fundamentals adapter (with HTTP doubles)."""

from __future__ import annotations

from datetime import date

import pytest

from alphalab.adapters.edgar import (
    EdgarClient,
    EdgarSettings,
    HttpResponse,
    RateLimiter,
)
from alphalab.domain.types import Ticker

# A minimal companyfacts payload shaped like the real XBRL JSON.
_COMPANYFACTS = {
    "facts": {
        "us-gaap": {
            "EarningsPerShareBasic": {
                "units": {
                    "USD/shares": [
                        {
                            "end": "2008-06-30",
                            "val": 1.00,
                            "filed": "2008-08-10",
                            "form": "10-Q",
                        },
                        {
                            "end": "2008-06-30",
                            "val": 0.40,
                            "filed": "2009-02-15",
                            "form": "10-Q/A",
                        },
                    ]
                }
            }
        }
    }
}


class _StubTransport:
    """Records requests and replays queued responses or errors."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        self.calls.append((url, headers))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, HttpResponse)
        return item


class _FakeClock:
    """Deterministic clock capturing sleeps instead of blocking."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _settings() -> EdgarSettings:
    return EdgarSettings(user_agent="alphalab-test test@example.com", max_rps=10.0)


class TestEdgarSettings:
    def test_user_agent_is_required_and_configurable(self) -> None:
        # Arrange / Act
        settings = EdgarSettings(user_agent="me me@example.com")

        # Assert — never hardcoded; comes from configuration.
        assert settings.user_agent == "me me@example.com"


class TestParsingAsFiled:
    def test_parses_each_filing_as_a_separate_record(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(200, _COMPANYFACTS)])
        client = EdgarClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        records = client.company_facts(Ticker("ACME"), cik="0000000001")

        # Assert — the restatement is a distinct record, as-filed.
        eps = [r for r in records if r.field == "EarningsPerShareBasic"]
        by_observed = {r.observed_at: r.value for r in eps}
        assert by_observed == {
            date(2008, 8, 10): 1.00,
            date(2009, 2, 15): 0.40,
        }

    def test_sets_as_of_and_observed_from_payload(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(200, _COMPANYFACTS)])
        client = EdgarClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        records = client.company_facts(Ticker("ACME"), cik="0000000001")

        # Assert
        first = min(records, key=lambda r: r.observed_at)
        assert first.as_of == date(2008, 6, 30)
        assert first.observed_at == date(2008, 8, 10)
        assert first.source == "SEC-EDGAR"

    def test_user_agent_header_is_sent(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(200, _COMPANYFACTS)])
        client = EdgarClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        client.company_facts(Ticker("ACME"), cik="0000000001")

        # Assert
        _, headers = transport.calls[0]
        assert headers["User-Agent"] == "alphalab-test test@example.com"


class TestBackoff:
    def test_retries_on_429_then_succeeds(self) -> None:
        # Arrange — a rate-limit response followed by success.
        transport = _StubTransport([HttpResponse(429, {}), HttpResponse(200, _COMPANYFACTS)])
        clock = _FakeClock()
        client = EdgarClient(_settings(), transport=transport, clock=clock)

        # Act
        records = client.company_facts(Ticker("ACME"), cik="0000000001")

        # Assert — it retried and backed off exponentially at least once.
        assert len(transport.calls) == 2
        assert clock.sleeps  # a backoff sleep occurred
        assert records

    def test_gives_up_after_max_retries(self) -> None:
        # Arrange — persistent 503s exhaust the retry budget.
        transport = _StubTransport([HttpResponse(503, {})] * 10)
        client = EdgarClient(_settings(), transport=transport, clock=_FakeClock(), max_retries=3)

        # Act / Assert
        with pytest.raises(RuntimeError, match="EDGAR request failed"):
            client.company_facts(Ticker("ACME"), cik="0000000001")
        assert len(transport.calls) == 4  # initial + 3 retries

    def test_backoff_grows_exponentially(self) -> None:
        # Arrange
        transport = _StubTransport(
            [HttpResponse(503, {}), HttpResponse(503, {}), HttpResponse(200, _COMPANYFACTS)]
        )
        clock = _FakeClock()
        client = EdgarClient(_settings(), transport=transport, clock=clock)

        # Act
        client.company_facts(Ticker("ACME"), cik="0000000001")

        # Assert — second backoff is larger than the first.
        assert clock.sleeps[1] > clock.sleeps[0]


class TestRateLimiter:
    def test_throttles_to_configured_rate(self) -> None:
        # Arrange — 2 req/s means a 0.5s minimum spacing.
        clock = _FakeClock()
        limiter = RateLimiter(max_rps=2.0, clock=clock)

        # Act — two back-to-back acquisitions.
        limiter.acquire()
        limiter.acquire()

        # Assert — the second call slept to honour the rate.
        assert clock.sleeps == [pytest.approx(0.5)]

    def test_no_sleep_when_enough_time_elapsed(self) -> None:
        # Arrange
        clock = _FakeClock()
        limiter = RateLimiter(max_rps=10.0, clock=clock)
        limiter.acquire()
        clock.now += 1.0  # plenty of time passes

        # Act
        limiter.acquire()

        # Assert
        assert clock.sleeps == []
