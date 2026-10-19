"""Unit tests for the Alpaca bars adapter (HTTP doubles, no network)."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import SecretStr

from alphalab.adapters.alpaca_bars import (
    AlpacaBarsClient,
    DataFeed,
    FeedSubscriptionError,
    RawBar,
)
from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.edgar import HttpResponse
from alphalab.domain.backtest.types import PriceBar
from alphalab.domain.types import Ticker

_SECRET = "do-not-leak-this-secret"


def _settings(rate: int = 150) -> AlpacaSettings:
    return AlpacaSettings(key_id="KID", secret_key=SecretStr(_SECRET), rate_limit_per_minute=rate)


def _bar(stamp: str, close: float, volume: float | None = 1000.0) -> dict[str, object]:
    row: dict[str, object] = {"t": stamp, "c": close}
    if volume is not None:
        row["v"] = volume
    return row


class _StubTransport:
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
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class TestPagination:
    def test_two_pages_are_concatenated_without_losing_bars(self) -> None:
        # Arrange — page 1 returns a token; page 2 closes the sequence.
        page1 = HttpResponse(
            200,
            {
                "bars": {"AAPL": [_bar("2021-01-04T05:00:00Z", 100.0)]},
                "next_page_token": "TOKEN123",
            },
        )
        page2 = HttpResponse(
            200,
            {
                "bars": {"AAPL": [_bar("2021-01-05T05:00:00Z", 101.0)]},
                "next_page_token": None,
            },
        )
        transport = _StubTransport([page1, page2])
        client = AlpacaBarsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        result = client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 31))

        # Assert — both days survive the pagination, none dropped.
        closes = [b.close for b in result[Ticker("AAPL")]]
        assert closes == [100.0, 101.0]
        # The second request carried the page token.
        assert "page_token=TOKEN123" in transport.calls[1][0]
        # The first request did not.
        assert "page_token" not in transport.calls[0][0]

    def test_stops_when_token_absent(self) -> None:
        # Arrange — a single page, no token key at all.
        transport = _StubTransport(
            [HttpResponse(200, {"bars": {"AAPL": [_bar("2021-01-04T05:00:00Z", 100.0)]}})]
        )
        client = AlpacaBarsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 2))

        # Assert — exactly one request was made.
        assert len(transport.calls) == 1


class TestRequestShape:
    def test_requests_adjustment_all_and_configured_feed(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(200, {"bars": {}})])
        client = AlpacaBarsClient(
            _settings(), transport=transport, feed=DataFeed.IEX, clock=_FakeClock()
        )

        # Act
        client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 2))

        # Assert — dividend+split adjustment and the IEX feed are in the query.
        url = transport.calls[0][0]
        assert "adjustment=all" in url
        assert "feed=iex" in url
        assert "timeframe=1Day" in url
        assert "data.alpaca.markets" in url

    def test_multiple_symbols_in_one_request(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(200, {"bars": {}})])
        client = AlpacaBarsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        client.daily_bars(
            [Ticker("AAPL"), Ticker("MSFT")], start=date(2021, 1, 1), end=date(2021, 1, 2)
        )

        # Assert — both symbols travel in a single request.
        url = transport.calls[0][0]
        assert "symbols=AAPL,MSFT" in url
        assert len(transport.calls) == 1

    def test_empty_symbols_makes_no_request(self) -> None:
        # Arrange
        transport = _StubTransport([])
        client = AlpacaBarsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        result = client.daily_bars([], start=date(2021, 1, 1), end=date(2021, 1, 2))

        # Assert
        assert result == {}
        assert transport.calls == []


class TestDomainConversion:
    def test_raw_bar_converts_to_price_bar(self) -> None:
        # Arrange
        raw = RawBar(symbol="AAPL", on=date(2021, 1, 4), close=100.0, volume=1000.0)

        # Act
        bar = raw.to_price_bar()

        # Assert
        assert isinstance(bar, PriceBar)
        assert bar.on == date(2021, 1, 4)
        assert bar.close == 100.0
        assert bar.volume == 1000.0

    def test_missing_volume_is_none_not_fabricated(self) -> None:
        # Arrange — a bar with no volume field.
        transport = _StubTransport(
            [HttpResponse(200, {"bars": {"AAPL": [_bar("2021-01-04T05:00:00Z", 100.0, None)]}})]
        )
        client = AlpacaBarsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        result = client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 2))

        # Assert — absence is preserved, never filled with a made-up number.
        assert result[Ticker("AAPL")][0].volume is None

    def test_malformed_rows_are_skipped_not_invented(self) -> None:
        # Arrange — one good bar and two unusable rows.
        rows = [
            _bar("2021-01-04T05:00:00Z", 100.0),
            {"t": "2021-01-05T05:00:00Z"},  # no close
            {"c": 5.0},  # no timestamp
        ]
        transport = _StubTransport([HttpResponse(200, {"bars": {"AAPL": rows}})])
        client = AlpacaBarsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        result = client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 6))

        # Assert — only the one valid bar; nothing fabricated for the gaps.
        assert len(result[Ticker("AAPL")]) == 1

    def test_bars_are_sorted_by_date(self) -> None:
        # Arrange — out-of-order rows.
        rows = [_bar("2021-01-06T05:00:00Z", 102.0), _bar("2021-01-04T05:00:00Z", 100.0)]
        transport = _StubTransport([HttpResponse(200, {"bars": {"AAPL": rows}})])
        client = AlpacaBarsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        result = client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 7))

        # Assert
        dates = [b.on for b in result[Ticker("AAPL")]]
        assert dates == [date(2021, 1, 4), date(2021, 1, 6)]


class TestThrottlingAndRetries:
    def test_throttles_according_to_rate_limit(self) -> None:
        # Arrange — 60/min => 1 req/s => 1.0s spacing between the two pages.
        page1 = HttpResponse(200, {"bars": {"AAPL": []}, "next_page_token": "T"})
        page2 = HttpResponse(200, {"bars": {"AAPL": []}})
        clock = _FakeClock()
        transport = _StubTransport([page1, page2])
        client = AlpacaBarsClient(_settings(rate=60), transport=transport, clock=clock)

        # Act
        client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 2))

        # Assert — the limiter imposed a spacing sleep.
        assert clock.sleeps == [pytest.approx(1.0)]

    def test_retries_on_429_then_succeeds(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(429, {}), HttpResponse(200, {"bars": {}})])
        clock = _FakeClock()
        client = AlpacaBarsClient(_settings(), transport=transport, clock=clock)

        # Act
        client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 2))

        # Assert
        assert len(transport.calls) == 2
        assert any(s > 0 for s in clock.sleeps)

    def test_retries_on_5xx(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(502, {}), HttpResponse(200, {"bars": {}})])
        client = AlpacaBarsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 2))

        # Assert
        assert len(transport.calls) == 2

    def test_non_retryable_4xx_raises_clearly(self) -> None:
        # Arrange — 404 is not retryable.
        transport = _StubTransport([HttpResponse(404, {})])
        client = AlpacaBarsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act / Assert
        with pytest.raises(RuntimeError, match="Alpaca bars request failed") as exc:
            client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 2))
        assert _SECRET not in str(exc.value)


class TestFeedSubscription:
    def test_403_is_actionable_feed_subscription_error(self) -> None:
        # Arrange — SIP requested on a plan that only allows IEX.
        transport = _StubTransport([HttpResponse(403, {})])
        client = AlpacaBarsClient(
            _settings(), transport=transport, feed=DataFeed.SIP, clock=_FakeClock()
        )

        # Act / Assert — the message names the fix, not an opaque 403.
        with pytest.raises(FeedSubscriptionError, match="IEX") as exc:
            client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 2))
        assert "sip" in str(exc.value).lower()
        assert _SECRET not in str(exc.value)

    def test_403_is_not_retried(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(403, {})])
        client = AlpacaBarsClient(
            _settings(), transport=transport, feed=DataFeed.SIP, clock=_FakeClock()
        )

        # Act / Assert — a subscription problem will not resolve by retrying.
        with pytest.raises(FeedSubscriptionError):
            client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 2))
        assert len(transport.calls) == 1


class TestSecrets:
    def test_auth_headers_sent_and_secret_not_in_url(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(200, {"bars": {}})])
        client = AlpacaBarsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 1), end=date(2021, 1, 2))

        # Assert
        url, headers = transport.calls[0]
        assert headers["APCA-API-SECRET-KEY"] == _SECRET
        assert _SECRET not in url


class TestHighLowAndRawClose:
    def test_high_low_are_parsed_and_raw_close_merged(self) -> None:
        adjusted = {
            "bars": {"DECK": [{"t": "2024-09-18T04:00:00Z", "c": 10.0, "h": 11.0, "l": 9.5}]}
        }
        quoted = {"bars": {"DECK": [{"t": "2024-09-18T04:00:00Z", "c": 100.0}]}}
        transport = _StubTransport([HttpResponse(200, adjusted), HttpResponse(200, quoted)])
        client = AlpacaBarsClient(
            _settings(), transport=transport, clock=_FakeClock(), with_raw_close=True
        )

        bars = client.daily_bars([Ticker("DECK")], start=date(2024, 9, 1), end=date(2024, 9, 30))

        bar = bars[Ticker("DECK")][0]
        assert (bar.close, bar.high, bar.low, bar.raw_close) == (10.0, 11.0, 9.5, 100.0)
        assert "adjustment=all" in transport.calls[0][0]
        assert "adjustment=raw" in transport.calls[1][0]

    def test_raw_close_off_by_default(self) -> None:
        transport = _StubTransport(
            [HttpResponse(200, {"bars": {"A": [_bar("2024-09-18T04:00:00Z", 5.0)]}})]
        )
        client = AlpacaBarsClient(_settings(), transport=transport, clock=_FakeClock())

        bar = client.daily_bars([Ticker("A")], start=date(2024, 9, 1), end=date(2024, 9, 30))[
            Ticker("A")
        ][0]

        assert bar.raw_close is None and len(transport.calls) == 1
