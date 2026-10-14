"""Unit tests for the ticker->CIK resolver (with HTTP doubles, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphalab.adapters.cik import CikResolver, CikSettings
from alphalab.adapters.edgar import HttpResponse
from alphalab.domain.types import Ticker

# A trimmed copy of the real ``company_tickers.json`` shape: an object keyed by
# arbitrary string indices, each with ``cik_str`` (int), ``ticker`` and ``title``.
_COMPANY_TICKERS: dict[str, object] = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    "2": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon Com Inc"},
}


class _StubTransport:
    """Replays queued responses and records the requests made."""

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


def _settings() -> CikSettings:
    return CikSettings(user_agent="alphalab-test test@example.com")


class TestResolution:
    def test_resolves_ticker_to_zero_padded_cik(self, tmp_path: Path) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(200, _COMPANY_TICKERS)])
        resolver = CikResolver(_settings(), transport=transport, cache_path=tmp_path / "cik.json")

        # Act
        cik = resolver.resolve(Ticker("AAPL"))

        # Assert — 10 digits, left-padded with zeros as companyfacts requires.
        assert cik == "0000320193"

    def test_resolution_is_case_insensitive(self, tmp_path: Path) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(200, _COMPANY_TICKERS)])
        resolver = CikResolver(_settings(), transport=transport, cache_path=tmp_path / "cik.json")

        # Act / Assert
        assert resolver.resolve(Ticker("msft")) == "0000789019"

    def test_unknown_ticker_raises(self, tmp_path: Path) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(200, _COMPANY_TICKERS)])
        resolver = CikResolver(_settings(), transport=transport, cache_path=tmp_path / "cik.json")

        # Act / Assert
        with pytest.raises(KeyError, match="NOPE"):
            resolver.resolve(Ticker("NOPE"))

    def test_user_agent_header_is_sent(self, tmp_path: Path) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(200, _COMPANY_TICKERS)])
        resolver = CikResolver(_settings(), transport=transport, cache_path=tmp_path / "cik.json")

        # Act
        resolver.resolve(Ticker("AAPL"))

        # Assert
        _, headers = transport.calls[0]
        assert headers["User-Agent"] == "alphalab-test test@example.com"


class TestCaching:
    def test_downloads_once_and_caches_to_disk(self, tmp_path: Path) -> None:
        # Arrange — only one response is queued; a second fetch would raise.
        transport = _StubTransport([HttpResponse(200, _COMPANY_TICKERS)])
        cache_path = tmp_path / "cik.json"
        resolver = CikResolver(_settings(), transport=transport, cache_path=cache_path)

        # Act — two lookups.
        resolver.resolve(Ticker("AAPL"))
        resolver.resolve(Ticker("MSFT"))

        # Assert — the network was hit exactly once; the cache file exists.
        assert len(transport.calls) == 1
        assert cache_path.exists()

    def test_reuses_cache_across_resolver_instances(self, tmp_path: Path) -> None:
        # Arrange — first resolver populates the cache.
        cache_path = tmp_path / "cik.json"
        first = CikResolver(
            _settings(),
            transport=_StubTransport([HttpResponse(200, _COMPANY_TICKERS)]),
            cache_path=cache_path,
        )
        first.resolve(Ticker("AAPL"))

        # A second resolver with a transport that would fail if used.
        empty_transport = _StubTransport([])
        second = CikResolver(_settings(), transport=empty_transport, cache_path=cache_path)

        # Act / Assert — served entirely from disk, no network.
        assert second.resolve(Ticker("AMZN")) == "0001018724"
        assert empty_transport.calls == []

    def test_corrupt_cache_is_ignored_and_refetched(self, tmp_path: Path) -> None:
        # Arrange — a cache file with invalid JSON.
        cache_path = tmp_path / "cik.json"
        cache_path.write_text("{not valid json", encoding="utf-8")
        transport = _StubTransport([HttpResponse(200, _COMPANY_TICKERS)])
        resolver = CikResolver(_settings(), transport=transport, cache_path=cache_path)

        # Act
        cik = resolver.resolve(Ticker("AAPL"))

        # Assert — it recovered by re-downloading.
        assert cik == "0000320193"
        assert len(transport.calls) == 1
        # And the cache was rewritten with valid JSON.
        json.loads(cache_path.read_text(encoding="utf-8"))


class TestErrors:
    def test_http_error_raises_runtime_error(self, tmp_path: Path) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(503, {})])
        resolver = CikResolver(
            _settings(),
            transport=transport,
            cache_path=tmp_path / "cik.json",
            max_retries=0,
        )

        # Act / Assert
        with pytest.raises(RuntimeError, match="CIK map request failed"):
            resolver.resolve(Ticker("AAPL"))
