"""Unit tests for AlpacaNewsClient and SEC EDGAR PIT filing disclosures."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import SecretStr

import alphalab.cli.ingest as ingest_mod
from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.alpaca_news import AlpacaNewsClient, NewsArticle
from alphalab.adapters.edgar import EdgarClient, EdgarSettings, HttpResponse
from alphalab.cli.ingest import load_live_builder
from alphalab.domain.types import Ticker


class _FakeClock:
    def __init__(self) -> None:
        self._now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self._now += seconds


class _QueueTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = list(responses)
        self.urls: list[str] = []

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        self.urls.append(url)
        return self._responses.pop(0)


def _alpaca_settings() -> AlpacaSettings:
    return AlpacaSettings(
        key_id="PKTEST",
        secret_key=SecretStr("SKTEST"),
    )


def test_alpaca_news_filters_pit_and_scrubs_currency_and_html() -> None:
    payload: dict[str, object] = {
        "news": [
            {
                "created_at": "2026-10-16T10:00:00Z",
                "headline": "Future leak headline after knowledge date",
                "summary": "Should be excluded by PIT filter",
                "source": "benzinga",
            },
            {
                "created_at": "2026-10-15T14:30:00Z",
                "headline": "<b>BMO Capital</b> Cuts Rating, Sets Target of $70",
                "summary": "<p>Software release 1.2.3 sees $1.5B demand.</p>",
                "source": "benzinga",
            },
            {
                "created_at": "2026-10-14T09:00:00Z",
                "headline": "Adobe Expands AI Partnership",
                "summary": "",
                "source": "reuters",
            },
        ]
    }
    transport = _QueueTransport([HttpResponse(status_code=200, json=payload)])
    client = AlpacaNewsClient(_alpaca_settings(), transport=transport, clock=_FakeClock())

    docs = client.recent_documents(Ticker("ADBE"), knowledge_date=date(2026, 10, 15), limit=5)

    assert len(docs) == 2
    assert "end=2026-10-15T23:59:59Z" in transport.urls[0]
    assert "Future leak" not in docs[0]
    assert "$70" not in docs[0]
    assert "[amount]" in docs[0]
    assert "[version]" in docs[0]
    assert "<b>" not in docs[0]
    assert docs[1] == "[Alpaca News 2026-10-14 | reuters] Adobe Expands AI Partnership"


def test_alpaca_news_retries_on_429_and_raises_on_400() -> None:
    clock = _FakeClock()
    transport = _QueueTransport(
        [
            HttpResponse(status_code=429, json={}),
            HttpResponse(
                status_code=200,
                json={
                    "news": [
                        {
                            "created_at": "2026-10-15T12:00:00Z",
                            "headline": "Valid headline",
                            "summary": "Valid summary",
                            "source": "benzinga",
                        }
                    ]
                },
            ),
        ]
    )
    client = AlpacaNewsClient(
        _alpaca_settings(),
        transport=transport,
        clock=clock,
        max_retries=2,
    )
    articles = client.recent_news(Ticker("AAPL"), knowledge_date=date(2026, 10, 15))
    assert len(articles) == 1
    assert articles[0] == NewsArticle(
        symbol="AAPL",
        created_on=date(2026, 10, 15),
        headline="Valid headline",
        summary="Valid summary",
        source="benzinga",
    )
    assert len(clock.slept) == 1

    bad_transport = _QueueTransport([HttpResponse(status_code=400, json={})])
    bad_client = AlpacaNewsClient(_alpaca_settings(), transport=bad_transport, clock=clock)
    with pytest.raises(RuntimeError, match="status=400"):
        bad_client.recent_news(Ticker("AAPL"), knowledge_date=date(2026, 10, 15))


def test_edgar_recent_filing_disclosures_filters_pit_and_maps_8k_items() -> None:
    submissions_payload: dict[str, object] = {
        "filings": {
            "recent": {
                "form": ["8-K", "8-K", "10-Q", "10-K", "4"],
                "filingDate": [
                    "2026-10-20",
                    "2026-09-10",
                    "2026-08-15",
                    "2026-02-20",
                    "2026-09-01",
                ],
                "items": ["2.02", "2.02,5.02,9.01", "", "", ""],
            }
        }
    }
    transport = _QueueTransport([HttpResponse(status_code=200, json=submissions_payload)])
    client = EdgarClient(
        EdgarSettings(user_agent="test test@example.com"),
        transport=transport,
        clock=_FakeClock(),
    )

    disclosures = client.recent_filing_disclosures(
        Ticker("ADBE"),
        cik="796343",
        knowledge_date=date(2026, 10, 15),
        limit=5,
    )

    assert len(disclosures) == 3
    assert "2026-10-20" not in " ".join(disclosures)
    assert (
        disclosures[0]
        == "[SEC EDGAR 8-K filed 2026-09-10] Current Report disclosures: "
        "Results of Operations and Financial Condition; "
        "Departure or Election of Directors or Principal Officers."
    )
    assert (
        disclosures[1]
        == "[SEC EDGAR 10-Q filed 2026-08-15] Quarterly Report (Form 10-Q) filed with the SEC."
    )
    assert (
        disclosures[2]
        == "[SEC EDGAR 10-K filed 2026-02-20] Annual Report (Form 10-K) filed with the SEC."
    )


def test_load_live_builder_attaches_sec_and_alpaca_news_documents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tests.unit.test_cli_ingest import _CIK_MAP, _full_facts

    monkeypatch.setenv("ALPHALAB_EDGAR_USER_AGENT", "alphalab-tests tests@example.com")
    monkeypatch.setattr(ingest_mod, "_cik_cache_path", lambda: tmp_path / "cik.json")
    monkeypatch.setattr(ingest_mod, "_pit_store_root", lambda: tmp_path / "pit")
    monkeypatch.setattr(
        ingest_mod, "_price_source", lambda: ingest_mod.CsvPriceSeriesSource(tmp_path / "prices")
    )

    submissions_payload: dict[str, object] = {
        "filings": {
            "recent": {
                "form": ["8-K"],
                "filingDate": ["2023-05-10"],
                "items": ["1.01,9.01"],
            }
        }
    }
    news_payload: dict[str, object] = {
        "news": [
            {
                "created_at": "2023-06-15T12:00:00Z",
                "headline": "Apple signs strategic supply agreement",
                "summary": "Expands domestic capacity",
                "source": "reuters",
            }
        ]
    }

    class _MultiTransport:
        def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
            if "company_tickers" in url:
                return HttpResponse(200, _CIK_MAP)
            if "/submissions/" in url:
                return HttpResponse(200, submissions_payload)
            if "/v1beta1/news" in url:
                return HttpResponse(200, news_payload)
            return HttpResponse(200, _full_facts())

    transport = _MultiTransport()
    monkeypatch.setattr(ingest_mod, "HttpxWriteTransport", lambda *a, **k: transport)
    news_client = AlpacaNewsClient(_alpaca_settings(), transport=transport, clock=_FakeClock())

    builder = load_live_builder(include_news=True, news_client=news_client)
    req = builder("AAPL", knowledge_date=date(2023, 6, 30))

    assert any("[SEC EDGAR 8-K filed 2023-05-10]" in d for d in req.documents)
    assert any("[Alpaca News 2023-06-15 | reuters]" in d for d in req.documents)
