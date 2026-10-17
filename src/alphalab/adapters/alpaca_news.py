"""Alpaca point-in-time news adapter (``/v1beta1/news``).

Correctness obligations
-----------------------
* **Strict point-in-time boundary.** Every query passes ``end={knowledge_date}T23:59:59Z``
  on the wire *and* filters ``created_on <= knowledge_date`` in memory before
  returning, so no headline published after ``knowledge_date`` can leak into an
  evaluation or backtest.
* **Clean qualitative text.** HTML tags and control whitespace are stripped so
  the LLM receives plain narrative text ready for qualitative extraction.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date, datetime

from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.edgar import Clock, RateLimiter, Transport, _SystemClock
from alphalab.domain.types import Ticker

__all__ = [
    "AlpacaNewsClient",
    "NewsArticle",
]

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_SECONDS_PER_MINUTE = 60.0
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CURRENCY_RE = re.compile(
    r"\$\s?\d[\d,.]*(?:\s?(?:billion|million|trillion|B|M|K))?",
    re.IGNORECASE,
)
_DOUBLE_DECIMAL_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)+\b")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class NewsArticle:
    """One point-in-time news item associated with a symbol."""

    symbol: str
    created_on: date
    headline: str
    summary: str
    source: str

    def to_document(self) -> str:
        """Render the article as a compact source document for LLM extraction."""
        prefix = f"[Alpaca News {self.created_on.isoformat()} | {self.source}]"
        if self.summary:
            return f"{prefix} {self.headline} — {self.summary}"
        return f"{prefix} {self.headline}"


class AlpacaNewsClient:
    """Fetches point-in-time news articles for a symbol up to ``knowledge_date``."""

    def __init__(
        self,
        settings: AlpacaSettings,
        *,
        transport: Transport,
        clock: Clock | None = None,
        max_retries: int = 3,
        base_backoff: float = 0.5,
        default_limit: int = 5,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._clock = clock or _SystemClock()
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._default_limit = default_limit
        self._limiter = RateLimiter(
            max_rps=settings.rate_limit_per_minute / _SECONDS_PER_MINUTE,
            clock=self._clock,
        )

    def recent_news(
        self,
        ticker: Ticker,
        *,
        knowledge_date: date,
        limit: int | None = None,
    ) -> tuple[NewsArticle, ...]:
        """Return up to ``limit`` news articles for ``ticker`` on or before ``knowledge_date``."""
        effective_limit = limit if limit is not None else self._default_limit
        if effective_limit <= 0:
            return ()
        url = self._build_url(ticker, knowledge_date=knowledge_date, limit=effective_limit)
        payload = self._get_with_retry(url)
        articles = _parse_news(str(ticker), payload, knowledge_date=knowledge_date)
        return tuple(articles[:effective_limit])

    def recent_documents(
        self,
        ticker: Ticker,
        *,
        knowledge_date: date,
        limit: int | None = None,
    ) -> tuple[str, ...]:
        """Return formatted narrative strings for ``ticker`` on or before ``knowledge_date``."""
        return tuple(
            article.to_document()
            for article in self.recent_news(ticker, knowledge_date=knowledge_date, limit=limit)
        )

    def _build_url(self, ticker: Ticker, *, knowledge_date: date, limit: int) -> str:
        end_iso = f"{knowledge_date.isoformat()}T23:59:59Z"
        return (
            f"{self._settings.data_base_url()}/v1beta1/news"
            f"?symbols={ticker}&end={end_iso}&limit={limit}&sort=desc"
        )

    def _get_with_retry(self, url: str) -> dict[str, object]:
        headers = self._settings.auth_headers()
        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            response = self._transport.get(url, headers=headers)
            if response.status_code == 200:
                return response.json
            last_status = response.status_code
            if response.status_code not in _RETRYABLE_STATUS:
                break
            if attempt < self._max_retries:
                self._clock.sleep(self._base_backoff * (2**attempt))
        raise RuntimeError(f"Alpaca news request failed (status={last_status})")


def _parse_news(
    symbol: str,
    payload: dict[str, object],
    *,
    knowledge_date: date,
) -> list[NewsArticle]:
    rows = payload.get("news")
    if not isinstance(rows, list):
        return []
    out: list[NewsArticle] = []
    for row in rows:
        article = _parse_one_article(symbol, row, knowledge_date=knowledge_date)
        if article is not None:
            out.append(article)
    return out


def _parse_one_article(
    symbol: str,
    row: object,
    *,
    knowledge_date: date,
) -> NewsArticle | None:
    if not isinstance(row, dict):
        return None
    stamp = row.get("created_at")
    headline_raw = row.get("headline")
    if not isinstance(stamp, str) or not isinstance(headline_raw, str):
        return None
    try:
        created_on = _parse_timestamp(stamp)
    except ValueError:
        return None
    if created_on > knowledge_date:
        return None
    headline = _clean_text(headline_raw)
    if not headline:
        return None
    summary_raw = row.get("summary")
    summary = _clean_text(summary_raw) if isinstance(summary_raw, str) else ""
    source_raw = row.get("source")
    source = (
        _clean_text(source_raw)
        if isinstance(source_raw, str) and source_raw.strip()
        else "AlpacaNews"
    )
    return NewsArticle(
        symbol=symbol,
        created_on=created_on,
        headline=headline,
        summary=summary,
        source=source,
    )


def _clean_text(text: str) -> str:
    unescaped = html.unescape(text)
    stripped = _HTML_TAG_RE.sub(" ", unescaped)
    no_currency = _CURRENCY_RE.sub("[amount]", stripped)
    no_double_dec = _DOUBLE_DECIMAL_RE.sub("[version]", no_currency)
    return _WHITESPACE_RE.sub(" ", no_double_dec).strip()


def _parse_timestamp(value: str) -> date:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).date()
