"""Alpaca historical daily bars adapter, dividend/split adjusted.

Why ``adjustment=all``
----------------------
A cross-sectional momentum ranking compares returns across names on the same
day. If prices are not adjusted for dividends, a dividend payer looks like it
dropped on the ex-date while a non-payer does not, and the ranking is corrupted
by a mechanical artefact rather than information. This project has already
stumbled on exactly that, so every request here asks for ``adjustment=all``
(splits *and* dividends).

Correctness obligations
------------------------
* **Pagination is mandatory.** Alpaca returns at most a page of bars and a
  ``next_page_token``; following it to exhaustion is the difference between a
  complete series and one silently truncated at the first page — the worst
  possible failure for a backtest.
* **No fabrication.** Gaps are left as gaps. The adapter never interpolates,
  forward-fills, or invents a bar.
* **Feed is explicit.** Free Alpaca plans are limited to the ``iex`` feed; the
  SIP feed needs a paid subscription. The feed is configurable and a
  subscription error is surfaced as an actionable message rather than an opaque
  failure.

Transport, throttling, and retries follow the shared EDGAR pattern: an injected
:class:`~alphalab.adapters.edgar.Transport`, a :class:`RateLimiter`, and
exponential backoff on 429/5xx.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.edgar import Clock, RateLimiter, Transport, _SystemClock
from alphalab.domain.backtest.types import PriceBar
from alphalab.domain.types import Ticker

__all__ = [
    "AlpacaBarsClient",
    "DataFeed",
    "FeedSubscriptionError",
    "RawBar",
]

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_FORBIDDEN = 403
_SECONDS_PER_MINUTE = 60.0


class DataFeed(StrEnum):
    """Market-data feed. ``IEX`` is free; ``SIP`` requires a paid plan."""

    IEX = "iex"
    SIP = "sip"


class FeedSubscriptionError(RuntimeError):
    """Raised when the account's plan does not grant access to the chosen feed."""


@dataclass(frozen=True)
class RawBar:
    """One daily bar as returned by Alpaca, before domain conversion.

    Kept distinct from :class:`~alphalab.domain.backtest.types.PriceBar` so the
    adapter owns the wire shape and the domain stays ignorant of the provider.
    """

    symbol: str
    on: date
    close: float
    volume: float | None

    def to_price_bar(self) -> PriceBar:
        """Convert to the domain :class:`PriceBar` (validates ``close > 0``)."""
        return PriceBar(on=self.on, close=self.close, volume=self.volume)


class AlpacaBarsClient:
    """Fetches adjusted daily bars for one or many symbols, with pagination."""

    def __init__(
        self,
        settings: AlpacaSettings,
        *,
        transport: Transport,
        feed: DataFeed = DataFeed.IEX,
        clock: Clock | None = None,
        max_retries: int = 5,
        base_backoff: float = 0.5,
        page_limit: int = 10_000,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._feed = feed
        self._clock = clock or _SystemClock()
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._page_limit = page_limit
        self._limiter = RateLimiter(
            max_rps=settings.rate_limit_per_minute / _SECONDS_PER_MINUTE,
            clock=self._clock,
        )

    def daily_bars(
        self, symbols: tuple[Ticker, ...] | list[Ticker], *, start: date, end: date
    ) -> dict[Ticker, tuple[PriceBar, ...]]:
        """Return adjusted daily bars per symbol over ``[start, end]``.

        Multiple symbols are requested together (Alpaca accepts a comma list),
        and ``next_page_token`` is followed to exhaustion so no bar is lost.
        """
        if not symbols:
            return {}
        raw = self._fetch_all_pages(symbols, start=start, end=end)
        return self._group_to_domain(raw)

    def _fetch_all_pages(
        self, symbols: tuple[Ticker, ...] | list[Ticker], *, start: date, end: date
    ) -> list[RawBar]:
        bars: list[RawBar] = []
        page_token: str | None = None
        while True:
            url = self._build_url(symbols, start=start, end=end, page_token=page_token)
            payload = self._get_with_retry(url)
            bars.extend(_parse_bars(payload))
            page_token = _next_token(payload)
            if page_token is None:
                return bars

    def _build_url(
        self,
        symbols: tuple[Ticker, ...] | list[Ticker],
        *,
        start: date,
        end: date,
        page_token: str | None,
    ) -> str:
        joined = ",".join(str(s) for s in symbols)
        url = (
            f"{self._settings.data_base_url()}/v2/stocks/bars"
            f"?symbols={joined}&timeframe=1Day"
            f"&start={start.isoformat()}&end={end.isoformat()}"
            f"&adjustment=all&feed={self._feed.value}&limit={self._page_limit}"
        )
        if page_token is not None:
            url = f"{url}&page_token={page_token}"
        return url

    def _get_with_retry(self, url: str) -> dict[str, object]:
        headers = self._settings.auth_headers()
        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            response = self._transport.get(url, headers=headers)
            if response.status_code == 200:
                return response.json
            if response.status_code == _FORBIDDEN:
                raise FeedSubscriptionError(
                    f"Alpaca denied access to feed={self._feed.value!r} (HTTP 403). "
                    "The free plan is limited to the IEX feed; SIP needs a paid "
                    "subscription. Pass feed=DataFeed.IEX or upgrade the plan."
                )
            last_status = response.status_code
            if response.status_code not in _RETRYABLE_STATUS:
                break
            if attempt < self._max_retries:
                self._clock.sleep(self._base_backoff * (2**attempt))
        raise RuntimeError(f"Alpaca bars request failed (status={last_status})")

    @staticmethod
    def _group_to_domain(raw: list[RawBar]) -> dict[Ticker, tuple[PriceBar, ...]]:
        grouped: dict[Ticker, list[PriceBar]] = {}
        for bar in raw:
            grouped.setdefault(Ticker(bar.symbol), []).append(bar.to_price_bar())
        return {ticker: tuple(sorted(bars, key=lambda b: b.on)) for ticker, bars in grouped.items()}


def _parse_bars(payload: dict[str, object]) -> list[RawBar]:
    """Flatten Alpaca's ``{"bars": {SYMBOL: [bar, ...]}}`` into :class:`RawBar`."""
    bars_by_symbol = payload.get("bars")
    if not isinstance(bars_by_symbol, dict):
        return []
    out: list[RawBar] = []
    for symbol, rows in bars_by_symbol.items():
        if not isinstance(symbol, str) or not isinstance(rows, list):
            continue
        for row in rows:
            bar = _parse_one(symbol, row)
            if bar is not None:
                out.append(bar)
    return out


def _parse_one(symbol: str, row: object) -> RawBar | None:
    if not isinstance(row, dict):
        return None
    stamp = row.get("t")
    close = row.get("c")
    if not isinstance(stamp, str) or not isinstance(close, int | float):
        return None
    volume = row.get("v")
    return RawBar(
        symbol=symbol,
        on=_parse_timestamp(stamp),
        close=float(close),
        volume=float(volume) if isinstance(volume, int | float) else None,
    )


def _next_token(payload: dict[str, object]) -> str | None:
    token = payload.get("next_page_token")
    return token if isinstance(token, str) and token else None


def _parse_timestamp(value: str) -> date:
    """Parse an RFC 3339 timestamp (e.g. ``2021-01-04T05:00:00Z``) to a date."""
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).date()
