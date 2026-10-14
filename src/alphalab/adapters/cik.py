"""Ticker -> CIK resolution against the SEC ``company_tickers.json`` map.

The SEC publishes the full ticker/CIK mapping at a single stable URL. We fetch
it once, cache it to disk as JSON, and serve every subsequent lookup from the
cache so a batch ingest hits the network at most once. The ``companyfacts`` API
requires a 10-digit, zero-padded CIK, so :meth:`CikResolver.resolve` returns
exactly that.

As with the EDGAR adapter, the one HTTP call sits behind the shared
:class:`~alphalab.adapters.edgar.Transport` protocol (tests use a double, never
the network), the identifying ``User-Agent`` is read from configuration — never
hardcoded — and retries back off exponentially on throttling/5xx.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from alphalab.adapters.edgar import Clock, Transport
from alphalab.domain.types import Ticker

_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_CIK_WIDTH = 10


@dataclass
class _SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class CikSettings(BaseSettings):
    """Configuration for the CIK resolver.

    ``user_agent`` is required and carries no default so the resolver cannot
    run anonymously against the SEC, matching the EDGAR adapter's policy.
    """

    model_config = SettingsConfigDict(env_prefix="ALPHALAB_EDGAR_")

    user_agent: str


@dataclass
class CikResolver:
    """Resolves a ticker symbol to its zero-padded SEC CIK, cached on disk."""

    settings: CikSettings
    transport: Transport
    cache_path: Path
    clock: Clock = field(default_factory=_SystemClock)
    max_retries: int = 5
    base_backoff: float = 0.5
    _by_ticker: dict[str, str] | None = field(default=None, init=False)

    def resolve(self, ticker: Ticker) -> str:
        """Return the 10-digit, zero-padded CIK for ``ticker``.

        Raises:
            KeyError: if the ticker is absent from the SEC map.
            RuntimeError: if the map cannot be fetched.
        """
        mapping = self._mapping()
        key = str(ticker).upper()
        try:
            return mapping[key]
        except KeyError as exc:
            raise KeyError(f"ticker {ticker!r} not found in SEC CIK map") from exc

    def _mapping(self) -> dict[str, str]:
        if self._by_ticker is None:
            self._by_ticker = self._load_or_fetch()
        return self._by_ticker

    def _load_or_fetch(self) -> dict[str, str]:
        cached = self._read_cache()
        if cached is not None:
            return cached
        payload = self._download()
        mapping = _parse_map(payload)
        self._write_cache(payload)
        return mapping

    def _read_cache(self) -> dict[str, str] | None:
        if not self.cache_path.exists():
            return None
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(raw, dict):
            return None
        return _parse_map(raw)

    def _write_cache(self, payload: dict[str, object]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(payload), encoding="utf-8")

    def _download(self) -> dict[str, object]:
        headers = {"User-Agent": self.settings.user_agent}
        last_status: int | None = None
        for attempt in range(self.max_retries + 1):
            response = self.transport.get(_MAP_URL, headers=headers)
            if response.status_code == 200:
                return response.json
            last_status = response.status_code
            if response.status_code not in _RETRYABLE_STATUS:
                break
            if attempt < self.max_retries:
                self.clock.sleep(self.base_backoff * (2**attempt))
        raise RuntimeError(f"CIK map request failed (status={last_status})")


def _parse_map(payload: dict[str, object]) -> dict[str, str]:
    """Flatten the SEC map into ``{TICKER: zero-padded CIK}``."""
    mapping: dict[str, str] = {}
    for entry in payload.values():
        if not isinstance(entry, dict):
            continue
        ticker = entry.get("ticker")
        cik_str = entry.get("cik_str")
        if not isinstance(ticker, str) or not isinstance(cik_str, int):
            continue
        mapping[ticker.upper()] = str(cik_str).zfill(_CIK_WIDTH)
    return mapping
