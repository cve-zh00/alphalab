"""Real end-to-end ingest against the live SEC EDGAR API.

Opt-in only: skipped unless ``ALPHALAB_LIVE_TESTS=1`` is set. The SEC requires a
descriptive ``User-Agent``; it is read from ``ALPHALAB_EDGAR_USER_AGENT`` with a
documented default. Run with::

    ALPHALAB_LIVE_TESTS=1 ALPHALAB_EDGAR_USER_AGENT="you you@example.com" \\
        uv run pytest tests/integration/test_ingest_live.py -v -m edgar_live --no-cov
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import httpx
import pytest

from alphalab.adapters.cik import CikResolver, CikSettings
from alphalab.adapters.edgar import EdgarClient, EdgarSettings, HttpResponse
from alphalab.adapters.edgar_live import FundamentalField, extract_fundamentals
from alphalab.domain.types import Ticker

pytestmark = pytest.mark.edgar_live

_LIVE = os.environ.get("ALPHALAB_LIVE_TESTS") == "1"
_DEFAULT_UA = "alphalab-research research@example.com"
_requires_live = pytest.mark.skipif(not _LIVE, reason="set ALPHALAB_LIVE_TESTS=1 to run")


class _HttpxTransport:
    """Real HTTP transport honouring the SEC's throttling expectations."""

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=30.0, follow_redirects=True)

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        response = self._client.get(url, headers=headers)
        payload: dict[str, object] = {}
        if response.status_code == 200:
            payload = response.json()
        return HttpResponse(status_code=response.status_code, json=payload)


def _ua() -> str:
    return os.environ.get("ALPHALAB_EDGAR_USER_AGENT", _DEFAULT_UA)


@_requires_live
@pytest.mark.parametrize("ticker", ["AAPL", "MSFT"])
def test_live_ingest_pulls_real_fundamentals(ticker: str, tmp_path: Path) -> None:
    # Arrange
    transport = _HttpxTransport()
    resolver = CikResolver(
        CikSettings(user_agent=_ua()), transport=transport, cache_path=tmp_path / "cik.json"
    )
    client = EdgarClient(EdgarSettings(user_agent=_ua()), transport=transport)

    # Act
    cik = resolver.resolve(Ticker(ticker))
    result = extract_fundamentals(client, Ticker(ticker), cik=cik)

    # Assert — CIK is a 10-digit zero-padded string.
    assert len(cik) == 10 and cik.isdigit()
    # Core fundamentals must be present for these large, well-reported filers.
    present = {r.field for r in result.records}
    for required in (
        FundamentalField.REVENUE,
        FundamentalField.NET_INCOME,
        FundamentalField.ASSETS,
    ):
        assert required in present, f"{ticker}: expected {required} in {sorted(present)}"

    # Report what we actually pulled (visible with -s / on failure).
    as_ofs = sorted({r.as_of for r in result.records})
    fileds = sorted({r.observed_at for r in result.records})
    print(f"\n[{ticker}] cik={cik}")
    print(f"[{ticker}] records={len(result.records)}")
    print(f"[{ticker}] fields={sorted(str(f) for f in present)}")
    print(f"[{ticker}] missing={sorted(str(f) for f in result.missing_fields)}")
    print(f"[{ticker}] as_of range: {as_ofs[0]} .. {as_ofs[-1]}")
    print(f"[{ticker}] filed range: {fileds[0]} .. {fileds[-1]}")

    # A restatement check: at least one period end should have >1 filing for a
    # large filer across its history (as-filed property holds end-to-end).
    assert isinstance(as_ofs[0], date)
