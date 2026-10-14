"""Real end-to-end test against the live Alpaca API (opt-in).

Skipped unless **both** ``ALPHALAB_ALPACA_LIVE_TESTS=1`` and valid credentials
(``ALPHALAB_ALPACA_KEY_ID`` / ``ALPHALAB_ALPACA_SECRET_KEY``) are present. In an
environment without credentials this test *skips* — it never fails. Run with::

    ALPHALAB_ALPACA_LIVE_TESTS=1 \\
        ALPHALAB_ALPACA_KEY_ID=... ALPHALAB_ALPACA_SECRET_KEY=... \\
        uv run pytest tests/integration/test_alpaca_live.py -v -m alpaca_live --no-cov

The test is read-only: it fetches the universe and a short bar window. It never
places an order.
"""

from __future__ import annotations

import os
from datetime import date

import httpx
import pytest
from pydantic import ValidationError

from alphalab.adapters.alpaca_assets import AlpacaAssetsClient
from alphalab.adapters.alpaca_bars import AlpacaBarsClient, DataFeed
from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.edgar import HttpResponse
from alphalab.domain.types import Ticker

pytestmark = pytest.mark.alpaca_live

_LIVE = os.environ.get("ALPHALAB_ALPACA_LIVE_TESTS") == "1"


def _settings_or_skip() -> AlpacaSettings:
    """Build settings from the environment, skipping if anything is missing."""
    if not _LIVE:
        pytest.skip("set ALPHALAB_ALPACA_LIVE_TESTS=1 to run")
    try:
        return AlpacaSettings()  # type: ignore[call-arg]  # values from env/.env
    except ValidationError:
        pytest.skip("Alpaca credentials not set (ALPHALAB_ALPACA_KEY_ID/SECRET_KEY)")


class _HttpxTransport:
    def __init__(self) -> None:
        self._client = httpx.Client(timeout=30.0, follow_redirects=True)

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        response = self._client.get(url, headers=headers)
        payload: object = {}
        if response.status_code == 200:
            payload = response.json()
        return HttpResponse(status_code=response.status_code, json=payload)  # type: ignore[arg-type]


def test_live_universe_includes_delisted_names() -> None:
    # Arrange
    settings = _settings_or_skip()
    client = AlpacaAssetsClient(settings, transport=_HttpxTransport())

    # Act
    snapshot = client.fetch_universe()

    # Assert — the whole point: delisted names are present and counted.
    assert snapshot.active_count > 0
    assert snapshot.inactive_count > 0
    print(
        f"\n[alpaca] active={snapshot.active_count} inactive={snapshot.inactive_count} "
        f"survivorship_ratio={snapshot.survivorship_ratio:.3f}"
    )


def test_live_bars_are_adjusted_and_non_empty() -> None:
    # Arrange
    settings = _settings_or_skip()
    client = AlpacaBarsClient(settings, transport=_HttpxTransport(), feed=DataFeed.IEX)

    # Act
    result = client.daily_bars([Ticker("AAPL")], start=date(2021, 1, 4), end=date(2021, 1, 15))

    # Assert
    bars = result.get(Ticker("AAPL"), ())
    assert bars, "expected non-empty AAPL history"
    assert all(b.close > 0 for b in bars)
    print(f"[alpaca] AAPL bars={len(bars)} first={bars[0].on} last={bars[-1].on}")
