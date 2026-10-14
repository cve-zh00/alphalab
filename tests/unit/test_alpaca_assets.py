"""Unit tests for the Alpaca assets adapter (HTTP doubles, no network)."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from alphalab.adapters.alpaca_assets import (
    AlpacaAsset,
    AlpacaAssetsClient,
    UniverseSnapshot,
)
from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.edgar import HttpResponse

_SECRET = "super-secret-value-should-never-leak"


def _settings() -> AlpacaSettings:
    return AlpacaSettings(key_id="KID", secret_key=SecretStr(_SECRET))


def _asset_row(symbol: str, status: str, cls: str = "us_equity") -> dict[str, object]:
    return {
        "symbol": symbol,
        "name": f"{symbol} Inc",
        "exchange": "NASDAQ",
        "class": cls,
        "status": status,
        "tradable": status == "active",
        "fractionable": True,
    }


class _StubTransport:
    """Replays queued responses keyed by call order; records requests."""

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


class TestFetchUniverse:
    def test_active_and_inactive_are_separated_and_marked(self) -> None:
        # Arrange — one page of active, one page of inactive (two calls).
        active = HttpResponse(200, [_asset_row("AAPL", "active")])  # type: ignore[arg-type]
        inactive = HttpResponse(200, [_asset_row("ENRN", "inactive")])  # type: ignore[arg-type]
        transport = _StubTransport([active, inactive])
        client = AlpacaAssetsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        snapshot = client.fetch_universe()

        # Assert — the delisted name is present and flagged.
        assert snapshot.active_count == 1
        assert snapshot.inactive_count == 1
        delisted = snapshot.inactive[0]
        assert delisted.symbol == "ENRN"
        assert delisted.is_delisted is True
        assert snapshot.active[0].is_delisted is False

    def test_survivorship_ratio_is_inactive_over_active(self) -> None:
        # Arrange
        active = HttpResponse(200, [_asset_row("A", "active"), _asset_row("B", "active")])  # type: ignore[arg-type]
        inactive = HttpResponse(200, [_asset_row("C", "inactive")])  # type: ignore[arg-type]
        transport = _StubTransport([active, inactive])
        client = AlpacaAssetsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        snapshot = client.fetch_universe()

        # Assert
        assert snapshot.survivorship_ratio == pytest.approx(0.5)
        assert len(snapshot.all_assets) == 3

    def test_non_us_equity_rows_are_filtered_out(self) -> None:
        # Arrange — a crypto row must not enter the equity universe.
        rows = [_asset_row("AAPL", "active"), _asset_row("BTCUSD", "active", cls="crypto")]
        transport = _StubTransport(
            [HttpResponse(200, rows), HttpResponse(200, [])]  # type: ignore[arg-type]
        )
        client = AlpacaAssetsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        snapshot = client.fetch_universe()

        # Assert
        assert [a.symbol for a in snapshot.active] == ["AAPL"]

    def test_request_targets_paper_host_without_live_ack(self) -> None:
        # Arrange
        transport = _StubTransport(
            [HttpResponse(200, []), HttpResponse(200, [])]  # type: ignore[arg-type]
        )
        client = AlpacaAssetsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        client.fetch_universe()

        # Assert — reads go to the paper trading host; both statuses requested.
        urls = [url for url, _ in transport.calls]
        assert all("paper-api.alpaca.markets" in u for u in urls)
        assert any("status=active" in u for u in urls)
        assert any("status=inactive" in u for u in urls)

    def test_empty_active_yields_zero_ratio_not_division_error(self) -> None:
        # Arrange
        transport = _StubTransport(
            [HttpResponse(200, []), HttpResponse(200, [_asset_row("X", "inactive")])]  # type: ignore[arg-type]
        )
        client = AlpacaAssetsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        snapshot = client.fetch_universe()

        # Assert
        assert snapshot.survivorship_ratio == 0.0


class TestAuthAndSecrets:
    def test_auth_headers_are_sent(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(200, [])])  # type: ignore[arg-type]
        client = AlpacaAssetsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        client.fetch_assets(status="active")

        # Assert
        _, headers = transport.calls[0]
        assert headers["APCA-API-KEY-ID"] == "KID"
        assert headers["APCA-API-SECRET-KEY"] == _SECRET

    def test_secret_never_appears_in_asset_repr(self) -> None:
        # Arrange / Act
        asset = AlpacaAsset(
            symbol="AAPL",
            name="Apple",
            exchange="NASDAQ",
            asset_class="us_equity",
            status="active",
            tradable=True,
            fractionable=True,
        )

        # Assert — the value object carries no credential.
        assert _SECRET not in repr(asset)

    def test_secret_not_in_snapshot_or_settings_repr(self) -> None:
        # Arrange
        settings = _settings()
        snapshot = UniverseSnapshot(active=(), inactive=())

        # Assert — SecretStr redacts in repr, and nothing echoes the raw value.
        assert _SECRET not in repr(settings)
        assert _SECRET not in repr(snapshot)


class TestRetries:
    def test_retries_on_429_then_succeeds(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(429, []), HttpResponse(200, [])])  # type: ignore[arg-type]
        clock = _FakeClock()
        client = AlpacaAssetsClient(_settings(), transport=transport, clock=clock)

        # Act
        client.fetch_assets(status="active")

        # Assert
        assert len(transport.calls) == 2
        assert clock.sleeps

    def test_retries_on_5xx(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(503, []), HttpResponse(200, [])])  # type: ignore[arg-type]
        client = AlpacaAssetsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act
        client.fetch_assets(status="active")

        # Assert
        assert len(transport.calls) == 2

    def test_non_retryable_4xx_raises_without_leaking_secret(self) -> None:
        # Arrange — 400 is not retryable.
        transport = _StubTransport([HttpResponse(400, [])])  # type: ignore[arg-type]
        client = AlpacaAssetsClient(_settings(), transport=transport, clock=_FakeClock())

        # Act / Assert
        with pytest.raises(RuntimeError, match="Alpaca assets request failed") as exc:
            client.fetch_assets(status="active")
        assert _SECRET not in str(exc.value)
        assert len(transport.calls) == 1

    def test_gives_up_after_max_retries(self) -> None:
        # Arrange
        transport = _StubTransport([HttpResponse(503, [])] * 10)  # type: ignore[arg-type]
        client = AlpacaAssetsClient(
            _settings(), transport=transport, clock=_FakeClock(), max_retries=2
        )

        # Act / Assert
        with pytest.raises(RuntimeError, match="status=503"):
            client.fetch_assets(status="active")
        assert len(transport.calls) == 3  # initial + 2 retries
