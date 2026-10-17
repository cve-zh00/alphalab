"""Unit tests for NOFX launch preflight checks (`alphalab.app.autopilot_preflight`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from alphalab.adapters.alpaca_broker import AlpacaBroker
from alphalab.adapters.alpaca_config import AlpacaEndpoint, AlpacaSettings
from alphalab.adapters.edgar import HttpResponse
from alphalab.adapters.kill_switch import KillSwitch
from alphalab.app.autopilot_preflight import run_preflight
from alphalab.domain.backtest import PriceBar
from alphalab.domain.types import Ticker


def _settings() -> AlpacaSettings:
    return AlpacaSettings(
        key_id="paper-key",
        secret_key="super-secret-value",
        endpoint=AlpacaEndpoint.PAPER,
    )


@dataclass
class FakePreflightTransport:
    equity: float = 100_000.0
    market_open: bool = True
    trading_blocked: bool = False
    account_blocked: bool = False
    account_error: bool = False
    clock_error: bool = False
    positions_payload: list[dict[str, object]] = field(default_factory=list)

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        if url.endswith("/v2/account"):
            if self.account_error:
                return HttpResponse(status_code=500, json={"message": "down"})
            return HttpResponse(
                status_code=200,
                json={
                    "equity": str(self.equity),
                    "buying_power": str(self.equity * 2),
                    "trading_blocked": self.trading_blocked,
                    "account_blocked": self.account_blocked,
                },
            )
        if url.endswith("/v2/clock"):
            if self.clock_error:
                return HttpResponse(status_code=503, json={"message": "clock down"})
            return HttpResponse(status_code=200, json={"is_open": self.market_open})
        return HttpResponse(status_code=200, json={})

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> HttpResponse:
        return HttpResponse(status_code=200, json={"id": "srv-1"})


@dataclass
class DictPrices:
    panel: Mapping[Ticker, Sequence[PriceBar]]
    fail: bool = False

    def recent_bars(self) -> Mapping[Ticker, Sequence[PriceBar]]:
        if self.fail:
            raise RuntimeError("bars unavailable")
        return self.panel


def _bars() -> dict[Ticker, Sequence[PriceBar]]:
    series = tuple(
        PriceBar(on=date(2024, 1, 1) + timedelta(days=i), close=100.0 + i, volume=5_000_000.0)
        for i in range(10)
    )
    return {Ticker("AAPL"): series, Ticker("SPY"): series}


def test_preflight_passes_when_all_dependencies_healthy(tmp_path: Path) -> None:
    transport = FakePreflightTransport()
    ks = KillSwitch(halt_path=tmp_path / "HALT", env={})
    broker = AlpacaBroker(_settings(), transport=transport, kill_switch=ks)

    report = run_preflight(
        broker=broker,
        kill_switch=ks,
        prices=DictPrices(_bars()),
        source="live",
        use_llm=True,
        edgar_user_agent="alphalab/0.1 (ops@example.com)",
        llm_probe=lambda: (True, "Ollama reachable (qwen3-vl:latest)"),
    )

    assert report.ready is True
    assert report.blocking_failures == ()
    assert len(report.checks) == 6


def test_preflight_blocks_on_engaged_kill_switch(tmp_path: Path) -> None:
    halt = tmp_path / "HALT"
    halt.write_text("stop", encoding="utf-8")
    ks = KillSwitch(halt_path=halt, env={})
    broker = AlpacaBroker(_settings(), transport=FakePreflightTransport(), kill_switch=ks)

    report = run_preflight(
        broker=broker,
        kill_switch=ks,
        prices=DictPrices(_bars()),
        source="fixture",
        use_llm=False,
    )

    assert report.ready is False
    names = [c.name for c in report.blocking_failures]
    assert "kill_switch" in names


def test_preflight_blocks_on_blocked_account_or_zero_equity(tmp_path: Path) -> None:
    ks = KillSwitch(halt_path=tmp_path / "HALT", env={})
    broker_blocked = AlpacaBroker(
        _settings(),
        transport=FakePreflightTransport(trading_blocked=True),
        kill_switch=ks,
    )
    report_blocked = run_preflight(
        broker=broker_blocked,
        kill_switch=ks,
        prices=DictPrices(_bars()),
        source="fixture",
        use_llm=False,
    )
    assert report_blocked.ready is False
    assert [c.name for c in report_blocked.blocking_failures] == ["broker_account"]

    broker_zero = AlpacaBroker(
        _settings(),
        transport=FakePreflightTransport(equity=0.0),
        kill_switch=ks,
    )
    report_zero = run_preflight(
        broker=broker_zero,
        kill_switch=ks,
        prices=DictPrices(_bars()),
        source="fixture",
        use_llm=False,
    )
    assert report_zero.ready is False
    assert [c.name for c in report_zero.blocking_failures] == ["broker_account"]


def test_preflight_market_closed_is_non_blocking_unless_required(tmp_path: Path) -> None:
    ks = KillSwitch(halt_path=tmp_path / "HALT", env={})
    broker = AlpacaBroker(
        _settings(),
        transport=FakePreflightTransport(market_open=False),
        kill_switch=ks,
    )

    non_blocking = run_preflight(
        broker=broker,
        kill_switch=ks,
        prices=DictPrices(_bars()),
        source="fixture",
        use_llm=False,
        require_market_open=False,
    )
    assert non_blocking.ready is True

    blocking = run_preflight(
        broker=broker,
        kill_switch=ks,
        prices=DictPrices(_bars()),
        source="fixture",
        use_llm=False,
        require_market_open=True,
    )
    assert blocking.ready is False
    assert [c.name for c in blocking.blocking_failures] == ["market_clock"]


def test_preflight_blocks_on_missing_spy_or_edgar_ua_or_llm(tmp_path: Path) -> None:
    ks = KillSwitch(halt_path=tmp_path / "HALT", env={})
    broker = AlpacaBroker(_settings(), transport=FakePreflightTransport(), kill_switch=ks)
    only_aapl = {Ticker("AAPL"): _bars()[Ticker("AAPL")]}

    report = run_preflight(
        broker=broker,
        kill_switch=ks,
        prices=DictPrices(only_aapl),
        source="live",
        use_llm=True,
        edgar_user_agent="",
        llm_probe=lambda: (False, "Ollama down"),
    )

    assert report.ready is False
    failed = {c.name for c in report.blocking_failures}
    assert failed == {"price_feed", "fundamentals_source", "ai_model"}
