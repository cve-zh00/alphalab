"""CLI unit tests for ``alphalab autopilot`` and ``paper --strategy agentic``."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pytest

import alphalab.cli.commands.autopilot as autopilot_cmd
from alphalab.adapters.alpaca_broker import AlpacaBroker
from alphalab.adapters.alpaca_config import AlpacaEndpoint, AlpacaSettings
from alphalab.adapters.edgar import HttpResponse
from alphalab.adapters.kill_switch import KillSwitch
from alphalab.cli.app import main
from alphalab.domain.backtest import PriceBar
from alphalab.domain.types import Ticker

_AS_OF = date(2024, 6, 10)


def _settings() -> AlpacaSettings:
    return AlpacaSettings(
        key_id="paper-key",
        secret_key="super-secret-value",
        endpoint=AlpacaEndpoint.PAPER,
    )


@dataclass
class FakeCliBrokerTransport:
    equity: float = 100_000.0
    market_open: bool = True
    positions_payload: list[dict[str, object]] = field(default_factory=list)
    posts: list[dict[str, object]] = field(default_factory=list)

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        if url.endswith("/v2/account"):
            return HttpResponse(
                status_code=200,
                json={
                    "equity": str(self.equity),
                    "buying_power": str(self.equity * 2),
                    "trading_blocked": False,
                    "account_blocked": False,
                },
            )
        if url.endswith("/v2/positions"):
            return HttpResponse(status_code=200, json=self.positions_payload)  # type: ignore[arg-type]
        if url.endswith("/v2/clock"):
            return HttpResponse(status_code=200, json={"is_open": self.market_open})
        return HttpResponse(status_code=200, json={})

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> HttpResponse:
        self.posts.append(json)
        return HttpResponse(status_code=200, json={"id": f"srv-{len(self.posts)}"})


@dataclass
class FakeCliPrices:
    panel: Mapping[Ticker, Sequence[PriceBar]]

    def recent_bars(self) -> Mapping[Ticker, Sequence[PriceBar]]:
        return self.panel


def _panel() -> dict[Ticker, Sequence[PriceBar]]:
    base = _AS_OF - timedelta(days=120)
    series = tuple(
        PriceBar(on=base + timedelta(days=i), close=100.0 + i * 0.2, volume=2_000_000.0)
        for i in range(121)
    )
    return {
        Ticker("AAPL"): series,
        Ticker("MSFT"): series,
        Ticker("SPY"): series,
    }


@pytest.fixture
def patched_autopilot_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> FakeCliBrokerTransport:
    transport = FakeCliBrokerTransport()

    def _broker_factory(ks: KillSwitch) -> AlpacaBroker:
        return AlpacaBroker(_settings(), transport=transport, kill_switch=ks)

    def _prices_factory(_prices_dir: Path, **_kwargs: object) -> FakeCliPrices:
        return FakeCliPrices(_panel())

    monkeypatch.setattr(autopilot_cmd, "_default_broker", _broker_factory)
    monkeypatch.setattr(autopilot_cmd, "_default_prices", _prices_factory)
    monkeypatch.setattr(
        autopilot_cmd,
        "_default_llm_probe",
        lambda: (True, "Ollama up (qwen3-vl:latest)"),
    )
    return transport


def test_autopilot_preflight_cli_text_and_json(
    tmp_path: Path, patched_autopilot_env: FakeCliBrokerTransport
) -> None:
    _ = patched_autopilot_env
    out = io.StringIO()
    err = io.StringIO()
    code = main(
        [
            "autopilot",
            "preflight",
            "--source",
            "fixture",
            "--no-llm",
            "--halt-file",
            str(tmp_path / "HALT"),
            "--no-color",
        ],
        out=out,
        err=err,
    )
    assert code == 0
    assert "AUTOPILOT LAUNCH PREFLIGHT" in out.getvalue()
    assert "READY" in out.getvalue()

    out_json = io.StringIO()
    code_json = main(
        [
            "autopilot",
            "preflight",
            "--source",
            "fixture",
            "--no-llm",
            "--halt-file",
            str(tmp_path / "HALT"),
            "--json",
        ],
        out=out_json,
        err=err,
    )
    assert code_json == 0
    payload = json.loads(out_json.getvalue())
    assert payload["ready"] is True
    assert len(payload["checks"]) == 6


def test_autopilot_status_and_run_dry_and_submit(
    tmp_path: Path, patched_autopilot_env: FakeCliBrokerTransport
) -> None:
    out = io.StringIO()
    err = io.StringIO()
    common = [
        "--execution-db",
        str(tmp_path / "exec.db"),
        "--db",
        str(tmp_path / "journal.db"),
        "--abstentions-db",
        str(tmp_path / "abstentions.db"),
        "--halt-file",
        str(tmp_path / "HALT"),
        "--as-of",
        "2024-06-10",
        "--source",
        "fixture",
        "--no-llm",
        "--assume-verdict",
        "buy",
        "--tickers",
        "AAPL,MSFT",
        "--no-color",
    ]
    code_status = main(["autopilot", "status", *common], out=out, err=err)
    assert code_status == 0
    assert "AUTOPILOT AGENT STATUS" in out.getvalue()
    assert patched_autopilot_env.posts == []

    out_dry = io.StringIO()
    code_dry = main(["autopilot", "run", *common], out=out_dry, err=err)
    assert code_dry == 0
    assert "AUTOPILOT CYCLE — DRY RUN" in out_dry.getvalue()
    assert patched_autopilot_env.posts == []

    out_sub = io.StringIO()
    code_sub = main(["autopilot", "run", "--submit", *common, "--json"], out=out_sub, err=err)
    assert code_sub == 0
    payload = json.loads(out_sub.getvalue())
    assert payload["submitted"] is True
    assert len(patched_autopilot_env.posts) == 2


def test_autopilot_loop_runs_bounded_cycles(
    tmp_path: Path, patched_autopilot_env: FakeCliBrokerTransport
) -> None:
    _ = patched_autopilot_env
    out = io.StringIO()
    err = io.StringIO()
    code = main(
        [
            "autopilot",
            "loop",
            "--execution-db",
            str(tmp_path / "exec.db"),
            "--db",
            str(tmp_path / "journal.db"),
            "--abstentions-db",
            str(tmp_path / "abstentions.db"),
            "--halt-file",
            str(tmp_path / "HALT"),
            "--as-of",
            "2024-06-10",
            "--source",
            "fixture",
            "--no-llm",
            "--tickers",
            "AAPL",
            "--max-cycles",
            "2",
            "--interval-seconds",
            "0",
            "--no-color",
        ],
        out=out,
        err=err,
    )
    assert code == 0
    assert out.getvalue().count("AUTOPILOT CYCLE — DRY RUN") == 2


def test_expand_ticker_aliases_resolves_spy500_and_deduplicates(tmp_path: Path) -> None:
    from alphalab.cli.commands.autopilot import expand_ticker_aliases, load_sp500_tickers

    cache = tmp_path / "sp500.json"
    cache.write_text(json.dumps(["AAPL", "BRK.B", "MSFT"]), encoding="utf-8")
    loaded = load_sp500_tickers(cache_path=cache)
    assert loaded == ("AAPL", "BRK.B", "MSFT")

    expanded = expand_ticker_aliases(
        ("spy500", "NVDA", "aapl"),
        sp500_loader=lambda: loaded,
    )
    assert expanded == ("AAPL", "BRK.B", "MSFT", "NVDA")


def test_cik_resolver_normalizes_dots_and_hyphens(tmp_path: Path) -> None:
    from alphalab.adapters.cik import CikResolver, CikSettings

    cache = tmp_path / "cik.json"
    cache.write_text(
        json.dumps(
            {
                "0": {"ticker": "BRK-B", "cik_str": 1067983},
                "1": {"ticker": "BF.A", "cik_str": 14693},
            }
        ),
        encoding="utf-8",
    )
    resolver = CikResolver(
        CikSettings(user_agent="test test@example.com"),
        transport=object(),  # type: ignore[arg-type]
        cache_path=cache,
    )
    assert resolver.resolve(Ticker("BRK.B")) == "0001067983"
    assert resolver.resolve(Ticker("BF-A")) == "0000014693"
