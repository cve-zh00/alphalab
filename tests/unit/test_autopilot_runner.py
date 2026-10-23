"""Unit tests for the autonomous agentic runner (`alphalab.app.autopilot_runner`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from alphalab.adapters.alpaca_broker import AlpacaBroker
from alphalab.adapters.alpaca_config import AlpacaEndpoint, AlpacaSettings
from alphalab.adapters.edgar import HttpResponse
from alphalab.adapters.execution_log import (
    ExecutionEvent,
    ExecutionEventKind,
    SqliteExecutionLog,
)
from alphalab.adapters.kill_switch import KillSwitch
from alphalab.app.autopilot_runner import (
    AGENTIC_STRATEGY_NAME,
    AutopilotRunner,
    select_liquid_candidates,
)
from alphalab.artifact import EquityResearchArtifact, PriceTarget, Verdict
from alphalab.domain.backtest import PriceBar
from alphalab.domain.execution import GuardActionKind
from alphalab.domain.forecast import Abstention
from alphalab.domain.types import Ticker

_NOW = datetime(2024, 6, 10, 15, 0, tzinfo=UTC)
_AS_OF = date(2024, 6, 10)


def _settings() -> AlpacaSettings:
    return AlpacaSettings(
        key_id="paper-key",
        secret_key="super-secret-value",
        endpoint=AlpacaEndpoint.PAPER,
    )


@dataclass
class FakeBrokerTransport:
    equity: float = 100_000.0
    market_open: bool = True
    trading_blocked: bool = False
    account_blocked: bool = False
    positions_payload: list[dict[str, object]] = field(default_factory=list)
    clock_payload: dict[str, object] | None = None
    posts: list[dict[str, object]] = field(default_factory=list)

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        if url.endswith("/v2/account"):
            return HttpResponse(
                status_code=200,
                json={
                    "equity": str(self.equity),
                    "buying_power": str(self.equity * 2),
                    "trading_blocked": self.trading_blocked,
                    "account_blocked": self.account_blocked,
                },
            )
        if url.endswith("/v2/positions"):
            return HttpResponse(status_code=200, json=self.positions_payload)  # type: ignore[arg-type]
        if url.endswith("/v2/clock"):
            if self.clock_payload is not None:
                return HttpResponse(status_code=200, json=self.clock_payload)
            return HttpResponse(status_code=200, json={"is_open": self.market_open})
        return HttpResponse(status_code=200, json={})

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> HttpResponse:
        self.posts.append(json)
        return HttpResponse(status_code=200, json={"id": f"srv-{len(self.posts)}"})


@dataclass
class DictPrices:
    panel: Mapping[Ticker, Sequence[PriceBar]]

    def recent_bars(self) -> Mapping[Ticker, Sequence[PriceBar]]:
        return self.panel


def _series(
    start: float, days: int = 120, step: float = 0.2, volume: float = 2_000_000.0
) -> tuple[PriceBar, ...]:
    base = _AS_OF - timedelta(days=days - 1)
    return tuple(
        PriceBar(on=base + timedelta(days=i), close=start + i * step, volume=volume)
        for i in range(days)
    )


def _panel() -> dict[Ticker, Sequence[PriceBar]]:
    return {
        Ticker("AAPL"): _series(100.0),
        Ticker("MSFT"): _series(200.0),
        Ticker("GOOG"): _series(150.0),
        Ticker("SPY"): _series(400.0),
    }


def _artifact(ticker: str, verdict: Verdict) -> EquityResearchArtifact:
    return EquityResearchArtifact(
        ticker=ticker,
        knowledge_date=_AS_OF,
        verdict=verdict,
        price_target=PriceTarget(value=140.0, unit="USD_per_share"),
        price_range=None,
        current_price=100.0,
        methods_used=("dcf",),
        method_values={},
        consensus_relative_span=0.1,
        narrative_slots={},
        fallback_slots={},
        contract_stamps=(),
        valuation_withheld=False,
        single_method_disclosed=False,
    )


def test_select_liquid_candidates_excludes_spy_and_keeps_held_positions() -> None:
    panel = _panel()
    # Add an illiquid held position (volume too low to pass screen on its own)
    panel[Ticker("HELD")] = _series(20.0, volume=100.0)

    candidates = select_liquid_candidates(_AS_OF, panel, held_tickers=[Ticker("HELD")], cap=2)

    assert Ticker("SPY") not in candidates
    assert Ticker("HELD") in candidates
    assert len(candidates) == 3  # top-2 liquid + 1 held


def test_run_cycle_evaluates_candidates_settles_and_submits_orders(tmp_path: Path) -> None:
    transport = FakeBrokerTransport()
    ks = KillSwitch(halt_path=tmp_path / "HALT", env={})
    broker = AlpacaBroker(_settings(), transport=transport, kill_switch=ks)
    log = SqliteExecutionLog(tmp_path / "exec.sqlite")
    verdicts = {
        Ticker("AAPL"): Verdict.BUY,
        Ticker("MSFT"): Verdict.HOLD,
        Ticker("GOOG"): Verdict.SELL,
    }

    runner = AutopilotRunner(
        broker=broker,
        prices=DictPrices(_panel()),
        execution_log=log,
        research=lambda t: _artifact(str(t), verdicts[t]),
        kill_switch=ks,
        forecast_recorder=lambda art, d: Abstention(
            ticker=art.ticker,
            knowledge_date=d,
            reason="verdict_hold",
        ),
        settlement_runner=lambda _d: 2,
        max_positions=5,
        now=_NOW,
    )

    result = runner.run_cycle(as_of=_AS_OF, submit=True)

    assert result.settled_predictions == 2
    assert result.abstentions_registered == 3
    assert result.cycle.submitted is True
    assert set(result.guarded.targets) == {Ticker("AAPL")}
    assert len(transport.posts) == 1
    assert transport.posts[0]["symbol"] == "AAPL"
    assert str(transport.posts[0]["client_order_id"]).startswith(f"{AGENTIC_STRATEGY_NAME}-")


def test_run_cycle_triggers_drawdown_exit_and_retains_hold_position(tmp_path: Path) -> None:
    # Open positions: AAPL (up then gave back >40% from peak) and MSFT (healthy, HOLD verdict)
    transport = FakeBrokerTransport(
        positions_payload=[
            {"symbol": "AAPL", "qty": "10", "current_price": "104.0"},
            {"symbol": "MSFT", "qty": "10", "current_price": "230.0"},
        ]
    )
    ks = KillSwitch(halt_path=tmp_path / "HALT", env={})
    broker = AlpacaBroker(_settings(), transport=transport, kill_switch=ks)
    log = SqliteExecutionLog(tmp_path / "exec.sqlite")
    # Prior AAPL buy at $100 -> peak ($123.80 in _panel) > +5% and current $104 gives back >80%
    log.append(
        ExecutionEvent(
            cycle_id="prior",
            recorded_at=datetime(2024, 5, 1, 15, 0, tzinfo=UTC),
            kind=ExecutionEventKind.SUBMITTED,
            ticker="AAPL",
            side="buy",
            shares=10.0,
            reference_price=100.0,
            client_order_id="prior-aapl-buy",
            broker_order_id="srv-0",
            detail="filled",
        )
    )
    verdicts = {
        Ticker("AAPL"): Verdict.BUY,
        Ticker("MSFT"): Verdict.HOLD,
        Ticker("GOOG"): Verdict.SELL,
    }

    runner = AutopilotRunner(
        broker=broker,
        prices=DictPrices(_panel()),
        execution_log=log,
        research=lambda t: _artifact(str(t), verdicts[t]),
        kill_switch=ks,
        max_positions=5,
        now=_NOW,
    )

    result = runner.run_cycle(as_of=_AS_OF, submit=False)

    assert Ticker("AAPL") not in result.guarded.targets
    assert Ticker("MSFT") in result.guarded.targets
    actions = {a.ticker: a.kind for a in result.guarded.actions}
    assert actions[Ticker("AAPL")] is GuardActionKind.DRAWDOWN_EXIT
    assert actions[Ticker("MSFT")] is GuardActionKind.HOLD_RETAINED


def test_run_cycle_enforces_reentry_cooldown_and_logs_rejection(tmp_path: Path) -> None:
    transport = FakeBrokerTransport()
    ks = KillSwitch(halt_path=tmp_path / "HALT", env={})
    broker = AlpacaBroker(_settings(), transport=transport, kill_switch=ks)
    log = SqliteExecutionLog(tmp_path / "exec.sqlite")
    # AAPL sold yesterday -> inside 3-day cooldown window
    log.append(
        ExecutionEvent(
            cycle_id="prior-sell",
            recorded_at=datetime(2024, 6, 9, 15, 0, tzinfo=UTC),
            kind=ExecutionEventKind.SUBMITTED,
            ticker="AAPL",
            side="sell",
            shares=10.0,
            reference_price=110.0,
            client_order_id="prior-aapl-sell",
            broker_order_id="srv-sell",
            detail="closed",
        )
    )

    runner = AutopilotRunner(
        broker=broker,
        prices=DictPrices(_panel()),
        execution_log=log,
        research=lambda t: _artifact(str(t), Verdict.BUY),
        kill_switch=ks,
        max_positions=5,
        now=_NOW,
    )

    result = runner.run_cycle(as_of=_AS_OF, submit=False)

    assert Ticker("AAPL") not in result.guarded.targets
    assert Ticker("MSFT") in result.guarded.targets
    rejected_events = [
        e for e in log if e.kind is ExecutionEventKind.REJECTED and e.ticker == "AAPL"
    ]
    assert len(rejected_events) == 1
    assert "trade throttle" in (rejected_events[0].detail or "")


def test_run_loop_respects_max_cycles_and_kill_switch(tmp_path: Path) -> None:
    transport = FakeBrokerTransport()
    halt_file = tmp_path / "HALT"
    ks = KillSwitch(halt_path=halt_file, env={})
    broker = AlpacaBroker(_settings(), transport=transport, kill_switch=ks)
    log = SqliteExecutionLog(tmp_path / "exec.sqlite")
    sleeps: list[float] = []

    runner = AutopilotRunner(
        broker=broker,
        prices=DictPrices(_panel()),
        execution_log=log,
        research=lambda t: _artifact(str(t), Verdict.HOLD),
        kill_switch=ks,
        now=_NOW,
    )

    results = runner.run_loop(
        submit=False,
        interval_seconds=15.0,
        max_cycles=2,
        sleep_fn=sleeps.append,
    )

    assert len(results) == 2
    assert sleeps == [15.0]


def test_run_cycle_uses_broker_avg_entry_price_and_live_position_price(tmp_path: Path) -> None:
    transport = FakeBrokerTransport(
        positions_payload=[
            {
                "symbol": "AAPL",
                "qty": "21.0",
                "avg_entry_price": "207.66",
                "current_price": "207.31",
            },
        ],
    )
    ks = KillSwitch(halt_path=tmp_path / "HALT", env={})
    broker = AlpacaBroker(_settings(), transport=transport, kill_switch=ks)
    log = SqliteExecutionLog(tmp_path / "exec.sqlite")
    # Yesterday's pre-trade bar reference_price ($195.77) differs from real fill ($207.66)
    log.append(
        ExecutionEvent(
            cycle_id="prior-buy",
            recorded_at=datetime(2024, 6, 9, 15, 0, tzinfo=UTC),
            kind=ExecutionEventKind.SUBMITTED,
            ticker="AAPL",
            side="buy",
            shares=21.0,
            reference_price=195.77,
            client_order_id="prior-aapl-buy",
            broker_order_id="srv-buy",
            detail="submitted",
        )
    )

    runner = AutopilotRunner(
        broker=broker,
        prices=DictPrices(_panel()),
        execution_log=log,
        research=lambda t: _artifact(str(t), Verdict.HOLD),
        kill_switch=ks,
        now=_NOW,
    )

    result = runner.run_cycle(as_of=_AS_OF, submit=False)
    aapl_risk = result.position_risks[Ticker("AAPL")]
    assert aapl_risk.entry_price == 207.66
    assert aapl_risk.current_price == 207.31
    assert aapl_risk.annualized_volatility is not None
    assert [i for i in result.cycle.plan.intents if i.ticker == Ticker("AAPL")] == []


def test_require_market_open_blocks_closed_market_and_aligns_pre_open_sleep(
    tmp_path: Path,
) -> None:
    transport = FakeBrokerTransport(
        market_open=False,
        clock_payload={
            "is_open": False,
            "timestamp": "2024-06-10T13:00:00Z",
            "next_open": "2024-06-10T13:30:00Z",
            "next_close": "2024-06-10T20:00:00Z",
        },
    )
    ks = KillSwitch(halt_path=tmp_path / "HALT", env={})
    broker = AlpacaBroker(_settings(), transport=transport, kill_switch=ks)
    log = SqliteExecutionLog(tmp_path / "exec.sqlite")

    runner = AutopilotRunner(
        broker=broker,
        prices=DictPrices(_panel()),
        execution_log=log,
        research=lambda t: _artifact(str(t), Verdict.BUY),
        kill_switch=ks,
        require_market_open=True,
        now=_NOW,
    )

    result = runner.run_cycle(as_of=_AS_OF, submit=True)
    assert result.cycle.submitted is False
    assert result.cycle.skipped_reason is not None
    assert "market is closed" in result.cycle.skipped_reason
    assert transport.posts == []

    # 30m (1800s) before open -> wakes 5m (300s) before open = 1500s sleep
    assert runner._next_sleep_seconds(3600.0, False) == 1500.0

    # 40m (2400s) before close while open -> wakes 10m (600s) before close = 1800s sleep
    transport.clock_payload = {
        "is_open": True,
        "timestamp": "2024-06-10T19:20:00Z",
        "next_open": "2024-06-11T13:30:00Z",
        "next_close": "2024-06-10T20:00:00Z",
    }
    assert runner._next_sleep_seconds(3600.0, True) == 1800.0


