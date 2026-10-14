"""Unit tests for the paper-trading runner.

The runner is exercised with a fake transport (so the real broker logic runs,
including idempotency and the kill switch) and a dict-backed price provider. No
test touches the network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from alphalab.adapters.alpaca_broker import AlpacaBroker
from alphalab.adapters.alpaca_config import AlpacaEndpoint, AlpacaSettings
from alphalab.adapters.edgar import HttpResponse
from alphalab.adapters.execution_log import ExecutionEventKind, SqliteExecutionLog
from alphalab.adapters.kill_switch import KillSwitch
from alphalab.app.paper_runner import PaperRunner, build_targets
from alphalab.domain.backtest import PriceBar
from alphalab.domain.types import Ticker

_NOW = datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
_AS_OF = date(2024, 1, 2)
#: A decision date inside the fake panel, used where the shadow benchmark's
#: anchor is under test. Must be after the panel's first bar.
_SHADOW_AS_OF = date(2023, 6, 1)


def _settings() -> AlpacaSettings:
    return AlpacaSettings(
        key_id="paper-key",
        secret_key="super-secret-value",
        endpoint=AlpacaEndpoint.PAPER,
    )


@dataclass
class FakeBrokerTransport:
    """Scripts account/positions/clock reads and records POSTs."""

    equity: float = 100_000.0
    market_open: bool = True
    trading_blocked: bool = False
    account_blocked: bool = False
    positions_payload: list[dict[str, object]] = field(default_factory=list)
    posts: list[dict[str, object]] = field(default_factory=list)
    post_status: int = 200

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
            return HttpResponse(status_code=200, json={"is_open": self.market_open})
        return HttpResponse(status_code=200, json={})

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> HttpResponse:
        self.posts.append(json)
        if self.post_status in (409, 422):
            return HttpResponse(status_code=self.post_status, json={"message": "duplicate"})
        return HttpResponse(status_code=self.post_status, json={"id": f"srv-{len(self.posts)}"})


@dataclass
class DictPrices:
    """A PriceProvider double returning a fixed bar panel."""

    panel: Mapping[Ticker, Sequence[PriceBar]]

    def recent_bars(self) -> Mapping[Ticker, Sequence[PriceBar]]:
        return self.panel


def _rising_series(
    start: float, days: int, step: float, volume: float = 10_000_000.0
) -> tuple[PriceBar, ...]:
    return tuple(
        PriceBar(on=date(2023, 1, 1) + timedelta(days=i), close=start + i * step, volume=volume)
        for i in range(days)
    )


def _panel() -> dict[Ticker, Sequence[PriceBar]]:
    # Six liquid names with distinct momentum, plus SPY for the shadow benchmark.
    return {
        Ticker("AAA"): _rising_series(10.0, 300, 0.10),
        Ticker("BBB"): _rising_series(10.0, 300, 0.09),
        Ticker("CCC"): _rising_series(10.0, 300, 0.08),
        Ticker("DDD"): _rising_series(10.0, 300, 0.07),
        Ticker("EEE"): _rising_series(10.0, 300, 0.06),
        Ticker("FFF"): _rising_series(10.0, 300, 0.01),
        Ticker("SPY"): _rising_series(400.0, 300, 0.20),
    }


def _runner(
    transport: FakeBrokerTransport,
    log: SqliteExecutionLog,
    *,
    panel: Mapping[Ticker, Sequence[PriceBar]] | None = None,
    kill_switch: KillSwitch | None = None,
) -> PaperRunner:
    broker = AlpacaBroker(_settings(), transport=transport, kill_switch=kill_switch)
    prices = DictPrices(panel=panel or _panel())
    return PaperRunner(broker=broker, prices=prices, execution_log=log, now=_NOW)


# -- target construction --------------------------------------------------


def test_build_targets_picks_top_n_equal_weight() -> None:
    # Arrange / Act
    targets = build_targets(as_of=_AS_OF, prices=_panel(), top_n=5)

    # Assert — five names, equal-weighted, summing to 1.0.
    assert len(targets) == 5
    assert all(w == pytest.approx(0.2) for w in targets.values())
    assert sum(targets.values()) == pytest.approx(1.0)


def test_build_targets_empty_when_no_prices() -> None:
    assert build_targets(as_of=_AS_OF, prices={}) == {}


# -- dry run is the default -----------------------------------------------


def test_dry_run_default_sends_no_orders(tmp_path: Path) -> None:
    # Arrange
    transport = FakeBrokerTransport()
    log = SqliteExecutionLog(tmp_path / "exec.db")

    # Act — note: submit defaults to False.
    result = _runner(transport, log).run_cycle()
    log.close()

    # Assert — a non-empty plan, but zero POSTs.
    assert not result.plan.is_empty
    assert transport.posts == []
    assert not result.submitted
    assert result.skipped_reason is not None


def test_dry_run_still_logs_intents(tmp_path: Path) -> None:
    # Arrange
    transport = FakeBrokerTransport()
    log = SqliteExecutionLog(tmp_path / "exec.db")

    # Act
    result = _runner(transport, log).run_cycle()
    kinds = {e.kind for e in log.list_events()}
    log.close()

    # Assert — intents are recorded even though nothing was sent.
    assert ExecutionEventKind.INTENT in kinds
    assert result.orders_reached_broker == 0


# -- gates: zero POST -----------------------------------------------------


def test_market_closed_sends_no_orders(tmp_path: Path) -> None:
    # Arrange
    transport = FakeBrokerTransport(market_open=False)
    log = SqliteExecutionLog(tmp_path / "exec.db")

    # Act — even with submit=True, a closed market blocks emission.
    result = _runner(transport, log).run_cycle(submit=True)
    log.close()

    # Assert
    assert transport.posts == []
    assert "closed" in (result.skipped_reason or "")


def test_trading_blocked_sends_no_orders(tmp_path: Path) -> None:
    # Arrange
    transport = FakeBrokerTransport(trading_blocked=True)
    log = SqliteExecutionLog(tmp_path / "exec.db")

    # Act
    result = _runner(transport, log).run_cycle(submit=True)
    log.close()

    # Assert
    assert transport.posts == []
    assert "trading_blocked" in (result.skipped_reason or "")


def test_account_blocked_sends_no_orders(tmp_path: Path) -> None:
    # Arrange
    transport = FakeBrokerTransport(account_blocked=True)
    log = SqliteExecutionLog(tmp_path / "exec.db")

    # Act
    result = _runner(transport, log).run_cycle(submit=True)
    log.close()

    # Assert
    assert transport.posts == []
    assert "account_blocked" in (result.skipped_reason or "")


def test_kill_switch_blocks_submission(tmp_path: Path) -> None:
    # Arrange — switch engaged via a present halt file.
    halt = tmp_path / "HALT"
    halt.write_text("", encoding="utf-8")
    transport = FakeBrokerTransport()
    log = SqliteExecutionLog(tmp_path / "exec.db")
    switch = KillSwitch(halt_path=halt, env={})

    # Act — submit=True, market open, account clear, yet the switch halts it.
    result = _runner(transport, log, kill_switch=switch).run_cycle(submit=True)
    halted_kinds = [e.kind for e in log.list_events() if e.kind is ExecutionEventKind.HALTED]
    log.close()

    # Assert — zero POSTs, and the halt is recorded for every intent.
    assert transport.posts == []
    assert result.halted_reason is not None
    assert halted_kinds


# -- submission + idempotency --------------------------------------------


def test_submit_sends_orders_when_everything_clear(tmp_path: Path) -> None:
    # Arrange
    transport = FakeBrokerTransport()
    log = SqliteExecutionLog(tmp_path / "exec.db")

    # Act
    result = _runner(transport, log).run_cycle(submit=True)
    log.close()

    # Assert — one POST per intent, all reached the broker.
    assert len(transport.posts) == len(result.plan.intents)
    assert result.orders_reached_broker == len(result.plan.intents)
    assert result.submitted


def test_running_the_cycle_twice_is_idempotent(tmp_path: Path) -> None:
    # Arrange — second cycle: the broker reports every order a duplicate.
    log = SqliteExecutionLog(tmp_path / "exec.db")
    first_transport = FakeBrokerTransport(post_status=200)
    first = _runner(first_transport, log).run_cycle(submit=True)

    second_transport = FakeBrokerTransport(post_status=409)
    second = _runner(second_transport, log).run_cycle(submit=True)
    log.close()

    # Assert — identical client_order_ids; the second cycle duplicates, not doubles.
    first_ids = [p["client_order_id"] for p in first_transport.posts]
    second_ids = [p["client_order_id"] for p in second_transport.posts]
    assert first_ids == second_ids
    assert all(o.status.value == "duplicate" for o in second.outcomes)
    assert first.orders_reached_broker == second.orders_reached_broker


# -- reconciliation and shadow benchmark ----------------------------------


def test_reconciliation_reports_drift_from_target(tmp_path: Path) -> None:
    # Arrange — hold a name the target does not want (pure drift).
    transport = FakeBrokerTransport(
        positions_payload=[{"symbol": "ZZZ", "qty": "100", "current_price": "100.0"}]
    )
    log = SqliteExecutionLog(tmp_path / "exec.db")

    # Act
    result = _runner(transport, log).run_cycle()
    log.close()

    # Assert — ZZZ shows up as an over-held deviation; the panel is not aligned.
    zzz = next(r for r in result.reconciliation.rows if r.ticker == Ticker("ZZZ"))
    assert zzz.target_weight == pytest.approx(0.0)
    assert zzz.deviation > 0.0
    assert not result.reconciliation.is_aligned


def test_shadow_benchmark_tracks_spy_without_orders(tmp_path: Path) -> None:
    # Arrange
    transport = FakeBrokerTransport()
    log = SqliteExecutionLog(tmp_path / "exec.db")

    # Act
    result = _runner(transport, log).run_cycle(as_of=_SHADOW_AS_OF)
    log.close()

    # Assert — the shadow exists, sends nothing, and is anchored at inception.
    #
    # The anchor is the whole point. Entering at the first bar of the fetched
    # window instead of at account inception reported a +719% benchmark
    # against a days-old portfolio, which makes comparison meaningless. On a
    # first cycle inception is the decision date, so the entry must be the
    # first bar on or after it — never the oldest bar of the lookback.
    assert result.shadow is not None
    assert result.shadow.symbol == Ticker("SPY")
    assert transport.posts == []
    spy = list(_panel()[Ticker("SPY")])
    expected = next(bar for bar in spy if bar.on >= _SHADOW_AS_OF)
    assert result.shadow.entry_price == pytest.approx(expected.close)
    assert result.shadow.entry_price != pytest.approx(spy[0].close), (
        "shadow entered at the window start rather than at inception"
    )


def test_shadow_benchmark_absent_without_spy(tmp_path: Path) -> None:
    # Arrange — a panel with no SPY.
    panel = {k: v for k, v in _panel().items() if k != Ticker("SPY")}
    transport = FakeBrokerTransport()
    log = SqliteExecutionLog(tmp_path / "exec.db")

    # Act
    result = _runner(transport, log, panel=panel).run_cycle()
    log.close()

    # Assert
    assert result.shadow is None


# -- secret safety --------------------------------------------------------


def test_secret_absent_from_logged_events(tmp_path: Path) -> None:
    # Arrange
    transport = FakeBrokerTransport()
    log = SqliteExecutionLog(tmp_path / "exec.db")

    # Act
    _runner(transport, log).run_cycle(submit=True)
    blob = "".join(repr(e) for e in log.list_events())
    log.close()

    # Assert — the account secret never reaches the execution log.
    assert "super-secret-value" not in blob
