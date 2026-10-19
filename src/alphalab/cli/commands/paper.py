"""The ``paper`` command family: status, run, log — paper trading only.

Three subcommands, following the project's CLI conventions (argparse, ``--json``,
colour off outside a TTY, exit codes 0/1/2, actionable errors without stack
traces):

* ``paper status`` — read-only: account, positions, P&L vs the shadow SPY
  benchmark, reconciliation against the target weights, kill-switch state and
  whether the market is open.
* ``paper run`` — run one cycle. **Dry-run is the default**: it prints the plan
  and sends nothing. Submitting real paper orders requires the explicit
  ``--submit`` flag, so the safe thing happens when you type the obvious thing.
* ``paper log`` — print the append-only execution history.

This module is the composition root for paper trading: it is the one place
allowed to construct the HTTP transport and reach the network. The runner and
broker it wires together are themselves network-agnostic and unit-tested with
doubles.
"""

from __future__ import annotations

import json as jsonlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TextIO

from alphalab.adapters.alpaca_broker import AlpacaBroker
from alphalab.adapters.alpaca_config import AlpacaSettings, LiveTradingNotAcknowledgedError
from alphalab.adapters.execution_log import SqliteExecutionLog
from alphalab.adapters.kill_switch import KillSwitch
from alphalab.app.paper_models import Reconciliation, ShadowBenchmark
from alphalab.app.paper_runner import CycleResult, PaperRunner, PriceProvider
from alphalab.cli.paper_render import render_log, render_run, render_status
from alphalab.domain.backtest import PriceBar
from alphalab.domain.types import Ticker
from alphalab.report.theme import theme_for

__all__ = [
    "PaperLogOptions",
    "PaperRunOptions",
    "PaperStatusOptions",
    "run_paper_log",
    "run_paper_run",
    "run_paper_status",
]


@dataclass(frozen=True)
class PaperStatusOptions:
    """Resolved options for ``paper status``."""

    execution_db: Path
    prices_dir: Path
    halt_path: Path | None
    as_json: bool
    no_color: bool


@dataclass(frozen=True)
class PaperRunOptions:
    """Resolved options for ``paper run``."""

    execution_db: Path
    prices_dir: Path
    halt_path: Path | None
    submit: bool
    as_json: bool
    no_color: bool

    #: Override the quarterly cadence gate. Trading off-schedule runs a
    #: strategy the backtest does not describe, so it must be asked for.
    force: bool = False


@dataclass(frozen=True)
class PaperLogOptions:
    """Resolved options for ``paper log``."""

    execution_db: Path
    as_json: bool
    no_color: bool


class _CsvPriceProvider:
    """Price provider backed by local ``<ticker>.csv`` close/volume series.

    Reads the ``volume`` column directly (not via the close-only price source)
    because the liquidity screen the strategy depends on is mandatory and needs
    traded value: without volume every name is treated as untradable and the
    plan is always empty.
    """

    def __init__(self, prices_dir: Path, *, tickers: Sequence[Ticker]) -> None:
        self._prices_dir = Path(prices_dir)
        self._tickers = tuple(tickers)

    def recent_bars(self) -> Mapping[Ticker, Sequence[PriceBar]]:
        out: dict[Ticker, Sequence[PriceBar]] = {}
        for ticker in self._tickers:
            bars = _read_csv_bars(self._prices_dir / f"{ticker}.csv")
            if bars:
                out[ticker] = bars
        return out


def _read_csv_bars(path: Path) -> tuple[PriceBar, ...]:
    """Parse a ``date,close,volume`` CSV into bars, skipping unusable rows."""
    import csv
    from datetime import datetime

    if not path.exists():
        return ()
    bars: list[PriceBar] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            close = _to_float(row.get("close"))
            if close is None or close <= 0.0:
                continue
            bars.append(
                PriceBar(
                    on=datetime.strptime(row["date"], "%Y-%m-%d").date(),
                    close=close,
                    volume=_to_float(row.get("volume")),
                )
            )
    return tuple(bars)


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _universe_tickers(prices_dir: Path) -> list[Ticker]:
    """Every ticker with a local price CSV, SPY always included for the shadow."""
    if not prices_dir.is_dir():
        return []
    tickers = {Ticker(path.stem) for path in prices_dir.glob("*.csv")}
    tickers.add(Ticker("SPY"))
    return sorted(tickers)


def _build_runner(
    *, settings: AlpacaSettings, execution_db: Path, prices_dir: Path, halt_path: Path | None
) -> tuple[PaperRunner, SqliteExecutionLog, KillSwitch]:
    """Wire a runner against the live paper API and local price CSVs."""
    from alphalab.cli.paper_transport import HttpxWriteTransport

    kill_switch = KillSwitch(halt_path=halt_path)
    broker = AlpacaBroker(settings, transport=HttpxWriteTransport(), kill_switch=kill_switch)
    provider: PriceProvider = _price_provider(settings, prices_dir)
    log = SqliteExecutionLog(execution_db)
    runner = PaperRunner(broker=broker, prices=provider, execution_log=log)
    return runner, log, kill_switch


def _price_provider(
    settings: AlpacaSettings,
    prices_dir: Path,
    *,
    tickers: Sequence[Ticker] | None = None,
) -> PriceProvider:
    """Prefer live Alpaca bars; fall back to local CSVs only if asked.

    A system left running must decide from fresh prices. The CSV provider reads
    a past download, so every cycle after it reuses stale closes while nothing
    reports a fault. Live bars are therefore the default, and the CSV path
    stays reachable through ``ALPHALAB_PAPER_PRICES=csv`` for offline work.
    """
    import os

    from alphalab.adapters.alpaca_bars import AlpacaBarsClient, DataFeed
    from alphalab.cli.alpaca_prices import AlpacaPriceProvider
    from alphalab.cli.paper_transport import HttpxWriteTransport

    resolved = list(tickers) if tickers is not None else _universe_tickers(prices_dir)
    if os.environ.get("ALPHALAB_PAPER_PRICES", "live").lower() == "csv":
        return _CsvPriceProvider(prices_dir, tickers=resolved)

    feed = DataFeed(os.environ.get("ALPHALAB_ALPACA_FEED", "sip").lower())
    client = AlpacaBarsClient(
        settings, transport=HttpxWriteTransport(), feed=feed, with_raw_close=True
    )
    return AlpacaPriceProvider(
        client,
        tickers=resolved,
    )


def _load_settings(err: TextIO) -> AlpacaSettings | None:
    """Build settings from env/.env, printing an actionable error on failure."""
    try:
        return AlpacaSettings()  # type: ignore[call-arg]  # values from env/.env
    except Exception as exc:
        print(
            "error: Alpaca credentials not configured. Set "
            "ALPHALAB_ALPACA_KEY_ID and ALPHALAB_ALPACA_SECRET_KEY "
            f"(env or .env). Detail: {_safe(exc)}",
            file=err,
        )
        return None


def run_paper_status(opts: PaperStatusOptions, *, out: TextIO, err: TextIO) -> int:
    """Print a read-only snapshot of the paper account. Exit 0 on success."""
    settings = _load_settings(err)
    if settings is None:
        return 2
    try:
        runner, log, kill_switch = _build_runner(
            settings=settings,
            execution_db=opts.execution_db,
            prices_dir=opts.prices_dir,
            halt_path=opts.halt_path,
        )
    except LiveTradingNotAcknowledgedError as exc:
        print(f"error: {exc}", file=err)
        return 2
    try:
        result = runner.run_cycle(submit=False)
    except Exception as exc:
        print(f"error: could not read paper account: {_safe(exc)}", file=err)
        return 1
    finally:
        log.close()
    theme = theme_for(out, force_disable=opts.no_color or opts.as_json)
    if opts.as_json:
        print(_status_json(result, kill_switch), file=out)
        return 0
    print(render_status(result, kill_switch.state(), theme), file=out)
    return 0


def run_paper_run(opts: PaperRunOptions, *, out: TextIO, err: TextIO) -> int:
    """Run one cycle. Dry-run unless ``--submit`` was passed. Exit 0 on success."""
    settings = _load_settings(err)
    if settings is None:
        return 2
    try:
        runner, log, _ = _build_runner(
            settings=settings,
            execution_db=opts.execution_db,
            prices_dir=opts.prices_dir,
            halt_path=opts.halt_path,
        )
    except LiveTradingNotAcknowledgedError as exc:
        print(f"error: {exc}", file=err)
        return 2
    submit = opts.submit
    if submit and not opts.force:
        due, last = _rebalance_due(log)
        if not due:
            submit = False
            print(
                f"note: last rebalance {last} is in the current quarter, so the "
                "measured quarterly schedule does not call for one today. "
                "Reconciling without submitting; pass --force to override.",
                file=out,
            )
    try:
        result = runner.run_cycle(submit=submit)
    except Exception as exc:
        print(f"error: cycle failed: {_safe(exc)}", file=err)
        return 1
    finally:
        log.close()
    theme = theme_for(out, force_disable=opts.no_color or opts.as_json)
    if opts.as_json:
        print(_run_json(result), file=out)
        return 0
    print(render_run(result, theme), file=out)
    return 0


def _rebalance_due(log: SqliteExecutionLog) -> tuple[bool, date | None]:
    """Whether the quarterly schedule calls for a rebalance, and when the last was.

    The measured strategy rebalances quarterly. The scheduler fires daily so
    that reconciliation stays current and a missed day is harmless, but trading
    every day would be a different, unmeasured strategy with roughly sixty
    times the turnover.

    Only ``SUBMITTED`` events count as a rebalance: intents and dry-runs did
    not change the portfolio, so treating them as one would skip the real
    trade.
    """
    from alphalab.adapters.execution_log import ExecutionEventKind
    from alphalab.domain.execution import is_rebalance_due

    last: date | None = None
    for event in log:
        if event.kind is not ExecutionEventKind.SUBMITTED:
            continue
        seen = event.recorded_at.date()
        if last is None or seen > last:
            last = seen
    return is_rebalance_due(last_rebalance=last, today=datetime.now(UTC).date()), last


def run_paper_log(opts: PaperLogOptions, *, out: TextIO, err: TextIO) -> int:
    """Print the append-only execution history. Read-only; exit 0 on success."""
    if not opts.execution_db.exists():
        print(
            f"error: no execution log at {opts.execution_db}. "
            "Run 'alphalab paper run' first to create it.",
            file=err,
        )
        return 2
    log = SqliteExecutionLog(opts.execution_db)
    try:
        events = log.list_events()
    finally:
        log.close()
    theme = theme_for(out, force_disable=opts.no_color or opts.as_json)
    if opts.as_json:
        print(_log_json(events, opts.execution_db), file=out)
        return 0
    print(render_log(events, theme), file=out)
    return 0


# -- JSON serialisation ---------------------------------------------------


def _status_json(result: CycleResult, kill_switch: KillSwitch) -> str:
    state = kill_switch.state()
    payload = {
        "equity": result.account.equity,
        "buying_power": result.account.buying_power,
        "market_open": result.market_open,
        "trading_blocked": result.account.trading_blocked,
        "account_blocked": result.account.account_blocked,
        "kill_switch_engaged": state.engaged,
        "kill_switch_reason": state.reason(),
        "reconciliation": _reconciliation_json(result.reconciliation),
        "shadow_benchmark": _shadow_json(result.shadow),
    }
    return jsonlib.dumps(payload, indent=2, sort_keys=True)


def _run_json(result: CycleResult) -> str:
    payload = {
        "cycle_id": result.cycle_id,
        "as_of": result.as_of.isoformat(),
        "submitted": result.submitted,
        "market_open": result.market_open,
        "skipped_reason": result.skipped_reason,
        "halted_reason": result.halted_reason,
        "equity": result.account.equity,
        "intents": [_intent_json(i, result.as_of) for i in result.plan.intents],
        "rejected": [{"ticker": str(r.ticker), "reason": r.reason} for r in result.plan.rejected],
        "outcomes": [
            {
                "ticker": str(o.intent.ticker),
                "status": o.status.value,
                "client_order_id": o.client_order_id,
                "broker_order_id": o.broker_order_id,
                "detail": o.detail,
            }
            for o in result.outcomes
        ],
        "reconciliation": _reconciliation_json(result.reconciliation),
        "shadow_benchmark": _shadow_json(result.shadow),
    }
    return jsonlib.dumps(payload, indent=2, sort_keys=True)


def _intent_json(intent: object, as_of: date) -> dict[str, object]:
    from alphalab.app.paper_runner import STRATEGY_NAME
    from alphalab.domain.execution import OrderIntent

    assert isinstance(intent, OrderIntent)
    return {
        "ticker": str(intent.ticker),
        "side": intent.side.value,
        "shares": intent.shares,
        "reference_price": intent.reference_price,
        "notional": intent.notional,
        "client_order_id": intent.client_order_id(strategy=STRATEGY_NAME, as_of=as_of),
    }


def _reconciliation_json(reconciliation: Reconciliation) -> list[dict[str, object]]:
    return [
        {
            "ticker": str(row.ticker),
            "target_weight": row.target_weight,
            "actual_weight": row.actual_weight,
            "deviation": row.deviation,
        }
        for row in reconciliation.rows
    ]


def _shadow_json(shadow: ShadowBenchmark | None) -> dict[str, object] | None:
    if shadow is None:
        return None
    return {
        "symbol": str(shadow.symbol),
        "initial_capital": shadow.initial_capital,
        "current_value": shadow.current_value,
        "total_return": shadow.total_return,
    }


def _log_json(events: Sequence[object], db_path: Path) -> str:
    from alphalab.adapters.execution_log import ExecutionEvent

    rows = []
    for event in events:
        assert isinstance(event, ExecutionEvent)
        rows.append(
            {
                "cycle_id": event.cycle_id,
                "recorded_at": event.recorded_at.isoformat(),
                "kind": event.kind.value,
                "ticker": event.ticker,
                "side": event.side,
                "shares": event.shares,
                "reference_price": event.reference_price,
                "client_order_id": event.client_order_id,
                "broker_order_id": event.broker_order_id,
                "detail": event.detail,
            }
        )
    return jsonlib.dumps({"execution_log": str(db_path), "events": rows}, indent=2, sort_keys=True)


def _safe(exc: Exception) -> str:
    """A one-line, secret-free rendering of an exception for the user."""
    text = str(exc) or exc.__class__.__name__
    return text.splitlines()[0][:200]
