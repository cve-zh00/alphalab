"""CLI tests for the ``paper`` command family.

The network is never touched: the runner is replaced by a fake, or the command
reads a real on-disk SQLite log it created itself. These tests nail down the
user-facing contract — dry-run is the default, exit codes, ``--json``, and that
``log`` is read-only.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

import alphalab.cli.commands.paper as paper_cmd
from alphalab.adapters.alpaca_broker import AccountSnapshot, OrderOutcome, OrderStatus
from alphalab.adapters.execution_log import (
    ExecutionEvent,
    ExecutionEventKind,
    SqliteExecutionLog,
)
from alphalab.adapters.kill_switch import KillSwitch
from alphalab.app.paper_models import Reconciliation, ReconciliationRow, ShadowBenchmark
from alphalab.app.paper_runner import CycleResult
from alphalab.cli.app import build_parser, main
from alphalab.domain.execution import OrderIntent, TradePlan
from alphalab.domain.types import Side, Ticker


def _account() -> AccountSnapshot:
    return AccountSnapshot(
        equity=100_000.0, buying_power=200_000.0, trading_blocked=False, account_blocked=False
    )


def _intent() -> OrderIntent:
    return OrderIntent(ticker=Ticker("AAA"), side=Side.BUY, shares=200.0, reference_price=100.0)


def _plan() -> TradePlan:
    return TradePlan(as_of=date(2024, 1, 2), intents=(_intent(),), rejected=(), equity=100_000.0)


def _cycle_result(*, submitted: bool, outcomes: tuple[OrderOutcome, ...] = ()) -> CycleResult:
    return CycleResult(
        cycle_id="rank_ensemble-2024-01-02T15:00:00+00:00",
        as_of=date(2024, 1, 2),
        account=_account(),
        market_open=True,
        submitted=submitted,
        plan=_plan(),
        outcomes=outcomes,
        reconciliation=Reconciliation(
            rows=(ReconciliationRow(ticker=Ticker("AAA"), target_weight=0.2, actual_weight=0.2),)
        ),
        shadow=ShadowBenchmark(
            symbol=Ticker("SPY"),
            initial_capital=100_000.0,
            entry_price=400.0,
            latest_price=440.0,
        ),
        skipped_reason=(
            None if submitted else "dry-run: no orders submitted (pass --submit to send)"
        ),
        targets={Ticker("AAA"): 0.2},
    )


class _FakeRunner:
    """Stands in for PaperRunner; records whether submit was requested."""

    def __init__(self) -> None:
        self.last_submit: bool | None = None

    def run_cycle(self, *, as_of: date | None = None, submit: bool = False) -> CycleResult:
        self.last_submit = submit
        outcomes: tuple[OrderOutcome, ...] = ()
        if submit:
            outcomes = (
                OrderOutcome(
                    intent=_intent(),
                    status=OrderStatus.SUBMITTED,
                    client_order_id="rank_ensemble-2024-01-02-AAA-buy",
                    broker_order_id="srv-1",
                ),
            )
        return _cycle_result(submitted=submit, outcomes=outcomes)


@pytest.fixture
def patched_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeRunner:
    """Replace _build_runner so no network or credentials are needed."""
    fake = _FakeRunner()
    log = SqliteExecutionLog(tmp_path / "exec.db")
    switch = KillSwitch(halt_path=tmp_path / "HALT", env={})

    def _fake_build_runner(**_kwargs: object) -> tuple[_FakeRunner, SqliteExecutionLog, KillSwitch]:
        return fake, log, switch

    monkeypatch.setattr(paper_cmd, "_build_runner", _fake_build_runner)
    monkeypatch.setattr(paper_cmd, "_load_settings", lambda _err: object())
    return fake


# -- argparse wiring: dry-run is the default ------------------------------


def test_run_defaults_to_dry_run_in_parser() -> None:
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args(["paper", "run"])

    # Assert — no --submit means submit is False.
    assert args.submit is False


def test_run_submit_flag_sets_submit() -> None:
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args(["paper", "run", "--submit"])

    # Assert
    assert args.submit is True


def test_submit_and_dry_run_are_mutually_exclusive() -> None:
    # Arrange
    parser = build_parser()

    # Act / Assert — argparse exits 2 on the conflict.
    with pytest.raises(SystemExit):
        parser.parse_args(["paper", "run", "--submit", "--dry-run"])


# -- run command: dry-run default -----------------------------------------


def test_paper_run_dry_run_is_default(patched_runner: _FakeRunner) -> None:
    # Arrange
    out, err = io.StringIO(), io.StringIO()

    # Act
    code = main(["paper", "run", "--no-color"], out=out, err=err)

    # Assert — exit 0, dry-run requested, output names the dry run.
    assert code == 0
    assert patched_runner.last_submit is False
    assert "DRY RUN" in out.getvalue()


def test_paper_run_submit_requested_explicitly(patched_runner: _FakeRunner) -> None:
    # Arrange
    out, err = io.StringIO(), io.StringIO()

    # Act
    code = main(["paper", "run", "--submit", "--no-color"], out=out, err=err)

    # Assert
    assert code == 0
    assert patched_runner.last_submit is True
    assert "SUBMITTED" in out.getvalue()


def test_paper_run_json_output(patched_runner: _FakeRunner) -> None:
    # Arrange
    out, err = io.StringIO(), io.StringIO()

    # Act
    code = main(["paper", "run", "--json"], out=out, err=err)

    # Assert — valid JSON reporting submitted=false by default.
    assert code == 0
    payload = json.loads(out.getvalue())
    assert payload["submitted"] is False
    assert payload["intents"][0]["ticker"] == "AAA"


# -- status command -------------------------------------------------------


def test_paper_status_is_read_only(patched_runner: _FakeRunner) -> None:
    # Arrange
    out, err = io.StringIO(), io.StringIO()

    # Act
    code = main(["paper", "status", "--no-color"], out=out, err=err)

    # Assert — status never submits; it ran a cycle with submit=False.
    assert code == 0
    assert patched_runner.last_submit is False
    assert "PAPER ACCOUNT STATUS" in out.getvalue()


def test_paper_status_json(patched_runner: _FakeRunner) -> None:
    # Arrange
    out, err = io.StringIO(), io.StringIO()

    # Act
    code = main(["paper", "status", "--json"], out=out, err=err)

    # Assert
    assert code == 0
    payload = json.loads(out.getvalue())
    assert payload["equity"] == pytest.approx(100_000.0)
    assert payload["kill_switch_engaged"] is False


# -- log command: read-only, no network -----------------------------------


def test_paper_log_missing_db_is_actionable(tmp_path: Path) -> None:
    # Arrange
    out, err = io.StringIO(), io.StringIO()
    missing = tmp_path / "nope.db"

    # Act
    code = main(["paper", "log", "--execution-db", str(missing)], out=out, err=err)

    # Assert — exit 2, actionable error, no stack trace.
    assert code == 2
    assert "no execution log" in err.getvalue()
    assert "Traceback" not in err.getvalue()


def test_paper_log_renders_events(tmp_path: Path) -> None:
    # Arrange — create a real log with one event.
    db = tmp_path / "exec.db"
    log = SqliteExecutionLog(db)
    log.append(
        ExecutionEvent(
            cycle_id="c1",
            recorded_at=datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
            kind=ExecutionEventKind.INTENT,
            ticker="AAA",
            side="buy",
            shares=200.0,
            reference_price=100.0,
            client_order_id="rank_ensemble-2024-01-02-AAA-buy",
            broker_order_id=None,
            detail=None,
        )
    )
    log.close()
    out, err = io.StringIO(), io.StringIO()

    # Act
    code = main(["paper", "log", "--execution-db", str(db), "--no-color"], out=out, err=err)

    # Assert
    assert code == 0
    assert "EXECUTION LOG" in out.getvalue()
    assert "AAA" in out.getvalue()


def test_paper_log_json(tmp_path: Path) -> None:
    # Arrange
    db = tmp_path / "exec.db"
    SqliteExecutionLog(db).close()
    out, err = io.StringIO(), io.StringIO()

    # Act
    code = main(["paper", "log", "--execution-db", str(db), "--json"], out=out, err=err)

    # Assert
    assert code == 0
    payload = json.loads(out.getvalue())
    assert payload["events"] == []


# -- subcommand required --------------------------------------------------


def test_paper_without_subcommand_is_usage_error() -> None:
    # Arrange
    out, err = io.StringIO(), io.StringIO()

    # Act
    code = main(["paper"], out=out, err=err)

    # Assert
    assert code == 2
    assert "usage: alphalab paper" in err.getvalue()


# -- credential and error paths -------------------------------------------


def test_status_bad_credentials_is_exit_2(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange — settings construction fails (no creds).
    def _raise() -> object:
        raise RuntimeError("missing key")

    monkeypatch.setattr(paper_cmd, "AlpacaSettings", lambda **_k: _raise())
    out, err = io.StringIO(), io.StringIO()

    # Act
    code = main(["paper", "status"], out=out, err=err)

    # Assert — actionable error, no stack trace.
    assert code == 2
    assert "credentials not configured" in err.getvalue()
    assert "Traceback" not in err.getvalue()


def test_run_cycle_failure_is_exit_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange — settings OK, but the runner raises mid-cycle (e.g. network down).
    class _Boom:
        def run_cycle(self, *, as_of: object = None, submit: bool = False) -> CycleResult:
            raise RuntimeError("connection refused")

    log = SqliteExecutionLog(tmp_path / "exec.db")
    switch = KillSwitch(halt_path=tmp_path / "HALT", env={})

    monkeypatch.setattr(paper_cmd, "_load_settings", lambda _err: object())
    monkeypatch.setattr(paper_cmd, "_build_runner", lambda **_k: (_Boom(), log, switch))
    out, err = io.StringIO(), io.StringIO()

    # Act
    code = main(["paper", "run"], out=out, err=err)

    # Assert
    assert code == 1
    assert "cycle failed" in err.getvalue()


def test_status_live_not_acknowledged_is_exit_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange — building the runner raises LiveTradingNotAcknowledgedError.
    from alphalab.adapters.alpaca_config import LiveTradingNotAcknowledgedError

    def _raise(**_k: object) -> object:
        raise LiveTradingNotAcknowledgedError("use paper")

    monkeypatch.setattr(paper_cmd, "_load_settings", lambda _err: object())
    monkeypatch.setattr(paper_cmd, "_build_runner", _raise)
    out, err = io.StringIO(), io.StringIO()

    # Act
    code = main(["paper", "status"], out=out, err=err)

    # Assert
    assert code == 2
    assert "use paper" in err.getvalue()


def test_build_runner_wires_a_broker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange — inject a fake transport so no socket is opened, and point at a
    # local prices dir with one CSV so the provider has data.
    from alphalab.adapters.alpaca_config import AlpacaEndpoint, AlpacaSettings
    from alphalab.adapters.edgar import HttpResponse

    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    (prices_dir / "SPY.csv").write_text(
        "date,close,volume\n2023-01-01,400.0,1000000\n2023-01-02,401.0,1000000\n",
        encoding="utf-8",
    )

    class _FakeTransport:
        def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
            if url.endswith("/v2/account"):
                return HttpResponse(
                    status_code=200,
                    json={
                        "equity": "100000",
                        "buying_power": "200000",
                        "trading_blocked": False,
                        "account_blocked": False,
                    },
                )
            if url.endswith("/v2/clock"):
                return HttpResponse(status_code=200, json={"is_open": False})
            if url.endswith("/v2/positions"):
                return HttpResponse(status_code=200, json=[])  # type: ignore[arg-type]
            return HttpResponse(status_code=200, json={})

        def post(
            self, url: str, *, headers: dict[str, str], json: dict[str, object]
        ) -> HttpResponse:
            raise AssertionError("status must never POST")

    import alphalab.cli.paper_transport as transport_mod

    monkeypatch.setattr(transport_mod, "HttpxWriteTransport", _FakeTransport)
    monkeypatch.setattr(
        paper_cmd,
        "_load_settings",
        lambda _err: AlpacaSettings(
            key_id="k",
            secret_key="super-secret-value",
            endpoint=AlpacaEndpoint.PAPER,
        ),
    )
    out, err = io.StringIO(), io.StringIO()

    # Act — real _build_runner + real runner against the fake transport.
    code = main(
        [
            "paper",
            "status",
            "--no-color",
            "--execution-db",
            str(tmp_path / "exec.db"),
            "--prices-dir",
            str(prices_dir),
        ],
        out=out,
        err=err,
    )

    # Assert — read-only status rendered, market reported closed, no POST made.
    assert code == 0
    assert "PAPER ACCOUNT STATUS" in out.getvalue()
    assert "CLOSED" in out.getvalue()
