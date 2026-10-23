"""Argparse builder and dispatcher for ``alphalab autopilot``."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import TextIO

from alphalab.app.agentic_portfolio import DEFAULT_MAX_POSITIONS
from alphalab.artifact import Verdict
from alphalab.cli.commands.autopilot import (
    DEFAULT_AUTOPILOT_CANDIDATE_CAP,
    AutopilotCycleOptions,
    AutopilotPreflightOptions,
    run_autopilot_loop,
    run_autopilot_preflight,
    run_autopilot_run,
    run_autopilot_status,
)
from alphalab.cli.forecast_paths import (
    default_abstention_db,
    default_journal_db,
    default_prices_dir,
)

__all__ = ["build_autopilot_parser", "dispatch_autopilot"]

_DEFAULT_JOURNAL_DB = default_journal_db()
_DEFAULT_ABSTENTIONS_DB = default_abstention_db()
_DEFAULT_EXECUTION_DB = _DEFAULT_JOURNAL_DB.parent / "execution.db"
_DEFAULT_PRICES_DIR = default_prices_dir()

_VERDICTS: dict[str, Verdict] = {
    "buy": Verdict.BUY,
    "sell": Verdict.SELL,
    "hold": Verdict.HOLD,
}


def _parse_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {raw!r} (use YYYY-MM-DD)") from exc


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colour even on a TTY."
    )


def _add_cycle_flags(parser: argparse.ArgumentParser, *, allow_submit: bool) -> None:
    parser.add_argument(
        "--execution-db",
        type=Path,
        default=_DEFAULT_EXECUTION_DB,
        help=f"Append-only execution log DB (default: {_DEFAULT_EXECUTION_DB}).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_JOURNAL_DB,
        help=f"Prediction journal DB (default: {_DEFAULT_JOURNAL_DB}).",
    )
    parser.add_argument(
        "--abstentions-db",
        type=Path,
        default=_DEFAULT_ABSTENTIONS_DB,
        help=f"Abstentions DB (default: {_DEFAULT_ABSTENTIONS_DB}).",
    )
    parser.add_argument(
        "--prices-dir",
        type=Path,
        default=_DEFAULT_PRICES_DIR,
        help=f"Fallback CSV prices directory (default: {_DEFAULT_PRICES_DIR}).",
    )
    parser.add_argument(
        "--halt-file",
        type=Path,
        default=None,
        help="Kill-switch halt file path (default: ./HALT at the repo root).",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_date,
        default=None,
        help="Decision date (YYYY-MM-DD; default: current/next NYSE session).",
    )
    parser.add_argument(
        "--source",
        choices=("live", "fixture"),
        default="live",
        help="Fundamentals source: 'live' (SEC EDGAR, default) or 'fixture' (local).",
    )
    parser.add_argument(
        "--debate",
        action="store_true",
        help="Use the Bull/Bear/Judge debate pipeline instead of single analyst.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use deterministic template narration (no Ollama wait).",
    )
    parser.add_argument(
        "--assume-verdict",
        choices=("buy", "sell", "hold"),
        default=None,
        help="Override verdict on --no-llm path to test BUY/SELL/HOLD execution flows.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated candidate tickers (default: liquidity-screened universe).",
    )
    parser.add_argument(
        "--candidate-cap",
        type=int,
        default=DEFAULT_AUTOPILOT_CANDIDATE_CAP,
        help=(
            "Max liquid candidates evaluated per cycle "
            f"(default: {DEFAULT_AUTOPILOT_CANDIDATE_CAP})."
        ),
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=DEFAULT_MAX_POSITIONS,
        help=f"Equal-weight portfolio slot count (default: {DEFAULT_MAX_POSITIONS}).",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=-8.0,
        help="Hard stop-loss threshold in %% from entry price (default: -8.0).",
    )
    parser.add_argument(
        "--drawdown-trigger-pct",
        type=float,
        default=5.0,
        help="Peak gain %% required to arm peak-drawdown protection (default: 5.0).",
    )
    parser.add_argument(
        "--drawdown-giveback-pct",
        type=float,
        default=40.0,
        help="Peak profit giveback %% that triggers auto-close (default: 40.0).",
    )
    parser.add_argument(
        "--cooldown-days",
        type=int,
        default=3,
        help="Re-entry cooldown in calendar days after a sell (default: 3).",
    )
    parser.add_argument(
        "--allow-closed-market",
        action="store_true",
        help="Allow submitting orders when the market clock is CLOSED (default: blocked).",
    )
    if allow_submit:
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--submit",
            action="store_true",
            help="Send market orders to Alpaca paper account (default is dry-run).",
        )
        group.add_argument(
            "--dry-run",
            action="store_true",
            help="Explicitly request a dry run (default).",
        )
    _add_output_flags(parser)


def build_autopilot_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``alphalab autopilot {preflight,status,run,loop}``."""
    p = sub.add_parser(
        "autopilot",
        help="Autonomous agentic trading (NOFX runtime guards + LLM research + Alpaca).",
        description=(
            "Operate the autonomous agentic trader ('The model proposes. The runtime "
            "disposes.'). Combines point-in-time EDGAR valuation, LLM Analyst/Debate "
            "verdicts, forward journal settlement, and NOFX deterministic risk guards "
            "(Safe Mode, stop-loss, peak-profit giveback, cooldown throttling)."
        ),
    )
    ap_sub = p.add_subparsers(dest="autopilot_command", metavar="SUBCOMMAND")

    pf = ap_sub.add_parser(
        "preflight",
        help="Verify KillSwitch, Alpaca account, market clock, bars, EDGAR, and Ollama.",
    )
    pf.add_argument("--prices-dir", type=Path, default=_DEFAULT_PRICES_DIR)
    pf.add_argument("--halt-file", type=Path, default=None)
    pf.add_argument("--source", choices=("live", "fixture"), default="live")
    pf.add_argument("--no-llm", action="store_true")
    pf.add_argument(
        "--require-market-open",
        action="store_true",
        help="Treat a closed market clock as a blocking preflight failure.",
    )
    _add_output_flags(pf)

    st = ap_sub.add_parser(
        "status",
        help="Read-only agentic assessment: position risk, AI verdicts, guards, reconciliation.",
    )
    _add_cycle_flags(st, allow_submit=False)

    rn = ap_sub.add_parser(
        "run",
        help="Execute one autonomous agentic cycle (dry-run unless --submit is given).",
    )
    _add_cycle_flags(rn, allow_submit=True)

    lp = ap_sub.add_parser(
        "loop",
        help="Run continuous autonomous cycles until stopped, halted, or --max-cycles.",
    )
    _add_cycle_flags(lp, allow_submit=True)
    lp.add_argument(
        "--interval-seconds",
        type=float,
        default=3600.0,
        help="Seconds to sleep between autonomous cycles (default: 3600).",
    )
    lp.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional maximum number of cycles before exiting.",
    )


def dispatch_autopilot(
    args: argparse.Namespace, *, out: TextIO, err: TextIO, no_color: bool
) -> int:
    """Route an ``autopilot`` invocation to its subcommand runner."""
    subcmd = getattr(args, "autopilot_command", None)
    if subcmd == "preflight":
        return run_autopilot_preflight(
            AutopilotPreflightOptions(
                prices_dir=args.prices_dir,
                halt_path=args.halt_file,
                source=args.source,
                use_llm=not args.no_llm,
                require_market_open=bool(args.require_market_open),
                as_json=args.json,
                no_color=no_color,
            ),
            out=out,
            err=err,
        )
    if subcmd in ("status", "run", "loop"):
        opts = _cycle_options_from_args(args, no_color=no_color, submit_allowed=subcmd != "status")
        if subcmd == "status":
            return run_autopilot_status(opts, out=out, err=err)
        if subcmd == "run":
            return run_autopilot_run(opts, out=out, err=err)
        return run_autopilot_loop(opts, out=out, err=err)
    print(
        "usage: alphalab autopilot {preflight|status|run|loop} [options]\n"
        "  run and loop are dry-run by default; pass --submit to send paper orders.",
        file=err,
    )
    return 2


def _cycle_options_from_args(
    args: argparse.Namespace, *, no_color: bool, submit_allowed: bool
) -> AutopilotCycleOptions:
    raw_tickers = getattr(args, "tickers", None)
    tickers = (
        tuple(t.strip().upper() for t in raw_tickers.split(",") if t.strip())
        if raw_tickers
        else None
    )
    raw_assume = getattr(args, "assume_verdict", None)
    assume = _VERDICTS.get(raw_assume) if raw_assume else None
    return AutopilotCycleOptions(
        execution_db=args.execution_db,
        journal_db=args.db,
        abstentions_db=args.abstentions_db,
        prices_dir=args.prices_dir,
        halt_path=args.halt_file,
        as_of=args.as_of,
        source=args.source,
        use_llm=not args.no_llm,
        debate=bool(args.debate),
        submit=bool(getattr(args, "submit", False)) if submit_allowed else False,
        candidate_cap=args.candidate_cap,
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss_pct,
        drawdown_trigger_pct=args.drawdown_trigger_pct,
        drawdown_giveback_pct=args.drawdown_giveback_pct,
        cooldown_days=args.cooldown_days,
        require_market_open=not bool(getattr(args, "allow_closed_market", False)),
        tickers=tickers,
        assume_verdict=assume,
        interval_seconds=float(getattr(args, "interval_seconds", 3600.0)),
        max_cycles=getattr(args, "max_cycles", None),
        as_json=args.json,
        no_color=no_color,
    )
