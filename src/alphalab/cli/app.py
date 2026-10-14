"""The alphalab CLI entrypoint: argparse wiring and dispatch.

Stdlib ``argparse`` only — no third-party CLI dependency. This module parses
``argv``, resolves it into a typed options object per subcommand, and delegates
to the matching runner in :mod:`alphalab.cli.commands`. It owns ``--help`` text
and exit-code policy (0 on success, non-zero on error); the runners own the
actual work.

Exit codes:
* ``0`` — success.
* ``1`` — a runtime failure (e.g. the pipeline raised, Ollama down with --llm).
* ``2`` — a usage / input error (bad args, unknown ticker, missing DB).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import TextIO

from alphalab.cli.commands import (
    CalibrationOptions,
    DoctorOptions,
    JournalOptions,
    PaperLogOptions,
    PaperRunOptions,
    PaperStatusOptions,
    ResearchOptions,
    ScheduleOptions,
    SettleOptions,
    run_calibration,
    run_doctor,
    run_journal,
    run_paper_log,
    run_paper_run,
    run_paper_status,
    run_predict,
    run_research,
    run_schedule_generate,
    run_settle,
)
from alphalab.cli.commands.predict import DEFAULT_HORIZON_DAYS, options_from_args
from alphalab.cli.forecast_paths import (
    default_abstention_db,
    default_journal_db,
    default_prices_dir,
)

_PROG = "alphalab"
_DEFAULT_DB = default_journal_db()
_DEFAULT_EXECUTION_DB = default_journal_db().parent / "execution.db"
_DEFAULT_PRICES_DIR = default_prices_dir()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _no_color_default() -> bool:
    """Honour the conventional ``NO_COLOR`` environment variable."""
    return bool(os.environ.get("NO_COLOR"))


def _add_common_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colour even on a TTY."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser with one subparser per command."""
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            "Alphalab: code-calculated equity research with LLM-assisted narrative. "
            "Numbers come from code; the model only argues and judges."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    _build_research_parser(sub)
    _build_predict_parser(sub)
    _build_settle_parser(sub)
    _build_journal_parser(sub)
    _build_calibration_parser(sub)
    _build_schedule_parser(sub)
    _build_doctor_parser(sub)
    _build_paper_parser(sub)
    _build_run_cadence_parser(sub)
    return parser


def _build_research_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "research",
        help="Run the research pipeline for a ticker and print the report.",
        description="Run the analyst (or --debate) pipeline and render the gated artifact.",
    )
    p.add_argument("ticker", help="Ticker symbol, e.g. AAPL.")
    p.add_argument(
        "--knowledge-date",
        type=_parse_date,
        default=date.today(),
        help="Point-in-time knowledge date (YYYY-MM-DD; default: today).",
    )
    p.add_argument(
        "--source",
        choices=("fixture", "live"),
        default="fixture",
        help="Data source: 'fixture' (local prices, demo) or 'live' (SEC EDGAR).",
    )
    p.add_argument("--debate", action="store_true", help="Use the Bull/Bear/Judge debate pipeline.")
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Use deterministic template narration (no Ollama, no 60s wait).",
    )
    p.add_argument(
        "--provenance",
        action="store_true",
        help="Append the derivation tree of the headline number.",
    )
    _add_common_output_flags(p)


def _build_predict_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "predict",
        help="Research a ticker and record a forward-dated prediction (or abstain).",
        description=(
            "Run the research pipeline and record a dated, scoreable forecast — or a "
            "stated abstention. An abstention is not an error (exit 0)."
        ),
    )
    p.add_argument("ticker", nargs="+", help="One or more ticker symbols, e.g. AAPL MSFT.")
    p.add_argument(
        "--horizon-days",
        type=int,
        default=DEFAULT_HORIZON_DAYS,
        help=f"Days until settlement (default: {DEFAULT_HORIZON_DAYS}).",
    )
    p.add_argument(
        "--knowledge-date",
        type=_parse_date,
        default=date.today(),
        help="Point-in-time knowledge date (YYYY-MM-DD; default: today).",
    )
    p.add_argument(
        "--llm-cutoff",
        type=_parse_date,
        default=None,
        help="Model training cutoff (YYYY-MM-DD); enables the anti-leakage gate.",
    )
    p.add_argument(
        "--source",
        choices=("fixture", "live"),
        default="fixture",
        help="Data source: 'fixture' (local prices, demo) or 'live' (SEC EDGAR).",
    )
    p.add_argument("--debate", action="store_true", help="Use the Bull/Bear/Judge debate pipeline.")
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Use deterministic template narration (no Ollama, no 60s wait).",
    )
    p.add_argument(
        "--assume-verdict",
        choices=("buy", "sell", "hold"),
        default=None,
        help="Override the offline (--no-llm) verdict to exercise the flow; ignored with the LLM.",
    )
    p.add_argument(
        "--db", type=Path, default=_DEFAULT_DB, help=f"Journal DB path (default: {_DEFAULT_DB})."
    )
    p.add_argument(
        "--abstentions-db",
        type=Path,
        default=default_abstention_db(),
        help=f"Abstention store path (default: {default_abstention_db()}).",
    )
    _add_common_output_flags(p)


def _build_settle_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "settle",
        help="Liquidate every due prediction from local prices (advances the clock).",
        description=(
            "Settle due predictions, pricing realized/benchmark moves from local close "
            "series. Reports settled, pending, and could-not-price (with reasons)."
        ),
    )
    p.add_argument(
        "--db", type=Path, default=_DEFAULT_DB, help=f"Journal DB path (default: {_DEFAULT_DB})."
    )
    p.add_argument(
        "--prices-dir",
        type=Path,
        default=default_prices_dir(),
        help=f"Directory of <ticker>.csv close series (default: {default_prices_dir()}).",
    )
    p.add_argument(
        "--as-of",
        type=_parse_date,
        default=date.today(),
        help="Settlement date (YYYY-MM-DD; default: today).",
    )
    _add_common_output_flags(p)


def _build_journal_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "journal",
        help="List open and settled predictions (read-only history view).",
        description="Show the accumulating forward record: open theses, settled outcomes.",
    )
    p.add_argument(
        "--db", type=Path, default=_DEFAULT_DB, help=f"Journal DB path (default: {_DEFAULT_DB})."
    )
    p.add_argument(
        "--abstentions-db",
        type=Path,
        default=default_abstention_db(),
        help=f"Abstention store path (default: {default_abstention_db()}).",
    )
    _add_common_output_flags(p)


def _build_calibration_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "calibration",
        help="Show the calibration report from the prediction journal.",
        description="Score Brier/ECE/hit-rate/overconfidence from settled predictions.",
    )
    p.add_argument(
        "--db", type=Path, default=_DEFAULT_DB, help=f"Journal DB path (default: {_DEFAULT_DB})."
    )
    _add_common_output_flags(p)


def _build_schedule_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "schedule",
        help="Generate launchd plists (never loads them).",
        description="Scheduler tooling. 'generate' writes plists and prints the install command.",
    )
    schedule_sub = p.add_subparsers(dest="schedule_command", metavar="SUBCOMMAND")
    gen = schedule_sub.add_parser(
        "generate",
        help="Write launchd plists and print the exact install command (no launchctl).",
    )
    gen.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "Library" / "LaunchAgents",
        help="Where to write the plists (default: ~/Library/LaunchAgents).",
    )


def _build_doctor_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "doctor",
        help="Read-only environment check (Ollama, fixtures, journal, EDGAR).",
        description="Report what is and is not configured. Modifies nothing.",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_DB,
        help=f"Journal DB path to probe (default: {_DEFAULT_DB}).",
    )
    _add_common_output_flags(p)


def _build_paper_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser(
        "paper",
        help="Paper trading: status, run (dry-run by default), log. Paper account only.",
        description=(
            "Operate the paper-trading loop. 'run' is a dry-run unless --submit is "
            "given, so the default never sends an order. No strategy here has a "
            "demonstrated edge; this is a forward test of execution, not a profit bet."
        ),
    )
    paper_sub = p.add_subparsers(dest="paper_command", metavar="SUBCOMMAND")
    _build_paper_status_parser(paper_sub)
    _build_paper_run_parser(paper_sub)
    _build_paper_log_parser(paper_sub)


def _add_paper_common(parser: argparse.ArgumentParser, *, with_prices: bool) -> None:
    parser.add_argument(
        "--execution-db",
        type=Path,
        default=_DEFAULT_EXECUTION_DB,
        help=f"Append-only execution log DB (default: {_DEFAULT_EXECUTION_DB}).",
    )
    if with_prices:
        parser.add_argument(
            "--prices-dir",
            type=Path,
            default=_DEFAULT_PRICES_DIR,
            help=f"Directory of <ticker>.csv series (default: {_DEFAULT_PRICES_DIR}).",
        )
        parser.add_argument(
            "--halt-file",
            type=Path,
            default=None,
            help="Kill-switch halt file path (default: ./HALT at the repo root).",
        )
    _add_common_output_flags(parser)


def _build_paper_status_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    s = sub.add_parser(
        "status",
        help="Read-only: account, positions, reconciliation, kill switch, market clock.",
    )
    _add_paper_common(s, with_prices=True)


def _build_paper_run_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    r = sub.add_parser(
        "run",
        help="Run one cycle. Dry-run by default; pass --submit to send paper orders.",
        description="Compute the plan and (only with --submit) send it to the paper account.",
    )
    group = r.add_mutually_exclusive_group()
    group.add_argument(
        "--submit",
        action="store_true",
        help="Actually send orders to the paper account. Default is a dry run.",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly request a dry run (this is already the default).",
    )
    r.add_argument(
        "--force",
        action="store_true",
        help=(
            "Submit even when the quarterly schedule does not call for a "
            "rebalance. This trades off the cadence the backtest measured."
        ),
    )
    _add_paper_common(r, with_prices=True)


def _build_paper_log_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    lg = sub.add_parser("log", help="Print the append-only execution history.")
    _add_paper_common(lg, with_prices=False)


def _build_run_cadence_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    # Hidden entrypoint fired by the launchd plists that `schedule generate`
    # writes. Not part of the user-facing surface, so it carries no help text.
    p = sub.add_parser("_run-cadence")
    p.add_argument("cadence", choices=("daily", "weekly", "monthly"))
    _add_common_output_flags(p)


def _parse_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {raw!r} (use YYYY-MM-DD)") from exc


def _dispatch_paper(args: argparse.Namespace, *, out: TextIO, err: TextIO, no_color: bool) -> int:
    """Route a ``paper`` invocation to its subcommand runner."""
    command = getattr(args, "paper_command", None)
    if command == "status":
        return run_paper_status(
            PaperStatusOptions(
                execution_db=args.execution_db,
                prices_dir=args.prices_dir,
                halt_path=args.halt_file,
                as_json=args.json,
                no_color=no_color,
            ),
            out=out,
            err=err,
        )
    if command == "run":
        return run_paper_run(
            PaperRunOptions(
                execution_db=args.execution_db,
                prices_dir=args.prices_dir,
                halt_path=args.halt_file,
                submit=bool(getattr(args, "submit", False)),
                force=bool(getattr(args, "force", False)),
                as_json=args.json,
                no_color=no_color,
            ),
            out=out,
            err=err,
        )
    if command == "log":
        return run_paper_log(
            PaperLogOptions(
                execution_db=args.execution_db,
                as_json=args.json,
                no_color=no_color,
            ),
            out=out,
            err=err,
        )
    print(
        "usage: alphalab paper {status|run|log} [options]\n"
        "  run is a dry-run by default; pass --submit to send paper orders.",
        file=err,
    )
    return 2


def main(
    argv: list[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None
) -> int:
    """Parse ``argv`` and dispatch to the matching command runner.

    Args:
        argv: Argument vector (excluding the program name). ``None`` uses
            ``sys.argv[1:]``.
        out: Output stream (defaults to ``sys.stdout``); injectable for tests.
        err: Error stream (defaults to ``sys.stderr``); injectable for tests.

    Returns:
        A process exit code (0 success, 1 runtime error, 2 usage error).
    """
    out = out or sys.stdout
    err = err or sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(out)
        return 0

    no_color = bool(getattr(args, "no_color", False)) or _no_color_default()

    if args.command == "research":
        return run_research(
            ResearchOptions(
                ticker=args.ticker.upper(),
                knowledge_date=args.knowledge_date,
                source=args.source,
                use_llm=not args.no_llm,
                debate=args.debate,
                as_json=args.json,
                show_provenance=args.provenance,
                no_color=no_color,
            ),
            out=out,
            err=err,
        )
    if args.command == "predict":
        return run_predict(options_from_args(args, no_color=no_color), out=out, err=err)
    if args.command == "settle":
        return run_settle(
            SettleOptions(
                db_path=args.db,
                prices_dir=args.prices_dir,
                as_of=args.as_of,
                as_json=args.json,
                no_color=no_color,
            ),
            out=out,
            err=err,
        )
    if args.command == "journal":
        return run_journal(
            JournalOptions(
                db_path=args.db,
                abstentions_path=args.abstentions_db,
                as_json=args.json,
                no_color=no_color,
            ),
            out=out,
            err=err,
        )
    if args.command == "calibration":
        return run_calibration(
            CalibrationOptions(db_path=args.db, as_json=args.json, no_color=no_color),
            out=out,
            err=err,
        )
    if args.command == "schedule":
        if getattr(args, "schedule_command", None) != "generate":
            print("usage: alphalab schedule generate [--out-dir DIR]", file=err)
            return 2
        return run_schedule_generate(
            ScheduleOptions(out_dir=args.out_dir, project_root=_project_root()),
            out=out,
            err=err,
        )
    if args.command == "doctor":
        return run_doctor(
            DoctorOptions(db_path=args.db, as_json=args.json, no_color=no_color),
            out=out,
            err=err,
        )
    if args.command == "paper":
        return _dispatch_paper(args, out=out, err=err, no_color=no_color)
    if args.command == "_run-cadence":
        from alphalab.cli.commands.run_cadence import run_cadence

        return run_cadence(args.cadence, out=out, err=err, no_color=no_color)
    parser.print_help(err)  # pragma: no cover — argparse rejects unknown commands first
    return 2
