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
    ResearchOptions,
    ScheduleOptions,
    run_calibration,
    run_doctor,
    run_research,
    run_schedule_generate,
)

_PROG = "alphalab"
_DEFAULT_DB = Path("journal.db")


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
    _build_calibration_parser(sub)
    _build_schedule_parser(sub)
    _build_doctor_parser(sub)
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


def _parse_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {raw!r} (use YYYY-MM-DD)") from exc


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
    parser.print_help(err)  # pragma: no cover — argparse rejects unknown commands first
    return 2
