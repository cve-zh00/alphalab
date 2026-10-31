"""CLI command runner for ``alphalab alpharesearch {run|history}``."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TextIO

from alphalab.adapters.alpharesearch_sources import (
    AcademicPaperScanner,
    LivePortfolioPostmortemScanner,
    LocalReferenceRepoScanner,
    RemoteGitHubRepoScanner,
)
from alphalab.adapters.alpharesearch_store import SQLiteAlphaResearchStore
from alphalab.app.alpharesearch_runner import AlphaResearchRunner
from alphalab.domain.alpharesearch.contracts import (
    AlphaResearchRunReport,
    DiscoverySourceKind,
)

__all__ = [
    "AlphaResearchHistoryOptions",
    "AlphaResearchRunOptions",
    "build_alpharesearch_parser",
    "dispatch_alpharesearch",
    "run_alpharesearch_history",
    "run_alpharesearch_run",
]

_SOURCE_ALIAS_MAP: dict[str, DiscoverySourceKind] = {
    "local": DiscoverySourceKind.LOCAL_REPO,
    "local_repo": DiscoverySourceKind.LOCAL_REPO,
    "remote": DiscoverySourceKind.REMOTE_REPO,
    "remote_repo": DiscoverySourceKind.REMOTE_REPO,
    "papers": DiscoverySourceKind.ACADEMIC_PAPER,
    "academic_paper": DiscoverySourceKind.ACADEMIC_PAPER,
    "postmortem": DiscoverySourceKind.LIVE_POSTMORTEM,
    "live_postmortem": DiscoverySourceKind.LIVE_POSTMORTEM,
}


@dataclass(frozen=True, slots=True)
class AlphaResearchRunOptions:
    """Resolved options for ``alphalab alpharesearch run``."""

    as_of: date
    db_path: Path
    reference_dir: Path
    execution_db: Path
    journal_db: Path
    plans_dir: Path
    out_plan_path: Path | None = None
    sources: tuple[DiscoverySourceKind, ...] = (
        DiscoverySourceKind.LOCAL_REPO,
        DiscoverySourceKind.REMOTE_REPO,
        DiscoverySourceKind.ACADEMIC_PAPER,
        DiscoverySourceKind.LIVE_POSTMORTEM,
    )
    ignore_watermarks: bool = False
    as_json: bool = False
    no_color: bool = False


@dataclass(frozen=True, slots=True)
class AlphaResearchHistoryOptions:
    """Resolved options for ``alphalab alpharesearch history``."""

    db_path: Path
    limit: int = 10
    as_json: bool = False
    no_color: bool = False


def parse_source_list(raw: str) -> tuple[DiscoverySourceKind, ...]:
    """Parse a comma-separated list of source names into :class:`DiscoverySourceKind`."""
    items: list[DiscoverySourceKind] = []
    for token in raw.split(","):
        clean = token.strip().lower()
        if not clean:
            continue
        if clean not in _SOURCE_ALIAS_MAP:
            valid = ", ".join(sorted(_SOURCE_ALIAS_MAP))
            raise ValueError(f"unknown alpharesearch source {token!r}; valid: {valid}")
        kind = _SOURCE_ALIAS_MAP[clean]
        if kind not in items:
            items.append(kind)
    if not items:
        raise ValueError("at least one source must be specified")
    return tuple(items)


def _render_report_console(report: AlphaResearchRunReport, *, out: TextIO) -> None:
    print(
        f"alpharesearch cycle: as_of={report.as_of.isoformat()}  "
        f"run_id={report.run_id}  "
        f"n_trials={report.n_trials_before}->{report.n_trials_after}",
        file=out,
    )
    print(
        f"  sources queried ({len(report.sources_queried)}): "
        + ", ".join(s.value for s in report.sources_queried),
        file=out,
    )
    print(f"  DISCOVERY SIGNALS ({len(report.signals_discovered)})", file=out)
    for sig in report.signals_discovered:
        print(
            f"    [{sig.source_kind.value}] {sig.source_name}: {sig.title}",
            file=out,
        )

    print(f"  3-ROLE COMMITTEE DISCERNMENT ({len(report.decisions)} proposals)", file=out)
    for d in report.decisions:
        prop = d.proposal
        sc = d.empirical_scorecard
        dsr_txt = f"DSR={sc.dsr:.3f}" if sc.dsr is not None else f"mode={sc.evaluation_mode}"
        print(
            f"    [{d.status.value.upper()}] {prop.proposal_id} ({prop.subsystem.value}) "
            f"{prop.title} — {dsr_txt}",
            file=out,
        )
        print(f"      reason: {d.summary_reason}", file=out)

    print(
        f"  SUMMARY: promoted={len(report.promoted_decisions)}  "
        f"vetoed={len(report.rejected_decisions)}  "
        f"plan={report.plan_output_path or 'none'}",
        file=out,
    )


def run_alpharesearch_run(
    opts: AlphaResearchRunOptions,
    *,
    out: TextIO,
    err: TextIO,
) -> int:
    """Execute one weekly AlphaResearch committee cycle."""
    try:
        store = SQLiteAlphaResearchStore(opts.db_path)
        runner = AlphaResearchRunner(
            store=store,
            local_scanner=LocalReferenceRepoScanner(opts.reference_dir),
            remote_scanner=RemoteGitHubRepoScanner(),
            paper_scanner=AcademicPaperScanner(),
            postmortem_scanner=LivePortfolioPostmortemScanner(
                opts.execution_db,
                opts.journal_db,
            ),
            plans_dir=opts.plans_dir,
        )
        report = runner.run_weekly_cycle(
            as_of=opts.as_of,
            sources=opts.sources,
            ignore_watermarks=opts.ignore_watermarks,
            output_plan_path=opts.out_plan_path,
        )
    except Exception as exc:
        print(f"error: alpharesearch run failed: {exc}", file=err)
        return 1

    if opts.as_json:
        print(report.model_dump_json(indent=2), file=out)
    else:
        _render_report_console(report, out=out)
    return 0


def run_alpharesearch_history(
    opts: AlphaResearchHistoryOptions,
    *,
    out: TextIO,
    err: TextIO,
) -> int:
    """List recent weekly AlphaResearch runs from SQLite."""
    try:
        store = SQLiteAlphaResearchStore(opts.db_path)
        runs = store.list_runs(limit=opts.limit)
    except Exception as exc:
        print(f"error: alpharesearch history failed: {exc}", file=err)
        return 1

    if opts.as_json:
        import json

        payload = [r.model_dump(mode="json") for r in runs]
        print(json.dumps(payload, indent=2), file=out)
        return 0

    print(f"alpharesearch history ({len(runs)} runs):", file=out)
    for r in runs:
        print(
            f"  {r.as_of.isoformat()}  run_id={r.run_id}  "
            f"signals={len(r.signals_discovered)}  "
            f"promoted={len(r.promoted_decisions)}/{len(r.decisions)}  "
            f"n_trials={r.n_trials_after}  plan={r.plan_output_path or 'none'}",
            file=out,
        )
    return 0


def build_alpharesearch_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    project_root: Path,
    default_execution_db: Path,
    default_journal_db: Path,
) -> None:
    """Register ``alphalab alpharesearch {run|history}`` subcommands."""
    default_ar_db = default_journal_db.parent / "alpharesearch.db"
    default_plans_dir = default_journal_db.parent / "alpharesearch" / "plans"
    default_ref_dir = project_root / "reference"

    p = sub.add_parser(
        "alpharesearch",
        help="Run the weekly 3-role AlphaResearch committee and discernment pipeline.",
    )
    ar_sub = p.add_subparsers(dest="alpharesearch_command", metavar="SUBCOMMAND")

    run_p = ar_sub.add_parser(
        "run",
        help="Scan sources, run 3-role discernment, and emit a certified plan.md.",
    )
    run_p.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),
        help="Knowledge date (YYYY-MM-DD, default: today).",
    )
    run_p.add_argument(
        "--sources",
        default="local,remote,papers,postmortem",
        help="Comma-separated sources: local,remote,papers,postmortem.",
    )
    run_p.add_argument(
        "--db",
        type=Path,
        default=default_ar_db,
        help="Path to alpharesearch SQLite DB.",
    )
    run_p.add_argument(
        "--reference-dir",
        type=Path,
        default=default_ref_dir,
        help="Directory containing local reference repositories.",
    )
    run_p.add_argument(
        "--execution-db",
        type=Path,
        default=default_execution_db,
        help="Path to execution.db for live portfolio postmortem.",
    )
    run_p.add_argument(
        "--journal-db",
        type=Path,
        default=default_journal_db,
        help="Path to journal.db for forecast calibration postmortem.",
    )
    run_p.add_argument(
        "--plans-dir",
        type=Path,
        default=default_plans_dir,
        help="Directory where dated plan.md files are written.",
    )
    run_p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional explicit output path for the generated plan.md.",
    )
    run_p.add_argument(
        "--ignore-watermarks",
        action="store_true",
        help="Scan all sources even if their commit/paper watermark was already seen.",
    )
    run_p.add_argument("--json", action="store_true", help="Emit JSON output.")
    run_p.add_argument("--no-color", action="store_true", help="Disable ANSI colour.")

    hist_p = ar_sub.add_parser(
        "history",
        help="List past weekly AlphaResearch committee runs.",
    )
    hist_p.add_argument(
        "--db",
        type=Path,
        default=default_ar_db,
        help="Path to alpharesearch SQLite DB.",
    )
    hist_p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of past runs to display.",
    )
    hist_p.add_argument("--json", action="store_true", help="Emit JSON output.")
    hist_p.add_argument("--no-color", action="store_true", help="Disable ANSI colour.")


def dispatch_alpharesearch(
    args: argparse.Namespace,
    *,
    out: TextIO,
    err: TextIO,
    no_color: bool,
) -> int:
    """Dispatch ``alphalab alpharesearch`` subcommands."""
    subcmd = getattr(args, "alpharesearch_command", None)
    if subcmd == "run":
        try:
            sources = parse_source_list(args.sources)
        except ValueError as exc:
            print(f"error: {exc}", file=err)
            return 2
        return run_alpharesearch_run(
            AlphaResearchRunOptions(
                as_of=args.as_of,
                db_path=args.db,
                reference_dir=args.reference_dir,
                execution_db=args.execution_db,
                journal_db=args.journal_db,
                plans_dir=args.plans_dir,
                out_plan_path=args.out,
                sources=sources,
                ignore_watermarks=args.ignore_watermarks,
                as_json=args.json,
                no_color=no_color,
            ),
            out=out,
            err=err,
        )
    if subcmd == "history":
        return run_alpharesearch_history(
            AlphaResearchHistoryOptions(
                db_path=args.db,
                limit=args.limit,
                as_json=args.json,
                no_color=no_color,
            ),
            out=out,
            err=err,
        )
    print("usage: alphalab alpharesearch {run|history} [options]", file=err)
    return 2
