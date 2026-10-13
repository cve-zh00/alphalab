#!/usr/bin/env python3
"""launchd entrypoint for the alphalab scheduler.

launchd invokes this script with a cadence name (``daily`` / ``weekly`` /
``monthly`` / ``event``); it runs that cadence and exits 0 on success or a
non-zero code with a clear structured log on failure, so a failed run is
visible in the launchd logs rather than silent.

It also offers ``generate-plists``, which writes the launchd plists to a
directory you choose. **This script never runs ``launchctl``** — loading an
agent modifies your environment. After generating the plists, install one with
the command printed by ``generate-plists`` (and shown in ``--help``):

    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alphalab.daily.plist

The default collaborators wired here are deterministic and offline (a no-op
refresh and a stub analyser) so the entrypoint is safe to run anywhere. The
composition root is the single place to swap in the real Ollama adapter and the
analyst/debate pipelines — see ``_build_*`` below.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Allow running as a bare script (launchd invokes it by path): ensure the
# package under ``src/`` is importable without an installed wheel.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alphalab.adapters.scheduler import (  # noqa: E402
    LaunchdJobSpec,
    install_command,
    write_plists,
)
from alphalab.app.cadences import (  # noqa: E402
    CadenceOutcome,
    run_daily,
    run_event,
    run_monthly,
    run_weekly,
)
from alphalab.domain.schedule import Cadence  # noqa: E402
from alphalab.obs import (  # noqa: E402
    RunRecord,
    RunStatus,
    bind_run_context,
    configure_logging,
    log_run_record,
)

_LABEL_PREFIX = "com.alphalab"
_DEFAULT_TICKERS = ("MSFT", "AAPL", "NVDA")


def _default_refresh(tickers: tuple[str, ...]) -> int:
    """Offline deterministic refresh stub. Replace at the composition root."""
    return len(tickers)


def _default_analyse(ticker: str) -> str:
    """Offline analyser stub. Replace with the analyst/debate pipeline."""
    return f"stub-verdict:{ticker}"


def _run_cadence(cadence: Cadence, now: datetime) -> CadenceOutcome:
    """Dispatch to the right cadence use case with default collaborators."""
    if cadence is Cadence.DAILY:
        return run_daily(tickers=_DEFAULT_TICKERS, refresh=_default_refresh, now=now)
    if cadence is Cadence.WEEKLY:
        return run_weekly(tickers=_DEFAULT_TICKERS[:3], analyse=_default_analyse, now=now)
    if cadence is Cadence.MONTHLY:
        return run_monthly(
            predictions=(),
            settled_ids=(),
            realized_for=lambda _p: (0.0, 0.0, 0.0),
            persist_settlement=lambda _s: None,
            as_of=now.date(),
            now=now,
        )
    return run_event(
        ticker=_DEFAULT_TICKERS[0],
        event_kind="manual",
        analyse=_default_analyse,
        now=now,
    )


def _execute(cadence: Cadence) -> int:
    """Run one cadence, emit a RunRecord, and return a process exit code."""
    configure_logging()
    run_id = uuid.uuid4().hex
    started = datetime.now()
    log = bind_run_context(run_id=run_id, cadence=cadence)
    log.info("cadence_started")
    try:
        outcome = _run_cadence(cadence, started)
    except Exception as exc:  # entrypoint boundary: never crash launchd silently
        finished = datetime.now()
        record = RunRecord(
            run_id=run_id,
            run_key=f"{cadence.value}:{started.date().isoformat()}",
            cadence=cadence,
            started_at=started,
            finished_at=finished,
            status=RunStatus.FAILED,
            error=f"{type(exc).__name__}: {exc}",
        )
        log_run_record(record)
        return 1
    finished = datetime.now()
    record = RunRecord(
        run_id=run_id,
        run_key=f"{cadence.value}:{started.date().isoformat()}",
        cadence=cadence,
        started_at=started,
        finished_at=finished,
        status=RunStatus.OK,
        produced=outcome.produced,
    )
    log_run_record(record)
    return 0


def _generate(out_dir: Path) -> int:
    """Write the plists and print the install commands. Never loads them."""
    spec = LaunchdJobSpec(
        label_prefix=_LABEL_PREFIX,
        program_arguments=(sys.executable, str(Path(__file__).resolve())),
        working_directory=_PROJECT_ROOT,
        log_directory=_PROJECT_ROOT / "logs",
    )
    written = write_plists(spec, out_dir)
    print(f"Wrote {len(written)} plist(s) to {out_dir}:")
    for path in written:
        print(f"  {path}")
        print(f"    install with: {install_command(path)}")
    print("\nThis tool did NOT modify your system. Run the commands above yourself to install.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    A single positional accepts either a cadence or ``generate-plists``. This
    avoids the subparser/positional ambiguity while keeping the invocation
    launchd uses dead simple: ``schedule_runner daily``.

    The epilog documents the manual install step so ``--help`` tells the user
    exactly how to load an agent without this tool ever doing it.
    """
    parser = argparse.ArgumentParser(
        prog="schedule_runner",
        description="Run an alphalab cadence (launchd entrypoint) or generate launchd plists.",
        epilog=(
            "Install (run yourself; this tool never does):\n"
            "  python scripts/schedule_runner.py generate-plists "
            "--out-dir ~/Library/LaunchAgents\n"
            "  launchctl bootstrap gui/$(id -u) "
            "~/Library/LaunchAgents/com.alphalab.daily.plist\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=[*[c.value for c in Cadence], "generate-plists"],
        help="A cadence to run (daily/weekly/monthly/event) or 'generate-plists'.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "Library" / "LaunchAgents",
        help="For generate-plists: where to write them (default: ~/Library/LaunchAgents).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-plists":
        return _generate(args.out_dir)
    return _execute(Cadence(args.command))


if __name__ == "__main__":
    raise SystemExit(main())
