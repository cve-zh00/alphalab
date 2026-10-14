"""The ``schedule generate`` command: write launchd plists, never load them.

Generates the launchd plists via the pure adapter and prints the **exact**
install command for the user to run themselves. It deliberately **never** calls
``launchctl``: loading an agent mutates the user's environment and requires
their explicit action. The output makes that boundary explicit.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from alphalab.adapters.scheduler import LaunchdJobSpec, install_command, write_plists

_LABEL_PREFIX = "com.alphalab"


@dataclass(frozen=True)
class ScheduleOptions:
    """Resolved options for one ``schedule generate`` invocation."""

    out_dir: Path
    project_root: Path


def run_schedule_generate(opts: ScheduleOptions, *, out: TextIO, err: TextIO) -> int:
    """Generate the launchd plists and print install commands. Never loads them."""
    spec = LaunchdJobSpec(
        label_prefix=_LABEL_PREFIX,
        program_arguments=(sys.executable, "-m", "alphalab.cli", "_run-cadence"),
        working_directory=opts.project_root,
        log_directory=opts.project_root / "logs",
    )
    try:
        written = write_plists(spec, opts.out_dir)
    except OSError as exc:
        print(f"error: could not write plists to {opts.out_dir}: {exc}", file=err)
        return 1

    print(f"Wrote {len(written)} launchd plist(s) to {opts.out_dir}:", file=out)
    for path in written:
        print(f"  {path}", file=out)
        print(f"    install with: {install_command(path)}", file=out)
    print(
        "\nThis command did NOT modify your system. "
        "launchctl was not run. Execute the commands above yourself to install.",
        file=out,
    )
    return 0
