"""launchd plist generator for the scheduler (adapter layer).

Why launchd and not cron (dossier §6): cron silently drops a job whose
scheduled instant passed while the Mac was asleep; launchd **enqueues** it and
fires on wake. For a pipeline that runs once or twice a day on a laptop that
sleeps, that difference is decisive.

This adapter only *generates artifacts*. It builds a correctly-escaped plist
with :mod:`plistlib` (never string templates) and can write the files to a
directory you choose. It **never** runs ``launchctl``: loading an agent
modifies the user's environment, which is a side effect we refuse to perform
without explicit action by the user. :func:`install_command` returns the exact
command string for the user to run themselves.

The ``StartCalendarInterval`` mapping:

* daily  → ``{Hour, Minute}`` (every day at that time);
* weekly → ``{Weekday, Hour, Minute}`` (launchd Weekday: Sunday=0);
* monthly→ ``{Day, Hour, Minute}`` (that day of each month).

The pure *decision* to run (and the missed-job handling) lives in
``domain.schedule``; launchd is just the trigger that wakes the entrypoint.
"""

from __future__ import annotations

import plistlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from alphalab.domain.schedule import Cadence

__all__ = [
    "LaunchdJobSpec",
    "build_plist",
    "install_command",
    "write_plists",
]

#: Default post-close-ish run time (local). 22:00 is comfortably after the US
#: cash close for a laptop in the Americas; it is only the launchd *trigger* —
#: the actual gating is done by ``due_runs``.
_DEFAULT_HOUR = 22
_DEFAULT_MINUTE = 0

#: launchd uses Sunday=0..Saturday=6 for ``Weekday``. The weekly cadence
#: targets Sunday, matching the dossier's "domingo" deep-dive slot.
_LAUNCHD_SUNDAY = 0

#: Day-of-month for the monthly trigger. launchd cannot express "last day of
#: the month", so we trigger on the 1st and let ``due_runs`` gate the actual
#: month-end semantics (a missed month-end still runs on the next wake).
_MONTHLY_TRIGGER_DAY = 1

#: Calendar cadences that get a plist. EVENT is excluded by design.
_CALENDAR_CADENCES = (Cadence.DAILY, Cadence.WEEKLY, Cadence.MONTHLY)


@dataclass(frozen=True, slots=True)
class LaunchdJobSpec:
    """Everything needed to render a launchd job for a cadence.

    Attributes:
        label_prefix: Reverse-DNS label prefix; the cadence is appended, e.g.
            ``com.alphalab.daily``.
        program_arguments: The argv launchd executes, *excluding* the trailing
            cadence argument which :func:`build_plist` appends.
        working_directory: ``WorkingDirectory`` for the job.
        log_directory: Directory for the per-cadence stdout/stderr logs.
        hour: Local hour of the trigger.
        minute: Local minute of the trigger.
    """

    label_prefix: str
    program_arguments: tuple[str, ...]
    working_directory: Path
    log_directory: Path
    hour: int = _DEFAULT_HOUR
    minute: int = _DEFAULT_MINUTE

    def __post_init__(self) -> None:
        if not self.label_prefix:
            raise ValueError("label_prefix must be non-empty")
        if not self.program_arguments:
            raise ValueError("program_arguments must be non-empty")
        if not 0 <= self.hour <= 23:
            raise ValueError(f"hour must be in [0, 23], got {self.hour}")
        if not 0 <= self.minute <= 59:
            raise ValueError(f"minute must be in [0, 59], got {self.minute}")


def _calendar_interval(spec: LaunchdJobSpec, cadence: Cadence) -> dict[str, int]:
    """Build the ``StartCalendarInterval`` dict for ``cadence``."""
    interval: dict[str, int] = {"Hour": spec.hour, "Minute": spec.minute}
    if cadence is Cadence.WEEKLY:
        interval["Weekday"] = _LAUNCHD_SUNDAY
    elif cadence is Cadence.MONTHLY:
        interval["Day"] = _MONTHLY_TRIGGER_DAY
    return interval


def _plist_dict(spec: LaunchdJobSpec, cadence: Cadence) -> dict[str, object]:
    """Assemble the plist mapping for one calendar cadence."""
    label = f"{spec.label_prefix}.{cadence.value}"
    argv = [*spec.program_arguments, cadence.value]
    out_log = spec.log_directory / f"{cadence.value}.out.log"
    err_log = spec.log_directory / f"{cadence.value}.err.log"
    return {
        "Label": label,
        "ProgramArguments": argv,
        "StartCalendarInterval": _calendar_interval(spec, cadence),
        "WorkingDirectory": str(spec.working_directory),
        "StandardOutPath": str(out_log),
        "StandardErrorPath": str(err_log),
        # Do not run immediately on load; the cadence is gated by due_runs and
        # fired by the calendar interval (plus enqueue-on-wake).
        "RunAtLoad": False,
    }


def build_plist(spec: LaunchdJobSpec, cadence: Cadence) -> bytes:
    """Return the launchd plist for ``cadence`` as correctly-escaped XML bytes.

    Raises:
        ValueError: If ``cadence`` is EVENT (event runs are not calendar-based).
    """
    if cadence is Cadence.EVENT:
        raise ValueError("event cadence is not calendar-scheduled; no plist is generated")
    return plistlib.dumps(_plist_dict(spec, cadence), fmt=plistlib.FMT_XML)


def write_plists(spec: LaunchdJobSpec, out_dir: Path) -> tuple[Path, ...]:
    """Write one plist per calendar cadence into ``out_dir`` and return paths.

    Creates ``out_dir`` if needed. This writes files only; it never loads them
    into launchd.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for cadence in _CALENDAR_CADENCES:
        path = out_dir / f"{spec.label_prefix}.{cadence.value}.plist"
        path.write_bytes(build_plist(spec, cadence))
        written.append(path)
    return tuple(written)


def install_command(plist_path: Path) -> str:
    """Return the exact command the *user* runs to load an agent.

    We never execute this: loading an agent mutates the user's environment.
    Modern launchd prefers ``bootstrap`` over the deprecated ``load``.
    """
    return f"launchctl bootstrap gui/$(id -u) {plist_path}"


# A read-only view of the launchd Weekday convention, exported for callers that
# want to document or assert on it without importing the private constant.
LAUNCHD_WEEKDAYS: Mapping[str, int] = {
    "sunday": 0,
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
}
