"""The scheduling spec, the idempotency key, and the pure ``due_runs`` rule.

Everything here is a pure function of its arguments. Time enters as a
parameter (``now``) and the last successful run enters as ``last_run`` — the
module never reads the wall clock and holds no state. That is what makes the
scheduler exhaustively testable on pinned dates.

Two ideas carry the design:

* **Run key** (``run_key``) — a string derived from *cadence + the period the
  moment falls in* (ISO day / ISO week / calendar month). Two moments in the
  same period produce the same key, so a run can be de-duplicated: running the
  daily cadence twice on the same day yields one key and therefore one unit of
  work. This is the idempotency primitive.

* **Missed-job policy** (``due_runs``) — the code-level counterpart of why we
  chose launchd over cron. cron drops a job whose instant passed while the Mac
  slept; launchd enqueues it and fires on wake. ``due_runs`` encodes the same
  policy: a cadence is due whenever its *current period's key differs from the
  key of the last run* — regardless of how long ago that was. If the daily job
  last ran five days ago, today's key differs, so it is due now.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from alphalab.domain.schedule.cadence import Cadence
from alphalab.domain.schedule.holidays import is_trading_day

__all__ = ["ScheduleSpec", "due_runs", "run_key"]

#: Default ISO weekday for the weekly cadence. ISO uses Monday=1 .. Sunday=7;
#: Sunday (7) matches the dossier's "domingo" deep-dive slot. ``ScheduleSpec``
#: stores Python's Monday=0..Sunday=6 convention, so Sunday is 6.
_DEFAULT_WEEKLY_WEEKDAY = 6  # Sunday (Python Mon=0..Sun=6)
_MAX_WEEKDAY = 6


@dataclass(frozen=True, slots=True)
class ScheduleSpec:
    """Immutable description of when a cadence should run.

    Attributes:
        cadence: Which cadence this spec governs.
        weekday: For the weekly cadence, the Python weekday (Mon=0..Sun=6) the
            run targets. Ignored for other cadences. Defaults to Sunday.
    """

    cadence: Cadence
    weekday: int = _DEFAULT_WEEKLY_WEEKDAY

    def __post_init__(self) -> None:
        if not 0 <= self.weekday <= _MAX_WEEKDAY:
            raise ValueError(f"weekday must be in [0, {_MAX_WEEKDAY}], got {self.weekday}")


def _last_day_number(year: int, month: int) -> int:
    """Return the day-of-month of the last day of ``year``/``month``."""
    next_first = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    return (next_first - timedelta(days=1)).day


def run_key(spec: ScheduleSpec, moment: datetime) -> str:
    """Return the idempotency key for running ``spec`` at ``moment``.

    The key is ``"<cadence>:<period>"`` where the period identifies the unit
    of de-duplication for the cadence:

    * daily → ISO calendar date (one run per day);
    * weekly → ISO year + ISO week (one run per week);
    * monthly → year + month (one run per month);
    * event → ISO *timestamp* (every event is distinct; events are never
      calendar-de-duplicated).

    Two moments in the same period yield identical keys; distinct cadences
    never collide because the cadence is part of the key.
    """
    cadence = spec.cadence
    if cadence is Cadence.DAILY:
        period = moment.date().isoformat()
    elif cadence is Cadence.WEEKLY:
        iso = moment.isocalendar()
        period = f"{iso.year:04d}-W{iso.week:02d}"
    elif cadence is Cadence.MONTHLY:
        period = f"{moment.year:04d}-{moment.month:02d}"
    else:  # Cadence.EVENT
        period = moment.isoformat()
    return f"{cadence.value}:{period}"


def due_runs(spec: ScheduleSpec, *, last_run: datetime | None, now: datetime) -> bool:
    """Decide whether ``spec`` is due to run at ``now``, given ``last_run``.

    Pure and total: for any inputs it returns a bool and never raises. The
    policy, cadence by cadence:

    * **daily** — due on a trading day whose run key differs from the last
      run's. A closed market (weekend or holiday) is never due. The missed-job
      case is automatic: if ``last_run`` was days ago, today's key differs, so
      a queued job runs on wake rather than being discarded.
    * **weekly** — due on the configured weekday (or any later day of a week
      that has not yet run), when this ISO week's key differs from the last
      run's. An overdue week (the Mac slept through the slot) is still due.
    * **monthly** — due on the last calendar day of a month whose key has not
      yet run; an overdue month-end is still due.
    * **event** — never calendar-due; event runs are triggered explicitly.
    """
    if spec.cadence is Cadence.EVENT:
        return False

    already_ran_this_period = last_run is not None and run_key(spec, last_run) == run_key(spec, now)
    if already_ran_this_period:
        return False

    if spec.cadence is Cadence.DAILY:
        return is_trading_day(now.date())

    if spec.cadence is Cadence.WEEKLY:
        return now.weekday() >= spec.weekday

    # Cadence.MONTHLY — due on the last calendar day of the month.
    return now.day == _last_day_number(now.year, now.month)
