"""Pure scheduling calendar for alphalab's cadences (dossier §6/§9).

This package is the pure, testable heart of the scheduler. It decides *when* a
cadence should run from injected time alone — it performs no I/O, holds no
state, and never calls ``datetime.now()``. The side-effecting pieces (building
launchd plists, executing a cadence) live in ``adapters``/``app``/``scripts``.

Public surface:

* :class:`Cadence` — DAILY / WEEKLY / MONTHLY / EVENT.
* :class:`ScheduleSpec` — frozen description of when a cadence runs.
* :func:`due_runs` — the pure "should this run now?" decision, including the
  missed-job policy that mirrors launchd's enqueue-on-wake behaviour.
* :func:`run_key` — the idempotency key (cadence + period) that de-duplicates
  work within a period.
* :func:`is_market_holiday` / :func:`is_trading_day` / :func:`market_holidays`
  — the explicit, versioned US market-holiday calendar (see
  ``holidays`` for its documented scope and limitations).
"""

from __future__ import annotations

from alphalab.domain.schedule.cadence import Cadence
from alphalab.domain.schedule.holidays import (
    CALENDAR_VERSION,
    is_market_holiday,
    is_trading_day,
    market_holidays,
)
from alphalab.domain.schedule.spec import ScheduleSpec, due_runs, run_key

__all__ = [
    "CALENDAR_VERSION",
    "Cadence",
    "ScheduleSpec",
    "due_runs",
    "is_market_holiday",
    "is_trading_day",
    "market_holidays",
    "run_key",
]
