"""Unit tests for ``due_runs`` — the pure "should this cadence run now?" rule.

The most important test here is the **missed-job** case: it is the code-level
counterpart of why we chose launchd over cron. cron silently drops a job if the
Mac was asleep at the scheduled instant; launchd enqueues it and runs it on
wake. ``due_runs`` encodes the same policy: if ``last_run`` is older than the
cadence demands, it returns True regardless of the exact scheduled instant.

All time is injected. Nothing reads the wall clock.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from alphalab.domain.schedule import Cadence, ScheduleSpec, due_runs


# ---------------------------------------------------------------------------
# First run (no prior run recorded)
# ---------------------------------------------------------------------------
def test_first_run_is_always_due_on_a_trading_day() -> None:
    # Arrange: Monday 2025-07-07, a trading day, never run before.
    spec = ScheduleSpec(cadence=Cadence.DAILY)
    now = datetime(2025, 7, 7, 22, 0)
    # Act / Assert
    assert due_runs(spec, last_run=None, now=now) is True


# ---------------------------------------------------------------------------
# Daily cadence
# ---------------------------------------------------------------------------
def test_daily_not_due_if_already_ran_today() -> None:
    # Arrange: ran this morning, now the evening of the same day.
    spec = ScheduleSpec(cadence=Cadence.DAILY)
    last = datetime(2025, 7, 7, 9, 0)
    now = datetime(2025, 7, 7, 22, 0)
    # Act / Assert
    assert due_runs(spec, last_run=last, now=now) is False


def test_daily_due_the_next_trading_day() -> None:
    # Arrange: last ran Monday, now it is Tuesday evening.
    spec = ScheduleSpec(cadence=Cadence.DAILY)
    last = datetime(2025, 7, 7, 22, 0)
    now = datetime(2025, 7, 8, 22, 0)
    # Act / Assert
    assert due_runs(spec, last_run=last, now=now) is True


def test_daily_does_not_run_on_market_holiday() -> None:
    # Arrange: Independence Day 2025 (Fri Jul 4) — market closed.
    spec = ScheduleSpec(cadence=Cadence.DAILY)
    now = datetime(2025, 7, 4, 22, 0)
    # Act / Assert: a closed market means no daily refresh.
    assert due_runs(spec, last_run=None, now=now) is False


def test_daily_does_not_run_on_weekend() -> None:
    # Arrange: Saturday 2025-07-05.
    spec = ScheduleSpec(cadence=Cadence.DAILY)
    now = datetime(2025, 7, 5, 22, 0)
    # Act / Assert
    assert due_runs(spec, last_run=None, now=now) is False


# ---------------------------------------------------------------------------
# The missed-job case — the whole reason for launchd
# ---------------------------------------------------------------------------
def test_daily_missed_job_runs_on_wake() -> None:
    # Arrange: last ran 5 days ago; the Mac was asleep since.
    spec = ScheduleSpec(cadence=Cadence.DAILY)
    last = datetime(2025, 7, 2, 22, 0)  # Wednesday
    now = datetime(2025, 7, 7, 8, 0)  # Monday — a trading day
    # Act / Assert: the queued job runs; it is not discarded.
    assert due_runs(spec, last_run=last, now=now) is True


def test_weekly_missed_job_runs_when_overdue() -> None:
    # Arrange: last weekly run was two ISO weeks ago.
    spec = ScheduleSpec(cadence=Cadence.WEEKLY)
    last = datetime(2025, 6, 29)  # ISO week 26
    now = datetime(2025, 7, 13)  # ISO week 28
    # Act / Assert
    assert due_runs(spec, last_run=last, now=now) is True


# ---------------------------------------------------------------------------
# Weekly cadence (default: Sunday)
# ---------------------------------------------------------------------------
def test_weekly_due_on_configured_weekday() -> None:
    # Arrange: Sunday 2025-07-13 is the configured run day (default Sunday=6).
    spec = ScheduleSpec(cadence=Cadence.WEEKLY)
    now = datetime(2025, 7, 13, 10, 0)
    # Act / Assert
    assert due_runs(spec, last_run=None, now=now) is True


def test_weekly_not_due_before_configured_weekday_in_a_fresh_week() -> None:
    # Arrange: Wednesday 2025-07-09, configured day is Sunday, no run this week.
    spec = ScheduleSpec(cadence=Cadence.WEEKLY)
    now = datetime(2025, 7, 9, 10, 0)
    # Act / Assert
    assert due_runs(spec, last_run=None, now=now) is False


def test_weekly_not_due_if_already_ran_this_week() -> None:
    # Arrange: ran Sunday, now Monday of the same ISO week.
    spec = ScheduleSpec(cadence=Cadence.WEEKLY)
    last = datetime(2025, 7, 13, 10, 0)  # Sunday, ISO week 28
    now = datetime(2025, 7, 14, 10, 0)  # Monday, same ISO week? No — new week.
    # Monday 2025-07-14 starts ISO week 29, so it would be a new week; use same.
    now_same_week = datetime(2025, 7, 13, 23, 0)
    # Act / Assert
    assert due_runs(spec, last_run=last, now=now_same_week) is False
    _ = now


# ---------------------------------------------------------------------------
# Monthly cadence (default: last day of month)
# ---------------------------------------------------------------------------
def test_monthly_due_on_last_day_of_month() -> None:
    # Arrange: 2025-07-31 is the last day of July.
    spec = ScheduleSpec(cadence=Cadence.MONTHLY)
    now = datetime(2025, 7, 31, 18, 0)
    # Act / Assert
    assert due_runs(spec, last_run=None, now=now) is True


def test_monthly_not_due_midmonth() -> None:
    # Arrange: 2025-07-15.
    spec = ScheduleSpec(cadence=Cadence.MONTHLY)
    now = datetime(2025, 7, 15, 18, 0)
    # Act / Assert
    assert due_runs(spec, last_run=None, now=now) is False


def test_monthly_not_due_if_already_ran_this_month() -> None:
    # Arrange: ran on the 31st; "now" is still the 31st (idempotent).
    spec = ScheduleSpec(cadence=Cadence.MONTHLY)
    last = datetime(2025, 7, 31, 18, 0)
    now = datetime(2025, 7, 31, 23, 0)
    # Act / Assert
    assert due_runs(spec, last_run=last, now=now) is False


def test_monthly_missed_job_runs_next_month_end() -> None:
    # Arrange: last ran end of June; now end of August (July was missed).
    spec = ScheduleSpec(cadence=Cadence.MONTHLY)
    last = datetime(2025, 6, 30, 18, 0)
    now = datetime(2025, 8, 31, 18, 0)
    # Act / Assert
    assert due_runs(spec, last_run=last, now=now) is True


# ---------------------------------------------------------------------------
# Event cadence is never calendar-driven
# ---------------------------------------------------------------------------
def test_event_cadence_is_never_calendar_due() -> None:
    # Arrange
    spec = ScheduleSpec(cadence=Cadence.EVENT)
    now = datetime(2025, 7, 7, 10, 0)
    # Act / Assert: event runs are triggered explicitly, not by the calendar.
    assert due_runs(spec, last_run=None, now=now) is False


# ---------------------------------------------------------------------------
# Custom spec configuration
# ---------------------------------------------------------------------------
def test_weekly_spec_accepts_custom_weekday() -> None:
    # Arrange: configure weekly to run on Friday (weekday=4).
    spec = ScheduleSpec(cadence=Cadence.WEEKLY, weekday=4)
    friday = datetime(2025, 7, 11, 10, 0)
    # Act / Assert
    assert due_runs(spec, last_run=None, now=friday) is True


def test_schedule_spec_rejects_bad_weekday() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="weekday"):
        ScheduleSpec(cadence=Cadence.WEEKLY, weekday=7)
