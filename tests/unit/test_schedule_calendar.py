"""Unit tests for the pure scheduling calendar (``alphalab.domain.schedule``).

Time is always injected: no test here reads the wall clock. Every decision is
pinned to a concrete, known date so the assertions are reproducible. This is
the contract that makes the scheduler testable at all — the domain never calls
``datetime.now()``.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from alphalab.domain.schedule import (
    Cadence,
    ScheduleSpec,
    is_market_holiday,
    is_trading_day,
    market_holidays,
    run_key,
)


# ---------------------------------------------------------------------------
# Cadence enum
# ---------------------------------------------------------------------------
def test_cadence_has_four_members() -> None:
    # Arrange / Act
    members = {c.value for c in Cadence}
    # Assert
    assert members == {"daily", "weekly", "monthly", "event"}


# ---------------------------------------------------------------------------
# Market holiday calendar
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2025, 1, 1), True),  # New Year's Day
        (date(2025, 1, 20), True),  # MLK Day (3rd Mon Jan)
        (date(2025, 2, 17), True),  # Presidents Day (3rd Mon Feb)
        (date(2025, 4, 18), True),  # Good Friday
        (date(2025, 5, 26), True),  # Memorial Day (last Mon May)
        (date(2025, 6, 19), True),  # Juneteenth
        (date(2025, 7, 4), True),  # Independence Day
        (date(2025, 9, 1), True),  # Labor Day (1st Mon Sep)
        (date(2025, 11, 27), True),  # Thanksgiving (4th Thu Nov)
        (date(2025, 12, 25), True),  # Christmas
        (date(2025, 7, 7), False),  # ordinary Monday
    ],
    ids=[
        "new_year",
        "mlk",
        "presidents",
        "good_friday",
        "memorial",
        "juneteenth",
        "independence",
        "labor",
        "thanksgiving",
        "christmas",
        "ordinary_monday",
    ],
)
def test_is_market_holiday_known_dates(day: date, expected: bool) -> None:
    # Arrange / Act / Assert
    assert is_market_holiday(day) is expected


def test_weekend_observance_saturday_rolls_back_to_friday() -> None:
    # Arrange: Independence Day 2026 (Jul 4) falls on a Saturday.
    saturday = date(2026, 7, 4)
    observed_friday = date(2026, 7, 3)
    # Act / Assert: the observed holiday is the preceding Friday.
    assert is_market_holiday(observed_friday) is True
    assert is_market_holiday(saturday) is False


def test_weekend_observance_sunday_rolls_forward_to_monday() -> None:
    # Arrange: New Year's Day 2023 (Jan 1) falls on a Sunday.
    sunday = date(2023, 1, 1)
    observed_monday = date(2023, 1, 2)
    # Act / Assert
    assert is_market_holiday(observed_monday) is True
    assert is_market_holiday(sunday) is False


def test_market_holidays_returns_sorted_unique_set_for_year() -> None:
    # Arrange / Act
    holidays_2025 = market_holidays(2025)
    # Assert: 11 federal/market holidays plus Good Friday = at least 10 entries.
    assert date(2025, 7, 4) in holidays_2025
    assert date(2025, 4, 18) in holidays_2025  # Good Friday
    assert len(holidays_2025) == len(set(holidays_2025))


# ---------------------------------------------------------------------------
# Trading-day predicate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2025, 7, 7), True),  # Monday, ordinary
        (date(2025, 7, 5), False),  # Saturday
        (date(2025, 7, 6), False),  # Sunday
        (date(2025, 7, 4), False),  # Independence Day (holiday)
    ],
    ids=["weekday", "saturday", "sunday", "holiday"],
)
def test_is_trading_day(day: date, expected: bool) -> None:
    # Arrange / Act / Assert
    assert is_trading_day(day) is expected


# ---------------------------------------------------------------------------
# Run keys (idempotency primitive)
# ---------------------------------------------------------------------------
def test_run_key_is_stable_within_the_same_period() -> None:
    # Arrange
    spec = ScheduleSpec(cadence=Cadence.DAILY)
    morning = datetime(2025, 7, 7, 9, 30)
    evening = datetime(2025, 7, 7, 23, 59)
    # Act / Assert: two moments on the same day share a key.
    assert run_key(spec, morning) == run_key(spec, evening)


def test_run_key_differs_across_days_for_daily() -> None:
    # Arrange
    spec = ScheduleSpec(cadence=Cadence.DAILY)
    # Act / Assert
    assert run_key(spec, datetime(2025, 7, 7, 9, 0)) != run_key(spec, datetime(2025, 7, 8, 9, 0))


def test_run_key_weekly_is_stable_within_iso_week() -> None:
    # Arrange: Mon 2025-07-07 and Sun 2025-07-13 are the same ISO week.
    spec = ScheduleSpec(cadence=Cadence.WEEKLY)
    # Act / Assert
    assert run_key(spec, datetime(2025, 7, 7)) == run_key(spec, datetime(2025, 7, 13))
    assert run_key(spec, datetime(2025, 7, 7)) != run_key(spec, datetime(2025, 7, 14))


def test_run_key_monthly_is_stable_within_month() -> None:
    # Arrange
    spec = ScheduleSpec(cadence=Cadence.MONTHLY)
    # Act / Assert
    assert run_key(spec, datetime(2025, 7, 1)) == run_key(spec, datetime(2025, 7, 31))
    assert run_key(spec, datetime(2025, 7, 31)) != run_key(spec, datetime(2025, 8, 1))


def test_run_key_encodes_cadence_so_distinct_cadences_never_collide() -> None:
    # Arrange
    when = datetime(2025, 7, 7, 9, 0)
    # Act / Assert
    keys = {run_key(ScheduleSpec(cadence=c), when) for c in Cadence}
    assert len(keys) == len(list(Cadence))
