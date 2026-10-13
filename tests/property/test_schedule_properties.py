"""Property-based tests for the pure scheduling calendar.

These assert the structural invariants that the example-based tests cannot
exhaustively cover: for *any* ``now`` and ``last_run``, the decision is
deterministic and total (never raises), and the idempotency key is stable
within a period so no cadence is ever decided to run twice for the same key.
"""

from __future__ import annotations

from datetime import datetime

from hypothesis import given
from hypothesis import strategies as st

from alphalab.domain.schedule import (
    Cadence,
    ScheduleSpec,
    due_runs,
    is_market_holiday,
    is_trading_day,
    run_key,
)

_cadences = st.sampled_from(list(Cadence))
_weekdays = st.integers(min_value=0, max_value=6)
_datetimes = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2050, 12, 31, 23, 59, 59),
)
_specs = st.builds(ScheduleSpec, cadence=_cadences, weekday=_weekdays)


@given(spec=_specs, now=_datetimes, last=st.none() | _datetimes)
def test_due_runs_is_total_and_boolean(
    spec: ScheduleSpec, now: datetime, last: datetime | None
) -> None:
    # Act
    result = due_runs(spec, last_run=last, now=now)
    # Assert: always a bool, never raises.
    assert isinstance(result, bool)


@given(spec=_specs, now=_datetimes, last=st.none() | _datetimes)
def test_due_runs_is_deterministic(
    spec: ScheduleSpec, now: datetime, last: datetime | None
) -> None:
    # Act
    first = due_runs(spec, last_run=last, now=now)
    second = due_runs(spec, last_run=last, now=now)
    # Assert
    assert first == second


@given(spec=_specs, now=_datetimes)
def test_once_run_in_period_never_due_again_in_same_period(
    spec: ScheduleSpec, now: datetime
) -> None:
    # Arrange: treat ``now`` itself as the last run — same period by construction.
    # Act / Assert: a cadence already run this period is never due again.
    assert due_runs(spec, last_run=now, now=now) is False


@given(spec=_specs, now=_datetimes)
def test_run_key_is_stable_for_a_single_moment(spec: ScheduleSpec, now: datetime) -> None:
    # Act / Assert: pure function of its inputs.
    assert run_key(spec, now) == run_key(spec, now)


@given(spec=_specs, now=_datetimes)
def test_run_key_is_prefixed_by_cadence(spec: ScheduleSpec, now: datetime) -> None:
    # Act / Assert
    assert run_key(spec, now).startswith(f"{spec.cadence.value}:")


@given(day=st.dates(min_value=datetime(2000, 1, 1).date(), max_value=datetime(2050, 12, 31).date()))
def test_trading_day_excludes_weekends_and_holidays(day: object) -> None:
    from datetime import date

    assert isinstance(day, date)
    # Act
    trading = is_trading_day(day)
    # Assert: a trading day is never a weekend and never a holiday.
    if trading:
        assert day.weekday() < 5
        assert not is_market_holiday(day)
