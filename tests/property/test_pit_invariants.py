"""Property-based tests for the point-in-time correctness invariants.

These are the structural anti-leakage guarantees. Hypothesis generates
arbitrary datasets and knowledge dates; the invariant under test is that no
filtered query ever surfaces a record that was not yet publicly knowable.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from alphalab.domain.pit import (
    KnowledgeDate,
    PITRecord,
    visible_records,
)
from alphalab.domain.types import Ticker

# A bounded calendar window keeps generated dates realistic and fast to shrink.
_MIN_DATE = date(1990, 1, 1)
_MAX_DATE = date(2030, 12, 31)

_dates = st.dates(min_value=_MIN_DATE, max_value=_MAX_DATE)
_tickers = st.sampled_from(["AAPL", "MSFT", "GOOG", "SPY", "IBM"]).map(Ticker)
_fields = st.sampled_from(["revenue", "eps", "assets", "net_income"])
_values = st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False)


@st.composite
def _pit_records(draw: st.DrawFn) -> PITRecord:
    """Build a valid ``PITRecord`` with ``observed_at >= as_of``."""
    as_of = draw(_dates)
    lag_days = draw(st.integers(min_value=0, max_value=400))
    observed_at = as_of + timedelta(days=lag_days)
    return PITRecord(
        ticker=draw(_tickers),
        field=draw(_fields),
        value=draw(_values),
        as_of=as_of,
        observed_at=observed_at,
    )


@given(records=st.lists(_pit_records(), max_size=50), knowledge=_dates)
def test_filter_never_leaks_future_knowledge(records: list[PITRecord], knowledge: date) -> None:
    # Arrange
    kd = KnowledgeDate(knowledge)

    # Act
    visible = list(visible_records(records, kd))

    # Assert — the central invariant: nothing observed after T is visible at T.
    assert all(record.observed_at <= knowledge for record in visible)


@given(records=st.lists(_pit_records(), max_size=50), knowledge=_dates)
def test_filter_is_a_subset_preserving_all_visible(
    records: list[PITRecord], knowledge: date
) -> None:
    # Arrange
    kd = KnowledgeDate(knowledge)

    # Act
    visible = list(visible_records(records, kd))

    # Assert — exactly the knowable records, no more and no fewer.
    expected = [r for r in records if r.observed_at <= knowledge]
    assert visible == expected


@given(record=_pit_records())
def test_observed_at_is_never_before_as_of(record: PITRecord) -> None:
    # Assert — construction guarantees the domain invariant.
    assert record.observed_at >= record.as_of


@given(as_of=_dates, lag=st.integers(min_value=1, max_value=400))
def test_observed_before_as_of_is_rejected(as_of: date, lag: int) -> None:
    # Arrange — observed strictly before the period it refers to.
    observed_at = as_of - timedelta(days=lag)

    # Act / Assert
    with pytest.raises(ValueError):
        PITRecord(
            ticker=Ticker("AAPL"),
            field="revenue",
            value=1.0,
            as_of=as_of,
            observed_at=observed_at,
        )
