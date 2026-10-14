"""Tests for the forward-versus-backfilled integrity check."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from alphalab.domain.forecast import is_forward, partition_by_integrity
from alphalab.domain.journal import Direction, Prediction


def _prediction(*, knowledge_date: date, recorded_at: datetime, pid: str = "p1") -> Prediction:
    return Prediction(
        prediction_id=pid,
        recorded_at=recorded_at,
        ticker="AAPL",
        direction=Direction.UP,
        probability=0.6,
        horizon=knowledge_date + timedelta(days=90),
        entry_price=200.0,
        knowledge_date=knowledge_date,
    )


def test_same_day_claim_is_forward() -> None:
    """Research run and committed the same day is a genuine forecast."""
    # Arrange
    day = date(2026, 10, 13)
    prediction = _prediction(
        knowledge_date=day, recorded_at=datetime(2026, 10, 13, 22, 0, tzinfo=UTC)
    )

    # Act / Assert
    assert is_forward(prediction)


def test_slightly_stale_data_is_still_forward() -> None:
    """A few days of data lag is normal research practice, not backfilling."""
    # Arrange — filings and prices are three days old.
    prediction = _prediction(
        knowledge_date=date(2026, 10, 10),
        recorded_at=datetime(2026, 10, 13, 9, 0, tzinfo=UTC),
    )

    # Act / Assert
    assert is_forward(prediction)


def test_long_gap_is_backfilled() -> None:
    """A claim about a long-past date is hindsight, whatever it is labelled.

    This is the case that would silently corrupt a calibration report.
    """
    # Arrange — "predicting" 2015 from 2026.
    prediction = _prediction(
        knowledge_date=date(2015, 6, 30),
        recorded_at=datetime(2026, 10, 13, 9, 0, tzinfo=UTC),
    )

    # Act / Assert
    assert not is_forward(prediction)


def test_claim_recorded_before_its_knowledge_date_is_rejected() -> None:
    """An impossible ordering disqualifies the record.

    Honest research cannot be committed before the data it used exists; such a
    row means a clock problem or a hand-edited entry.
    """
    # Arrange
    prediction = _prediction(
        knowledge_date=date(2026, 10, 20),
        recorded_at=datetime(2026, 10, 13, 9, 0, tzinfo=UTC),
    )

    # Act / Assert
    assert not is_forward(prediction)


def test_boundary_at_tolerance_is_accepted() -> None:
    """Exactly at the tolerance still counts; the threshold is inclusive."""
    # Arrange — a seven-day gap, equal to FORWARD_TOLERANCE.
    prediction = _prediction(
        knowledge_date=date(2026, 10, 6),
        recorded_at=datetime(2026, 10, 13, 0, 0, tzinfo=UTC),
    )

    # Act / Assert
    assert is_forward(prediction)


def test_partition_separates_and_preserves_both_halves() -> None:
    """Backfilled entries are set aside and counted, never dropped.

    The excluded count belongs in the report; hiding it would overstate how
    much evidence the calibration rests on.
    """
    # Arrange
    good = _prediction(
        knowledge_date=date(2026, 10, 12),
        recorded_at=datetime(2026, 10, 13, 9, 0, tzinfo=UTC),
        pid="good",
    )
    stale = _prediction(
        knowledge_date=date(2019, 1, 2),
        recorded_at=datetime(2026, 10, 13, 9, 0, tzinfo=UTC),
        pid="stale",
    )

    # Act
    forward, backfilled = partition_by_integrity([good, stale])

    # Assert
    assert [p.prediction_id for p in forward] == ["good"]
    assert [p.prediction_id for p in backfilled] == ["stale"]


def test_custom_tolerance_is_honoured() -> None:
    """A stricter tolerance reclassifies borderline entries."""
    # Arrange
    prediction = _prediction(
        knowledge_date=date(2026, 10, 10),
        recorded_at=datetime(2026, 10, 13, 9, 0, tzinfo=UTC),
    )

    # Act / Assert
    assert not is_forward(prediction, tolerance=timedelta(days=1))
