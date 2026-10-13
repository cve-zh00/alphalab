"""Unit tests for the pure PIT domain primitives."""

from __future__ import annotations

from datetime import date

import pytest

from alphalab.domain.pit import (
    KnowledgeDate,
    LeakageError,
    PITRecord,
    UnregisteredFeatureError,
    visible_records,
)
from alphalab.domain.types import Ticker


class TestPITRecord:
    def test_is_frozen(self) -> None:
        # Arrange
        record = PITRecord(
            ticker=Ticker("AAPL"),
            field="revenue",
            value=100.0,
            as_of=date(2008, 6, 30),
            observed_at=date(2008, 8, 15),
        )

        # Act / Assert — mutation is rejected by the frozen model.
        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
            record.value = 200.0  # type: ignore[misc]

    def test_accepts_observed_equal_to_as_of(self) -> None:
        # Arrange / Act
        record = PITRecord(
            ticker=Ticker("AAPL"),
            field="revenue",
            value=1.0,
            as_of=date(2008, 6, 30),
            observed_at=date(2008, 6, 30),
        )

        # Assert
        assert record.observed_at == record.as_of

    def test_rejects_observed_before_as_of(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="observed_at"):
            PITRecord(
                ticker=Ticker("AAPL"),
                field="revenue",
                value=1.0,
                as_of=date(2008, 6, 30),
                observed_at=date(2008, 6, 29),
            )

    def test_optional_source(self) -> None:
        # Arrange / Act
        record = PITRecord(
            ticker=Ticker("AAPL"),
            field="revenue",
            value=1.0,
            as_of=date(2008, 6, 30),
            observed_at=date(2008, 8, 1),
            source="SEC-EDGAR",
        )

        # Assert
        assert record.source == "SEC-EDGAR"


class TestKnowledgeDate:
    def test_wraps_date(self) -> None:
        # Arrange / Act
        kd = KnowledgeDate(date(2008, 10, 1))

        # Assert
        assert kd.value == date(2008, 10, 1)

    def test_is_frozen(self) -> None:
        # Arrange
        kd = KnowledgeDate(date(2008, 10, 1))

        # Act / Assert
        with pytest.raises(Exception):  # noqa: B017
            kd.value = date(2009, 1, 1)  # type: ignore[misc]

    def test_ordering(self) -> None:
        # Arrange
        earlier = KnowledgeDate(date(2008, 1, 1))
        later = KnowledgeDate(date(2008, 12, 31))

        # Assert
        assert earlier < later


class TestVisibleRecords:
    def test_filters_out_future_observations(self) -> None:
        # Arrange
        early = PITRecord(
            ticker=Ticker("AAPL"),
            field="eps",
            value=1.0,
            as_of=date(2008, 3, 31),
            observed_at=date(2008, 5, 1),
        )
        late = PITRecord(
            ticker=Ticker("AAPL"),
            field="eps",
            value=2.0,
            as_of=date(2008, 6, 30),
            observed_at=date(2008, 11, 1),
        )

        # Act
        visible = list(visible_records([early, late], KnowledgeDate(date(2008, 10, 1))))

        # Assert
        assert visible == [early]

    def test_includes_observation_on_the_boundary(self) -> None:
        # Arrange
        boundary = PITRecord(
            ticker=Ticker("AAPL"),
            field="eps",
            value=1.0,
            as_of=date(2008, 3, 31),
            observed_at=date(2008, 10, 1),
        )

        # Act
        visible = list(visible_records([boundary], KnowledgeDate(date(2008, 10, 1))))

        # Assert
        assert visible == [boundary]


class TestDomainErrors:
    def test_leakage_error_is_an_exception(self) -> None:
        assert issubclass(LeakageError, Exception)

    def test_unregistered_feature_error_is_an_exception(self) -> None:
        assert issubclass(UnregisteredFeatureError, Exception)
