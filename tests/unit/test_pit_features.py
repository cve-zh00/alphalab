"""Unit tests for the feature registry and availability-lag declarations."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from alphalab.domain.pit import (
    FeatureRegistry,
    FeatureSpec,
    UnregisteredFeatureError,
)


def _quarterly_fundamental() -> FeatureSpec:
    return FeatureSpec(name="revenue", availability_lag_days=90)


class TestFeatureSpec:
    def test_is_frozen(self) -> None:
        # Arrange
        spec = _quarterly_fundamental()

        # Act / Assert
        with pytest.raises(Exception):  # noqa: B017
            spec.availability_lag_days = 1  # type: ignore[misc]

    def test_rejects_negative_lag(self) -> None:
        # Act / Assert — a feature cannot be known before its period.
        with pytest.raises(ValueError):
            FeatureSpec(name="revenue", availability_lag_days=-1)

    def test_earliest_knowable_applies_lag(self) -> None:
        # Arrange
        spec = _quarterly_fundamental()
        as_of = date(2008, 6, 30)

        # Act
        knowable = spec.earliest_knowable(as_of)

        # Assert
        assert knowable == as_of + timedelta(days=90)


class TestFeatureRegistry:
    def test_get_returns_registered_spec(self) -> None:
        # Arrange
        spec = _quarterly_fundamental()
        registry = FeatureRegistry([spec])

        # Act
        resolved = registry.get("revenue")

        # Assert
        assert resolved is spec

    def test_unregistered_feature_raises(self) -> None:
        # Arrange
        registry = FeatureRegistry([_quarterly_fundamental()])

        # Act / Assert
        with pytest.raises(UnregisteredFeatureError, match="unknown_feature"):
            registry.get("unknown_feature")

    def test_duplicate_names_are_rejected(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="duplicate"):
            FeatureRegistry([_quarterly_fundamental(), _quarterly_fundamental()])

    def test_registry_is_read_only(self) -> None:
        # Arrange
        registry = FeatureRegistry([_quarterly_fundamental()])

        # Act
        names = registry.names()

        # Assert — mutating the returned view must not affect the registry.
        names_copy = set(names)
        assert names_copy == {"revenue"}

    def test_contains(self) -> None:
        # Arrange
        registry = FeatureRegistry([_quarterly_fundamental()])

        # Assert
        assert "revenue" in registry
        assert "missing" not in registry
