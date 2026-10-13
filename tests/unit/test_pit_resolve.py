"""Unit tests for the structural leakage gate :func:`resolve_feature`.

These tests pin the contract that makes look-ahead *inexpressible*: the only
sanctioned path from a stored :class:`PITRecord` to a usable feature value
raises :class:`LeakageError` whenever the datum was not knowable at the active
knowledge date, and :class:`UnregisteredFeatureError` whenever the feature has
no audited availability lag.
"""

from __future__ import annotations

from datetime import date

import pytest

from alphalab.domain.pit import (
    FeatureRegistry,
    FeatureSpec,
    KnowledgeDate,
    LeakageError,
    PITRecord,
    UnregisteredFeatureError,
    resolve_feature,
)
from alphalab.domain.types import Ticker

_AS_OF = date(2008, 6, 30)
_LAG_DAYS = 45
_EARLIEST = date(2008, 8, 14)  # _AS_OF + 45 days


def _registry() -> FeatureRegistry:
    return FeatureRegistry([FeatureSpec(name="eps", availability_lag_days=_LAG_DAYS)])


def _record(*, observed_at: date, value: float = 1.0, field: str = "eps") -> PITRecord:
    return PITRecord(
        ticker=Ticker("ACME"),
        field=field,
        value=value,
        as_of=_AS_OF,
        observed_at=observed_at,
        source="SEC-EDGAR",
    )


class TestResolveFeatureHappyPath:
    def test_returns_value_when_knowable(self) -> None:
        # Arrange — zero-lag feature observed on its as_of, queried later.
        registry = FeatureRegistry([FeatureSpec(name="eps", availability_lag_days=0)])
        record = _record(observed_at=_AS_OF, value=1.23)
        knowledge = KnowledgeDate(date(2008, 10, 1))

        # Act
        value = resolve_feature(record, registry=registry, knowledge_date=knowledge)

        # Assert
        assert value == 1.23

    def test_returns_value_on_the_knowledge_boundary(self) -> None:
        # Arrange — observed exactly on the knowledge date is still knowable.
        record = _record(observed_at=_EARLIEST, value=2.0)
        knowledge = KnowledgeDate(_EARLIEST)

        # Act
        value = resolve_feature(record, registry=_registry(), knowledge_date=knowledge)

        # Assert — observed_at == knowledge_date is on the legal side.
        assert value == 2.0


class TestResolveFeatureRejectsLeakage:
    def test_future_observation_raises_leakage(self) -> None:
        # Arrange — the datum became knowable AFTER the knowledge date.
        record = _record(observed_at=date(2008, 11, 1))
        knowledge = KnowledgeDate(date(2008, 10, 1))

        # Act / Assert — surfacing it would be look-ahead.
        with pytest.raises(LeakageError, match="look-ahead"):
            resolve_feature(record, registry=_registry(), knowledge_date=knowledge)

    def test_observation_before_registered_lag_raises(self) -> None:
        # Arrange — a record claiming to be knowable before its availability
        # lag allows is malformed and must be rejected structurally.
        too_early = date(2008, 7, 1)  # only 1 day after as_of; lag is 45 days
        record = _record(observed_at=too_early)
        knowledge = KnowledgeDate(date(2009, 1, 1))

        # Act / Assert
        with pytest.raises(LeakageError, match="availability lag"):
            resolve_feature(record, registry=_registry(), knowledge_date=knowledge)


class TestResolveFeatureRequiresRegistration:
    def test_unregistered_feature_raises(self) -> None:
        # Arrange — a feature with no audited lag cannot be resolved at all.
        record = _record(observed_at=_EARLIEST, field="unaudited")
        knowledge = KnowledgeDate(date(2009, 1, 1))

        # Act / Assert
        with pytest.raises(UnregisteredFeatureError, match="unaudited"):
            resolve_feature(record, registry=_registry(), knowledge_date=knowledge)
