"""Property-based tests for the trial ledger (Hypothesis)."""

from __future__ import annotations

import random
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from alphalab.adapters.ledger_sqlite import SqliteTrialLedger
from alphalab.domain.ledger import TrialMetrics, TrialRecord, canonicalize, spec_hash

# JSON-safe scalar values the canonical hasher accepts.
_scalars = st.one_of(
    st.booleans(),
    st.integers(min_value=-(10**9), max_value=10**9),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=20),
    st.none(),
)

# Flat parameter dictionaries with string keys.
_param_dicts = st.dictionaries(
    keys=st.text(min_size=1, max_size=10),
    values=_scalars,
    max_size=8,
)


def _metrics() -> TrialMetrics:
    return TrialMetrics(
        sharpe_gross=1.0,
        sharpe_net=0.8,
        return_gross=0.1,
        return_net=0.07,
        max_drawdown=-0.05,
    )


def _record(params: dict[str, Any], family: str = "fam") -> TrialRecord:
    return TrialRecord.create(
        spec=params,
        dataset_hash="ds",
        family=family,
        window_start=date(2020, 1, 1),
        window_end=date(2021, 1, 1),
        params=params,
        metrics=_metrics(),
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[SqliteTrialLedger]:
    with SqliteTrialLedger(tmp_path / "prop.db") as led:
        yield led


class TestCanonicalizationProperties:
    @given(params=_param_dicts)
    @settings(max_examples=300)
    def test_any_key_permutation_is_stable(self, params: dict[str, Any]) -> None:
        # Arrange: build a shuffled copy with identical content.
        items = list(params.items())
        random.Random(len(items)).shuffle(items)
        shuffled = dict(items)

        # Act / Assert
        assert spec_hash(params) == spec_hash(shuffled)

    @given(a=_param_dicts, b=_param_dicts)
    @settings(max_examples=300)
    def test_hash_agrees_with_canonical_form(self, a: dict[str, Any], b: dict[str, Any]) -> None:
        # Property: the hash is a faithful function of the canonical form.
        # Equivalent canonical forms (e.g. permuted keys) hash equally;
        # distinct canonical forms hash differently (no reasonable collision).
        # We compare on the canonical form rather than Python ``==`` because
        # the hasher deliberately distinguishes ``1``/``1.0``/``True`` even
        # though they compare equal in Python.
        if canonicalize(a) == canonicalize(b):
            assert spec_hash(a) == spec_hash(b)
        else:
            assert spec_hash(a) != spec_hash(b)


class TestCountProperties:
    @given(n=st.integers(min_value=0, max_value=60))
    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_count_equals_number_of_appends(self, ledger: SqliteTrialLedger, n: int) -> None:
        # Arrange: start from whatever is there (fixture is fresh per example
        # is not guaranteed across examples, so measure the delta).
        base = ledger.trial_count("countfam")

        # Act
        for i in range(n):
            ledger.append(_record({"i": i}, family="countfam"))

        # Assert
        assert ledger.trial_count("countfam") == base + n


class TestImmutabilityProperty:
    @given(params=_param_dicts)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_public_api_never_mutates_existing_record(
        self, ledger: SqliteTrialLedger, params: dict[str, Any]
    ) -> None:
        # Arrange
        record = _record(params, family="immfam")
        ledger.append(record)
        before = ledger.get(record.trial_id)

        # Act: append another record (the only mutating op available).
        ledger.append(_record({"other": 1}, family="immfam"))
        after = ledger.get(record.trial_id)

        # Assert: the original record is byte-for-byte unchanged.
        assert before is not None
        assert after is not None
        assert before == after
