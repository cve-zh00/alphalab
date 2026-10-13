"""Property-based tests for the A/B experiment.

Invariants under test (hypothesis-generated inputs, always seeded):

* The runner never raises on a well-formed decision set, whatever the verdicts
  or forward returns.
* The result is deterministic given a fixed seed.
* The emitted structures are immutable (frozen).
* The three arms always share the same out-of-sample observation count, and
  ``n_executed <= n_proposed`` for every arm.
* The paired-difference statistic for a series against itself is exactly zero.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alphalab.app.ab_experiment import ABExperimentRunner
from alphalab.artifact.models import Verdict
from alphalab.domain.experiment.difference import paired_sharpe_difference
from tests.unit._ab_runner_doubles import (
    make_analyst_pipeline,
    make_debate_pipeline,
    make_decision,
)


class _FakeLedger:
    def __init__(self) -> None:
        self.records: list[object] = []

    def append(self, record: object) -> str:
        self.records.append(record)
        return "id"


_VERDICTS = st.sampled_from([Verdict.BUY, Verdict.SELL, Verdict.HOLD])


def _build_decisions(forwards: list[float], *, accept: bool) -> list:
    base = date(2020, 1, 1)
    return [
        make_decision(
            knowledge_date=base + timedelta(days=i),
            forward_return=f,
            accept=accept,
            predicted_prob=0.5,
            seed=i,
        )
        for i, f in enumerate(forwards)
    ]


def _runner(verdict: Verdict) -> ABExperimentRunner:
    return ABExperimentRunner(
        analyst=make_analyst_pipeline(verdict=verdict),
        debate=make_debate_pipeline(verdict=verdict),
        ledger=_FakeLedger(),
        skeptic_rng=np.random.default_rng(0),
    )


@settings(max_examples=25, deadline=None)
@given(
    forwards=st.lists(
        st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
        min_size=6,
        max_size=20,
    ),
    verdict=_VERDICTS,
    accept=st.booleans(),
)
def test_runner_never_raises_and_arms_are_consistent(
    forwards: list[float], verdict: Verdict, accept: bool
) -> None:
    # Arrange
    decisions = _build_decisions(forwards, accept=accept)
    oos_start = len(decisions) // 2
    runner = _runner(verdict)

    # Act
    comp = runner.run(
        decisions,
        oos_start=oos_start,
        rng=np.random.default_rng(42),
        recorded_at=datetime(2024, 1, 1),
        n_resamples=400,
    )

    # Assert — arm-length and proposed/executed invariants.
    n = comp.n_observations
    assert n == len(decisions) - oos_start
    for arm in (comp.arm_a, comp.arm_b, comp.arm_c):
        assert len(arm.executed_returns) == n
        assert arm.n_executed <= arm.n_proposed
    assert comp.arm_c.n_executed <= comp.arm_b.n_executed


@settings(max_examples=15, deadline=None)
@given(
    forwards=st.lists(
        st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
        min_size=6,
        max_size=16,
    ),
)
def test_runner_is_deterministic_with_fixed_seed(forwards: list[float]) -> None:
    # Arrange
    decisions = _build_decisions(forwards, accept=False)
    oos_start = len(decisions) // 2

    # Act — two independent runs with the same seeds.
    c1 = _runner(Verdict.BUY).run(
        decisions,
        oos_start=oos_start,
        rng=np.random.default_rng(7),
        recorded_at=datetime(2024, 1, 1),
        n_resamples=400,
    )
    c2 = _runner(Verdict.BUY).run(
        decisions,
        oos_start=oos_start,
        rng=np.random.default_rng(7),
        recorded_at=datetime(2024, 1, 1),
        n_resamples=400,
    )

    # Assert — identical verdicts and CIs.
    assert c1.debate_vs_single.ci_low == c2.debate_vs_single.ci_low
    assert c1.debate_vs_single.ci_high == c2.debate_vs_single.ci_high
    assert c1.debate_verdict == c2.debate_verdict
    assert c1.skeptic_sharpe_verdict == c2.skeptic_sharpe_verdict


@settings(max_examples=30, deadline=None)
@given(
    series=st.lists(
        st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=40,
    ),
)
def test_self_difference_is_exactly_zero(series: list[float]) -> None:
    # Arrange / Act — a series against itself: paired resampling cancels.
    tup = tuple(series)
    diff = paired_sharpe_difference(
        tup, tup, name_a="X", name_b="Y", rng=np.random.default_rng(1), n_resamples=300
    )

    # Assert
    assert diff.delta == pytest.approx(0.0, abs=1e-12)
    assert diff.ci_low == pytest.approx(0.0, abs=1e-12)
    assert diff.ci_high == pytest.approx(0.0, abs=1e-12)
    assert diff.winner is None


@settings(max_examples=20, deadline=None)
@given(
    forwards=st.lists(
        st.floats(min_value=-0.03, max_value=0.03, allow_nan=False, allow_infinity=False),
        min_size=6,
        max_size=14,
    ),
)
def test_comparison_result_is_immutable(forwards: list[float]) -> None:
    # Arrange
    decisions = _build_decisions(forwards, accept=False)
    comp = _runner(Verdict.BUY).run(
        decisions,
        oos_start=len(decisions) // 2,
        rng=np.random.default_rng(3),
        recorded_at=datetime(2024, 1, 1),
        n_resamples=300,
    )

    # Act / Assert
    with pytest.raises(Exception, match=r"frozen|Instance is frozen"):
        comp.conclusive = True  # type: ignore[misc]
