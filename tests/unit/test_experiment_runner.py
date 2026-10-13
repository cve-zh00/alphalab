"""Unit tests for the ABExperimentRunner orchestration.

These exercise the full three-arm wiring with deterministic doubles (no LLM):
same inputs to every arm, costs applied uniformly, the skeptic veto diverging
arm C from arm B, trials recorded in a ledger double, and an out-of-sample-only
verdict.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pytest

from alphalab.app.ab_experiment import ABExperimentRunner
from alphalab.artifact.models import Verdict
from alphalab.domain.experiment import Comparison
from alphalab.domain.ledger import TrialRecord
from tests.unit._ab_runner_doubles import (
    make_analyst_pipeline,
    make_debate_pipeline,
    make_decision,
)


class FakeLedger:
    """In-memory ledger double recording appended trials by family."""

    def __init__(self) -> None:
        self.records: list[TrialRecord] = []

    def append(self, record: TrialRecord) -> str:
        self.records.append(record)
        return record.trial_id

    def trial_count(self, family: str | None = None) -> int:
        if family is None:
            return len(self.records)
        return sum(1 for r in self.records if r.family == family)


def _decisions(n: int, *, accept: bool, forward: float) -> list:
    base = date(2020, 1, 1)
    rng = np.random.default_rng(abs(hash(("fwd", n, accept))) % (2**32))
    jitter = rng.normal(0.0, 0.003, size=n)
    return [
        make_decision(
            knowledge_date=base + timedelta(days=i),
            forward_return=forward + float(jitter[i]),
            accept=accept,
            predicted_prob=0.6,
            seed=i,
        )
        for i in range(n)
    ]


def _runner(ledger: FakeLedger, *, verdict: Verdict = Verdict.BUY) -> ABExperimentRunner:
    return ABExperimentRunner(
        analyst=make_analyst_pipeline(verdict=verdict),
        debate=make_debate_pipeline(verdict=verdict),
        ledger=ledger,
        skeptic_rng=np.random.default_rng(0),
    )


def test_run_returns_an_immutable_comparison() -> None:
    # Arrange — 8 decisions, OOS from index 4; vetoed candidates for arm C.
    ledger = FakeLedger()
    runner = _runner(ledger)
    decisions = _decisions(8, accept=False, forward=0.01)

    # Act
    comp = runner.run(
        decisions,
        oos_start=4,
        rng=np.random.default_rng(1),
        recorded_at=datetime(2024, 1, 1),
        n_resamples=1_000,
    )

    # Assert — a frozen Comparison over the 4 OOS decisions.
    assert isinstance(comp, Comparison)
    assert comp.n_observations == 4
    with pytest.raises(Exception, match=r"frozen|Instance is frozen"):
        comp.n_observations = 1  # type: ignore[misc]


def test_skeptic_vetoes_make_arm_c_execute_fewer_than_arm_b() -> None:
    # Arrange — every candidate is vetoed, so arm C executes nothing while A/B
    # (BUY on every decision) execute everything.
    ledger = FakeLedger()
    runner = _runner(ledger)
    decisions = _decisions(10, accept=False, forward=0.01)

    # Act
    comp = runner.run(
        decisions,
        oos_start=5,
        rng=np.random.default_rng(2),
        recorded_at=datetime(2024, 1, 1),
        n_resamples=1_000,
    )

    # Assert — arm C proposed but executed zero; A and B executed all 5 OOS.
    assert comp.arm_b.n_executed == 5
    assert comp.arm_c.n_proposed == 5
    assert comp.arm_c.n_executed == 0
    assert comp.skeptic_analysis.n_vetoed == 5


def test_accepting_candidates_let_arm_c_match_arm_b() -> None:
    # Arrange — the skeptic accepts every candidate, so C == B exactly.
    ledger = FakeLedger()
    runner = _runner(ledger)
    decisions = _decisions(8, accept=True, forward=0.01)

    # Act
    comp = runner.run(
        decisions,
        oos_start=4,
        rng=np.random.default_rng(3),
        recorded_at=datetime(2024, 1, 1),
        n_resamples=1_000,
    )

    # Assert — identical executed series: skeptic made no difference.
    assert comp.arm_c.n_executed == comp.arm_b.n_executed
    assert comp.arm_c.executed_returns == comp.arm_b.executed_returns
    assert comp.skeptic_analysis.n_vetoed == 0


def test_costs_are_applied_so_net_sharpe_not_above_gross() -> None:
    # Arrange — positive forward return with a real cost drag.
    ledger = FakeLedger()
    runner = _runner(ledger)
    decisions = _decisions(10, accept=True, forward=0.02)

    # Act
    comp = runner.run(
        decisions,
        oos_start=0,
        rng=np.random.default_rng(4),
        recorded_at=datetime(2024, 1, 1),
        n_resamples=1_000,
    )

    # Assert — with a constant positive series the gross and net Sharpe are
    # both degenerate-high, but net must never exceed gross after costs.
    assert comp.arm_a.sharpe_net <= comp.arm_a.sharpe_gross + 1e-9


def test_each_arm_and_each_comparison_is_recorded_as_a_trial() -> None:
    # Arrange
    ledger = FakeLedger()
    runner = _runner(ledger)
    decisions = _decisions(8, accept=False, forward=0.01)

    # Act
    runner.run(
        decisions,
        oos_start=4,
        rng=np.random.default_rng(5),
        recorded_at=datetime(2024, 1, 1),
        n_resamples=1_000,
    )

    # Assert — 4 OOS trials per arm (3 arms) + 3 comparison trials.
    assert ledger.trial_count("ab_arm_a") == 4
    assert ledger.trial_count("ab_arm_b") == 4
    assert ledger.trial_count("ab_arm_c") == 4
    assert ledger.trial_count("ab_comparisons") == 3


def test_hold_verdict_proposes_nothing() -> None:
    # Arrange — every artifact resolves to HOLD: no position, nothing executes.
    ledger = FakeLedger()
    runner = _runner(ledger, verdict=Verdict.HOLD)
    decisions = _decisions(8, accept=True, forward=0.01)

    # Act
    comp = runner.run(
        decisions,
        oos_start=4,
        rng=np.random.default_rng(6),
        recorded_at=datetime(2024, 1, 1),
        n_resamples=1_000,
    )

    # Assert
    assert comp.arm_a.n_proposed == 0
    assert comp.arm_a.n_executed == 0


def test_insufficient_oos_window_raises() -> None:
    # Arrange — only one decision left out of sample.
    ledger = FakeLedger()
    runner = _runner(ledger)
    decisions = _decisions(4, accept=False, forward=0.01)

    # Act / Assert
    with pytest.raises(ValueError, match="out-of-sample"):
        runner.run(
            decisions,
            oos_start=3,
            rng=np.random.default_rng(7),
            recorded_at=datetime(2024, 1, 1),
        )
