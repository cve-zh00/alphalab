"""Unit tests for the skeptic aggregator and the two mandated demos.

The aggregator must: run every criterion (no short-circuit), reject on any
single failure, never raise, and name every violated criterion with numbers.
The two demos prove the skeptic *discriminates* — it rejects a clearly bad
candidate enumerating all violations, and accepts a defensible one.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from alphalab.domain.compute import AuditFinding, Severity
from alphalab.domain.costs import CostSpec
from alphalab.domain.skeptic import (
    DataProvenance,
    SkepticVerdict,
    TradeCandidate,
    judge_candidate,
)
from alphalab.domain.types import Side, Ticker

_N_CRITERIA = 8  # seven deterministic + statistical significance


@pytest.fixture
def defensible_candidate() -> TradeCandidate:
    """A candidate that should survive every criterion."""
    rng = np.random.default_rng(42)
    returns = tuple(float(x) for x in rng.normal(0.004, 0.01, 500))
    return TradeCandidate(
        ticker=Ticker("GOOD"),
        side=Side.BUY,
        proposed_size=50.0,
        horizon_days=5,
        backtest_returns=returns,
        cost_fractions=tuple(0.0001 for _ in returns),
        n_trials=2,
        trial_sharpe_std=0.0,
        cost_spec=CostSpec(impact_eta=1.0, impact_alpha=0.5),
        gross_edge_bps=40.0,
        adv=1_000_000.0,
        sigma_bps=100.0,
        per_slice_sharpe=(0.3, 0.35, 0.25, 0.4),
        provenance=DataProvenance(
            knowledge_date=date(2020, 1, 1),
            includes_delisted=True,
            pit_compliant=True,
            latest_observed_at=date(2019, 12, 1),
        ),
    )


@pytest.fixture
def worthless_candidate() -> TradeCandidate:
    """A candidate that violates every single criterion."""
    rng = np.random.default_rng(7)
    returns = tuple(float(x) for x in rng.normal(0.0, 0.02, 40))
    hard = AuditFinding(
        check="ev_bridge_check",
        passed=False,
        severity=Severity.HARD,
        message="EV bridge mismatch",
    )
    return TradeCandidate(
        ticker=Ticker("ZZZ"),
        side=Side.BUY,
        proposed_size=5_000_000.0,
        horizon_days=5,
        backtest_returns=returns,
        cost_fractions=tuple(0.01 for _ in returns),
        n_trials=500,
        trial_sharpe_std=0.5,
        cost_spec=CostSpec(impact_eta=1.0, impact_alpha=0.5),
        gross_edge_bps=5.0,
        adv=10_000.0,
        sigma_bps=200.0,
        per_slice_sharpe=(0.1, -0.3, -0.2, -0.1),
        provenance=DataProvenance(
            knowledge_date=date(2020, 1, 1),
            includes_delisted=False,
            pit_compliant=False,
            latest_observed_at=date(2021, 1, 1),
        ),
        audit_findings=(hard,),
    )


def test_verdict_lists_every_criterion(defensible_candidate: TradeCandidate) -> None:
    # Arrange / Act
    verdict = judge_candidate(defensible_candidate, rng=np.random.default_rng(1))

    # Assert: all criteria ran (no short-circuit), one finding each.
    assert len(verdict.findings) == _N_CRITERIA


def test_single_hard_failure_rejects(defensible_candidate: TradeCandidate) -> None:
    # Arrange: flip exactly one axis (survivorship) to failing.
    bad_prov = defensible_candidate.provenance.model_copy(update={"includes_delisted": False})
    candidate = defensible_candidate.model_copy(update={"provenance": bad_prov})

    # Act
    verdict = judge_candidate(candidate, rng=np.random.default_rng(1))

    # Assert: one failure is enough to reject, and it is reported.
    assert verdict.accepted is False
    assert verdict.n_criteria_failed == 1
    assert any("survivorship" in r for r in verdict.rejection_criteria)


def test_rejection_criteria_name_criterion_with_numbers(
    worthless_candidate: TradeCandidate,
) -> None:
    # Arrange / Act
    verdict = judge_candidate(worthless_candidate, rng=np.random.default_rng(2))

    # Assert: every rejection string names its criterion, and every failed
    # finding carries numeric evidence in its ``values`` (the auditable number,
    # as opposed to a bare generic message).
    assert verdict.rejection_criteria
    for reason in verdict.rejection_criteria:
        assert reason.startswith("skeptic.")
    for finding in verdict.findings:
        if not finding.passed:
            assert finding.values, f"{finding.check} rejected without numeric evidence"
            for _key, number in finding.values:
                assert isinstance(number, float)


def test_safe_run_converts_exception_to_hard_finding() -> None:
    # Arrange: a criterion that explodes.
    from alphalab.domain.skeptic.aggregator import _safe_run

    def _boom(_candidate: object) -> None:
        raise RuntimeError("kaboom")

    # Act
    finding = _safe_run("skeptic.explodes", _boom, object())

    # Assert: the explosion becomes a stamped HARD finding, not a crash.
    assert finding.passed is False
    assert finding.severity is Severity.HARD
    assert "kaboom" in finding.message


def test_aggregator_never_raises_on_empty_series() -> None:
    # Arrange: a maximally degenerate candidate (empty series, negative size).
    candidate = TradeCandidate(
        ticker=Ticker("NAN"),
        side=Side.SELL,
        proposed_size=-1.0,
        horizon_days=0,
        backtest_returns=(),
        cost_fractions=(),
        n_trials=0,
        provenance=DataProvenance(knowledge_date=date(2020, 1, 1), includes_delisted=False),
    )

    # Act
    verdict = judge_candidate(candidate, rng=np.random.default_rng(0))

    # Assert: a verdict, not an exception.
    assert isinstance(verdict, SkepticVerdict)
    assert verdict.accepted is False


# --------------------------------------------------------------------------
# DEMO 1 — a clearly worthless candidate is rejected, enumerating all failures
# --------------------------------------------------------------------------
def test_demo_worthless_candidate_rejected_enumerating_all(
    worthless_candidate: TradeCandidate,
) -> None:
    # Arrange / Act
    verdict = judge_candidate(worthless_candidate, rng=np.random.default_rng(2))

    # Assert: rejected, and *every* criterion fired.
    assert verdict.accepted is False
    assert verdict.n_criteria_failed == _N_CRITERIA
    fired = {r.split(":", 1)[0] for r in verdict.rejection_criteria}
    assert fired == {
        "skeptic.sample_sufficiency",
        "skeptic.cost_survival",
        "skeptic.capacity",
        "skeptic.survivorship",
        "skeptic.leakage",
        "skeptic.cross_robustness",
        "skeptic.contradictory_data",
        "skeptic.significance",
    }


# --------------------------------------------------------------------------
# DEMO 2 — a defensible candidate is accepted (the skeptic discriminates)
# --------------------------------------------------------------------------
def test_demo_defensible_candidate_accepted(defensible_candidate: TradeCandidate) -> None:
    # Arrange / Act
    verdict = judge_candidate(defensible_candidate, rng=np.random.default_rng(1))

    # Assert: accepted with zero objections.
    assert verdict.accepted is True
    assert verdict.n_criteria_failed == 0
    assert verdict.rejection_criteria == ()
