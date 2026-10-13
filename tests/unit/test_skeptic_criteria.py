"""Unit tests for the individual deterministic skeptic criteria.

Each criterion is tested in **both directions**: a case that triggers the
objection *and* a clean case that does not. A skeptic that rejects everything
is as useless as one that accepts everything, so every test file asserts the
negative as well as the positive. AAA structure throughout.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from alphalab.domain.compute import AuditFinding, Severity
from alphalab.domain.costs import CostSpec
from alphalab.domain.skeptic import (
    DataProvenance,
    TradeCandidate,
    check_capacity,
    check_contradictory_data,
    check_cost_survival,
    check_cross_robustness,
    check_leakage,
    check_sample_sufficiency,
    check_statistical_significance,
    check_survivorship,
)
from alphalab.domain.types import Side, Ticker


def _clean_provenance() -> DataProvenance:
    return DataProvenance(
        knowledge_date=date(2020, 1, 1),
        includes_delisted=True,
        pit_compliant=True,
        latest_observed_at=date(2019, 12, 1),
    )


def _candidate(
    *,
    returns: tuple[float, ...],
    cost_fractions: tuple[float, ...] | None = None,
    proposed_size: float = 100.0,
    n_trials: int = 2,
    trial_sharpe_std: float = 0.0,
    gross_edge_bps: float = 40.0,
    adv: float = 1_000_000.0,
    sigma_bps: float = 100.0,
    per_slice_sharpe: tuple[float, ...] = (0.3, 0.35, 0.25, 0.4),
    provenance: DataProvenance | None = None,
    audit_findings: tuple[AuditFinding, ...] = (),
) -> TradeCandidate:
    """Build a candidate that is *clean by default*; override one axis per test."""
    if cost_fractions is None:
        cost_fractions = tuple(0.0001 for _ in returns)
    return TradeCandidate(
        ticker=Ticker("AAA"),
        side=Side.BUY,
        proposed_size=proposed_size,
        horizon_days=5,
        backtest_returns=returns,
        cost_fractions=cost_fractions,
        n_trials=n_trials,
        trial_sharpe_std=trial_sharpe_std,
        cost_spec=CostSpec(impact_eta=1.0, impact_alpha=0.5),
        gross_edge_bps=gross_edge_bps,
        adv=adv,
        sigma_bps=sigma_bps,
        per_slice_sharpe=per_slice_sharpe,
        provenance=provenance or _clean_provenance(),
        audit_findings=audit_findings,
    )


@pytest.fixture
def strong_returns() -> tuple[float, ...]:
    rng = np.random.default_rng(42)
    return tuple(float(x) for x in rng.normal(0.004, 0.01, 500))


# --------------------------------------------------------------------------
# S1 — sample sufficiency
# --------------------------------------------------------------------------
def test_sample_sufficiency_rejects_thin_series() -> None:
    # Arrange: a short series whose T_eff is below the floor.
    rng = np.random.default_rng(1)
    candidate = _candidate(returns=tuple(float(x) for x in rng.normal(0.0, 0.01, 30)))

    # Act
    finding = check_sample_sufficiency(candidate)

    # Assert
    assert finding.passed is False
    assert finding.check == "skeptic.sample_sufficiency"


def test_sample_sufficiency_accepts_long_series(strong_returns: tuple[float, ...]) -> None:
    # Arrange / Act
    finding = check_sample_sufficiency(_candidate(returns=strong_returns))

    # Assert
    assert finding.passed is True


# --------------------------------------------------------------------------
# S2 — cost survival
# --------------------------------------------------------------------------
def test_cost_survival_rejects_when_costs_erase_edge(strong_returns: tuple[float, ...]) -> None:
    # Arrange: huge per-period cost drag erases the edge and flips the sign.
    candidate = _candidate(
        returns=strong_returns,
        cost_fractions=tuple(0.02 for _ in strong_returns),
    )

    # Act
    finding = check_cost_survival(candidate)

    # Assert
    assert finding.passed is False


def test_cost_survival_accepts_when_edge_survives(strong_returns: tuple[float, ...]) -> None:
    # Arrange / Act
    finding = check_cost_survival(_candidate(returns=strong_returns))

    # Assert
    assert finding.passed is True


# --------------------------------------------------------------------------
# S3 — capacity
# --------------------------------------------------------------------------
def test_capacity_rejects_oversized_order(strong_returns: tuple[float, ...]) -> None:
    # Arrange: tiny ADV + enormous size pushes well past the ceiling.
    candidate = _candidate(returns=strong_returns, proposed_size=5_000_000.0, adv=10_000.0)

    # Act
    finding = check_capacity(candidate)

    # Assert
    assert finding.passed is False


def test_capacity_accepts_small_order(strong_returns: tuple[float, ...]) -> None:
    # Arrange / Act
    finding = check_capacity(_candidate(returns=strong_returns, proposed_size=10.0))

    # Assert
    assert finding.passed is True


def test_capacity_rejects_unmeasurable_market(strong_returns: tuple[float, ...]) -> None:
    # Arrange: non-positive ADV makes the capacity ceiling unmeasurable.
    candidate = _candidate(returns=strong_returns, adv=0.0)

    # Act
    finding = check_capacity(candidate)

    # Assert: a HARD rejection rather than an exception.
    assert finding.passed is False
    assert finding.severity is Severity.HARD


def test_capacity_rejects_nan_size(strong_returns: tuple[float, ...]) -> None:
    # Arrange: a degenerate (NaN) proposed size.
    candidate = _candidate(returns=strong_returns, proposed_size=float("nan"))

    # Act
    finding = check_capacity(candidate)

    # Assert
    assert finding.passed is False


# --------------------------------------------------------------------------
# S4 — survivorship
# --------------------------------------------------------------------------
def test_survivorship_rejects_universe_without_delisted(
    strong_returns: tuple[float, ...],
) -> None:
    # Arrange
    prov = DataProvenance(knowledge_date=date(2020, 1, 1), includes_delisted=False)
    candidate = _candidate(returns=strong_returns, provenance=prov)

    # Act
    finding = check_survivorship(candidate)

    # Assert
    assert finding.passed is False


def test_survivorship_accepts_universe_with_delisted(strong_returns: tuple[float, ...]) -> None:
    # Arrange / Act
    finding = check_survivorship(_candidate(returns=strong_returns))

    # Assert
    assert finding.passed is True


# --------------------------------------------------------------------------
# S5 — leakage
# --------------------------------------------------------------------------
def test_leakage_rejects_future_dated_lineage(strong_returns: tuple[float, ...]) -> None:
    # Arrange: lineage observed after the knowledge date.
    prov = DataProvenance(
        knowledge_date=date(2020, 1, 1),
        includes_delisted=True,
        latest_observed_at=date(2021, 1, 1),
    )

    # Act
    finding = check_leakage(_candidate(returns=strong_returns, provenance=prov))

    # Assert
    assert finding.passed is False


def test_leakage_rejects_declared_non_pit(strong_returns: tuple[float, ...]) -> None:
    # Arrange
    prov = DataProvenance(
        knowledge_date=date(2020, 1, 1), includes_delisted=True, pit_compliant=False
    )

    # Act
    finding = check_leakage(_candidate(returns=strong_returns, provenance=prov))

    # Assert
    assert finding.passed is False


def test_leakage_accepts_clean_lineage(strong_returns: tuple[float, ...]) -> None:
    # Arrange / Act
    finding = check_leakage(_candidate(returns=strong_returns))

    # Assert
    assert finding.passed is True


# --------------------------------------------------------------------------
# S6 — statistical significance
# --------------------------------------------------------------------------
def test_significance_rejects_pure_noise() -> None:
    # Arrange: pure noise with many trials cannot clear DSR / HLZ.
    rng = np.random.default_rng(3)
    candidate = _candidate(
        returns=tuple(float(x) for x in rng.normal(0.0, 0.02, 300)),
        n_trials=500,
        trial_sharpe_std=0.5,
    )

    # Act
    finding = check_statistical_significance(candidate, rng=np.random.default_rng(99))

    # Assert
    assert finding.passed is False


def test_significance_accepts_strong_edge(strong_returns: tuple[float, ...]) -> None:
    # Arrange / Act
    finding = check_statistical_significance(
        _candidate(returns=strong_returns), rng=np.random.default_rng(5)
    )

    # Assert
    assert finding.passed is True


def test_significance_rejects_weak_edge_via_ci_or_dsr() -> None:
    # Arrange: a weak positive drift — enough to not be pure noise, but its
    # bootstrap CI should straddle zero / fail DSR, exercising those branches.
    rng = np.random.default_rng(11)
    candidate = _candidate(
        returns=tuple(float(x) for x in rng.normal(0.0005, 0.02, 120)),
        n_trials=50,
        trial_sharpe_std=0.3,
    )

    # Act
    finding = check_statistical_significance(candidate, rng=np.random.default_rng(13))

    # Assert
    assert finding.passed is False
    assert finding.values  # numeric evidence attached


# --------------------------------------------------------------------------
# S7 — cross-asset / cross-period robustness
# --------------------------------------------------------------------------
def test_cross_robustness_rejects_single_slice_edge(strong_returns: tuple[float, ...]) -> None:
    # Arrange: edge positive in only a quarter of the slices.
    candidate = _candidate(returns=strong_returns, per_slice_sharpe=(0.3, -0.2, -0.1, -0.15))

    # Act
    finding = check_cross_robustness(candidate)

    # Assert
    assert finding.passed is False


def test_cross_robustness_accepts_broad_edge(strong_returns: tuple[float, ...]) -> None:
    # Arrange / Act
    finding = check_cross_robustness(_candidate(returns=strong_returns))

    # Assert
    assert finding.passed is True


# --------------------------------------------------------------------------
# S8 — contradictory data
# --------------------------------------------------------------------------
def test_contradictory_data_rejects_hard_finding(strong_returns: tuple[float, ...]) -> None:
    # Arrange: a blocking HARD compute-audit finding attached to the candidate.
    hard = AuditFinding(
        check="ev_bridge_check",
        passed=False,
        severity=Severity.HARD,
        message="EV bridge mismatch",
    )

    # Act
    finding = check_contradictory_data(_candidate(returns=strong_returns, audit_findings=(hard,)))

    # Assert
    assert finding.passed is False


def test_contradictory_data_ignores_soft_finding(strong_returns: tuple[float, ...]) -> None:
    # Arrange: a SOFT finding must NOT trigger S8.
    soft = AuditFinding(
        check="magnitude_sanity",
        passed=False,
        severity=Severity.SOFT,
        message="borderline magnitude",
    )

    # Act
    finding = check_contradictory_data(_candidate(returns=strong_returns, audit_findings=(soft,)))

    # Assert
    assert finding.passed is True


def test_contradictory_data_accepts_no_findings(strong_returns: tuple[float, ...]) -> None:
    # Arrange / Act
    finding = check_contradictory_data(_candidate(returns=strong_returns))

    # Assert
    assert finding.passed is True


# --------------------------------------------------------------------------
# rejection_criteria must NAME the criterion with its numeric value
# --------------------------------------------------------------------------
def test_findings_carry_numeric_values(strong_returns: tuple[float, ...]) -> None:
    # Arrange: oversized order so the capacity finding fires with numbers.
    candidate = _candidate(returns=strong_returns, proposed_size=5_000_000.0, adv=10_000.0)

    # Act
    finding = check_capacity(candidate)

    # Assert: the finding stamps the concrete size and ceiling, not a generic string.
    value_keys = {k for k, _ in finding.values}
    assert "proposed_size" in value_keys
    assert "capacity_ceiling" in value_keys
    assert any(char.isdigit() for char in finding.message)
