"""Unit tests for the three-arm comparison and its verdicts.

The two most important tests here are the **negative control** (identical arms
must never declare a winner) and the **skeptic honesty** tests (vetoing losers
must help, vetoing winners must hurt — no thumb on the scale).
"""

from __future__ import annotations

import numpy as np
import pytest

from alphalab.domain.experiment.comparison import (
    Verdict,
    analyse_skeptic,
    compare_arms,
)
from tests.unit._experiment_doubles import (
    executed_outcome,
    make_arm_result,
    veto_outcome,
)


def _edged_series(n: int, *, mu: float, seed: int) -> tuple[float, ...]:
    rng = np.random.default_rng(seed)
    return tuple(float(x) for x in rng.normal(mu, 0.01, size=n))


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL — identical arms must not declare a winner.
# ---------------------------------------------------------------------------


def test_identical_arms_are_never_declared_a_winner() -> None:
    # Arrange — three arms with the EXACT same return series.
    series = _edged_series(400, mu=0.0008, seed=1)
    arm_a = make_arm_result("A", series)
    arm_b = make_arm_result("B", series)
    arm_c = make_arm_result("C", series)
    outcomes = tuple(executed_outcome(r) for r in series)

    # Act
    comp = compare_arms(
        arm_a,
        arm_b,
        arm_c,
        arm_c_outcomes=outcomes,
        rng=np.random.default_rng(0),
        n_resamples=2_000,
    )

    # Assert — no pairwise test may exclude zero; no winner anywhere.
    assert comp.debate_vs_single.winner is None
    assert comp.skeptic_vs_debate.winner is None
    assert comp.single_vs_skeptic.winner is None
    assert comp.debate_verdict in {Verdict.INDISTINGUISHABLE, Verdict.INCONCLUSIVE}
    assert comp.skeptic_sharpe_verdict in {Verdict.INDISTINGUISHABLE, Verdict.INCONCLUSIVE}
    assert comp.debate_vs_single.delta == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# POSITIVE CONTROL — a real, wide edge must be detected, both directions.
# ---------------------------------------------------------------------------


def test_detects_debate_improvement_when_edge_is_real() -> None:
    # Arrange — arm B dominates arm A by a wide, consistent margin.
    rng = np.random.default_rng(4)
    base = rng.normal(0.0, 0.005, size=400)
    weak = tuple(float(x) for x in base)
    strong = tuple(float(x + 0.01) for x in base)
    arm_a = make_arm_result("A", weak)
    arm_b = make_arm_result("B", strong)
    arm_c = make_arm_result("C", strong)
    outcomes = tuple(executed_outcome(r) for r in strong)

    # Act
    comp = compare_arms(
        arm_a,
        arm_b,
        arm_c,
        arm_c_outcomes=outcomes,
        rng=np.random.default_rng(1),
        n_resamples=2_000,
    )

    # Assert — the debate's advantage is detected and attributed to B.
    assert comp.debate_vs_single.winner == "B"
    assert comp.debate_verdict is Verdict.IMPROVES


def test_detects_debate_regression_when_debate_is_worse() -> None:
    # Arrange — arm B is clearly worse than arm A.
    rng = np.random.default_rng(6)
    base = rng.normal(0.0, 0.005, size=400)
    strong = tuple(float(x + 0.01) for x in base)
    weak = tuple(float(x) for x in base)
    arm_a = make_arm_result("A", strong)
    arm_b = make_arm_result("B", weak)
    arm_c = make_arm_result("C", weak)
    outcomes = tuple(executed_outcome(r) for r in weak)

    # Act
    comp = compare_arms(
        arm_a,
        arm_b,
        arm_c,
        arm_c_outcomes=outcomes,
        rng=np.random.default_rng(2),
        n_resamples=2_000,
    )

    # Assert — challenger (B) is worse: WORSENS, winner is the baseline A.
    assert comp.debate_vs_single.winner == "A"
    assert comp.debate_verdict is Verdict.WORSENS


# ---------------------------------------------------------------------------
# SKEPTIC HONESTY — vetoes are scored by what they actually avoided.
# ---------------------------------------------------------------------------


def test_skeptic_that_vetoes_losers_reduces_drawdown() -> None:
    # Arrange — arm C has a shallower (less negative) drawdown than B, and its
    # vetoes avoided net-losing trades.
    series_b = (0.01, -0.08, 0.01, -0.06)
    series_c = (0.01, 0.0, 0.01, 0.0)  # the two losers were vetoed -> flat
    arm_b = make_arm_result("B", series_b)
    arm_c = make_arm_result("C", series_c)
    vetoes = (veto_outcome(-0.08), veto_outcome(-0.06))

    # Act
    analysis = analyse_skeptic(arm_b, arm_c, arm_c_outcomes=vetoes)

    # Assert — drawdown improved and the vetoed trades would have lost money.
    assert analysis.drawdown_verdict is Verdict.IMPROVES
    assert analysis.drawdown_delta > 0.0
    assert analysis.n_vetoed == 2
    assert analysis.n_vetoed_losers == 2
    assert analysis.n_vetoed_winners == 0
    assert analysis.avoided_return < 0.0
    assert analysis.avoids_edgeless_trades is True


def test_skeptic_that_vetoes_winners_does_not_claim_credit() -> None:
    # Arrange — arm C vetoed trades that would have WON; it worsened results.
    series_b = (0.05, 0.04, 0.05)
    series_c = (0.0, 0.0, 0.05)  # two winners vetoed -> flat, deeper relative dd
    arm_b = make_arm_result("B", series_b, max_drawdown=-0.02)
    arm_c = make_arm_result("C", series_c, max_drawdown=-0.10)
    vetoes = (veto_outcome(0.05), veto_outcome(0.04))

    # Act
    analysis = analyse_skeptic(arm_b, arm_c, arm_c_outcomes=vetoes)

    # Assert — the skeptic skipped winners: it must NOT look good.
    assert analysis.n_vetoed_winners == 2
    assert analysis.n_vetoed_losers == 0
    assert analysis.avoided_return > 0.0
    assert analysis.avoids_edgeless_trades is False
    assert analysis.drawdown_verdict is Verdict.WORSENS
    assert analysis.drawdown_delta < 0.0


def test_no_vetoes_means_no_edgeless_claim() -> None:
    # Arrange — the skeptic vetoed nothing.
    arm_b = make_arm_result("B", (0.01, 0.02, 0.01), max_drawdown=-0.05)
    arm_c = make_arm_result("C", (0.01, 0.02, 0.01), max_drawdown=-0.05)

    # Act
    analysis = analyse_skeptic(arm_b, arm_c, arm_c_outcomes=())

    # Assert
    assert analysis.n_vetoed == 0
    assert analysis.avoided_return == 0.0
    assert analysis.avoids_edgeless_trades is False
    assert analysis.drawdown_verdict is Verdict.INDISTINGUISHABLE


# ---------------------------------------------------------------------------
# GUARDS
# ---------------------------------------------------------------------------


def test_mismatched_arm_lengths_raise() -> None:
    # Arrange — arms run over different numbers of decisions.
    arm_a = make_arm_result("A", (0.01, 0.02, 0.03, 0.04))
    arm_b = make_arm_result("B", (0.01, 0.02, 0.03))
    arm_c = make_arm_result("C", (0.01, 0.02, 0.03))

    # Act / Assert
    with pytest.raises(ValueError, match="same number"):
        compare_arms(arm_a, arm_b, arm_c, arm_c_outcomes=(), rng=np.random.default_rng(0))


def test_comparison_is_frozen() -> None:
    # Arrange
    series = _edged_series(50, mu=0.001, seed=9)
    comp = compare_arms(
        make_arm_result("A", series),
        make_arm_result("B", series),
        make_arm_result("C", series),
        arm_c_outcomes=tuple(executed_outcome(r) for r in series),
        rng=np.random.default_rng(0),
        n_resamples=1_000,
    )

    # Act / Assert
    with pytest.raises(Exception, match=r"frozen|Instance is frozen"):
        comp.debate_verdict = Verdict.IMPROVES  # type: ignore[misc]
