"""ACCEPTANCE — the decisive three-arm A/B experiment, end to end.

This is the deliverable of Task 14. It runs the full experiment with
deterministic doubles (no LLM), over a shared set of decisions whose
out-of-sample forward returns are **driftless GBM — edgeless by construction**
(dossier §0: honest synthetic data with no alpha), applies costs identically to
all three arms, records every arm's trials and every pairwise comparison in a
real ``SqliteTrialLedger``, and emits a :class:`Comparison`.

It then renders the **comparative table of the three arms** (Sharpe gross/net,
max drawdown, proposed vs executed, DSR, PBO, Brier/ECE) and the **verdict with
the confidence interval of the Sharpe difference**. The table is the entregable.

The expected honest result, given edgeless inputs and a small out-of-sample
window: the arms are **indistinguishable or the result is inconclusive** — the
sample cannot manufacture an edge that is not there, and the experiment reports
that rather than inventing a winner. A negative result is a success.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from alphalab.adapters.ledger_sqlite import SqliteTrialLedger
from alphalab.app.ab_experiment import ABExperimentRunner
from alphalab.artifact.models import Verdict as ArtifactVerdict
from alphalab.domain.experiment.comparison import Comparison, Verdict
from alphalab.domain.stats.synthetic import gbm_log_returns
from tests.unit._ab_runner_doubles import (
    make_analyst_pipeline,
    make_debate_pipeline,
    make_decision,
)

pytestmark = pytest.mark.acceptance

_N_DECISIONS = 40
_OOS_START = 20  # 20 in-sample (search), 20 out-of-sample (verdict)


def _edgeless_forwards(n: int) -> np.ndarray:
    """Driftless GBM per-period returns: no edge by construction, seeded."""
    rng = np.random.default_rng(2024)
    return gbm_log_returns(rng=rng, n_periods=n, mu=0.0, sigma=0.01)


def _build_decisions() -> list:
    forwards = _edgeless_forwards(_N_DECISIONS)
    base = date(2021, 1, 1)
    rng = np.random.default_rng(99)
    probs = rng.uniform(0.45, 0.65, size=_N_DECISIONS)
    return [
        make_decision(
            knowledge_date=base + timedelta(days=i),
            forward_return=float(forwards[i]),
            cost_fraction=0.0005,
            accept=False,  # edgeless candidates -> skeptic vetoes (honest)
            predicted_prob=float(probs[i]),
            seed=i,
        )
        for i in range(_N_DECISIONS)
    ]


def _render_table(comp: Comparison) -> str:
    """Render the three-arm comparative table and the verdict block."""
    header = (
        f"{'arm':<34}{'Sh.gross':>10}{'Sh.net':>10}{'maxDD':>9}"
        f"{'prop':>6}{'exec':>6}{'DSR':>7}{'PBO':>7}{'Brier':>8}{'ECE':>8}"
    )
    rows = [header, "-" * len(header)]
    for arm in (comp.arm_a, comp.arm_b, comp.arm_c):
        rows.append(
            f"{arm.arm.name + ' ' + arm.arm.label:<34}"
            f"{arm.sharpe_gross:>10.3f}{arm.sharpe_net:>10.3f}{arm.max_drawdown:>9.3f}"
            f"{arm.n_proposed:>6}{arm.n_executed:>6}"
            f"{arm.certification.dsr:>7.3f}{arm.certification.pbo:>7.3f}"
            f"{arm.calibration.brier:>8.3f}{arm.calibration.ece:>8.3f}"
        )
    dbs = comp.debate_vs_single
    svd = comp.skeptic_vs_debate
    sa = comp.skeptic_analysis
    rows.append("")
    rows.append(
        f"VERDICT debate(B) vs single(A): {comp.debate_verdict.value.upper()}  "
        f"delta_SR={dbs.delta:+.4f}  "
        f"CI[{dbs.ci_low:+.4f}, {dbs.ci_high:+.4f}] @ conf={dbs.confidence:.4f}"
    )
    rows.append(
        f"VERDICT skeptic(C) vs debate(B): {comp.skeptic_sharpe_verdict.value.upper()}  "
        f"delta_SR={svd.delta:+.4f}  "
        f"CI[{svd.ci_low:+.4f}, {svd.ci_high:+.4f}] @ conf={svd.confidence:.4f}"
    )
    rows.append(
        f"SKEPTIC: vetoed={sa.n_vetoed} (losers={sa.n_vetoed_losers}, "
        f"winners={sa.n_vetoed_winners})  avoided_return={sa.avoided_return:+.4f}  "
        f"drawdown={sa.drawdown_verdict.value}  avoids_edgeless={sa.avoids_edgeless_trades}"
    )
    rows.append(
        f"POWER: n_oos={comp.n_observations}  "
        f"min_detectable_delta_SR={comp.min_detectable_sharpe_diff:.4f}  "
        f"conclusive={comp.conclusive}  (Bonferroni across 3 pairs)"
    )
    return "\n".join(rows)


def test_ab_experiment_emits_the_three_arm_verdict(tmp_path: Path) -> None:
    """Run the whole experiment and emit the comparative table + verdict."""
    # Arrange — shared decisions, identical inputs for every arm.
    decisions = _build_decisions()
    with SqliteTrialLedger(tmp_path / "ab_trials.sqlite3") as ledger:
        runner = ABExperimentRunner(
            analyst=make_analyst_pipeline(verdict=ArtifactVerdict.BUY),
            debate=make_debate_pipeline(verdict=ArtifactVerdict.BUY),
            ledger=ledger,
            skeptic_rng=np.random.default_rng(0),
        )

        # Act
        comp = runner.run(
            decisions,
            oos_start=_OOS_START,
            rng=np.random.default_rng(1234),
            recorded_at=datetime(2024, 1, 1),
            n_resamples=2_000,
        )

        # Assert — every arm recorded one trial per OOS decision + 3 comparisons.
        n_oos = _N_DECISIONS - _OOS_START
        assert ledger.trial_count("ab_arm_a") == n_oos
        assert ledger.trial_count("ab_arm_b") == n_oos
        assert ledger.trial_count("ab_arm_c") == n_oos
        assert ledger.trial_count("ab_comparisons") == 3

    # Emit the deliverable table. Capture-and-reprint so it survives capsys.
    table = _render_table(comp)
    assert "VERDICT debate(B) vs single(A)" in table
    assert "VERDICT skeptic(C) vs debate(B)" in table
    assert "POWER" in table

    # Assert — the experiment emitted a well-formed verdict over OOS only.
    assert comp.n_observations == n_oos
    valid = {
        Verdict.IMPROVES,
        Verdict.WORSENS,
        Verdict.INDISTINGUISHABLE,
        Verdict.INCONCLUSIVE,
    }
    assert comp.debate_verdict in valid
    assert comp.skeptic_sharpe_verdict in valid

    # Assert — same decisions in, same window: arms are comparable.
    assert len(comp.arm_a.executed_returns) == n_oos
    assert len(comp.arm_b.executed_returns) == n_oos
    assert len(comp.arm_c.executed_returns) == n_oos

    # Assert — the honest finding on edgeless data: no fabricated winner. With a
    # 20-observation OOS window the test is underpowered, so the verdict must be
    # a non-committal "indistinguishable" or "inconclusive", never a win.
    assert comp.debate_verdict in {Verdict.INDISTINGUISHABLE, Verdict.INCONCLUSIVE}

    # Assert — the skeptic analysis is coherent: it vetoed the edgeless
    # candidates and its claim to avoid edgeless trades is measured, not assumed.
    assert comp.skeptic_analysis.n_vetoed == comp.arm_c.n_proposed

    # Print the deliverable table to the terminal (``make acid-test`` runs -v).
    print("\n" + table)
