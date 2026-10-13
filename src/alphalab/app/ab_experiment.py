"""The decisive three-arm A/B runner (orchestration layer).

METHODOLOGICAL HONESTY — READ BEFORE TRUSTING ANY NUMBER THIS EMITS
===================================================================
This runner exists to **falsify the project's own additions**, not to make the
debate or the skeptic look good. A negative result is a success of this module.

What it does
------------
Over the **same ordered set of decisions, the same time window, and the same
``knowledge_date`` per decision**, it runs three interchangeable arms:

* **Arm A (baseline)** — ``AnalystPipeline`` (single analyst).
* **Arm B (debate)** — ``DebatePipeline`` (bull / bear / judge).
* **Arm C (debate + skeptic)** — arm B plus the deterministic
  ``judge_candidate`` gate, which can **veto** a trade.

Each arm turns a gated artifact's directional ``verdict`` into a position
(BUY -> +1, SELL -> -1, HOLD -> 0) and earns the decision's realised
out-of-sample forward return. **Costs are subtracted identically for all three
arms** (the same per-decision ``cost_fraction``), so no arm is flattered by
hidden frictions (dossier §0, *The Alpha Illusion*: TradingAgents fell from
Sharpe 0.43 to 0.22 once frictions were modelled). Arm C only differs by the
veto: a vetoed decision contributes **zero** to arm C's traded return but its
avoided return is recorded so the skeptic can be judged on whether it dodged
losers or skipped winners.

Each arm records one trial per decision in the ``SqliteTrialLedger`` so the DSR
sees an honest trial count. The three pairwise comparisons are themselves
trials and are recorded too (dossier §4: it would be ironic to build a harness
against multiple testing and then commit it in our own experiment).

In-sample vs out-of-sample
--------------------------
Decisions before ``oos_start`` are the **in-sample** search; the verdict is
emitted **only** over the out-of-sample tail (``decisions[oos_start:]``). The
split index is explicit and recorded.

Sample size and its consequence (power)
---------------------------------------
With few out-of-sample decisions the difference between arms is undetectable
*by construction*. The emitted :class:`~alphalab.domain.experiment.Comparison`
carries the minimum detectable Sharpe difference at the Bonferroni-corrected
confidence and an explicit ``conclusive`` flag. When the sample cannot detect a
material effect the verdict is **"inconclusive"**, a stronger and different
caveat than "indistinguishable" (a true tie). Read that flag before the verdict.

What this experiment does NOT prove
-----------------------------------
* It runs with **deterministic doubles**, not the real 27B LLM: it measures the
  *architecture* (debate vs single, skeptic gate), which the Agent Market Arena
  (arXiv:2510.11695) found dominates outcome variance — but it is **not**
  evidence about any particular model's live skill.
* It runs over a **synthetic or bounded historical window**, so it is **not**
  evidence of deployable profitability (dossier §0, *The Alpha Illusion*).
* Costs are modelled, but market impact, borrow and capacity are only as
  realistic as the injected ``cost_fraction``.

The honest benchmark remains **SPY buy-and-hold, net of costs**; every arm is
measured against it and against each other, and the result is reported as-is.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from alphalab.app.ab_metrics import (
    annualised_sharpe,
    calibrate,
    certify_series,
    max_drawdown,
    position_sign,
)
from alphalab.app.debate_pipeline import DebatePipeline
from alphalab.app.inputs import AnalystRequest
from alphalab.app.pipeline import AnalystPipeline
from alphalab.artifact.models import Verdict as ArtifactVerdict
from alphalab.domain.experiment import (
    ArmResult,
    Comparison,
    DecisionOutcome,
    ExperimentArm,
    compare_arms,
)
from alphalab.domain.ledger import TrialMetrics, TrialRecord
from alphalab.domain.skeptic import SkepticVerdict, TradeCandidate, judge_candidate

__all__ = ["ABExperimentRunner", "ArmArtifacts", "ExperimentDecision"]

#: Stable arm identities used across the ledger and the comparison.
ARM_A = ExperimentArm(name="A", label="baseline single analyst", family="ab_arm_a")
ARM_B = ExperimentArm(name="B", label="bull/bear/judge debate", family="ab_arm_b")
ARM_C = ExperimentArm(name="C", label="debate + deterministic skeptic", family="ab_arm_c")


class ExperimentDecision(BaseModel):
    """One decision shared by all three arms, with its out-of-sample truth.

    Attributes:
        request: The identical pipeline input every arm consumes.
        forward_return: Realised per-period forward return observed *after*
            ``request.knowledge_date`` (the out-of-sample truth). Signed by the
            arm's position; never read before the decision is made.
        cost_fraction: Per-period cost drag applied identically to every arm.
        candidate: The ``TradeCandidate`` the skeptic (arm C) judges. Only the
            skeptic consumes it; A and B ignore it.
        predicted_prob: Pre-trade win probability for calibration (optional).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    request: AnalystRequest
    forward_return: float
    cost_fraction: float = Field(ge=0.0)
    candidate: TradeCandidate
    predicted_prob: float | None = None


class _ArmOutcome(BaseModel):
    """Internal per-decision record carrying both the public outcome and sign."""

    model_config = ConfigDict(frozen=True)

    outcome: DecisionOutcome
    sign: int


class ArmArtifacts(BaseModel):
    """The per-decision records of one arm (pre-aggregation)."""

    model_config = ConfigDict(frozen=True)

    arm: ExperimentArm
    records: tuple[_ArmOutcome, ...]

    @property
    def outcomes(self) -> tuple[DecisionOutcome, ...]:
        """The public per-decision outcomes."""
        return tuple(r.outcome for r in self.records)


def _gross_return(sign: int, decision: ExperimentDecision) -> float:
    """Signed realised return before costs (0 when flat)."""
    return sign * decision.forward_return


def _net_return(sign: int, decision: ExperimentDecision) -> float:
    """Signed, cost-adjusted realised return (0 when flat)."""
    if sign == 0:
        return 0.0
    return sign * decision.forward_return - decision.cost_fraction


def _build_outcome(
    verdict: ArtifactVerdict,
    decision: ExperimentDecision,
    *,
    vetoed: bool,
) -> _ArmOutcome:
    """Build the per-decision record, applying the skeptic veto if any."""
    sign = position_sign(verdict)
    proposed = sign != 0
    executed = proposed and not vetoed
    net = _net_return(sign, decision)
    outcome = DecisionOutcome(
        proposed=proposed,
        executed=executed,
        # Executed trades record what they earned; vetoed trades record the
        # net return they avoided so the skeptic can be scored on it.
        realized_return=net if proposed else 0.0,
        predicted_prob=decision.predicted_prob,
        was_winner=(net > 0.0) if proposed else None,
    )
    return _ArmOutcome(outcome=outcome, sign=sign)


class ABExperimentRunner:
    """Run the three arms over one shared decision set and compare them.

    Dependencies (pipelines, skeptic rng, ledger) are injected so the runner is
    fully testable with deterministic doubles and never calls a real LLM.
    """

    __slots__ = ("_analyst", "_debate", "_ledger", "_skeptic_rng")

    def __init__(
        self,
        *,
        analyst: AnalystPipeline,
        debate: DebatePipeline,
        ledger: object,
        skeptic_rng: np.random.Generator,
    ) -> None:
        self._analyst = analyst
        self._debate = debate
        self._ledger = ledger
        self._skeptic_rng = skeptic_rng

    def run(
        self,
        decisions: Sequence[ExperimentDecision],
        *,
        oos_start: int,
        rng: np.random.Generator,
        recorded_at: datetime,
        n_resamples: int = 10_000,
    ) -> Comparison:
        """Execute all arms and return the immutable out-of-sample comparison.

        Args:
            decisions: Shared, ordered decisions (in-sample then out-of-sample).
            oos_start: Index at which the out-of-sample window begins; the
                verdict is emitted only over ``decisions[oos_start:]``.
            rng: Seeded generator for the comparison bootstrap.
            recorded_at: Timestamp stamped on every ledger trial.
            n_resamples: Bootstrap resamples per pairwise comparison.

        Raises:
            ValueError: If ``oos_start`` leaves fewer than two OOS decisions.
        """
        if not 0 <= oos_start <= len(decisions) - 2:
            raise ValueError("oos_start must leave at least two out-of-sample decisions")

        arm_a = self._run_arm(ARM_A, decisions, self._decide_analyst)
        arm_b = self._run_arm(ARM_B, decisions, self._decide_debate)
        arm_c = self._run_arm(ARM_C, decisions, self._decide_skeptic)

        oos = slice(oos_start, None)
        result_a = self._aggregate(arm_a, decisions, oos, recorded_at)
        result_b = self._aggregate(arm_b, decisions, oos, recorded_at)
        result_c = self._aggregate(arm_c, decisions, oos, recorded_at)

        comparison = compare_arms(
            result_a,
            result_b,
            result_c,
            arm_c_outcomes=arm_c.outcomes[oos],
            rng=rng,
            n_resamples=n_resamples,
        )
        self._record_comparison_trials(comparison, decisions, oos_start, recorded_at)
        return comparison

    # -- per-arm decision functions --------------------------------------

    def _decide_analyst(self, decision: ExperimentDecision) -> _ArmOutcome:
        verdict = self._analyst.run(decision.request).artifact.verdict
        return _build_outcome(verdict, decision, vetoed=False)

    def _decide_debate(self, decision: ExperimentDecision) -> _ArmOutcome:
        verdict = self._debate.run(decision.request).artifact.verdict
        return _build_outcome(verdict, decision, vetoed=False)

    def _decide_skeptic(self, decision: ExperimentDecision) -> _ArmOutcome:
        verdict = self._debate.run(decision.request).artifact.verdict
        skeptic: SkepticVerdict = judge_candidate(decision.candidate, rng=self._skeptic_rng)
        return _build_outcome(verdict, decision, vetoed=not skeptic.accepted)

    # -- arm execution + aggregation -------------------------------------

    @staticmethod
    def _run_arm(
        arm: ExperimentArm,
        decisions: Sequence[ExperimentDecision],
        run: Callable[[ExperimentDecision], _ArmOutcome],
    ) -> ArmArtifacts:
        """Run one arm across every decision, producing its record stream."""
        return ArmArtifacts(arm=arm, records=tuple(run(d) for d in decisions))

    def _aggregate(
        self,
        arm: ArmArtifacts,
        decisions: Sequence[ExperimentDecision],
        oos: slice,
        recorded_at: datetime,
    ) -> ArmResult:
        """Aggregate an arm's out-of-sample outcomes into a frozen result."""
        oos_records = arm.records[oos]
        oos_decisions = decisions[oos]
        # One return per period: executed trades earn their return, flat/vetoed
        # periods earn exactly zero, so every arm's series has equal length.
        net_series = tuple(
            _net_return(r.sign, d) if r.outcome.executed else 0.0
            for r, d in zip(oos_records, oos_decisions, strict=True)
        )
        gross_series = tuple(
            _gross_return(r.sign, d) if r.outcome.executed else 0.0
            for r, d in zip(oos_records, oos_decisions, strict=True)
        )
        n_proposed = sum(1 for r in oos_records if r.outcome.proposed)
        n_executed = sum(1 for r in oos_records if r.outcome.executed)
        winners = sum(1 for n in net_series if n > 0.0)
        hit_rate = winners / n_executed if n_executed else 0.0
        self._record_arm_trials(arm.arm, decisions, oos, recorded_at, net_series)
        return ArmResult(
            arm=arm.arm,
            sharpe_gross=annualised_sharpe(gross_series),
            sharpe_net=annualised_sharpe(net_series),
            max_drawdown=max_drawdown(net_series),
            hit_rate=hit_rate,
            n_proposed=n_proposed,
            n_executed=n_executed,
            executed_returns=net_series,
            certification=certify_series(net_series),
            calibration=calibrate(arm.outcomes[oos]),
        )

    # -- ledger ----------------------------------------------------------

    def _record_arm_trials(
        self,
        arm: ExperimentArm,
        decisions: Sequence[ExperimentDecision],
        oos: slice,
        recorded_at: datetime,
        series: tuple[float, ...],
    ) -> None:
        """Append one ledger trial per out-of-sample decision for this arm."""
        window = _window(decisions, oos)
        for i, ret in enumerate(series):
            self._append_trial(arm.family, i, window, ret, recorded_at)

    def _record_comparison_trials(
        self,
        comparison: Comparison,
        decisions: Sequence[ExperimentDecision],
        oos_start: int,
        recorded_at: datetime,
    ) -> None:
        """Record each pairwise comparison as its own trial (multiple-testing)."""
        window = _window(decisions, slice(oos_start, None))
        for diff in (
            comparison.debate_vs_single,
            comparison.skeptic_vs_debate,
            comparison.single_vs_skeptic,
        ):
            config = f"{diff.name_a}_vs_{diff.name_b}"
            self._append_trial("ab_comparisons", config, window, diff.delta, recorded_at)

    def _append_trial(
        self,
        family: str,
        config: object,
        window: tuple[date, date],
        sharpe: float,
        recorded_at: datetime,
    ) -> None:
        """Append a single trial record through the injected ledger."""
        record = TrialRecord.create(
            spec={"family": family, "config": config},
            dataset_hash="ab-experiment-v1",
            family=family,
            window_start=window[0],
            window_end=window[1],
            params={"config": config},
            metrics=TrialMetrics(
                sharpe_gross=sharpe,
                sharpe_net=sharpe,
                return_gross=sharpe,
                return_net=sharpe,
                max_drawdown=0.0,
            ),
            recorded_at=recorded_at,
        )
        self._ledger.append(record)  # type: ignore[attr-defined]


def _window(decisions: Sequence[ExperimentDecision], oos: slice) -> tuple[date, date]:
    """First and last ``knowledge_date`` of the out-of-sample window."""
    dates = sorted(d.request.knowledge_date for d in decisions[oos])
    return dates[0], dates[-1]
