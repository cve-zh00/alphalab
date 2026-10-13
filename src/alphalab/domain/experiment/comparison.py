"""The three-arm comparison: explicit verdicts, honestly qualified.

``compare_arms`` consumes the three measured :class:`ArmResult` objects plus the
per-decision outcomes of arm C (needed to judge the skeptic's vetoes) and emits
an immutable :class:`Comparison` that answers the user's three questions:

1. Does the **debate** (arm B) beat the **single agent** (arm A), net-of-costs
   and out-of-sample?
2. Does the **skeptic** (arm C) **reduce drawdown** versus the debate (arm B)?
3. Does the skeptic **avoid edgeless trades** — of the trades it vetoed, how
   many would have lost money net-of-costs?

Every Sharpe claim is backed by a paired-bootstrap CI of the *difference*
(:mod:`alphalab.domain.experiment.difference`) at a Bonferroni-corrected
confidence; a difference is a "win" only when that CI excludes zero. When it
does not, the verdict is explicitly "indistinguishable". When the sample is too
small to detect even a material effect (:mod:`alphalab.domain.experiment.power`),
the verdict is "inconclusive", which is a *different and stronger* caveat than
"indistinguishable".

This module is pure: numpy/scipy/pydantic only, no I/O, no pipelines.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.experiment.arms import ArmResult, DecisionOutcome
from alphalab.domain.experiment.difference import (
    N_PAIRWISE_COMPARISONS,
    PairedDifference,
    paired_sharpe_difference,
)
from alphalab.domain.experiment.power import (
    is_conclusive,
    minimum_detectable_sharpe_difference,
)


class Verdict(StrEnum):
    """Possible outcomes of a pairwise or feature verdict."""

    #: The challenger arm is better with a CI excluding zero.
    IMPROVES = "improves"
    #: The challenger arm is worse with a CI excluding zero.
    WORSENS = "worsens"
    #: CI includes zero but the sample could have detected a material effect.
    INDISTINGUISHABLE = "indistinguishable"
    #: CI includes zero and the sample is too small to detect a material effect.
    INCONCLUSIVE = "inconclusive"


class SkepticAnalysis(BaseModel):
    """Did the skeptic's vetoes help? Measured, not assumed.

    Attributes:
        n_vetoed: Number of proposed trades the skeptic vetoed in arm C.
        n_vetoed_losers: Of those, how many had a negative realised return
            (losses correctly avoided).
        n_vetoed_winners: Of those, how many had a positive realised return
            (gains wrongly skipped).
        avoided_return: Net realised return summed over all vetoed trades. A
            *negative* number means the vetoed trades, taken together, would
            have lost money — the skeptic added value. A positive number means
            it skipped net-profitable trades.
        drawdown_verdict: Whether arm C reduced drawdown vs arm B.
        drawdown_delta: ``max_drawdown(C) - max_drawdown(B)``. Both drawdowns
            are non-positive; a *positive* delta means C's drawdown is shallower
            (less negative) — an improvement.
    """

    model_config = ConfigDict(frozen=True)

    n_vetoed: int = Field(ge=0)
    n_vetoed_losers: int = Field(ge=0)
    n_vetoed_winners: int = Field(ge=0)
    avoided_return: float
    drawdown_verdict: Verdict
    drawdown_delta: float

    @property
    def avoids_edgeless_trades(self) -> bool:
        """True iff the vetoed trades would, net, have lost money."""
        return self.n_vetoed > 0 and self.avoided_return < 0.0


class Comparison(BaseModel):
    """Immutable, auditable comparison of the three arms with verdicts.

    Attributes:
        arm_a: Baseline (single agent) result.
        arm_b: Debate result.
        arm_c: Debate + skeptic result.
        debate_vs_single: Paired Sharpe-difference test B - A.
        skeptic_vs_debate: Paired Sharpe-difference test C - B.
        single_vs_skeptic: Paired Sharpe-difference test A - C (completes the
            family of three; recorded for the multiple-comparison accounting).
        debate_verdict: Verdict for question 1 (does debate improve?).
        skeptic_sharpe_verdict: Verdict for the skeptic on Sharpe terms.
        skeptic_analysis: Drawdown and edgeless-trade analysis (questions 2-3).
        n_observations: Per-arm out-of-sample observation count.
        min_detectable_sharpe_diff: Approximate MDE at the corrected confidence.
        conclusive: Whether the sample could detect a material effect at all.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    arm_a: ArmResult
    arm_b: ArmResult
    arm_c: ArmResult
    debate_vs_single: PairedDifference
    skeptic_vs_debate: PairedDifference
    single_vs_skeptic: PairedDifference
    debate_verdict: Verdict
    skeptic_sharpe_verdict: Verdict
    skeptic_analysis: SkepticAnalysis
    n_observations: int = Field(ge=2)
    min_detectable_sharpe_diff: float
    conclusive: bool


def _verdict_from(diff: PairedDifference, *, conclusive: bool) -> Verdict:
    """Map a paired difference (challenger - baseline) to a verdict.

    ``diff.name_a`` is the challenger and ``diff.name_b`` the baseline, so a
    positive ``delta`` with a CI excluding zero means the challenger improves.
    """
    if diff.excludes_zero:
        return Verdict.IMPROVES if diff.delta > 0.0 else Verdict.WORSENS
    return Verdict.INDISTINGUISHABLE if conclusive else Verdict.INCONCLUSIVE


def _max_drawdown(returns: tuple[float, ...]) -> float:
    """Worst peak-to-trough drawdown of a compounded return series (<= 0)."""
    if not returns:
        return 0.0
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    return float(drawdowns.min())


def analyse_skeptic(
    arm_b: ArmResult,
    arm_c: ArmResult,
    *,
    arm_c_outcomes: tuple[DecisionOutcome, ...],
) -> SkepticAnalysis:
    """Measure whether the skeptic's vetoes reduced drawdown and avoided losers.

    Args:
        arm_b: The debate arm (skeptic's drawdown baseline).
        arm_c: The debate + skeptic arm.
        arm_c_outcomes: Every arm-C decision, so vetoes (proposed and not
            executed) can be scored by the return they avoided.
    """
    vetoed = [o for o in arm_c_outcomes if o.proposed and not o.executed]
    n_vetoed_losers = sum(1 for o in vetoed if o.realized_return < 0.0)
    n_vetoed_winners = sum(1 for o in vetoed if o.realized_return > 0.0)
    avoided_return = float(sum(o.realized_return for o in vetoed))
    drawdown_delta = arm_c.max_drawdown - arm_b.max_drawdown
    if drawdown_delta > 0.0:
        drawdown_verdict = Verdict.IMPROVES
    elif drawdown_delta < 0.0:
        drawdown_verdict = Verdict.WORSENS
    else:
        drawdown_verdict = Verdict.INDISTINGUISHABLE
    return SkepticAnalysis(
        n_vetoed=len(vetoed),
        n_vetoed_losers=n_vetoed_losers,
        n_vetoed_winners=n_vetoed_winners,
        avoided_return=avoided_return,
        drawdown_verdict=drawdown_verdict,
        drawdown_delta=drawdown_delta,
    )


def compare_arms(
    arm_a: ArmResult,
    arm_b: ArmResult,
    arm_c: ArmResult,
    *,
    arm_c_outcomes: tuple[DecisionOutcome, ...],
    rng: np.random.Generator,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    power: float = 0.80,
) -> Comparison:
    """Compare the three arms and emit the honest verdicts.

    All three pairwise Sharpe-difference tests share the Bonferroni family size
    (:data:`N_PAIRWISE_COMPARISONS`) so the multiple-comparison correction is
    applied uniformly. The ``rng`` is split deterministically per comparison so
    the result is reproducible yet each test uses an independent resample draw.

    Args:
        arm_a: Baseline single-agent result.
        arm_b: Debate result.
        arm_c: Debate + skeptic result.
        arm_c_outcomes: Per-decision outcomes of arm C (for the veto analysis).
        rng: Seeded generator (reproducibility is mandatory).
        alpha: Family-wise significance level before Bonferroni correction.
        n_resamples: Bootstrap resamples per pairwise test.
        power: Target power for the detectable-effect analysis.

    Raises:
        ValueError: If the three arms' executed series differ in length (the
            arms must have been run over the same decisions).
    """
    series = (arm_a.executed_returns, arm_b.executed_returns, arm_c.executed_returns)
    lengths = {len(s) for s in series}
    if len(lengths) != 1:
        raise ValueError("arms must share the same number of executed observations")
    n = lengths.pop()

    seeds = rng.integers(0, 2**32 - 1, size=3)

    debate_vs_single = paired_sharpe_difference(
        arm_b.executed_returns,
        arm_a.executed_returns,
        name_a=arm_b.arm.name,
        name_b=arm_a.arm.name,
        rng=np.random.default_rng(int(seeds[0])),
        alpha=alpha,
        n_comparisons=N_PAIRWISE_COMPARISONS,
        n_resamples=n_resamples,
    )
    skeptic_vs_debate = paired_sharpe_difference(
        arm_c.executed_returns,
        arm_b.executed_returns,
        name_a=arm_c.arm.name,
        name_b=arm_b.arm.name,
        rng=np.random.default_rng(int(seeds[1])),
        alpha=alpha,
        n_comparisons=N_PAIRWISE_COMPARISONS,
        n_resamples=n_resamples,
    )
    single_vs_skeptic = paired_sharpe_difference(
        arm_a.executed_returns,
        arm_c.executed_returns,
        name_a=arm_a.arm.name,
        name_b=arm_c.arm.name,
        rng=np.random.default_rng(int(seeds[2])),
        alpha=alpha,
        n_comparisons=N_PAIRWISE_COMPARISONS,
        n_resamples=n_resamples,
    )

    confidence = debate_vs_single.confidence
    conclusive = is_conclusive(
        arm_a.certification.observed_sharpe, n, confidence=confidence, power=power
    )
    mde = minimum_detectable_sharpe_difference(
        arm_a.certification.observed_sharpe, n, confidence=confidence, power=power
    )

    return Comparison(
        arm_a=arm_a,
        arm_b=arm_b,
        arm_c=arm_c,
        debate_vs_single=debate_vs_single,
        skeptic_vs_debate=skeptic_vs_debate,
        single_vs_skeptic=single_vs_skeptic,
        debate_verdict=_verdict_from(debate_vs_single, conclusive=conclusive),
        skeptic_sharpe_verdict=_verdict_from(skeptic_vs_debate, conclusive=conclusive),
        skeptic_analysis=analyse_skeptic(arm_b, arm_c, arm_c_outcomes=arm_c_outcomes),
        n_observations=n,
        min_detectable_sharpe_diff=mde,
        conclusive=conclusive,
    )
