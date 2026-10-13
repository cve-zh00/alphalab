"""Immutable value types for the three-arm A/B experiment.

This module is **pure** (stdlib + pydantic/numpy only, no I/O, no adapters,
no app layer) so it stays legal under the ``domain`` purity guard. The runner
in :mod:`alphalab.app.ab_experiment` is responsible for driving the actual
pipelines and transaction-cost accounting; everything here only *describes* an
arm and *carries* the metrics that were measured, so the comparison logic can
be unit-tested with hand-verifiable numbers and never touches a pipeline.

Design notes
------------
* An arm is identified by a stable ``name`` ("A"/"B"/"C") and a human label.
* :class:`DecisionOutcome` is the atomic unit: one trade decision, whether it
  was *proposed* by the pipeline and whether it was *executed* (the skeptic in
  arm C can veto, so proposed and executed diverge), plus the per-period
  realised return it contributed. ``realized_return`` is the return the
  decision *would have* earned if executed; the arm's executed return series
  only includes decisions where ``executed is True``.
* :class:`ArmResult` is frozen and carries both gross and net Sharpe, max
  drawdown, hit rate, proposed/executed counts, the ``CertificationResult``
  and the calibration metrics. It is a pure data record — all numbers are
  computed upstream and injected.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.journal.calibration import CalibrationReport
from alphalab.domain.stats.certification import CertificationResult


class ExperimentArm(BaseModel):
    """Identity and configuration of one experiment arm.

    Attributes:
        name: Stable short identifier (e.g. ``"A"``, ``"B"``, ``"C"``).
        label: Human-readable description of the arm.
        family: Trial-ledger family string this arm records trials under.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    label: str = Field(min_length=1)
    family: str = Field(min_length=1)


class DecisionOutcome(BaseModel):
    """One trade decision and the return it did (or would have) earned.

    Attributes:
        proposed: Whether the pipeline proposed a trade at all.
        executed: Whether the trade actually executed (False iff vetoed).
        realized_return: Per-period realised return the decision contributed
            if executed. For a vetoed decision this is the return that was
            *avoided*; the skeptic analysis uses its sign to decide whether the
            veto helped (avoided a loss) or hurt (skipped a gain).
        predicted_prob: Model's pre-trade probability the trade is a winner,
            used for calibration. ``None`` when no probability was emitted.
        was_winner: Whether the decision's realised return was positive, used
            as the calibration outcome. ``None`` mirrors ``predicted_prob``.
    """

    model_config = ConfigDict(frozen=True)

    proposed: bool
    executed: bool
    realized_return: float
    predicted_prob: float | None = None
    was_winner: bool | None = None


class ArmResult(BaseModel):
    """Frozen metrics of a single arm over the out-of-sample window.

    Attributes:
        arm: The arm this result belongs to.
        sharpe_gross: Annualised gross Sharpe of the executed return series.
        sharpe_net: Annualised net (after-cost) Sharpe of the executed series.
        max_drawdown: Worst peak-to-trough drawdown (non-positive fraction).
        hit_rate: Fraction of executed decisions that were winners.
        n_proposed: Number of decisions the pipeline proposed.
        n_executed: Number of decisions that actually executed.
        executed_returns: The per-period net return series actually traded.
        certification: The honest-evaluation certification of this arm.
        calibration: Calibration metrics for this arm's predictions.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    arm: ExperimentArm
    sharpe_gross: float
    sharpe_net: float
    max_drawdown: float = Field(le=0.0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    n_proposed: int = Field(ge=0)
    n_executed: int = Field(ge=0)
    executed_returns: tuple[float, ...]
    certification: CertificationResult
    calibration: CalibrationReport

    @property
    def n_vetoed(self) -> int:
        """Number of proposed decisions that did not execute (vetoes)."""
        return self.n_proposed - self.n_executed
