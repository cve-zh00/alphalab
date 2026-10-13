"""The decisive three-arm A/B experiment (pure domain).

This package holds the **pure comparison logic** for the Phase-3 experiment
that puts the project's own additions on trial: does the debate (arm B) beat
the single analyst (arm A), and does the deterministic skeptic (arm C) earn
its complexity by reducing drawdown and vetoing edgeless trades?

Nothing here runs a pipeline or touches I/O — that is the runner's job
(:mod:`alphalab.app.ab_experiment`). This package only *measures*: given the
arms' return series and metrics it emits an immutable :class:`Comparison` with
explicit, honestly-qualified verdicts (improves / worsens / indistinguishable /
inconclusive), each backed by a Bonferroni-corrected paired-bootstrap CI of the
Sharpe *difference*.
"""

from __future__ import annotations

from alphalab.domain.experiment.arms import (
    ArmResult,
    DecisionOutcome,
    ExperimentArm,
)
from alphalab.domain.experiment.comparison import (
    Comparison,
    SkepticAnalysis,
    Verdict,
    analyse_skeptic,
    compare_arms,
)
from alphalab.domain.experiment.difference import (
    N_PAIRWISE_COMPARISONS,
    PairedDifference,
    paired_sharpe_difference,
)
from alphalab.domain.experiment.power import (
    MATERIAL_PERIOD_SHARPE_GAP,
    is_conclusive,
    minimum_detectable_sharpe_difference,
    sharpe_standard_error,
)

__all__ = [
    "MATERIAL_PERIOD_SHARPE_GAP",
    "N_PAIRWISE_COMPARISONS",
    "ArmResult",
    "Comparison",
    "DecisionOutcome",
    "ExperimentArm",
    "PairedDifference",
    "SkepticAnalysis",
    "Verdict",
    "analyse_skeptic",
    "compare_arms",
    "is_conclusive",
    "minimum_detectable_sharpe_difference",
    "paired_sharpe_difference",
    "sharpe_standard_error",
]
