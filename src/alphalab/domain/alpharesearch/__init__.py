"""Pure domain package for the AlphaResearch 3-role weekly research committee."""

from __future__ import annotations

from alphalab.domain.alpharesearch.contracts import (
    AblationEvidence,
    AlphaResearchRunReport,
    CommitteeDecision,
    CommitteeDeliberationNote,
    DiscoverySignal,
    DiscoverySourceKind,
    EmpiricalAblationPanel,
    EmpiricalScorecard,
    EmpiricalTradeSample,
    ProposalDecisionStatus,
    ResearchProposal,
    SkepticCritique,
    TargetSubsystem,
    compute_proposal_id,
)
from alphalab.domain.alpharesearch.gate import (
    MAX_COMPLEXITY_LOC_DELTA,
    MAX_DRAWDOWN_DEGRADATION_TOLERANCE,
    MAX_TURNOVER_INCREASE_PCT,
    judge_proposal,
    run_empirical_gatekeeper,
    run_skeptic_review,
)
from alphalab.domain.alpharesearch.plan_renderer import render_weekly_plan_md

__all__ = [
    "MAX_COMPLEXITY_LOC_DELTA",
    "MAX_DRAWDOWN_DEGRADATION_TOLERANCE",
    "MAX_TURNOVER_INCREASE_PCT",
    "AblationEvidence",
    "AlphaResearchRunReport",
    "CommitteeDecision",
    "CommitteeDeliberationNote",
    "DiscoverySignal",
    "DiscoverySourceKind",
    "EmpiricalAblationPanel",
    "EmpiricalScorecard",
    "EmpiricalTradeSample",
    "ProposalDecisionStatus",
    "ResearchProposal",
    "SkepticCritique",
    "TargetSubsystem",
    "compute_proposal_id",
    "judge_proposal",
    "render_weekly_plan_md",
    "run_empirical_gatekeeper",
    "run_skeptic_review",
]
