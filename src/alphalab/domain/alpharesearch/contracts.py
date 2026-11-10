"""Immutable domain contracts for the AlphaResearch 3-role weekly committee.

The committee enforces a strict 3-role adversarial discernment pipeline before
any software or quantitative improvement is promoted into an executable
``plan.md``:

1. **Role 1 — ScoutProposer**: synthesises raw :class:`DiscoverySignal` items
   from local reference repositories, remote GitHub repositories, academic
   papers (arXiv / Semantic Scholar), and live portfolio postmortems into a
   falsifiable :class:`ResearchProposal` with explicit mathematical formulas
   and target files.
2. **Role 2 — SkepticReviewer**: red-teams every proposal without
   short-circuiting across 6 methodological failure modes (PIT look-ahead,
   LLM numeric boundary, graveyard repetition, transaction-cost/turnover drag,
   NOFX guard weakening, and Occam's Razor complexity penalty), producing an
   immutable :class:`SkepticCritique`.
3. **Role 3 — EmpiricalGatekeeper**: increments the honest trial count
   (``n_trials``) and evaluates empirical / invariant evidence via
   :func:`~alphalab.domain.stats.certification.certify_strategy` (DSR, hardened
   PBO, stationary block bootstrap CI) and paired Sharpe difference tests,
   producing an immutable :class:`EmpiricalScorecard`.

Pure domain module: no I/O, no filesystem, no network, no global state.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.compute import AuditFinding
from alphalab.domain.ledger.hashing import spec_hash

__all__ = [
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
]


class DiscoverySourceKind(StrEnum):
    """Origin category of a weekly research discovery signal."""

    LOCAL_REPO = "local_repo"
    REMOTE_REPO = "remote_repo"
    ACADEMIC_PAPER = "academic_paper"
    LIVE_POSTMORTEM = "live_postmortem"


class TargetSubsystem(StrEnum):
    """Alphalab subsystem targeted by a research proposal."""

    VALUATION = "valuation"
    PRESCREEN = "prescreen"
    SIZING = "sizing"
    RISK_GUARDS = "risk_guards"
    EXECUTION = "execution"
    CALIBRATION = "calibration"


class ProposalDecisionStatus(StrEnum):
    """Final committee disposition for a candidate research proposal."""

    PROMOTED_TO_PLAN = "promoted_to_plan"
    HOLD_FOR_EVIDENCE = "hold_for_evidence"
    REJECTED_BY_SKEPTIC = "rejected_by_skeptic"
    REJECTED_BY_EMPIRICAL_GATE = "rejected_by_empirical_gate"


class DiscoverySignal(BaseModel):
    """Immutable raw finding harvested by one of the four weekly scanners."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str = Field(min_length=1)
    source_kind: DiscoverySourceKind
    source_name: str = Field(min_length=1, description="Repo name, journal/arXiv category, or DB.")
    title: str = Field(min_length=1)
    reference_uri: str = Field(min_length=1, description="Local path, commit URL, or paper URL.")
    observed_at: date
    watermark_token: str = Field(
        min_length=1,
        description="Commit SHA, paper ID, or cycle timestamp for incremental deduplication.",
    )
    summary: str = Field(min_length=1)
    quantitative_mechanism: str = Field(
        default="",
        description="Key formula, algorithm, or empirical metric extracted from the source.",
    )
    suggested_subsystem: TargetSubsystem = TargetSubsystem.VALUATION


class AblationEvidence(BaseModel):
    """Empirical backtest or invariant ablation series attached to a proposal.

    When ``baseline_returns`` and ``challenger_returns`` are non-empty, the
    :class:`EmpiricalGatekeeper` runs full statistical certification (DSR,
    hardened PBO, and paired Politis-Romano block-bootstrap Sharpe difference).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_returns: tuple[float, ...] = ()
    challenger_returns: tuple[float, ...] = ()
    #: Flattened row-major ``T x M`` matrix of tried configurations (optional).
    configuration_matrix_rows: tuple[tuple[float, ...], ...] = ()
    cost_bps_assumed: float = Field(default=10.0, ge=0.0)
    turnover_delta_pct: float = 0.0
    invariants_verified: tuple[str, ...] = ()


def compute_proposal_id(*, subsystem: TargetSubsystem, title: str, formula: str) -> str:
    """Compute a canonical deterministic proposal hash for anti-snooping memory."""
    canonical = {
        "subsystem": str(subsystem),
        "title": " ".join(title.lower().split()),
        "formula": " ".join(formula.lower().split()),
    }
    return spec_hash(canonical)[:16]


class ResearchProposal(BaseModel):
    """Formal hypothesis produced by Role 1 (``ScoutProposer``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    subsystem: TargetSubsystem
    source_signals: tuple[DiscoverySignal, ...] = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    mathematical_formulation: str = Field(min_length=1)
    falsifiable_hypothesis: str = Field(min_length=1)
    affected_files: tuple[str, ...] = Field(min_length=1)
    implementation_todos: tuple[str, ...] = Field(min_length=1)
    verification_commands: tuple[str, ...] = ("pytest -q", "ruff check src tests")

    #: Explicit architectural & methodological declarations checked by Role 2.
    pit_safe: bool = True
    domain_pure: bool = True
    llm_numeric_boundary_safe: bool = True
    preserves_nofx_guards: bool = True
    estimated_loc_delta: int = Field(default=40)
    expected_sharpe_gain: float = Field(default=0.10)
    ablation_evidence: AblationEvidence = Field(default_factory=AblationEvidence)

    @classmethod
    def create(
        cls,
        *,
        title: str,
        subsystem: TargetSubsystem,
        source_signals: tuple[DiscoverySignal, ...],
        problem_statement: str,
        mathematical_formulation: str,
        falsifiable_hypothesis: str,
        affected_files: tuple[str, ...],
        implementation_todos: tuple[str, ...],
        verification_commands: tuple[str, ...] = ("pytest -q", "ruff check src tests"),
        pit_safe: bool = True,
        domain_pure: bool = True,
        llm_numeric_boundary_safe: bool = True,
        preserves_nofx_guards: bool = True,
        estimated_loc_delta: int = 40,
        expected_sharpe_gain: float = 0.10,
        ablation_evidence: AblationEvidence | None = None,
    ) -> ResearchProposal:
        """Build a proposal with a canonically derived ``proposal_id``."""
        pid = compute_proposal_id(
            subsystem=subsystem,
            title=title,
            formula=mathematical_formulation,
        )
        return cls(
            proposal_id=pid,
            title=title,
            subsystem=subsystem,
            source_signals=source_signals,
            problem_statement=problem_statement,
            mathematical_formulation=mathematical_formulation,
            falsifiable_hypothesis=falsifiable_hypothesis,
            affected_files=affected_files,
            implementation_todos=implementation_todos,
            verification_commands=verification_commands,
            pit_safe=pit_safe,
            domain_pure=domain_pure,
            llm_numeric_boundary_safe=llm_numeric_boundary_safe,
            preserves_nofx_guards=preserves_nofx_guards,
            estimated_loc_delta=estimated_loc_delta,
            expected_sharpe_gain=expected_sharpe_gain,
            ablation_evidence=ablation_evidence or AblationEvidence(),
        )


class CommitteeDeliberationNote(BaseModel):
    """Immutable LLM deliberation note for Role 1 (Scout) and Role 2 (Skeptic)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_index: int = Field(ge=0)
    scout_rationale: str = Field(min_length=1)
    skeptic_audit_note: str = Field(min_length=1)
    recommended_safeguard: str = Field(min_length=1)


class SkepticCritique(BaseModel):
    """Immutable verdict from Role 2 (``SkepticReviewer``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    findings: tuple[AuditFinding, ...]
    veto_reasons: tuple[str, ...] = ()
    required_safeguards: tuple[str, ...] = ()


class EmpiricalScorecard(BaseModel):
    """Immutable verdict from Role 3 (``EmpiricalGatekeeper``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: bool
    evaluation_mode: str = Field(
        description="'backtest_certified', 'structural_invariant', or 'insufficient_evidence'."
    )
    n_trials_total: int = Field(ge=1)
    trial_id: str | None = None
    observed_sharpe_baseline: float | None = None
    observed_sharpe_challenger: float | None = None
    sharpe_delta: float | None = None
    sharpe_diff_ci_lower: float | None = None
    sharpe_diff_ci_upper: float | None = None
    dsr: float | None = None
    pbo: float | None = None
    max_drawdown_baseline: float | None = None
    max_drawdown_challenger: float | None = None
    drawdown_delta: float | None = None
    rejection_reasons: tuple[str, ...] = ()


class CommitteeDecision(BaseModel):
    """Complete 3-role judgement for a single :class:`ResearchProposal`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal: ResearchProposal
    skeptic_critique: SkepticCritique
    empirical_scorecard: EmpiricalScorecard
    status: ProposalDecisionStatus
    summary_reason: str = Field(min_length=1)

    @property
    def promoted(self) -> bool:
        """Return True iff the proposal passed all 3 committee roles."""
        return self.status is ProposalDecisionStatus.PROMOTED_TO_PLAN


class AlphaResearchRunReport(BaseModel):
    """Immutable weekly report of the AlphaResearch committee."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    as_of: date
    started_at: datetime
    finished_at: datetime
    sources_queried: tuple[DiscoverySourceKind, ...]
    signals_discovered: tuple[DiscoverySignal, ...]
    decisions: tuple[CommitteeDecision, ...]
    n_trials_before: int = Field(ge=0)
    n_trials_after: int = Field(ge=0)
    plan_output_path: str | None = None

    @property
    def promoted_decisions(self) -> tuple[CommitteeDecision, ...]:
        """Return only proposals that survived the full discernment pipeline."""
        return tuple(d for d in self.decisions if d.promoted)

    @property
    def rejected_decisions(self) -> tuple[CommitteeDecision, ...]:
        """Return proposals vetoed by the Skeptic or Empirical Gatekeeper."""
        return tuple(d for d in self.decisions if not d.promoted)


class EmpiricalTradeSample(BaseModel):
    """Immutable Point-in-Time settled trade sample for empirical ablations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    knowledge_date: date
    probability: float = Field(ge=0.0, le=1.0)
    alpha_gross: float
    alpha_net: float
    accrual_ratio: float = 0.0
    fcf_roe: float = 0.0
    roa_operating: float = 0.0
    net_leverage: float = 0.0
    had_protective_stop_exit: bool = False


class EmpiricalAblationPanel(BaseModel):
    """Immutable cross-sectional cohort panel for out-of-sample ablations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: tuple[EmpiricalTradeSample, ...] = ()
    cohort_size: int = Field(default=6, ge=2)

    @property
    def n_cohorts(self) -> int:
        """Return the number of complete sequential cohorts in the panel."""
        return len(self.samples) // self.cohort_size

