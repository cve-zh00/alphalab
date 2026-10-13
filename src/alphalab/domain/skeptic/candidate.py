"""Immutable contract types submitted to and returned by the skeptic.

:class:`TradeCandidate` is everything a trade/thesis puts on trial;
:class:`SkepticVerdict` is the auditable judgement. Both are **frozen** — a
candidate cannot be mutated mid-evaluation and a verdict cannot be doctored
after the fact. The verdict never carries a naked ``bool``: every rejection is
accompanied by the concrete criteria it violated, each legible and numeric,
plus the full list of structured findings (reusing the compute layer's
:class:`~alphalab.domain.compute.AuditFinding`).

Pure: numpy/pydantic only, no I/O, no global state.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.compute import AuditFinding
from alphalab.domain.costs import CostSpec
from alphalab.domain.types import Side, Ticker


class DataProvenance(BaseModel):
    """Immutable description of *how clean* the data behind a candidate is.

    These are the structural anti-leakage / anti-survivorship facts the
    skeptic cannot derive from returns alone and must be told explicitly. A
    candidate that lies here is out of scope: the skeptic judges what it is
    given, honestly declared.
    """

    model_config = ConfigDict(frozen=True)

    knowledge_date: date
    includes_delisted: bool
    #: True iff every feature was resolved strictly along the point-in-time
    #: path (no future information leaked into the information set).
    pit_compliant: bool = True
    #: The latest ``observed_at`` in the data lineage. When it is strictly
    #: after ``knowledge_date`` the candidate used data that was not yet
    #: knowable at decision time — a hard leakage violation.
    latest_observed_at: date | None = None


class TradeCandidate(BaseModel):
    """Immutable description of a trade/thesis submitted for judgement."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    ticker: Ticker
    side: Side
    proposed_size: float = Field(description="Order size in shares (may be degenerate).")
    horizon_days: int = Field(description="Intended holding horizon in days.")

    #: Per-period (non-annualised) backtest returns backing the thesis.
    backtest_returns: tuple[float, ...]
    #: Per-period cost drag (fractions) aligned with ``backtest_returns``.
    cost_fractions: tuple[float, ...]

    #: The *honest* trial count from the ledger (``N`` configurations tried).
    n_trials: int
    #: ``sqrt(Var(SR))`` across the tried configurations (for the DSR).
    trial_sharpe_std: float = 0.0

    cost_spec: CostSpec = Field(default_factory=CostSpec)
    #: Expected gross edge of the signal, in basis points.
    gross_edge_bps: float = 0.0
    #: Average daily volume in shares (for the capacity ceiling).
    adv: float = 0.0
    #: Daily volatility in bps (for the capacity ceiling).
    sigma_bps: float = 0.0

    provenance: DataProvenance

    #: Per-slice (asset or sub-window) per-period Sharpe ratios, used to test
    #: cross-asset / cross-period robustness. Empty when not supplied.
    per_slice_sharpe: tuple[float, ...] = ()

    #: Structured findings already attached to the candidate by the compute
    #: audit operators. A HARD finding here is a contradiction in the data.
    audit_findings: tuple[AuditFinding, ...] = ()


class SkepticVerdict(BaseModel):
    """Immutable, auditable judgement of a :class:`TradeCandidate`.

    ``accepted`` is ``True`` only when *no* rejection criterion fired. The
    full list of findings is always present (one per criterion evaluated, in a
    fixed order) so the Phase-3 A/B experiment can count *how many* and *which*
    criteria failed, not merely the first. ``rejection_criteria`` names each
    violated criterion with its numbers.
    """

    model_config = ConfigDict(frozen=True)

    accepted: bool
    findings: tuple[AuditFinding, ...]
    rejection_criteria: tuple[str, ...] = ()

    @property
    def n_criteria_failed(self) -> int:
        """Number of distinct criteria that triggered a rejection."""
        return len(self.rejection_criteria)
