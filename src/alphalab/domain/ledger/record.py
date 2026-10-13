"""The :class:`TrialRecord` domain model and its result metrics.

A *trial* is a single backtest execution. The whole point of this ledger is
to count trials honestly: an autonomous agent inflates the trial count
silently, and that count is exactly what sets statistical significance via the
Deflated Sharpe Ratio. Under-counting trials makes the DSR certify noise, so a
``TrialRecord`` is immutable (``frozen``) and always carries *both* gross and
net metrics — you cannot record gross alone and quietly flatter a strategy by
hiding transaction costs.

``llm_version`` / ``llm_cutoff`` are first-class metadata, not an
afterthought. A frontier API updates itself silently and its knowledge cutoff
advances; recording them is what keeps the backtest reproducible and its
temporal hygiene auditable. We name them ``llm_*`` rather than ``model_*`` to
sidestep Pydantic v2's protected ``model_`` namespace cleanly, without
disabling the guard with ``protected_namespaces=()``.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alphalab.domain.ledger.hashing import spec_hash

__all__ = ["TrialMetrics", "TrialRecord", "new_trial_id"]


def new_trial_id() -> str:
    """Generate a fresh, globally-unique trial id.

    We use a random UUID4 rather than a content-derived id *on purpose*: a
    trial identifies an *execution*, not a specification. Two runs of the same
    strategy spec are two distinct trials and must each count toward the DSR's
    ``SR0``. A content hash would collapse re-runs into one id and silently
    under-count — the exact failure mode this ledger exists to prevent. Spec
    identity is tracked separately via ``spec_hash``.
    """
    return uuid.uuid4().hex


class TrialMetrics(BaseModel):
    """Resulting metrics of a backtest trial, gross and net of costs.

    Both the gross and the net Sharpe are required. Costs are the difference
    between a strategy that looks good on paper and one that survives
    execution; forcing both keeps that gap visible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sharpe_gross: float = Field(description="Sharpe ratio before costs.")
    sharpe_net: float = Field(description="Sharpe ratio after costs.")
    return_gross: float = Field(description="Total return before costs.")
    return_net: float = Field(description="Total return after costs.")
    max_drawdown: float = Field(
        le=0.0,
        description="Worst peak-to-trough drawdown as a non-positive fraction.",
    )


class TrialRecord(BaseModel):
    """An immutable record of a single backtest trial."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_id: str = Field(default_factory=new_trial_id)
    recorded_at: datetime
    spec_hash: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    family: str = Field(min_length=1, description="Strategy family; groups trials for the DSR.")
    window_start: date
    window_end: date
    params: Mapping[str, Any]
    metrics: TrialMetrics
    llm_version: str | None = Field(
        default=None,
        description="Version of the LLM that proposed the strategy, if any.",
    )
    llm_cutoff: date | None = Field(
        default=None,
        description="Knowledge cutoff of that LLM; guards temporal hygiene.",
    )

    @field_validator("window_end")
    @classmethod
    def _window_is_ordered(cls, value: date, info: Any) -> date:
        start = info.data.get("window_start")
        if start is not None and value < start:
            raise ValueError("window_end must not precede window_start")
        return value

    @classmethod
    def create(
        cls,
        *,
        spec: Mapping[str, Any],
        dataset_hash: str,
        family: str,
        window_start: date,
        window_end: date,
        params: Mapping[str, Any],
        metrics: TrialMetrics,
        recorded_at: datetime,
        llm_version: str | None = None,
        llm_cutoff: date | None = None,
    ) -> TrialRecord:
        """Build a record, deriving ``spec_hash`` canonically from ``spec``.

        This is the preferred constructor: it guarantees the spec hash is
        computed with the canonical hasher so equivalent specs always share a
        hash regardless of key ordering.
        """
        return cls(
            recorded_at=recorded_at,
            spec_hash=spec_hash(spec),
            dataset_hash=dataset_hash,
            family=family,
            window_start=window_start,
            window_end=window_end,
            params=dict(params),
            metrics=metrics,
            llm_version=llm_version,
            llm_cutoff=llm_cutoff,
        )
