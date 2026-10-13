"""Typed inputs to the analyst pipeline.

``ValuationBundle`` groups the already-provenanced method estimates computed by
``domain/compute``; the pipeline aggregates them into a consensus and injects
the resulting point as the artifact's headline. Keeping the bundle explicit
makes the pipeline's contract clear: it consumes provenanced numbers, it never
invents them.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.analyst.contracts import RunMetadata
from alphalab.domain.compute import ProvenancedValue


class ValuationBundle(BaseModel):
    """A set of provenanced per-method value-per-share estimates.

    Every estimate is a :class:`ProvenancedValue` produced by a compute
    operator (DCF/DDM/multiples). The pipeline runs ``consensus`` over them and
    injects the resulting point as the artifact's price target.

    Attributes:
        estimates: Non-empty method estimates, all in the same unit.
        unit: The shared unit (e.g. ``"USD_per_share"``).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    estimates: tuple[ProvenancedValue, ...] = Field(min_length=1)
    unit: str = Field(min_length=1)


class AnalystRequest(BaseModel):
    """Everything the pipeline needs to produce one research artifact.

    Attributes:
        ticker: The instrument under analysis.
        knowledge_date: Point-in-time knowledge date for the whole run.
        current_price: Reference current price per share.
        documents: Source texts handed to the extraction step.
        valuation: Provenanced method estimates to aggregate.
        run_metadata: Mandatory model/cutoff/dataset provenance for the run.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    ticker: str = Field(min_length=1)
    knowledge_date: date
    current_price: float
    documents: tuple[str, ...] = Field(min_length=1)
    valuation: ValuationBundle
    run_metadata: RunMetadata
