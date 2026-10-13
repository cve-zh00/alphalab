"""Pure, traceable compute layer.

Thesis: *numbers are computed by code; judgement is supplied elsewhere.* Every
number this layer produces is a
:class:`~alphalab.domain.compute.provenance.ProvenancedValue` that can be
traced back to its raw-input leaves, and every coherence check returns a
structured :class:`~alphalab.domain.compute.operators.audit.findings.AuditFinding`
(never a bare bool).

The two types the Task 7 contract gate consumes — ``ProvenancedValue`` and
``AuditFinding`` (with its ``Severity``) — are re-exported here for a stable
import path.
"""

from __future__ import annotations

from alphalab.domain.compute.errors import (
    ComputeError,
    GrowthExceedsDiscountError,
    InvalidValuationInputError,
)
from alphalab.domain.compute.operators import (
    ConsensusRange,
    consensus,
    dcf_value_per_share,
    ddm_value_per_share,
    ev_bridge,
    ev_ebitda_implied_value,
    pb_implied_value,
    pe_implied_value,
    wacc,
)
from alphalab.domain.compute.operators.audit import (
    AuditFinding,
    Severity,
    currency_caliber,
    ev_bridge_check,
    magnitude_sanity,
    sector_sign_check,
    ttm_period_check,
)
from alphalab.domain.compute.provenance import (
    ProvenanceDepthError,
    ProvenancedValue,
    ProvenanceError,
    combine,
    leaf,
    provenanced,
)

__all__ = [
    # Audit
    "AuditFinding",
    # Errors
    "ComputeError",
    "ConsensusRange",
    "GrowthExceedsDiscountError",
    "InvalidValuationInputError",
    "ProvenanceDepthError",
    "ProvenanceError",
    # Provenance core
    "ProvenancedValue",
    "Severity",
    "combine",
    "consensus",
    "currency_caliber",
    "dcf_value_per_share",
    "ddm_value_per_share",
    "ev_bridge",
    "ev_bridge_check",
    "ev_ebitda_implied_value",
    "leaf",
    "magnitude_sanity",
    "pb_implied_value",
    "pe_implied_value",
    "provenanced",
    "sector_sign_check",
    "ttm_period_check",
    # Valuation operators
    "wacc",
]
