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
    cost_of_equity_capm,
    dcf_value_per_share,
    ddm_value_per_share,
    ev_bridge,
    ev_ebitda_implied_value,
    ffo_implied_value,
    pb_implied_value,
    pe_implied_value,
    residual_income_value_per_share,
    reverse_dcf_implied_growth,
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
from alphalab.domain.compute.volatility import (
    GarchVolatilityForecast,
    forecast_garch11_volatility,
)

__all__ = [
    # Audit
    "AuditFinding",
    # Errors
    "ComputeError",
    "ConsensusRange",
    "GarchVolatilityForecast",
    "GrowthExceedsDiscountError",
    "InvalidValuationInputError",
    "ProvenanceDepthError",
    "ProvenanceError",
    # Provenance core
    "ProvenancedValue",
    "Severity",
    "combine",
    "consensus",
    "cost_of_equity_capm",
    "currency_caliber",
    "dcf_value_per_share",
    "ddm_value_per_share",
    "ev_bridge",
    "ev_bridge_check",
    "ev_ebitda_implied_value",
    "ffo_implied_value",
    "forecast_garch11_volatility",
    "leaf",
    "magnitude_sanity",
    "pb_implied_value",
    "pe_implied_value",
    "provenanced",
    "residual_income_value_per_share",
    "reverse_dcf_implied_growth",
    "sector_sign_check",
    "ttm_period_check",
    # Valuation operators
    "wacc",
]
