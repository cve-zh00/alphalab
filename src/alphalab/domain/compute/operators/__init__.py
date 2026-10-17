"""Pure valuation operators, each returning a provenanced value."""

from __future__ import annotations

from alphalab.domain.compute.operators.consensus import ConsensusRange, consensus
from alphalab.domain.compute.operators.dcf import (
    cost_of_equity_capm,
    dcf_value_per_share,
    reverse_dcf_implied_growth,
    wacc,
)
from alphalab.domain.compute.operators.ddm import ddm_value_per_share
from alphalab.domain.compute.operators.multiples import (
    ev_bridge,
    ev_ebitda_implied_value,
    ffo_implied_value,
    pb_implied_value,
    pe_implied_value,
    residual_income_value_per_share,
)

__all__ = [
    "ConsensusRange",
    "consensus",
    "cost_of_equity_capm",
    "dcf_value_per_share",
    "ddm_value_per_share",
    "ev_bridge",
    "ev_ebitda_implied_value",
    "ffo_implied_value",
    "pb_implied_value",
    "pe_implied_value",
    "residual_income_value_per_share",
    "reverse_dcf_implied_growth",
    "wacc",
]
