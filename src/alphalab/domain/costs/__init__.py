"""Transaction cost models.

Public API for the pure transaction-cost domain. Costs are modelled as a
constraint applied at every stage (never an ex-post filter), and gross figures
are always paired with their net counterparts via :class:`GrossNetResult`.
"""

from __future__ import annotations

from alphalab.domain.costs.capacity import (
    StressedStop,
    capacity_curve,
    capacity_size,
    net_edge_bps,
    stressed_stop,
)
from alphalab.domain.costs.components import (
    borrow_cost,
    commission_cost,
    market_impact_bps,
    market_impact_cost,
    slippage_cost,
    spread_cost,
)
from alphalab.domain.costs.result import (
    CostBreakdown,
    CostSpec,
    GrossNetResult,
    cost_breakdown_for_trade,
    sharpe_gross_net,
)

__all__ = [
    "CostBreakdown",
    "CostSpec",
    "GrossNetResult",
    "StressedStop",
    "borrow_cost",
    "capacity_curve",
    "capacity_size",
    "commission_cost",
    "cost_breakdown_for_trade",
    "market_impact_bps",
    "market_impact_cost",
    "net_edge_bps",
    "sharpe_gross_net",
    "slippage_cost",
    "spread_cost",
    "stressed_stop",
]
