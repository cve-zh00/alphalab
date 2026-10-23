"""Order planning with pre-trade risk limits.

The planner is the last point at which a bad number can be stopped before it
reaches a broker. It is pure by design: no network, no clock, no account
lookup, so every risk rule is exhaustively testable without an account.

Defaults are conservative on purpose — long-only, whole shares, capped single
position and single order. A limit that must be loosened deliberately is safe;
one that must be tightened after an incident is not.
"""

from __future__ import annotations

from alphalab.domain.execution.cadence import is_rebalance_due, quarter_of
from alphalab.domain.execution.nofx_guards import (
    AutopilotGuardConfig,
    GuardAction,
    GuardActionKind,
    GuardedTargets,
    PositionRiskSnapshot,
    apply_nofx_guards,
    effective_stop_loss_pct,
    should_drawdown_close,
)
from alphalab.domain.execution.planner import plan_trades
from alphalab.domain.execution.types import (
    Holding,
    OrderIntent,
    RejectedOrder,
    RiskLimits,
    TradePlan,
)

__all__ = [
    "AutopilotGuardConfig",
    "GuardAction",
    "GuardActionKind",
    "GuardedTargets",
    "Holding",
    "OrderIntent",
    "PositionRiskSnapshot",
    "RejectedOrder",
    "RiskLimits",
    "TradePlan",
    "apply_nofx_guards",
    "effective_stop_loss_pct",
    "is_rebalance_due",
    "plan_trades",
    "quarter_of",
    "should_drawdown_close",
]
