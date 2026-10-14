"""Immutable value types for the backtest engine.

The engine is a pure domain component: every price it consumes arrives through
these types as plain arguments, never through I/O. Adapters are responsible for
translating their own row shapes into :class:`PriceBar`.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.types import Ticker

__all__ = [
    "BacktestResult",
    "PerformanceMetrics",
    "PriceBar",
    "RebalanceRecord",
]


class PriceBar(BaseModel):
    """One observation of a tradable price.

    ``volume`` is optional because some sources omit it; cost models that need
    average daily volume must handle its absence rather than assume a value.
    """

    model_config = ConfigDict(frozen=True)

    on: date
    close: float = Field(gt=0.0)
    volume: float | None = Field(default=None, ge=0.0)


class RebalanceRecord(BaseModel):
    """Audit trail for a single rebalance.

    ``signal_date`` is when the decision was made; ``execution_date`` is the
    strictly later bar at which it was filled. Keeping both lets a reader
    verify the one-bar gap that prevents same-bar look-ahead.
    """

    model_config = ConfigDict(frozen=True)

    signal_date: date
    execution_date: date
    selected: tuple[Ticker, ...]
    turnover: float = Field(ge=0.0)
    cost_fraction: float = Field(ge=0.0)


class PerformanceMetrics(BaseModel):
    """Summary statistics for one return stream."""

    model_config = ConfigDict(frozen=True)

    sharpe_annualized: float
    total_return: float
    cagr: float
    volatility_annualized: float = Field(ge=0.0)
    max_drawdown: float = Field(le=0.0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    n_periods: int = Field(ge=0)


class BacktestResult(BaseModel):
    """Outcome of a backtest, with gross and net inseparably paired.

    Both ``gross`` and ``net`` are required fields, so no caller can obtain or
    report a gross figure without the net one alongside it. This mirrors the
    guarantee already enforced by :class:`alphalab.domain.costs.GrossNetResult`
    and upholds the project-wide "gross never without net" invariant.
    """

    model_config = ConfigDict(frozen=True)

    gross: PerformanceMetrics
    net: PerformanceMetrics
    gross_returns: tuple[float, ...]
    net_returns: tuple[float, ...]
    equity_curve: tuple[float, ...]
    dates: tuple[date, ...]
    rebalances: tuple[RebalanceRecord, ...]
    total_cost_fraction: float = Field(ge=0.0)

    @property
    def mean_turnover(self) -> float:
        """Average traded notional per rebalance, as a fraction of equity."""
        if not self.rebalances:
            return 0.0
        return sum(r.turnover for r in self.rebalances) / len(self.rebalances)
