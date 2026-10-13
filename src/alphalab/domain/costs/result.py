"""Immutable gross/net result structures and the cost specification.

The central design rule of TAREA 3.4: it must be **impossible to report a
gross figure without its net counterpart**. We enforce this at the type level
by never exposing a "gross-only" return. The only way to obtain a gross value
is through :class:`GrossNetResult`, which always carries the matching net
value and the per-component cost breakdown.

All structures are frozen Pydantic models (immutable); mutation raises
``ValidationError``. All functions are pure and use only numpy/scipy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.costs.components import (
    borrow_cost,
    commission_cost,
    market_impact_cost,
    slippage_cost,
    spread_cost,
)
from alphalab.domain.types import Side

_TRADING_DAYS_PER_YEAR = 252


class CostSpec(BaseModel):
    """Frozen specification of all transaction-cost parameters.

    Groups the knobs for every cost component so a single spec can be threaded
    through costing, capacity and Sharpe calculations consistently.
    """

    model_config = ConfigDict(frozen=True)

    half_spread_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    impact_eta: float = Field(default=1.0, ge=0.0)
    impact_alpha: float = Field(default=0.5, gt=0.0)
    per_share: float = Field(default=0.0, ge=0.0)
    min_commission: float = Field(default=0.0, ge=0.0)
    ad_valorem_bps: float = Field(default=0.0, ge=0.0)
    borrow_annual_rate: float = Field(default=0.0, ge=0.0)


class CostBreakdown(BaseModel):
    """Frozen per-component breakdown of a single trade's costs (currency).

    The :pyattr:`total` is the sum of all components and is validated to be
    non-negative on construction.
    """

    model_config = ConfigDict(frozen=True)

    spread: float = Field(ge=0.0)
    slippage: float = Field(ge=0.0)
    market_impact: float = Field(ge=0.0)
    commission: float = Field(ge=0.0)
    borrow: float = Field(ge=0.0)

    @property
    def total(self) -> float:
        """Total cost in currency units (sum of all components)."""
        return self.spread + self.slippage + self.market_impact + self.commission + self.borrow


class GrossNetResult(BaseModel):
    """Immutable pairing of a gross figure with its net counterpart.

    This type is the *only* public carrier of a gross value. Because ``net``
    and ``costs`` are required fields, no caller can obtain or report gross in
    isolation — satisfying the "gross never without net" invariant.
    """

    model_config = ConfigDict(frozen=True)

    gross: float
    net: float
    costs: CostBreakdown

    @property
    def total_cost(self) -> float:
        """Convenience accessor for the aggregate cost that was deducted."""
        return self.costs.total


def cost_breakdown_for_trade(
    *,
    side: Side,
    price: float,
    shares: float,
    adv: float,
    sigma_bps: float,
    holding_days: float,
    spec: CostSpec,
) -> CostBreakdown:
    """Compute the full per-component cost of one trade (currency units).

    Pure composition of the component functions in
    :mod:`alphalab.domain.costs.components` under a single :class:`CostSpec`.

    Args:
        side: Trade side (drives borrow cost for shorts).
        price: Execution reference price per share (>= 0).
        shares: Number of shares traded (>= 0).
        adv: Average daily volume in shares (> 0).
        sigma_bps: Daily volatility in bps.
        holding_days: Holding period in days (>= 0).
        spec: Cost specification.

    Returns:
        A frozen :class:`CostBreakdown`.
    """
    notional = price * shares
    return CostBreakdown(
        spread=spread_cost(notional, spec.half_spread_bps),
        slippage=slippage_cost(notional, spec.slippage_bps),
        market_impact=market_impact_cost(
            price, shares, adv, sigma_bps, eta=spec.impact_eta, alpha=spec.impact_alpha
        ),
        commission=commission_cost(
            shares,
            notional,
            per_share=spec.per_share,
            min_commission=spec.min_commission,
            ad_valorem_bps=spec.ad_valorem_bps,
        ),
        borrow=borrow_cost(side, notional, holding_days, spec.borrow_annual_rate),
    )


def _sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    """Annualised Sharpe ratio of a return series.

    A series with no meaningful dispersion has an undefined Sharpe; by
    convention we return ``0.0``. We compare the sample std against the scale
    of the series (not against exact zero) so that floating-point noise in a
    constant series does not blow the ratio up to a spurious huge value.
    """
    if returns.size < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    mean = float(np.mean(returns))
    scale = max(abs(mean), float(np.max(np.abs(returns))), 1.0)
    if std <= 1e-12 * scale:
        return 0.0
    return (mean / std) * float(np.sqrt(periods_per_year))


def sharpe_gross_net(
    gross_returns: Sequence[float],
    cost_fractions: Sequence[float],
    *,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> GrossNetResult:
    """Annualised gross and net Sharpe ratios in one immutable result.

    ``net_return[t] = gross_return[t] - cost_fraction[t]`` where both series
    are expressed as per-period return fractions. Returning both inside a
    :class:`GrossNetResult` makes it impossible to report gross Sharpe without
    the net Sharpe alongside it.

    Args:
        gross_returns: Per-period gross returns (fractions).
        cost_fractions: Per-period cost drag (fractions), same length.
        periods_per_year: Annualisation factor (default 252 trading days).

    Returns:
        A :class:`GrossNetResult` whose ``gross``/``net`` are Sharpe ratios and
        whose ``costs`` records the aggregate return drag as its slippage term.
    """
    gross = np.asarray(gross_returns, dtype=float)
    drag = np.asarray(cost_fractions, dtype=float)
    if gross.shape != drag.shape:
        raise ValueError("gross_returns and cost_fractions must have equal length")
    net = gross - drag
    gross_sharpe = _sharpe(gross, periods_per_year)
    net_sharpe = _sharpe(net, periods_per_year)
    total_drag = float(np.sum(np.abs(drag)))
    breakdown = CostBreakdown(
        spread=0.0,
        slippage=total_drag,
        market_impact=0.0,
        commission=0.0,
        borrow=0.0,
    )
    return GrossNetResult(gross=gross_sharpe, net=net_sharpe, costs=breakdown)
