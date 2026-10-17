"""Discounted-cash-flow valuation operators.

Exposes two operators, each returning a fully-traced
:class:`~alphalab.domain.compute.provenance.ProvenancedValue`:

* :func:`wacc` — the weighted-average cost of capital, exposed as its own
  operator so the discount rate itself is traceable rather than an opaque
  input to the DCF.
* :func:`dcf_value_per_share` — projected free cash flows plus a Gordon
  terminal value, discounted at the WACC, divided by shares outstanding.

The terminal growth ``g`` must be strictly below the discount rate ``r``;
otherwise the geometric terminal series diverges and we raise
:class:`GrowthExceedsDiscountError` instead of emitting an absurd number.

Pure: numpy only, no I/O, no state.
"""

from __future__ import annotations

from collections.abc import Sequence

from alphalab.domain.compute.errors import (
    GrowthExceedsDiscountError,
    InvalidValuationInputError,
)
from alphalab.domain.compute.operators.multiples import ev_bridge
from alphalab.domain.compute.provenance import ProvenancedValue, combine
from alphalab.domain.compute.thresholds import MIN_GROWTH_DISCOUNT_SPREAD

_WACC_VERSION = "1.0.0"
_CAPM_VERSION = "1.0.0"
_DCF_VERSION = "1.0.0"


def cost_of_equity_capm(
    *,
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float,
) -> ProvenancedValue:
    """Required return on equity via the Capital Asset Pricing Model.

    ``Ke = Rf + beta * ERP``.

    Args:
        risk_free_rate: Risk-free yield (fraction, >= 0).
        beta: Equity sensitivity to market excess return (> 0).
        equity_risk_premium: Expected market excess return (fraction, > 0).

    Returns:
        A provenanced cost of equity in ``ratio`` units.
    """
    if risk_free_rate < 0.0:
        raise InvalidValuationInputError("risk_free_rate must be non-negative")
    if beta <= 0.0:
        raise InvalidValuationInputError("beta must be positive")
    if equity_risk_premium <= 0.0:
        raise InvalidValuationInputError("equity_risk_premium must be positive")

    value = risk_free_rate + beta * equity_risk_premium
    return combine(
        value=value,
        operator="cost_of_equity_capm",
        version=_CAPM_VERSION,
        unit="ratio",
        parents=(),
        extra_inputs={
            "risk_free_rate": risk_free_rate,
            "beta": beta,
            "equity_risk_premium": equity_risk_premium,
        },
    )


def wacc(
    *,
    equity_value: float,
    debt_value: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float,
) -> ProvenancedValue:
    """Weighted-average cost of capital.

    ``WACC = E/V * Re + D/V * Rd * (1 - Tc)`` where ``V = E + D``.

    Args:
        equity_value: Market value of equity (> 0).
        debt_value: Market value of debt (>= 0).
        cost_of_equity: Required return on equity (fraction).
        cost_of_debt: Pre-tax cost of debt (fraction).
        tax_rate: Marginal tax rate (fraction in [0, 1)).

    Returns:
        A provenanced WACC in ``ratio`` units.
    """
    if equity_value <= 0.0:
        raise InvalidValuationInputError("equity_value must be positive")
    if debt_value < 0.0:
        raise InvalidValuationInputError("debt_value must be non-negative")
    if not 0.0 <= tax_rate < 1.0:
        raise InvalidValuationInputError("tax_rate must be in [0, 1)")

    total = equity_value + debt_value
    equity_weight = equity_value / total
    debt_weight = debt_value / total
    value = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1.0 - tax_rate)

    return combine(
        value=value,
        operator="wacc",
        version=_WACC_VERSION,
        unit="ratio",
        parents=(),
        extra_inputs={
            "equity_value": equity_value,
            "debt_value": debt_value,
            "cost_of_equity": cost_of_equity,
            "cost_of_debt": cost_of_debt,
            "tax_rate": tax_rate,
        },
    )


def _check_growth_below_discount(growth: float, discount: float) -> None:
    """Guard the g < r precondition shared by DCF terminal value and DDM."""
    if discount - growth < MIN_GROWTH_DISCOUNT_SPREAD:
        raise GrowthExceedsDiscountError(
            f"terminal growth g={growth:.6f} must be below discount rate "
            f"r={discount:.6f} by at least {MIN_GROWTH_DISCOUNT_SPREAD:.6f}"
        )


def dcf_value_per_share(
    *,
    cash_flows: Sequence[float],
    discount_rate: ProvenancedValue,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float | None = None,
) -> ProvenancedValue:
    """Intrinsic value per share from projected free cash flows.

    Supports both consistent valuation modes:
    * **FCFE mode** (``net_debt=None``): ``cash_flows`` are levered Free Cash
      Flows to Equity (e.g. ``OCF - CapEx``) discounted at the cost of equity;
      the present value is already equity value and is divided directly by
      ``shares_outstanding`` without subtracting net debt.
    * **FCFF mode** (``net_debt`` supplied): ``cash_flows`` are unlevered Free
      Cash Flows to the Firm (``EBIT*(1-t) + D&A - CapEx``) discounted at WACC;
      the enterprise present value is bridged to equity via :func:`ev_bridge`
      (``EV - net_debt``) before dividing by ``shares_outstanding``.

    Args:
        cash_flows: Projected free cash flows, year 1..N (non-empty).
        discount_rate: Provenanced discount rate (``ratio``); becomes a parent
            so the discount rate is part of the derivation chain.
        terminal_growth: Perpetual growth rate ``g`` applied after year N.
        shares_outstanding: Diluted shares outstanding (> 0).
        net_debt: Optional net debt (``total_debt - cash``) for FCFF -> equity
            bridging. Keep ``None`` when discounting FCFE at cost of equity.

    Returns:
        A provenanced value per share in ``USD_per_share`` units.

    Raises:
        GrowthExceedsDiscountError: if ``g`` is not strictly below ``r``.
        InvalidValuationInputError: on empty flows or non-positive shares.
    """
    if len(cash_flows) == 0:
        raise InvalidValuationInputError("cash_flows must be non-empty")
    if shares_outstanding <= 0.0:
        raise InvalidValuationInputError("shares_outstanding must be positive")

    r = discount_rate.value
    _check_growth_below_discount(terminal_growth, r)

    pv_explicit = 0.0
    for year, cf in enumerate(cash_flows, start=1):
        pv_explicit += cf / (1.0 + r) ** year

    terminal_cf = cash_flows[-1] * (1.0 + terminal_growth)
    terminal_value = terminal_cf / (r - terminal_growth)
    pv_terminal = terminal_value / (1.0 + r) ** len(cash_flows)

    total_pv = pv_explicit + pv_terminal
    parents: tuple[ProvenancedValue, ...] = (discount_rate,)
    if net_debt is not None:
        equity_node = ev_bridge(enterprise_value=total_pv, net_debt=net_debt)
        equity_pv = equity_node.value
        parents = (discount_rate, equity_node)
    else:
        equity_pv = total_pv

    value_per_share = equity_pv / shares_outstanding

    extra = {f"cf_{i}": cf for i, cf in enumerate(cash_flows, start=1)}
    extra["terminal_growth"] = terminal_growth
    extra["shares_outstanding"] = shares_outstanding
    if net_debt is not None:
        extra["net_debt"] = net_debt

    return combine(
        value=value_per_share,
        operator="dcf_value_per_share",
        version=_DCF_VERSION,
        unit="USD_per_share",
        parents=parents,
        extra_inputs=extra,
    )


def reverse_dcf_implied_growth(
    *,
    current_price: float,
    base_cash_flow: float,
    discount_rate: float,
    terminal_growth: float,
    shares_outstanding: float,
    projection_years: int = 5,
    net_debt: float = 0.0,
) -> float | None:
    """Solve for the explicit-period annual growth rate implied by ``current_price``.

    Uses bisection over ``[-0.30, 0.50]`` to find ``g*`` such that a
    ``projection_years`` DCF starting from ``base_cash_flow`` equals the market
    equity value ``current_price * shares_outstanding``. Returns ``None`` if
    inputs are non-positive or ``terminal_growth`` violates the discount spread.
    """
    if (
        current_price <= 0.0
        or base_cash_flow <= 0.0
        or shares_outstanding <= 0.0
        or projection_years <= 0
        or discount_rate - terminal_growth < MIN_GROWTH_DISCOUNT_SPREAD
    ):
        return None

    target_pv = current_price * shares_outstanding + net_debt
    if target_pv <= 0.0:
        return None

    def _pv_at(g: float) -> float:
        cfs = [base_cash_flow * ((1.0 + g) ** yr) for yr in range(1, projection_years + 1)]
        pv_exp = sum(cf / ((1.0 + discount_rate) ** yr) for yr, cf in enumerate(cfs, start=1))
        tv = cfs[-1] * (1.0 + terminal_growth) / (discount_rate - terminal_growth)
        pv_tv = tv / ((1.0 + discount_rate) ** projection_years)
        return pv_exp + pv_tv

    low, high = -0.30, 0.50
    for _ in range(40):
        mid = 0.5 * (low + high)
        if _pv_at(mid) < target_pv:
            low = mid
        else:
            high = mid
    return round(0.5 * (low + high), 4)
