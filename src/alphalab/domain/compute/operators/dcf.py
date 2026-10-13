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
from alphalab.domain.compute.provenance import ProvenancedValue, combine
from alphalab.domain.compute.thresholds import MIN_GROWTH_DISCOUNT_SPREAD

_WACC_VERSION = "1.0.0"
_DCF_VERSION = "1.0.0"


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
) -> ProvenancedValue:
    """Intrinsic value per share from projected free cash flows.

    Discounts each projected free cash flow at ``r`` and adds a Gordon terminal
    value computed on the final year's cash flow, then divides the enterprise
    present value by shares outstanding.

    Args:
        cash_flows: Projected free cash flows, year 1..N (non-empty).
        discount_rate: Provenanced WACC (``ratio``); becomes the parent so the
            discount rate is part of the derivation chain.
        terminal_growth: Perpetual growth rate ``g`` applied after year N.
        shares_outstanding: Diluted shares outstanding (> 0).

    Returns:
        A provenanced value per share in ``USD_per_share`` units whose parents
        include the WACC node.

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

    enterprise_pv = pv_explicit + pv_terminal
    value_per_share = enterprise_pv / shares_outstanding

    extra = {f"cf_{i}": cf for i, cf in enumerate(cash_flows, start=1)}
    extra["terminal_growth"] = terminal_growth
    extra["shares_outstanding"] = shares_outstanding

    return combine(
        value=value_per_share,
        operator="dcf_value_per_share",
        version=_DCF_VERSION,
        unit="USD_per_share",
        parents=(discount_rate,),
        extra_inputs=extra,
    )
