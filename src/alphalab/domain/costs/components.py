"""Pure transaction-cost component functions.

Every function here is a pure function of its arguments: no I/O, no global
state, no mutation. Costs are expressed either in **basis points** (bps,
1 bps = 0.01%) of notional or in absolute **currency** units, as noted per
function. The domain layer may use :mod:`numpy` and :mod:`scipy` because they
are pure computation, not I/O.

Design intent (TAREA 3): costs are a *constraint applied at every stage*,
never an ex-post filter. Each component below is deliberately small and
independently testable so it can be composed into gross/net results.
"""

from __future__ import annotations

from alphalab.domain.types import Side

#: One basis point as a fraction (1 bps = 0.0001).
_BPS = 1e-4


def _require_non_negative(name: str, value: float) -> None:
    """Raise ``ValueError`` if ``value`` is negative (helper for validation)."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def spread_cost(notional: float, half_spread_bps: float) -> float:
    """Cost of crossing the spread when *taking* liquidity.

    A liquidity taker pays the **half-spread**: buying lifts the ask, selling
    hits the bid, each a half-spread away from the mid price.

    Args:
        notional: Absolute traded notional in currency units (price * qty).
        half_spread_bps: Half of the quoted spread, in basis points.

    Returns:
        The spread cost in currency units (always >= 0).
    """
    _require_non_negative("notional", notional)
    _require_non_negative("half_spread_bps", half_spread_bps)
    return notional * half_spread_bps * _BPS


def slippage_cost(notional: float, slippage_bps: float) -> float:
    """Base per-order slippage, independent of size (in bps of notional).

    This captures the fixed deslizamiento of an order beyond the spread and
    the non-linear impact term, e.g. latency and discretisation effects.

    Args:
        notional: Absolute traded notional in currency units.
        slippage_bps: Base slippage in basis points.

    Returns:
        Slippage cost in currency units (always >= 0).
    """
    _require_non_negative("notional", notional)
    _require_non_negative("slippage_bps", slippage_bps)
    return notional * slippage_bps * _BPS


def market_impact_bps(
    quantity: float,
    adv: float,
    sigma: float,
    *,
    eta: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """Non-linear market impact in basis points (square-root law).

    Functional form::

        impact_bps = eta * sigma * (Q / ADV) ** alpha

    This is the **square-root law** of market impact (Almgren et al., 2005;
    Grinold & Kahn): temporary impact scales with the square root of the
    participation ratio ``Q / ADV``. With ``alpha < 1`` the *per-share* cost
    grows with size, so **doubling the order more than doubles the total
    impact cost** — the mechanism that caps strategy capacity.

    Args:
        quantity: Order size ``Q`` in shares (>= 0).
        adv: Average daily volume in shares (> 0).
        sigma: Daily volatility of the asset, expressed in bps.
        eta: Dimensionless impact coefficient (>= 0, default 1.0).
        alpha: Power-law exponent (default 0.5 = square root). ``0 < alpha``.

    Returns:
        Temporary market impact in basis points (always >= 0).
    """
    _require_non_negative("quantity", quantity)
    _require_non_negative("sigma", sigma)
    _require_non_negative("eta", eta)
    if adv <= 0:
        raise ValueError(f"adv must be positive, got {adv}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    participation = quantity / adv
    return float(eta * sigma * participation**alpha)


def market_impact_cost(
    price: float,
    quantity: float,
    adv: float,
    sigma: float,
    *,
    eta: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """Market-impact cost in currency units from the square-root law.

    Converts :func:`market_impact_bps` to currency by applying the impact (in
    bps) to the traded notional ``price * quantity``.

    Args:
        price: Execution reference price per share (>= 0).
        quantity: Order size in shares (>= 0).
        adv: Average daily volume in shares (> 0).
        sigma: Daily volatility in bps.
        eta: Impact coefficient (>= 0).
        alpha: Power-law exponent (> 0, default 0.5).

    Returns:
        Market-impact cost in currency units (always >= 0).
    """
    _require_non_negative("price", price)
    impact_bps = market_impact_bps(quantity, adv, sigma, eta=eta, alpha=alpha)
    notional = price * quantity
    return notional * impact_bps * _BPS


def commission_cost(
    shares: float,
    notional: float,
    *,
    per_share: float = 0.0,
    min_commission: float = 0.0,
    ad_valorem_bps: float = 0.0,
) -> float:
    """Broker commission with a per-order minimum plus an ad-valorem part.

    The commission is ``max(per_share * shares, min_commission)`` plus an
    optional ad-valorem component charged in bps of notional.

    Args:
        shares: Number of shares traded (>= 0).
        notional: Absolute traded notional in currency units (>= 0).
        per_share: Commission charged per share (>= 0).
        min_commission: Per-order floor applied to the per-share part (>= 0).
        ad_valorem_bps: Proportional fee in bps of notional (>= 0).

    Returns:
        Total commission in currency units (always >= 0).
    """
    _require_non_negative("shares", shares)
    _require_non_negative("notional", notional)
    _require_non_negative("per_share", per_share)
    _require_non_negative("min_commission", min_commission)
    _require_non_negative("ad_valorem_bps", ad_valorem_bps)
    per_share_part = max(per_share * shares, min_commission)
    ad_valorem_part = notional * ad_valorem_bps * _BPS
    return per_share_part + ad_valorem_part


def borrow_cost(
    side: Side,
    notional: float,
    holding_days: float,
    annual_rate: float,
    *,
    days_per_year: float = 365.0,
) -> float:
    """Short-borrow financing cost, prorated by holding period.

    Only applies to **short** positions (``Side.SELL``); a long position pays
    nothing here. The annual borrow rate is prorated linearly by the number of
    days the position is held.

    Args:
        side: Order/position side. Cost is charged only for ``Side.SELL``.
        notional: Absolute borrowed notional in currency units (>= 0).
        holding_days: Days the short is held (>= 0).
        annual_rate: Annualised borrow rate as a fraction (e.g. 0.05 = 5%/yr).
        days_per_year: Day-count denominator (> 0, default 365).

    Returns:
        Borrow cost in currency units (0 for longs, >= 0 for shorts).
    """
    _require_non_negative("notional", notional)
    _require_non_negative("holding_days", holding_days)
    _require_non_negative("annual_rate", annual_rate)
    if days_per_year <= 0:
        raise ValueError(f"days_per_year must be positive, got {days_per_year}")
    if side is Side.BUY:
        return 0.0
    return notional * annual_rate * (holding_days / days_per_year)
