"""Capacity (edge-degradation) model and the stressed-stop cost.

Two pure helpers implementing the capacity side of TAREA 3:

* :func:`net_edge_bps` / :func:`capacity_curve` — how a fixed gross edge is
  eroded by size-dependent costs, and the size at which net edge crosses zero.
* :func:`stressed_stop` — the *real* expected cost of a triggered stop, which
  is a market order paying slippage and taker fees on both legs rather than
  the nominal trigger price (pattern from HyperGrok's risk sizing).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.costs.components import market_impact_bps

_BPS = 1e-4


class StressedStop(BaseModel):
    """Frozen result of a stress-adjusted stop calculation (currency/unit).

    ``risk_per_unit`` is the trigger distance *plus* the frictions a market
    order actually pays, so position sizing is done on what the stop will
    cost, not on its nominal trigger.
    """

    model_config = ConfigDict(frozen=True)

    nominal_distance: float = Field(ge=0.0)
    slippage: float = Field(ge=0.0)
    fees: float = Field(ge=0.0)
    risk_per_unit: float = Field(ge=0.0)


def net_edge_bps(
    gross_edge_bps: float,
    quantity: float,
    adv: float,
    sigma_bps: float,
    *,
    fixed_cost_bps: float = 0.0,
    eta: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """Net edge in bps after fixed costs and size-dependent market impact.

    ``net = gross_edge - fixed_cost - impact(quantity)`` where the impact term
    follows the square-root law. For ``alpha < 1`` the impact grows super-
    linearly in total cost, so net edge decreases monotonically with size and
    eventually turns negative — the capacity ceiling.

    Args:
        gross_edge_bps: Expected gross edge of the signal, in bps.
        quantity: Order size in shares (>= 0).
        adv: Average daily volume in shares (> 0).
        sigma_bps: Daily volatility in bps.
        fixed_cost_bps: Size-independent round-trip cost in bps (spread etc.).
        eta: Impact coefficient (>= 0).
        alpha: Power-law exponent (> 0, default 0.5).

    Returns:
        Net edge in bps (may be negative once costs exceed the edge).
    """
    impact = market_impact_bps(quantity, adv, sigma_bps, eta=eta, alpha=alpha)
    return gross_edge_bps - fixed_cost_bps - impact


def capacity_curve(
    gross_edge_bps: float,
    sizes: tuple[float, ...],
    adv: float,
    sigma_bps: float,
    *,
    fixed_cost_bps: float = 0.0,
    eta: float = 1.0,
    alpha: float = 0.5,
) -> tuple[tuple[float, float], ...]:
    """Edge-degradation curve: ``(size, net_edge_bps)`` for each size.

    Args:
        gross_edge_bps: Expected gross edge in bps.
        sizes: Candidate order sizes in shares.
        adv: Average daily volume in shares (> 0).
        sigma_bps: Daily volatility in bps.
        fixed_cost_bps: Size-independent round-trip cost in bps.
        eta: Impact coefficient (>= 0).
        alpha: Power-law exponent (> 0).

    Returns:
        A tuple of ``(size, net_edge_bps)`` pairs, one per input size.
    """
    return tuple(
        (
            size,
            net_edge_bps(
                gross_edge_bps,
                size,
                adv,
                sigma_bps,
                fixed_cost_bps=fixed_cost_bps,
                eta=eta,
                alpha=alpha,
            ),
        )
        for size in sizes
    )


def capacity_size(
    gross_edge_bps: float,
    adv: float,
    sigma_bps: float,
    *,
    fixed_cost_bps: float = 0.0,
    eta: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """Order size at which net edge hits exactly zero (the capacity ceiling).

    Solving ``gross - fixed - eta*sigma*(Q/ADV)**alpha = 0`` for ``Q`` gives::

        Q* = ADV * ((gross - fixed) / (eta * sigma)) ** (1 / alpha)

    Args:
        gross_edge_bps: Expected gross edge in bps.
        adv: Average daily volume in shares (> 0).
        sigma_bps: Daily volatility in bps (> 0).
        fixed_cost_bps: Size-independent round-trip cost in bps.
        eta: Impact coefficient (> 0).
        alpha: Power-law exponent (> 0).

    Returns:
        The break-even size ``Q*`` in shares. ``0.0`` if the fixed costs
        alone already exhaust the edge.

    Raises:
        ValueError: If ``adv``, ``sigma_bps``, ``eta`` or ``alpha`` are not
            strictly positive.
    """
    if adv <= 0:
        raise ValueError(f"adv must be positive, got {adv}")
    if sigma_bps <= 0:
        raise ValueError(f"sigma_bps must be positive, got {sigma_bps}")
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    budget = gross_edge_bps - fixed_cost_bps
    if budget <= 0:
        return 0.0
    ratio = budget / (eta * sigma_bps)
    return float(adv * ratio ** (1.0 / alpha))


def stressed_stop(
    trigger_distance: float,
    *,
    slippage_per_leg: float,
    taker_fee_per_leg: float,
) -> StressedStop:
    """Real expected cost of a triggered stop (a market order on both legs).

    A triggered stop does not execute at its nominal trigger price: it becomes
    a market order and pays slippage and taker fees on *both* the entry/exit
    legs. Sizing on the trigger distance alone understates risk, so this
    returns the stressed risk-per-unit instead.

    Reference example: ``100`` (trigger) ``+ 3.00`` (slippage) ``+ 2.65``
    (fees) ``= 105.65`` risk per unit.

    Args:
        trigger_distance: Nominal distance from entry to the stop trigger
            (currency per unit, >= 0).
        slippage_per_leg: Expected slippage per leg (currency per unit, >= 0).
            Charged on both legs.
        taker_fee_per_leg: Taker fee per leg (currency per unit, >= 0).
            Charged on both legs.

    Returns:
        A frozen :class:`StressedStop`. ``risk_per_unit`` is strictly greater
        than ``trigger_distance`` whenever any friction is positive.
    """
    if trigger_distance < 0:
        raise ValueError(f"trigger_distance must be non-negative, got {trigger_distance}")
    if slippage_per_leg < 0:
        raise ValueError(f"slippage_per_leg must be non-negative, got {slippage_per_leg}")
    if taker_fee_per_leg < 0:
        raise ValueError(f"taker_fee_per_leg must be non-negative, got {taker_fee_per_leg}")
    slippage = 2.0 * slippage_per_leg
    fees = 2.0 * taker_fee_per_leg
    risk = trigger_distance + slippage + fees
    return StressedStop(
        nominal_distance=trigger_distance,
        slippage=slippage,
        fees=fees,
        risk_per_unit=risk,
    )
