"""Translating traded notional into a drag on returns.

The engine knows how much it traded (turnover); it does not know spreads or
venue fees. That knowledge lives here, behind a protocol, so the engine stays
agnostic and a caller can swap in a harsher model without touching the
simulation loop.
"""

from __future__ import annotations

from typing import Protocol

from alphalab.domain.costs import CostSpec, market_impact_bps, slippage_cost, spread_cost

__all__ = ["ProportionalCostModel", "TurnoverCostModel"]

_BPS = 1e-4


class TurnoverCostModel(Protocol):
    """Maps traded notional to the fraction of equity it consumes."""

    def cost_fraction(self, *, turnover: float) -> float: ...


class ProportionalCostModel:
    """Costs proportional to traded notional, from a :class:`CostSpec`.

    ``turnover`` is the *total* traded notional as a fraction of equity, so a
    full switch from one holding to another counts 2.0 (one sell plus one buy)
    and is charged twice. Charging the full traded notional rather than a
    one-way convention keeps the estimate conservative.

    Per-share commissions are deliberately not applied: this model works in
    fraction-of-equity space and has no share counts, so inventing them would
    fabricate precision. Use ``ad_valorem_bps`` for value-proportional fees.

    When ``participation`` is supplied, square-root market impact is added on
    top via :func:`market_impact_bps`, modelling the extra cost of demanding
    liquidity. Left at ``None``, impact is omitted rather than guessed.
    """

    __slots__ = ("_impact_bps", "_spec")

    def __init__(
        self,
        spec: CostSpec,
        *,
        participation: float | None = None,
        sigma_bps: float | None = None,
    ) -> None:
        self._spec = spec
        self._impact_bps = _impact(spec, participation, sigma_bps)

    def cost_fraction(self, *, turnover: float) -> float:
        """Fraction of equity lost to trading ``turnover`` notional."""
        if turnover <= 0.0:
            return 0.0
        spread = spread_cost(turnover, self._spec.half_spread_bps)
        slippage = slippage_cost(turnover, self._spec.slippage_bps)
        ad_valorem = turnover * self._spec.ad_valorem_bps * _BPS
        impact = turnover * self._impact_bps * _BPS
        return spread + slippage + ad_valorem + impact


def _impact(spec: CostSpec, participation: float | None, sigma_bps: float | None) -> float:
    """Impact in basis points, or zero when its inputs are unavailable."""
    if participation is None or sigma_bps is None:
        return 0.0
    return market_impact_bps(
        participation,
        1.0,
        sigma_bps,
        eta=spec.impact_eta,
        alpha=spec.impact_alpha,
    )
