"""Turning target weights into order intents, under pre-trade risk limits.

This is the last place where a bad number can be stopped before it reaches a
broker, so every rule here is enforced rather than advisory, and every refusal
is recorded rather than dropped.

The function is pure. It takes the portfolio as reported, the targets a strategy
asked for, and the limits in force, and returns what should be sent. No network,
no clock, no account lookup — which means the risk rules can be tested
exhaustively without touching an account.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from alphalab.domain.execution.types import (
    Holding,
    OrderIntent,
    RejectedOrder,
    RiskLimits,
    TradePlan,
)
from alphalab.domain.types import Side, Ticker

__all__ = ["plan_trades"]


def plan_trades(
    *,
    as_of: date,
    equity: float,
    holdings: Mapping[Ticker, Holding],
    targets: Mapping[Ticker, float],
    prices: Mapping[Ticker, float],
    limits: RiskLimits | None = None,
) -> TradePlan:
    """Compute the orders that move ``holdings`` toward ``targets``.

    ``targets`` are fractions of equity. A ticker absent from ``targets`` is
    treated as a target of zero, so dropping a name from a strategy's selection
    closes the position rather than leaving it stranded — the failure mode that
    quietly turns a rebalancing strategy into a buy-and-hold one.

    Raises:
        ValueError: if ``equity`` is not positive. Sizing against zero or
            negative equity is meaningless and must fail loudly rather than
            produce zero-share orders that look like a quiet no-op.
    """
    if equity <= 0.0:
        raise ValueError("equity must be positive to size orders")

    rules = limits or RiskLimits()
    rejected: list[RejectedOrder] = []
    capped = _cap_targets(targets, rules, rejected)
    capped = _cap_gross(capped, rules, rejected)

    intents: list[OrderIntent] = []
    for ticker in sorted(set(capped) | set(holdings)):
        intent = _intent_for(
            ticker=ticker,
            target_weight=capped.get(ticker, 0.0),
            holding=holdings.get(ticker),
            price=prices.get(ticker),
            equity=equity,
            rules=rules,
            rejected=rejected,
        )
        if intent is not None:
            intents.append(intent)

    return TradePlan(
        as_of=as_of,
        intents=tuple(intents),
        rejected=tuple(rejected),
        equity=equity,
    )


def _cap_targets(
    targets: Mapping[Ticker, float], rules: RiskLimits, rejected: list[RejectedOrder]
) -> dict[Ticker, float]:
    """Clip any single target to the position limit, recording the clip."""
    out: dict[Ticker, float] = {}
    for ticker, weight in targets.items():
        if weight < 0.0 and not rules.allow_short:
            rejected.append(
                RejectedOrder(
                    ticker=ticker,
                    reason=f"negative target {weight:.4f} but shorting is disabled",
                )
            )
            continue
        if abs(weight) > rules.max_position_weight:
            rejected.append(
                RejectedOrder(
                    ticker=ticker,
                    reason=(
                        f"target {weight:.4f} exceeds max_position_weight "
                        f"{rules.max_position_weight:.4f}; clipped"
                    ),
                )
            )
            weight = rules.max_position_weight * (1.0 if weight > 0 else -1.0)
        out[ticker] = weight
    return out


def _cap_gross(
    targets: Mapping[Ticker, float], rules: RiskLimits, rejected: list[RejectedOrder]
) -> dict[Ticker, float]:
    """Scale all targets down proportionally if gross exposure is exceeded.

    Scaling rather than dropping names keeps the strategy's intended
    composition intact; dropping would silently change which bets are held.
    """
    gross = sum(abs(w) for w in targets.values())
    if gross <= rules.max_gross_exposure or gross == 0.0:
        return dict(targets)

    scale = rules.max_gross_exposure / gross
    rejected.append(
        RejectedOrder(
            ticker=Ticker("*"),
            reason=(
                f"gross exposure {gross:.4f} exceeds "
                f"{rules.max_gross_exposure:.4f}; all targets scaled by {scale:.4f}"
            ),
        )
    )
    return {t: w * scale for t, w in targets.items()}


def _intent_for(
    *,
    ticker: Ticker,
    target_weight: float,
    holding: Holding | None,
    price: float | None,
    equity: float,
    rules: RiskLimits,
    rejected: list[RejectedOrder],
) -> OrderIntent | None:
    """Build one order, or ``None`` when no trade is warranted."""
    reference = price if price is not None else (holding.price if holding else None)
    if reference is None or reference <= 0.0:
        rejected.append(RejectedOrder(ticker=ticker, reason="no usable reference price"))
        return None

    current_shares = holding.shares if holding else 0.0
    target_shares = equity * target_weight / reference
    delta = target_shares - current_shares

    if not rules.allow_short and current_shares + delta < -1e-9:
        rejected.append(RejectedOrder(ticker=ticker, reason="order would open a short; blocked"))
        return None

    if rules.whole_shares_only:
        delta = float(int(delta))

    if abs(delta) < 1e-9:
        return None

    notional = abs(delta) * reference
    if notional < rules.min_order_weight * equity:
        return None  # dust: spread would cost more than the drift it corrects

    if notional > rules.max_order_weight * equity:
        allowed = rules.max_order_weight * equity / reference
        if rules.whole_shares_only:
            allowed = float(int(allowed))
        rejected.append(
            RejectedOrder(
                ticker=ticker,
                reason=(
                    f"order notional {notional:.2f} exceeds max_order_weight "
                    f"{rules.max_order_weight:.4f} of equity; clipped"
                ),
            )
        )
        if allowed < 1.0:
            return None
        delta = allowed * (1.0 if delta > 0 else -1.0)

    return OrderIntent(
        ticker=ticker,
        side=Side.BUY if delta > 0 else Side.SELL,
        shares=abs(delta),
        reference_price=reference,
    )
