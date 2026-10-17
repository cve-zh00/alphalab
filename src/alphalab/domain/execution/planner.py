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
    fractionable: Mapping[Ticker, bool] | None = None,
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
    capped = _cap_targets(
        targets, rules, rejected, holdings=holdings, equity=equity, prices=prices
    )
    capped = _cap_gross(capped, rules, rejected)

    intents: list[OrderIntent] = []
    for ticker in sorted(set(capped) | set(holdings)):
        ticker_whole_only = rules.whole_shares_only or (
            fractionable is not None and not fractionable.get(ticker, True)
        )
        intent = _intent_for(
            ticker=ticker,
            target_weight=capped.get(ticker, 0.0),
            holding=holdings.get(ticker),
            price=prices.get(ticker),
            equity=equity,
            rules=rules,
            whole_shares_only=ticker_whole_only,
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
    targets: Mapping[Ticker, float],
    rules: RiskLimits,
    rejected: list[RejectedOrder],
    *,
    holdings: Mapping[Ticker, Holding] | None = None,
    equity: float | None = None,
    prices: Mapping[Ticker, float] | None = None,
) -> dict[Ticker, float]:
    """Clip any single target to the position limit, recording the clip.

    A held position that drifted above ``max_position_weight`` through
    appreciation may keep its current weight up to ``max_drift_weight``;
    increases are still capped at ``max_position_weight``.
    """
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
        cap = rules.max_position_weight
        current = _current_weight(ticker, holdings, equity, prices)
        if weight > 0.0 and current > cap:
            # Never force a sell below current weight, up to the drift ceiling.
            cap = max(cap, min(current, rules.max_drift_weight))
        if abs(weight) > cap + 1e-12:
            rejected.append(
                RejectedOrder(
                    ticker=ticker,
                    reason=(
                        f"target {weight:.4f} exceeds position limit "
                        f"{cap:.4f} (max_position_weight "
                        f"{rules.max_position_weight:.4f}, max_drift_weight "
                        f"{rules.max_drift_weight:.4f}); clipped"
                    ),
                )
            )
            weight = cap * (1.0 if weight > 0 else -1.0)
        out[ticker] = weight
    return out


def _current_weight(
    ticker: Ticker,
    holdings: Mapping[Ticker, Holding] | None,
    equity: float | None,
    prices: Mapping[Ticker, float] | None,
) -> float:
    """Current fraction of equity held in ``ticker`` (0.0 when unknown)."""
    if not holdings or not equity or equity <= 0.0:
        return 0.0
    holding = holdings.get(ticker)
    if holding is None or holding.shares <= 0.0:
        return 0.0
    price = (prices or {}).get(ticker, holding.price)
    if price is None or price <= 0.0:
        return 0.0
    return holding.shares * price / equity


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
    whole_shares_only: bool,
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

    if whole_shares_only:
        delta = float(int(delta))
    elif target_weight > 0.0:
        delta = round(delta, 4)
    else:
        # Full exit: keep exact position shares (up to 4 decimals) so no dust remains
        delta = -round(current_shares, 4)

    if abs(delta) < 1e-9:
        return None

    notional = abs(delta) * reference
    if notional < rules.min_order_weight * equity:
        return None  # dust: spread would cost more than the drift it corrects

    if notional > rules.max_order_weight * equity:
        allowed = rules.max_order_weight * equity / reference
        allowed = float(int(allowed)) if whole_shares_only else round(allowed, 4)
        rejected.append(
            RejectedOrder(
                ticker=ticker,
                reason=(
                    f"order notional {notional:.2f} exceeds max_order_weight "
                    f"{rules.max_order_weight:.4f} of equity; clipped"
                ),
            )
        )
        if (whole_shares_only and allowed < 1.0) or allowed <= 0.0:
            return None
        delta = allowed * (1.0 if delta > 0 else -1.0)

    return OrderIntent(
        ticker=ticker,
        side=Side.BUY if delta > 0 else Side.SELL,
        shares=abs(delta),
        reference_price=reference,
    )
