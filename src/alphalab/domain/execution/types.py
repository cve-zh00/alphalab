"""Value types for order planning.

The planner is pure: it receives the current portfolio and a set of target
weights, and returns intents. It never talks to a broker, so every risk rule
below is testable without a network and without an account.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.types import Side, Ticker

__all__ = [
    "Holding",
    "OrderIntent",
    "RejectedOrder",
    "RiskLimits",
    "TradePlan",
]


class Holding(BaseModel):
    """A position currently held, as reported by the broker."""

    model_config = ConfigDict(frozen=True)

    ticker: Ticker
    shares: float
    price: float = Field(gt=0.0)

    @property
    def market_value(self) -> float:
        """Current value of the position, signed with the share count."""
        return self.shares * self.price


class RiskLimits(BaseModel):
    """Pre-trade constraints applied to every plan.

    These are deliberately conservative defaults rather than permissive ones.
    A limit that has to be loosened on purpose is safe; a limit that has to be
    tightened after an incident is not.
    """

    model_config = ConfigDict(frozen=True)

    #: Largest fraction of equity any single position may reach.
    #:
    #: 0.10 leaves headroom above the 5% target weight of a twenty-name
    #: portfolio, so ordinary price drift never trips the limit, while still
    #: blocking a sizing error that would concentrate the book. Measured on the
    #: de-biased universe, concentration is the dominant risk: five names drew
    #: down 72.1% where twenty drew down 43.9%.
    max_position_weight: float = Field(default=0.10, gt=0.0, le=1.0)

    #: Ceiling for already-held positions that grew via appreciation. New buys
    #: and increases are still capped by ``max_position_weight``; a held
    #: position above that limit is only trimmed once it exceeds this ceiling.
    max_drift_weight: float = Field(default=0.12, gt=0.0, le=1.0)

    #: Tolerance band for held positions: when the current weight is within
    #: this fraction of the target weight, no rebalancing trade is made. Avoids
    #: churn from relative-sizing noise. Exits (target 0) are never suppressed.
    rebalance_band: float = Field(default=0.30, ge=0.0, lt=1.0)

    #: Largest notional a single order may carry, as a fraction of equity.
    #: Caps the damage a mispriced or stale signal can do in one step.
    max_order_weight: float = Field(default=0.25, gt=0.0, le=1.0)

    #: Ceiling on the sum of absolute position weights.
    max_gross_exposure: float = Field(default=1.0, gt=0.0)

    #: Orders below this fraction of equity are dropped as dust: their
    #: commission and spread outweigh the tracking error they fix.
    min_order_weight: float = Field(default=0.002, ge=0.0)

    #: Shorting is off by default. The research in this project is long-only,
    #: and a sign error that silently opens shorts is unbounded risk.
    allow_short: bool = False

    #: Fractional shares complicate order types and venue rules; whole shares
    #: keep execution simple and predictable.
    whole_shares_only: bool = True


class OrderIntent(BaseModel):
    """An order the planner believes should be sent."""

    model_config = ConfigDict(frozen=True)

    ticker: Ticker
    side: Side
    shares: float = Field(gt=0.0)
    reference_price: float = Field(gt=0.0)

    @property
    def notional(self) -> float:
        """Approximate value of the order at the reference price."""
        return self.shares * self.reference_price

    def client_order_id(self, *, strategy: str, as_of: date) -> str:
        """Deterministic broker-side identifier for idempotency.

        Built from strategy, date, ticker and side so that re-running the same
        rebalance produces the *same* id. Brokers reject duplicates, which
        turns an accidental double invocation into a no-op instead of double
        exposure. This is the cheapest protection against the most common
        operational failure in automated trading.
        """
        return f"{strategy}-{as_of.isoformat()}-{self.ticker}-{self.side.value}"


class RejectedOrder(BaseModel):
    """An intent that a risk rule refused, with the reason recorded."""

    model_config = ConfigDict(frozen=True)

    ticker: Ticker
    reason: str = Field(min_length=1)


class TradePlan(BaseModel):
    """Orders to send, alongside what was refused and why.

    Rejections are part of the plan rather than discarded, so an operator can
    see that the planner declined to act and on what grounds. A plan that
    silently dropped orders would be indistinguishable from one that had
    nothing to do.
    """

    model_config = ConfigDict(frozen=True)

    as_of: date
    intents: tuple[OrderIntent, ...]
    rejected: tuple[RejectedOrder, ...]
    equity: float = Field(gt=0.0)

    @property
    def gross_notional(self) -> float:
        """Total value to be traded if every intent fills."""
        return sum(intent.notional for intent in self.intents)

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to send."""
        return not self.intents
