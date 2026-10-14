"""Value types for one paper-trading cycle: reconciliation and shadow benchmark.

Kept separate from the runner so the orchestration file stays small and these
plain, I/O-free dataclasses can be asserted on directly in tests.

Reconciliation
--------------
The place real systems break silently is the gap between what a strategy
*intended* to hold and what the account *actually* holds. A fill can be partial,
an order can be rejected downstream, a position can drift with price. So every
cycle compares the realised position weights against the target weights and
reports the per-name deviation. Making the gap visible is the whole point;
a cycle that only logged "orders sent" would hide exactly the failure that
matters.

Shadow benchmark
----------------
A strategy return means nothing without a reference. Each cycle also records
what a buy-and-hold of SPY with the same initial capital would be worth now,
*without sending any order*. It is a paper computation only, so the forward
comparison is honest from day one rather than reconstructed later.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphalab.domain.types import Ticker

__all__ = ["Reconciliation", "ReconciliationRow", "ShadowBenchmark"]


@dataclass(frozen=True)
class ReconciliationRow:
    """Target vs. actual weight for one name, with the signed gap."""

    ticker: Ticker
    target_weight: float
    actual_weight: float

    @property
    def deviation(self) -> float:
        """Actual minus target: positive means over-held, negative under-held."""
        return self.actual_weight - self.target_weight


@dataclass(frozen=True)
class Reconciliation:
    """The full set of target-vs-actual rows for one cycle."""

    rows: tuple[ReconciliationRow, ...]

    @property
    def max_abs_deviation(self) -> float:
        """Largest absolute gap across all names (0.0 when perfectly aligned)."""
        if not self.rows:
            return 0.0
        return max(abs(row.deviation) for row in self.rows)

    @property
    def is_aligned(self) -> bool:
        """Whether every name is within a half-percent of its target.

        A small tolerance avoids flagging the unavoidable drift between whole-
        share sizing and a fractional target weight as if it were a fault.
        """
        return self.max_abs_deviation <= 0.005


@dataclass(frozen=True)
class ShadowBenchmark:
    """A paper buy-and-hold of SPY, never routed to the broker.

    ``shares`` is fixed at inception from ``initial_capital`` and the entry
    price; ``current_value`` marks it to the latest price. No order is ever
    placed, so this is a pure yardstick for the strategy's forward record.
    """

    symbol: Ticker
    initial_capital: float
    entry_price: float
    latest_price: float

    @property
    def shares(self) -> float:
        """Shares the initial capital bought at the entry price."""
        if self.entry_price <= 0.0:
            return 0.0
        return self.initial_capital / self.entry_price

    @property
    def current_value(self) -> float:
        """Mark-to-market value of the held shares at the latest price."""
        return self.shares * self.latest_price

    @property
    def total_return(self) -> float:
        """Fractional return since inception."""
        if self.initial_capital <= 0.0:
            return 0.0
        return self.current_value / self.initial_capital - 1.0
