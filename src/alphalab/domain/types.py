"""Core domain value types shared across the pure domain layer.

These primitives carry no behaviour that performs I/O and depend only on
the standard library, keeping the domain layer self-contained.
"""

from __future__ import annotations

from enum import Enum
from typing import NewType

#: Nominal type for an instrument symbol (e.g. ``Ticker("AAPL")``).
#: Using ``NewType`` gives static type checkers a distinct type without any
#: runtime overhead, preventing accidental mixing with plain ``str`` values.
Ticker = NewType("Ticker", str)


class Side(Enum):
    """Direction of an order or position."""

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """Return the signed multiplier for this side (+1 buy, -1 sell)."""
        return 1 if self is Side.BUY else -1

    def opposite(self) -> Side:
        """Return the opposing side."""
        return Side.SELL if self is Side.BUY else Side.BUY
