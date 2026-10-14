"""The signal's window onto the past — and the structural anti-look-ahead gate.

Why this module exists
----------------------
A backtest can be made to look profitable by letting the signal peek at data it
could not have had. Documenting "do not look ahead" is not a control: the whole
premise of this project is that anti-leakage must be *structural*, because a
leakage oracle passes Deflated Sharpe and PBO cleanly.

So the engine never hands a signal the full price history. It hands it a
:class:`SignalContext` built by slicing each series at the decision date. The
future is not withheld by convention; it is **absent from the object**. A
signal that tries to read tomorrow's price finds nothing there, because the
engine never put it in scope.

Two independent guarantees stack here:

1. **Truncation** (this module): the context contains no bar after ``as_of``.
2. **Execution delay** (:mod:`alphalab.domain.backtest.engine`): fills happen at
   the first bar *strictly after* ``as_of``, so even a signal that perfectly
   knew the ``as_of`` close could not trade on it at that close.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Protocol

from alphalab.domain.backtest.types import PriceBar
from alphalab.domain.types import Ticker

__all__ = ["SignalContext", "SignalFunction"]


class SignalContext:
    """An immutable, past-only view of the universe at one decision date.

    Construction truncates every series at ``as_of`` inclusive. The truncated
    bars are not stored, so no accessor — present or future — can surface them.
    """

    __slots__ = ("_as_of", "_history")

    def __init__(
        self,
        *,
        as_of: date,
        prices: Mapping[Ticker, Sequence[PriceBar]],
    ) -> None:
        self._as_of = as_of
        history: dict[Ticker, tuple[PriceBar, ...]] = {}
        for ticker, bars in prices.items():
            visible = _truncate(bars, as_of)
            if visible:
                history[ticker] = visible
        self._history = history

    @property
    def as_of(self) -> date:
        """The decision date. No data after this date exists in this object."""
        return self._as_of

    def tickers(self) -> tuple[Ticker, ...]:
        """Tickers with at least one bar on or before :attr:`as_of`, sorted."""
        return tuple(sorted(self._history))

    def history(self, ticker: Ticker) -> tuple[PriceBar, ...]:
        """Bars for ``ticker`` up to and including :attr:`as_of`.

        Returns an empty tuple for an unknown ticker rather than raising, so a
        signal can probe the universe without special-casing new listings.
        """
        return self._history.get(ticker, ())

    def closes(self, ticker: Ticker) -> tuple[float, ...]:
        """Closing prices up to :attr:`as_of`, oldest first."""
        return tuple(bar.close for bar in self.history(ticker))

    def latest_close(self, ticker: Ticker) -> float | None:
        """Most recent close on or before :attr:`as_of`, or ``None``."""
        bars = self.history(ticker)
        return bars[-1].close if bars else None


def _truncate(bars: Sequence[PriceBar], as_of: date) -> tuple[PriceBar, ...]:
    """Keep only bars dated on or before ``as_of``.

    Assumes ``bars`` is sorted ascending by date, which the engine guarantees,
    and uses binary search so building a context stays cheap when the engine
    does it once per rebalance across a long history.
    """
    dates = [bar.on for bar in bars]
    cut = bisect_right(dates, as_of)
    return tuple(bars[:cut])


class SignalFunction(Protocol):
    """Scores the universe from a past-only context.

    Implementations return a score per ticker; higher means more attractive.
    Tickers may be omitted to express "no opinion" — the engine will not select
    them. The context is the *only* channel through which market data arrives,
    which is what makes the no-look-ahead guarantee structural.
    """

    def __call__(self, context: SignalContext) -> Mapping[Ticker, float]: ...
