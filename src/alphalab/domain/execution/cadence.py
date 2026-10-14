"""Deciding when a rebalance is actually due.

Why a daily heartbeat must not mean daily trading
-------------------------------------------------
The strategy's out-of-sample behaviour was measured with **quarterly**
rebalancing. Scheduling a submit every day would run a different strategy:
turnover roughly sixty times higher, with costs to match, and no measurement
behind it. The backtest would no longer describe what the live account does,
which is the precise failure this project exists to avoid.

So the scheduler fires daily — frequent enough that a missed day is harmless
and reconciliation stays current — while this module decides whether today is
one of the days the measured strategy would have traded. Every other day the
runner reconciles and reports without sending anything.

The last rebalance date comes from the append-only execution log, so the
decision survives restarts, and a machine that was asleep on the quarter
boundary simply rebalances on the next day it wakes — late, but recorded as
such, rather than skipped silently.
"""

from __future__ import annotations

from datetime import date

__all__ = ["is_rebalance_due", "quarter_of"]


def quarter_of(day: date) -> tuple[int, int]:
    """Return ``(year, quarter)`` for ``day``, quarters numbered 1-4."""
    return (day.year, (day.month - 1) // 3 + 1)


def is_rebalance_due(*, last_rebalance: date | None, today: date) -> bool:
    """Whether the measured quarterly schedule calls for a rebalance today.

    ``None`` means the account has never traded, so the first invocation
    establishes the position — otherwise a fresh account would sit in cash
    until an arbitrary calendar boundary arrived.

    A ``last_rebalance`` dated in the future is treated as due. That ordering
    should be impossible, and refusing to act on it would leave an account
    frozen indefinitely because of a clock error; rebalancing is recoverable,
    an indefinite freeze is not visible.
    """
    if last_rebalance is None:
        return True
    if last_rebalance > today:
        return True
    return quarter_of(last_rebalance) != quarter_of(today)
