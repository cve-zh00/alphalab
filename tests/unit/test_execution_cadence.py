"""Tests for the quarterly rebalance cadence gate.

The gate exists so that a daily scheduler does not silently turn a quarterly
strategy into a daily one. Trading every day would raise turnover roughly
sixtyfold and run a strategy with no measurement behind it.
"""

from __future__ import annotations

from datetime import date

from alphalab.domain.execution import is_rebalance_due, quarter_of


def test_never_traded_is_due() -> None:
    """A fresh account establishes its position immediately.

    Waiting for an arbitrary calendar boundary would leave new capital idle in
    cash for up to three months for no reason.
    """
    assert is_rebalance_due(last_rebalance=None, today=date(2026, 10, 14))


def test_same_quarter_is_not_due() -> None:
    """Having traded this quarter, the schedule calls for nothing today."""
    assert not is_rebalance_due(last_rebalance=date(2026, 10, 1), today=date(2026, 10, 14))
    assert not is_rebalance_due(last_rebalance=date(2026, 10, 14), today=date(2026, 12, 31))


def test_next_quarter_is_due() -> None:
    """Crossing into a new quarter triggers the rebalance."""
    assert is_rebalance_due(last_rebalance=date(2026, 12, 31), today=date(2027, 1, 1))


def test_year_boundary_is_handled() -> None:
    """Q4 to Q1 is a new quarter despite the quarter number decreasing."""
    assert quarter_of(date(2026, 12, 31)) == (2026, 4)
    assert quarter_of(date(2027, 1, 1)) == (2027, 1)
    assert is_rebalance_due(last_rebalance=date(2026, 10, 5), today=date(2027, 1, 5))


def test_late_wakeup_still_rebalances() -> None:
    """A machine asleep over the boundary rebalances when it wakes.

    Late and recorded as such is better than skipped silently.
    """
    assert is_rebalance_due(last_rebalance=date(2026, 7, 2), today=date(2026, 10, 20))


def test_future_last_rebalance_is_due() -> None:
    """An impossible ordering must not freeze the account indefinitely.

    A clock error that leaves ``last_rebalance`` in the future would otherwise
    block every future rebalance, and an indefinite freeze is invisible,
    whereas an extra rebalance is recoverable and logged.
    """
    assert is_rebalance_due(last_rebalance=date(2027, 1, 1), today=date(2026, 10, 14))


def test_all_quarter_boundaries() -> None:
    """Each calendar quarter maps to its own number."""
    assert quarter_of(date(2026, 1, 1)) == (2026, 1)
    assert quarter_of(date(2026, 3, 31)) == (2026, 1)
    assert quarter_of(date(2026, 4, 1)) == (2026, 2)
    assert quarter_of(date(2026, 6, 30)) == (2026, 2)
    assert quarter_of(date(2026, 7, 1)) == (2026, 3)
    assert quarter_of(date(2026, 9, 30)) == (2026, 3)
    assert quarter_of(date(2026, 10, 1)) == (2026, 4)
