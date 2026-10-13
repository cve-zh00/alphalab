"""A pure, explicit, versioned US equity-market holiday calendar.

Scope and honesty (read this before trusting it)
-------------------------------------------------
This is a **static, rule-based table**, not an official exchange calendar. It
computes the regular NYSE/Nasdaq full-day closes from first principles:

* fixed-date holidays (New Year, Juneteenth, Independence, Christmas) with the
  standard **weekend-observance** rule — a Saturday holiday is observed the
  preceding Friday, a Sunday holiday the following Monday;
* floating-Monday/Thursday holidays (MLK, Presidents, Memorial, Labor,
  Thanksgiving) by nth-weekday-of-month arithmetic;
* **Good Friday**, computed from the Gregorian (Western) Easter algorithm.

What it deliberately does **not** cover, and why you must not treat it as
authoritative:

* **Ad-hoc closures** — national days of mourning (e.g. a presidential
  funeral), weather closures (Hurricane Sandy, 2012), or any one-off the
  exchange declares. These have no rule and are not in this table.
* **Half-days / early closes** (the day after Thanksgiving, Christmas Eve).
  This table models full-day closes only; a half day is still a trading day
  here.
* **History before Juneteenth (2021)** — Juneteenth became a market holiday in
  2022; applying this table to earlier years marks a false holiday.

Keep this table **maintained**: it is a convenience for a laptop scheduler, not
a source of truth. When in doubt, defer to the exchange.

Pure module: stdlib ``datetime`` only. No I/O, no ``datetime.now()``.
"""

from __future__ import annotations

from datetime import date, timedelta

__all__ = [
    "CALENDAR_VERSION",
    "is_market_holiday",
    "is_trading_day",
    "market_holidays",
]

#: Version of the holiday ruleset. Bump when the rules change (e.g. a new
#: federal holiday is added) so a stored run can be attributed to the table
#: that produced its schedule decisions.
CALENDAR_VERSION = 1

# Weekday constants (Python's Monday=0 .. Sunday=6).
_SATURDAY = 5
_SUNDAY = 6


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the ``n``-th ``weekday`` (Mon=0) of ``month`` in ``year``.

    ``n`` is 1-based (``n=1`` is the first occurrence).
    """
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last ``weekday`` (Mon=0) of ``month`` in ``year``."""
    next_month_first = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = next_month_first - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def _observed(day: date) -> date:
    """Apply the exchange weekend-observance rule to a fixed-date holiday.

    Saturday rolls back to Friday; Sunday rolls forward to Monday.
    """
    if day.weekday() == _SATURDAY:
        return day - timedelta(days=1)
    if day.weekday() == _SUNDAY:
        return day + timedelta(days=1)
    return day


def _easter_sunday(year: int) -> date:
    """Return Western (Gregorian) Easter Sunday via the Anonymous algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    length = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * length) // 451
    month = (h + length - 7 * m + 114) // 31
    day = ((h + length - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _good_friday(year: int) -> date:
    """Good Friday is the Friday two days before Easter Sunday."""
    return _easter_sunday(year) - timedelta(days=2)


def market_holidays(year: int) -> frozenset[date]:
    """Return the set of full-day US market closes for ``year``.

    The result is the *observed* dates (weekend rule already applied), so a
    membership test answers "is the market closed on this calendar day".
    """
    holidays = {
        _observed(date(year, 1, 1)),  # New Year's Day
        _nth_weekday(year, 1, 0, 3),  # MLK Day — 3rd Monday of January
        _nth_weekday(year, 2, 0, 3),  # Presidents Day — 3rd Monday of February
        _good_friday(year),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day — last Monday of May
        _observed(date(year, 6, 19)),  # Juneteenth
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day — 1st Monday of September
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving — 4th Thursday of November
        _observed(date(year, 12, 25)),  # Christmas Day
    }
    return frozenset(holidays)


def is_market_holiday(day: date) -> bool:
    """Return True if ``day`` is a full-day US market close (observed)."""
    return day in market_holidays(day.year)


def is_trading_day(day: date) -> bool:
    """Return True if ``day`` is a regular trading day.

    A trading day is a weekday that is not a market holiday. Half-days are
    treated as trading days (this table models full-day closes only).
    """
    if day.weekday() >= _SATURDAY:
        return False
    return not is_market_holiday(day)
