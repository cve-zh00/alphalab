"""Domain exceptions for the prediction journal.

The journal exists to make the system's track record *measurable* and
*honest*. These errors fire when something would corrupt that record - a
lesson that overflows its hard length budget, or an attempt to settle a
prediction whose horizon has not yet elapsed.
"""

from __future__ import annotations

__all__ = ["JournalError", "LessonTooLongError", "NotYetDueError"]


class JournalError(Exception):
    """Base class for all prediction-journal domain errors."""


class LessonTooLongError(JournalError):
    """Raised when a reflection exceeds the hard length budget.

    Settle-then-reflect deliberately caps lessons at a few prose sentences:
    inflating the context with a growing wall of text degrades the model that
    later reads it. A lesson that breaks the budget is rejected rather than
    silently truncated, so the discipline stays visible.
    """


class NotYetDueError(JournalError):
    """Raised when settling a prediction whose horizon has not yet elapsed.

    Liquidating early would record an outcome against an incomplete holding
    period and poison the calibration statistics.
    """
