"""Domain-level exceptions for the point-in-time layer.

These are pure signalling types: they carry no I/O and let the adapter and
application layers distinguish anti-leakage violations from ordinary errors.
"""

from __future__ import annotations


class PITError(Exception):
    """Base class for all point-in-time domain errors."""


class UnregisteredFeatureError(PITError):
    """Raised when a feature is requested that was never declared.

    Forcing every feature through an explicit registry makes an undeclared
    (and therefore unaudited) availability lag impossible to use by accident.
    """


class LeakageError(PITError):
    """Raised when data not yet publicly knowable would be surfaced.

    This is the structural tripwire used by the acid test: if any code path
    ever returns a record whose ``observed_at`` is after the active knowledge
    date, that is look-ahead leakage and must fail loudly.
    """
