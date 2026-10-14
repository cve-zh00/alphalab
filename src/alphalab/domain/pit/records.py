"""Immutable point-in-time records and the knowledge-date wrapper.

The two timestamps on :class:`PITRecord` are the heart of the anti-leakage
design. ``as_of`` is the period the datum describes; ``observed_at`` is when
it first became publicly knowable. The validator makes it impossible to build
a record that was knowable before the period it refers to.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, ConfigDict, model_validator

from alphalab.domain.types import Ticker


@dataclass(frozen=True, order=True)
class KnowledgeDate:
    """Explicit wrapper around the instant of knowledge.

    Wrapping the date in a nominal type stops a plain ``date`` from being
    passed by accident into a positional slot that expects "what is knowable
    now", which is exactly the kind of silent mistake that causes look-ahead.
    The wrapper is frozen and ordered so knowledge dates compare naturally.
    """

    value: date


class PITRecord(BaseModel):
    """A single, immutable point-in-time observation of one field.

    Attributes:
        ticker: Instrument the datum belongs to.
        field: Name of the observed field (e.g. ``"revenue"``).
        value: Numeric value, stored as-filed (never reconstructed).
        as_of: Date the datum refers to (e.g. fiscal quarter end).
        observed_at: Date the datum became publicly knowable (e.g. filing).
        source: Optional provenance tag (e.g. ``"SEC-EDGAR"``).
        period_start: For a *flow* measured over a span (revenue, cash flow,
            earnings per share), the first day of that span; ``as_of`` is its
            last. ``None`` for a *stock* measured at an instant (assets, equity,
            shares outstanding), which has no duration.

            Without this, a quarter of revenue and a year of revenue are
            indistinguishable — both are just a value with an end date — and
            whichever was filed most recently wins. That silently feeds a
            three-month figure to a model expecting twelve, understating every
            flow by roughly four. Keeping the span makes the two separable at
            the point where the choice is actually made.
    """

    model_config = ConfigDict(frozen=True)

    ticker: Ticker
    field: str
    value: float
    as_of: date
    observed_at: date
    source: str | None = None
    period_start: date | None = None

    @model_validator(mode="after")
    def _observed_not_before_as_of(self) -> PITRecord:
        if self.observed_at < self.as_of:
            raise ValueError(
                "observed_at must be on or after as_of "
                f"(got observed_at={self.observed_at!r} < as_of={self.as_of!r}): "
                "a datum cannot be knowable before the period it describes"
            )
        if self.period_start is not None and self.period_start > self.as_of:
            raise ValueError(
                "period_start must be on or before as_of "
                f"(got period_start={self.period_start!r} > as_of={self.as_of!r}): "
                "a period cannot start after it ends"
            )
        return self

    @property
    def period_days(self) -> int | None:
        """Length of the span this datum covers, or ``None`` for an instant."""
        if self.period_start is None:
            return None
        return (self.as_of - self.period_start).days
