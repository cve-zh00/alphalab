"""Point-in-time investable universe, including delisted names.

Survivorship bias arises when a backtest's universe only contains companies
that still exist today. To avoid it, :class:`UniverseMember` carries both a
``listed_on`` and an optional ``delisted_on`` date, and membership is resolved
*as of a knowledge date* — a name delisted in 2008 is still a constituent for
any knowledge date before its delisting.

LIMITATION: there is no free, authoritative source of historical delistings
wired in here. :class:`InMemoryUniverseSource` implements the
:class:`UniverseSource` protocol over hand-supplied (or test) data. A
production deployment must back this protocol with a real delisting-aware
feed (e.g. a licensed vendor); until then, universes built from this adapter
are only as complete as the delisted rows provided to it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from alphalab.domain.pit import KnowledgeDate
from alphalab.domain.types import Ticker


@dataclass(frozen=True)
class UniverseMember:
    """One instrument's listing lifecycle.

    Attributes:
        ticker: The instrument symbol.
        listed_on: First date the instrument was investable.
        delisted_on: Date the instrument stopped trading, or ``None`` if it
            is still listed. Keeping delisted names is what prevents
            survivorship bias.
    """

    ticker: Ticker
    listed_on: date
    delisted_on: date | None

    def is_active_on(self, when: date) -> bool:
        """Return whether the instrument was investable on ``when``."""
        if when < self.listed_on:
            return False
        return self.delisted_on is None or when < self.delisted_on


class UniverseSource(Protocol):
    """Protocol for resolving universe constituents at a knowledge date."""

    def constituents(self, knowledge_date: KnowledgeDate) -> list[UniverseMember]: ...


class InMemoryUniverseSource:
    """Universe source backed by an in-memory list of members."""

    def __init__(self, members: Iterable[UniverseMember]) -> None:
        self._members = tuple(members)

    def constituents(self, knowledge_date: KnowledgeDate) -> list[UniverseMember]:
        """Return members active as of ``knowledge_date`` (delisted included)."""
        when = knowledge_date.value
        return [m for m in self._members if m.is_active_on(when)]
