"""Ports (typing ``Protocol``s) and role-isolated requests for the debate.

Typed *only* against the pure contracts, so the domain never imports
PydanticAI (mirrors ``domain/analyst/ports.py``). The adapter layer implements
these; tests implement them with deterministic doubles.

Context isolation by role (the TradingAgents sub-graph pattern, dossier §4)
---------------------------------------------------------------------------
Each role gets its **own** request type carrying only the context that role is
allowed to see — never one shared history blob:

* :class:`BullRequest` sees the shared evidence and the *bear's prior public
  arguments* (so it can rebut), but **not** the bear's private reasoning.
* :class:`BearRequest` is the mirror: it sees the *bull's prior public
  arguments*, not the bull's private reasoning.
* :class:`JudgeRequest` sees the full ordered transcript of public turns — it
  is the only role that reads the whole exchange, because resolving requires
  it.

A debater therefore cannot inherit the other side's tool-calls or private
state: the type system makes the leak impossible, not merely discouraged. The
"public arguments" a side sees are plain strings extracted from the transcript
by the orchestrator, so no structured private field crosses the boundary.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.debate.contracts import (
    BearCase,
    BullCase,
    DebateTurn,
    JudgeRuling,
)


class BullRequest(BaseModel):
    """Context handed to the bull — shared evidence plus the bear's public line.

    Attributes:
        ticker: The instrument under debate.
        evidence: Shared, neutral evidence both sides may cite.
        opponent_arguments: The bear's prior *public* arguments, for rebuttal.
            Never the bear's private case object or confidence.
        valuation_summary: Pre-computed numeric context (name -> value) the
            debater narrates around but cannot change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()
    opponent_arguments: tuple[str, ...] = ()
    valuation_summary: tuple[tuple[str, float], ...] = ()


class BearRequest(BaseModel):
    """Context handed to the bear — shared evidence plus the bull's public line.

    Attributes:
        ticker: The instrument under debate.
        evidence: Shared, neutral evidence both sides may cite.
        opponent_arguments: The bull's prior *public* arguments, for rebuttal.
            Never the bull's private case object or confidence.
        valuation_summary: Pre-computed numeric context (name -> value) the
            debater narrates around but cannot change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()
    opponent_arguments: tuple[str, ...] = ()
    valuation_summary: tuple[tuple[str, float], ...] = ()


class JudgeRequest(BaseModel):
    """Context handed to the arbiter — the full ordered public transcript.

    The judge is the only role that reads the whole exchange, because resolving
    the debate requires weighing both sides. It receives the ordered turns and
    the computed numeric context; it never emits a number.

    Attributes:
        ticker: The instrument under debate.
        turns: The full ordered sequence of debater turns.
        valuation_summary: Pre-computed numeric context the judge narrates
            around but cannot change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    turns: tuple[DebateTurn, ...]
    valuation_summary: tuple[tuple[str, float], ...] = ()


@runtime_checkable
class BullProvider(Protocol):
    """Produces the bullish case (quick-think)."""

    def argue(self, request: BullRequest) -> BullCase:
        """Return the bull's structured argument for ``request``."""
        ...


@runtime_checkable
class BearProvider(Protocol):
    """Produces the bearish case (quick-think)."""

    def argue(self, request: BearRequest) -> BearCase:
        """Return the bear's structured argument for ``request``."""
        ...


@runtime_checkable
class JudgeProvider(Protocol):
    """Resolves the debate into a directional ruling (deep-think)."""

    def rule(self, request: JudgeRequest) -> JudgeRuling:
        """Return the arbiter's ruling for ``request``."""
        ...
