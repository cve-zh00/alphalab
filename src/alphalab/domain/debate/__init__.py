"""Pure Bull / Bear / Judge debate domain (Phase 3 variant of the analyst).

This package holds *only* immutable Pydantic models, typing ``Protocol``s, and
pure turn-taking logic — no PydanticAI, no I/O, no ``adapters``/``app`` imports
(enforced by ``tests/unit/test_architecture.py``).

It replicates the load-bearing ideas from TradingAgents' debate (dossier §4),
chosen deliberately over emergent consensus:

* A **hard turn cap** router (:mod:`router`): the debate ends by *count*
  (``2 * max_rounds`` debater turns), never by agreement. The cap is a
  guaranteed invariant, proved by a property test.
* An **exhaustive route map**: the router is a total function of state and
  degrades explicitly (to "resolve") on any impossible state rather than
  crashing with a ``KeyError``.
* **Two tiers**: bull/bear are quick-think, the judge is deep-think (wired in
  the adapter).
* **Role isolation** (:mod:`ports`): each debater's request carries only the
  opponent's *public* arguments, never the other side's private case — the
  sub-graph isolation that stops cross-contamination of context.

As with the analyst, a valuation number is *structurally* impossible for any
role to emit: the contracts have no numeric valuation field. The headline
target is injected downstream from ``domain/compute``.
"""

from __future__ import annotations

from alphalab.domain.debate.conductor import conduct_debate
from alphalab.domain.debate.contracts import (
    BearCase,
    BullCase,
    DebateTranscript,
    DebateTurn,
    JudgeRuling,
)
from alphalab.domain.debate.ports import (
    BearProvider,
    BearRequest,
    BullProvider,
    BullRequest,
    JudgeProvider,
    JudgeRequest,
)
from alphalab.domain.debate.roles import DebateRole
from alphalab.domain.debate.router import (
    RouterAction,
    RouterDecision,
    next_decision,
)

__all__ = [
    "BearCase",
    "BearProvider",
    "BearRequest",
    "BullCase",
    "BullProvider",
    "BullRequest",
    "DebateRole",
    "DebateTranscript",
    "DebateTurn",
    "JudgeProvider",
    "JudgeRequest",
    "JudgeRuling",
    "RouterAction",
    "RouterDecision",
    "conduct_debate",
    "next_decision",
]
