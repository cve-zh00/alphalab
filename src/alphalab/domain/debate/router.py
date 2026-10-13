"""The deterministic turn router — the engineering core of the debate.

This module decides, from the turns emitted so far, **who speaks next** or
whether it is time to **resolve** with the Judge. It is pure and has no
knowledge of the LLM: it reasons only over the sequence of
:class:`~alphalab.domain.debate.roles.DebateRole` values already spoken.

Design, after the TradingAgents router (dossier §4)
----------------------------------------------------
* **Hard turn cap, not emergent consensus.** The debate alternates
  ``BULL, BEAR, BULL, BEAR, ...`` and ends by *count*: once ``2 * max_rounds``
  debater turns exist, the next decision is always "resolve with the Judge".
  Agreement is never the stop condition — it may never come and only burns
  tokens. This makes the budget a hard guarantee: see the property test.
* **Exhaustive route map, no implicit ``else``.** The hardening TradingAgents
  added after a real label-drift crash was a *total* mapping of router states.
  Here the decision is a total function of ``(count, parity, resolved)``; an
  impossible or malformed state degrades *explicitly* to
  :data:`RouterDecision.RESOLVE` with a recorded ``reason`` rather than raising
  a ``KeyError`` and crashing the pipeline.

The router never raises on an unexpected state. The only guard is a defensive
assertion-free fallback that routes to resolution and explains itself.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.debate.contracts import DebateTurn
from alphalab.domain.debate.roles import DebateRole


class RouterAction(StrEnum):
    """What the router instructs the orchestrator to do next."""

    SPEAK = "speak"
    RESOLVE = "resolve"


class RouterDecision(BaseModel):
    """The router's total, explainable decision for a debate state.

    Attributes:
        action: Whether a debater should speak or the Judge should resolve.
        speaker: The role to speak next when ``action`` is ``SPEAK``; ``None``
            when resolving.
        reason: A short, documented explanation of why this route was taken —
            especially load-bearing for the explicit degradation paths.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: RouterAction
    speaker: DebateRole | None = None
    reason: str = Field(min_length=1)


def _count_debater_turns(turns: tuple[DebateTurn, ...]) -> int:
    """Count only BULL/BEAR turns; a stray JUDGE turn never inflates the count."""
    return sum(1 for t in turns if t.role in (DebateRole.BULL, DebateRole.BEAR))


def next_decision(
    *,
    turns: tuple[DebateTurn, ...],
    max_rounds: int,
) -> RouterDecision:
    """Return the next action for the debate, as a total function of state.

    The debate runs ``BULL, BEAR`` per round for ``max_rounds`` rounds, i.e. at
    most ``2 * max_rounds`` debater turns, after which the Judge resolves.

    Routing table (exhaustive over the reachable state space):

    * ``max_rounds < 1``            → RESOLVE  (no budget to debate; degrade)
    * ``count >= 2 * max_rounds``   → RESOLVE  (hard cap reached)
    * ``count`` even                → SPEAK BULL  (bull opens each round)
    * ``count`` odd                 → SPEAK BEAR  (bear answers)

    Any state that somehow escapes the above (it cannot, by construction, since
    the three branches partition the integers) degrades to RESOLVE with an
    explicit reason instead of raising.

    Args:
        turns: The turns emitted so far, in order.
        max_rounds: The per-side round budget (``>= 1`` for a real debate).

    Returns:
        A :class:`RouterDecision` that is always well-formed.
    """
    if max_rounds < 1:
        return RouterDecision(
            action=RouterAction.RESOLVE,
            speaker=None,
            reason=f"non-positive max_rounds={max_rounds}; nothing to debate, resolving",
        )

    count = _count_debater_turns(turns)
    cap = 2 * max_rounds

    if count >= cap:
        return RouterDecision(
            action=RouterAction.RESOLVE,
            speaker=None,
            reason=f"hard cap reached: {count} debater turns >= 2*max_rounds={cap}",
        )

    if count % 2 == 0:
        return RouterDecision(
            action=RouterAction.SPEAK,
            speaker=DebateRole.BULL,
            reason=f"round opener: {count} turns taken, bull speaks",
        )

    if count % 2 == 1:
        return RouterDecision(
            action=RouterAction.SPEAK,
            speaker=DebateRole.BEAR,
            reason=f"round reply: {count} turns taken, bear answers",
        )

    # Unreachable: the three branches above partition every non-negative int.
    # We keep an explicit, documented degradation rather than an implicit
    # ``else`` that could crash the pipeline on a future refactor.
    return RouterDecision(
        action=RouterAction.RESOLVE,
        speaker=None,
        reason=f"unexpected router state (count={count}); degrading to resolve",
    )
