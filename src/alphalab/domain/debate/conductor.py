"""The pure debate conductor: drive the router + providers into a transcript.

This is the turn-taking *logic* of the debate, kept pure and in the domain. It
knows the router and the three ports but nothing about PydanticAI or I/O — it
calls ``provider.argue(...)`` / ``provider.rule(...)`` through the typed
``Protocol``s, so it runs identically with real LLM adapters or deterministic
test doubles.

What it guarantees
------------------
* **Hard turn cap.** The loop is bounded by the router: it keeps requesting
  decisions until the router says ``RESOLVE`` (which it must, by
  ``count >= 2*max_rounds``). A belt-and-braces counter also caps iterations at
  ``2*max_rounds`` so a hypothetically misbehaving router can never spin
  forever. The number of debater turns therefore never exceeds
  ``2 * max_rounds``.
* **Role isolation.** Each debater is handed a request built from only the
  *public arguments* of its opponent — the conductor extracts those strings
  from the transcript and never passes one side's private
  :class:`~alphalab.domain.debate.contracts.BullCase` /
  :class:`~alphalab.domain.debate.contracts.BearCase` object to the other.
"""

from __future__ import annotations

from alphalab.domain.debate.contracts import (
    BearCase,
    BullCase,
    DebateTranscript,
    DebateTurn,
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
from alphalab.domain.debate.router import RouterAction, next_decision

ValuationSummary = tuple[tuple[str, float], ...]


def _public_arguments(turns: tuple[DebateTurn, ...], role: DebateRole) -> tuple[str, ...]:
    """Return, in order, the public argument strings a given role has spoken."""
    return tuple(t.argument for t in turns if t.role is role)


def _append_bull(
    turns: tuple[DebateTurn, ...],
    *,
    ticker: str,
    evidence: tuple[str, ...],
    valuation_summary: ValuationSummary,
    provider: BullProvider,
) -> tuple[tuple[DebateTurn, ...], BullCase]:
    """Ask the bull to argue, seeing only the bear's public line, and record it."""
    request = BullRequest(
        ticker=ticker,
        evidence=evidence,
        opponent_arguments=_public_arguments(turns, DebateRole.BEAR),
        valuation_summary=valuation_summary,
    )
    case = provider.argue(request)
    turn = DebateTurn(
        role=DebateRole.BULL,
        turn_number=len(turns),
        argument=case.argument,
        evidence=case.evidence,
    )
    return (*turns, turn), case


def _append_bear(
    turns: tuple[DebateTurn, ...],
    *,
    ticker: str,
    evidence: tuple[str, ...],
    valuation_summary: ValuationSummary,
    provider: BearProvider,
) -> tuple[tuple[DebateTurn, ...], BearCase]:
    """Ask the bear to argue, seeing only the bull's public line, and record it."""
    request = BearRequest(
        ticker=ticker,
        evidence=evidence,
        opponent_arguments=_public_arguments(turns, DebateRole.BULL),
        valuation_summary=valuation_summary,
    )
    case = provider.argue(request)
    turn = DebateTurn(
        role=DebateRole.BEAR,
        turn_number=len(turns),
        argument=case.argument,
        evidence=case.evidence,
    )
    return (*turns, turn), case


def conduct_debate(
    *,
    ticker: str,
    evidence: tuple[str, ...],
    valuation_summary: ValuationSummary,
    max_rounds: int,
    bull: BullProvider,
    bear: BearProvider,
    judge: JudgeProvider,
) -> DebateTranscript:
    """Run a bounded debate and return its auditable transcript.

    The conductor alternates bull/bear strictly as the router dictates, stops
    at the hard cap, then asks the judge to resolve from the full transcript.

    Args:
        ticker: The instrument under debate.
        evidence: Shared, neutral evidence both debaters may cite.
        valuation_summary: Pre-computed numeric context (never model-emitted).
        max_rounds: Per-side round budget (``>= 1``).
        bull: Provider for the bullish case.
        bear: Provider for the bearish case.
        judge: Provider for the arbiter's ruling.

    Returns:
        A frozen :class:`DebateTranscript` with the ordered turns and the ruling.
    """
    effective_rounds = max(1, max_rounds)
    turns: tuple[DebateTurn, ...] = ()

    # Belt-and-braces bound: even a misbehaving router cannot exceed this.
    for _ in range(2 * effective_rounds):
        decision = next_decision(turns=turns, max_rounds=effective_rounds)
        if decision.action is RouterAction.RESOLVE:
            break
        if decision.speaker is DebateRole.BULL:
            turns, _ = _append_bull(
                turns,
                ticker=ticker,
                evidence=evidence,
                valuation_summary=valuation_summary,
                provider=bull,
            )
        elif decision.speaker is DebateRole.BEAR:
            turns, _ = _append_bear(
                turns,
                ticker=ticker,
                evidence=evidence,
                valuation_summary=valuation_summary,
                provider=bear,
            )
        else:  # pragma: no cover - router never emits SPEAK without a debater speaker
            break

    ruling = judge.rule(
        JudgeRequest(ticker=ticker, turns=turns, valuation_summary=valuation_summary)
    )
    return DebateTranscript(
        ticker=ticker,
        turns=turns,
        ruling=ruling,
        max_rounds=effective_rounds,
    )
