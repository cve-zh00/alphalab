"""Property tests for the debate's hard turn cap (Hypothesis).

The single most important invariant of the whole debate design (dossier §4,
the TradingAgents hard-count stop): for *any* ``max_rounds`` and *any* sequence
of emitted turns, the debate can never run more than ``2 * max_rounds`` debater
turns. We prove it two ways:

1. **Router level**: once ``2 * max_rounds`` debater turns exist, the router
   always says RESOLVE — regardless of how the turns were arranged.
2. **Conductor level**: a full driven debate never produces more than
   ``2 * max_rounds`` turns, with deterministic doubles standing in for the LLM.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from alphalab.domain.debate.conductor import conduct_debate
from alphalab.domain.debate.contracts import DebateTurn
from alphalab.domain.debate.roles import DebateRole
from alphalab.domain.debate.router import RouterAction, next_decision
from tests.unit._debate_doubles import StubBear, StubBull, StubJudge

_DEBATER_ROLES = st.sampled_from([DebateRole.BULL, DebateRole.BEAR, DebateRole.JUDGE])


@st.composite
def _turn_sequences(draw: st.DrawFn) -> tuple[DebateTurn, ...]:
    """Arbitrary (possibly non-alternating) turn sequences, including JUDGE."""
    roles = draw(st.lists(_DEBATER_ROLES, min_size=0, max_size=40))
    return tuple(
        DebateTurn(role=role, turn_number=i, argument=f"arg {i}") for i, role in enumerate(roles)
    )


@given(max_rounds=st.integers(min_value=1, max_value=20), turns=_turn_sequences())
def test_router_resolves_once_cap_reached(max_rounds: int, turns: tuple[DebateTurn, ...]) -> None:
    """If debater turns have reached the cap, the router must RESOLVE."""
    cap = 2 * max_rounds
    debater_turns = sum(1 for t in turns if t.role in (DebateRole.BULL, DebateRole.BEAR))
    decision = next_decision(turns=turns, max_rounds=max_rounds)
    if debater_turns >= cap:
        assert decision.action is RouterAction.RESOLVE
        assert decision.speaker is None
    else:
        assert decision.action is RouterAction.SPEAK
        assert decision.speaker in (DebateRole.BULL, DebateRole.BEAR)


@given(max_rounds=st.integers(min_value=1, max_value=12))
def test_conducted_debate_never_exceeds_cap(max_rounds: int) -> None:
    """A fully driven debate never emits more than ``2 * max_rounds`` turns."""
    transcript = conduct_debate(
        ticker="SYNTH",
        evidence=("ev",),
        valuation_summary=(("consensus_mid", 10.0),),
        max_rounds=max_rounds,
        bull=StubBull(),
        bear=StubBear(),
        judge=StubJudge(),
    )
    assert len(transcript.turns) <= 2 * max_rounds
    # And it ran exactly the budget (doubles never agree early; stop is by count).
    assert len(transcript.turns) == 2 * max_rounds


@given(max_rounds=st.integers(min_value=-10, max_value=0))
def test_non_positive_rounds_never_crash(max_rounds: int) -> None:
    """Non-positive budgets degrade safely: router resolves, never raises."""
    decision = next_decision(turns=(), max_rounds=max_rounds)
    assert decision.action is RouterAction.RESOLVE
