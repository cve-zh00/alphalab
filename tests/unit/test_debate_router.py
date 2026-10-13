"""Unit tests for the deterministic turn router (no LLM).

The router is the engineering core of the debate: a total function of state
that alternates BULL/BEAR, resolves at the hard cap, and degrades explicitly on
any impossible state rather than raising. These tests pin the state machine.
"""

from __future__ import annotations

import pytest

from alphalab.domain.debate.contracts import DebateTurn
from alphalab.domain.debate.roles import DebateRole
from alphalab.domain.debate.router import RouterAction, next_decision


def _turn(role: DebateRole, n: int) -> DebateTurn:
    return DebateTurn(role=role, turn_number=n, argument=f"arg {n}", evidence=())


def _alternating(count: int) -> tuple[DebateTurn, ...]:
    """Build ``count`` turns alternating BULL, BEAR, BULL, BEAR, ..."""
    roles = (DebateRole.BULL, DebateRole.BEAR)
    return tuple(_turn(roles[i % 2], i) for i in range(count))


class TestAlternation:
    def test_bull_opens_from_empty(self) -> None:
        decision = next_decision(turns=(), max_rounds=2)
        assert decision.action is RouterAction.SPEAK
        assert decision.speaker is DebateRole.BULL

    def test_bear_answers_after_bull(self) -> None:
        decision = next_decision(turns=_alternating(1), max_rounds=2)
        assert decision.action is RouterAction.SPEAK
        assert decision.speaker is DebateRole.BEAR

    def test_bull_opens_second_round(self) -> None:
        decision = next_decision(turns=_alternating(2), max_rounds=2)
        assert decision.action is RouterAction.SPEAK
        assert decision.speaker is DebateRole.BULL

    @pytest.mark.parametrize("count", [0, 2, 4, 6])
    def test_even_count_routes_to_bull(self, count: int) -> None:
        decision = next_decision(turns=_alternating(count), max_rounds=10)
        assert decision.speaker is DebateRole.BULL

    @pytest.mark.parametrize("count", [1, 3, 5, 7])
    def test_odd_count_routes_to_bear(self, count: int) -> None:
        decision = next_decision(turns=_alternating(count), max_rounds=10)
        assert decision.speaker is DebateRole.BEAR


class TestHardCap:
    def test_resolves_exactly_at_cap(self) -> None:
        # 2*max_rounds debater turns => resolve, not speak.
        decision = next_decision(turns=_alternating(4), max_rounds=2)
        assert decision.action is RouterAction.RESOLVE
        assert decision.speaker is None

    def test_resolves_beyond_cap(self) -> None:
        decision = next_decision(turns=_alternating(10), max_rounds=2)
        assert decision.action is RouterAction.RESOLVE

    def test_speaks_one_before_cap(self) -> None:
        decision = next_decision(turns=_alternating(3), max_rounds=2)
        assert decision.action is RouterAction.SPEAK


class TestTotality:
    """Any state yields a well-formed decision; the router never raises."""

    def test_non_positive_max_rounds_degrades_to_resolve(self) -> None:
        decision = next_decision(turns=(), max_rounds=0)
        assert decision.action is RouterAction.RESOLVE
        assert "non-positive" in decision.reason

    def test_negative_max_rounds_degrades_to_resolve(self) -> None:
        decision = next_decision(turns=_alternating(3), max_rounds=-5)
        assert decision.action is RouterAction.RESOLVE

    def test_stray_judge_turn_does_not_inflate_count(self) -> None:
        # A JUDGE turn mixed in must not count toward the debater cap; the
        # router still routes on the two debater turns present (-> bull next).
        turns = (
            _turn(DebateRole.BULL, 0),
            _turn(DebateRole.JUDGE, 1),
            _turn(DebateRole.BEAR, 2),
        )
        decision = next_decision(turns=turns, max_rounds=3)
        assert decision.action is RouterAction.SPEAK
        assert decision.speaker is DebateRole.BULL

    def test_every_decision_has_a_reason(self) -> None:
        for count in range(0, 9):
            decision = next_decision(turns=_alternating(count), max_rounds=3)
            assert decision.reason
