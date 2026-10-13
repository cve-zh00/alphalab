"""Unit tests for the pure debate conductor (deterministic doubles, no LLM).

Covers turn alternation, the hard cap on debater turns, the role-isolation
guarantee (the bull never receives the bear's private case), and the
auditability of the resulting transcript.
"""

from __future__ import annotations

import pytest

from alphalab.artifact.models import Verdict
from alphalab.domain.debate.conductor import conduct_debate
from alphalab.domain.debate.contracts import DebateTranscript
from alphalab.domain.debate.roles import DebateRole
from tests.unit._debate_doubles import StubBear, StubBull, StubJudge

_SUMMARY = (("consensus_mid", 42.0), ("current_price", 40.0))
_EVIDENCE = ("Shared filing excerpt.",)


def _run(*, max_rounds: int) -> tuple[DebateTranscript, StubBull, StubBear, StubJudge]:
    bull, bear, judge = StubBull(), StubBear(), StubJudge(verdict=Verdict.BUY)
    transcript = conduct_debate(
        ticker="SYNTH",
        evidence=_EVIDENCE,
        valuation_summary=_SUMMARY,
        max_rounds=max_rounds,
        bull=bull,
        bear=bear,
        judge=judge,
    )
    return transcript, bull, bear, judge


class TestTurnTaking:
    @pytest.mark.parametrize("max_rounds", [1, 2, 3])
    def test_turns_never_exceed_cap(self, max_rounds: int) -> None:
        transcript, _, _, _ = _run(max_rounds=max_rounds)
        assert len(transcript.turns) == 2 * max_rounds

    def test_turns_alternate_bull_bear(self) -> None:
        transcript, _, _, _ = _run(max_rounds=2)
        roles = [t.role for t in transcript.turns]
        assert roles == [
            DebateRole.BULL,
            DebateRole.BEAR,
            DebateRole.BULL,
            DebateRole.BEAR,
        ]

    def test_judge_is_always_invoked_once(self) -> None:
        _, _, _, judge = _run(max_rounds=2)
        assert len(judge.requests) == 1

    def test_zero_or_negative_rounds_clamped_to_one(self) -> None:
        # The conductor clamps to a single round so a debate always happens.
        transcript, _, _, _ = _run(max_rounds=0)
        assert len(transcript.turns) == 2
        assert transcript.max_rounds == 1


class TestContextIsolation:
    """Each debater sees only its opponent's PUBLIC arguments, never the case."""

    def test_bull_first_request_has_no_bear_arguments(self) -> None:
        _, bull, _, _ = _run(max_rounds=2)
        first = bull.requests[0]
        assert first.opponent_arguments == ()

    def test_bear_sees_only_bull_public_argument_strings(self) -> None:
        _, _bull, bear, _ = _run(max_rounds=2)
        # The bear's first request must carry the bull's public argument text,
        # and nothing structured (no BullCase, no confidence).
        first_bear = bear.requests[0]
        assert len(first_bear.opponent_arguments) == 1
        assert first_bear.opponent_arguments[0].startswith("Bull argument #1")
        # Only plain strings cross the boundary.
        assert all(isinstance(a, str) for a in first_bear.opponent_arguments)

    def test_bull_second_round_sees_prior_bear_argument(self) -> None:
        _, bull, _, _ = _run(max_rounds=2)
        # On its 2nd turn the bull should see the bear's 1st public argument.
        second_bull = bull.requests[1]
        assert any("Bear argument #1" in a for a in second_bull.opponent_arguments)

    def test_bull_never_receives_bear_private_confidence(self) -> None:
        _, bull, _, _ = _run(max_rounds=2)
        # The request type simply has no field for the opponent's confidence.
        assert not hasattr(bull.requests[0], "opponent_confidence")


class TestAuditability:
    def test_transcript_reconstructs_who_said_what_in_order(self) -> None:
        transcript, _, _, _ = _run(max_rounds=2)
        reconstructed = [(t.turn_number, t.role.value, t.argument) for t in transcript.turns]
        assert reconstructed == [
            (0, "bull", "Bull argument #1: durable franchise, expanding margins."),
            (1, "bear", "Bear argument #1: regulatory overhang, margin pressure."),
            (2, "bull", "Bull argument #2: durable franchise, expanding margins."),
            (3, "bear", "Bear argument #2: regulatory overhang, margin pressure."),
        ]

    def test_ruling_cites_both_sides(self) -> None:
        transcript, _, _, _ = _run(max_rounds=1)
        ruling = transcript.ruling
        assert ruling.bull_points_weighed
        assert ruling.bear_points_weighed
        assert ruling.verdict is Verdict.BUY
