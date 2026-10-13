"""Unit tests for the pure debate contracts (no LLM).

Pins the structural guarantees: the debater cases carry no numeric valuation
field (so a model cannot emit a price), everything is frozen, ``extra`` is
forbidden, and the transcript is a faithful ordered record.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alphalab.artifact.models import Verdict
from alphalab.domain.debate.contracts import (
    BearCase,
    BullCase,
    DebateTranscript,
    DebateTurn,
    JudgeRuling,
)
from alphalab.domain.debate.roles import DebateRole


def _ruling() -> JudgeRuling:
    return JudgeRuling(
        verdict=Verdict.BUY,
        rationale="Bull margins beat bear risk.",
        bull_points_weighed=("margins",),
        bear_points_weighed=("regulation",),
    )


class TestNoNumericValuationField:
    """A valuation number must be structurally impossible to emit."""

    def test_bull_case_rejects_price_target_field(self) -> None:
        with pytest.raises(ValidationError):
            BullCase(
                argument="a",
                catalysts="c",
                evidence=(),
                confidence=0.5,
                price_target=42.0,  # type: ignore[call-arg]
            )

    def test_bear_case_rejects_fair_value_field(self) -> None:
        with pytest.raises(ValidationError):
            BearCase(
                argument="a",
                risks="r",
                evidence=(),
                confidence=0.5,
                fair_value=1.0,  # type: ignore[call-arg]
            )

    def test_judge_ruling_rejects_extra_numeric_field(self) -> None:
        with pytest.raises(ValidationError):
            JudgeRuling(
                verdict=Verdict.HOLD,
                rationale="x",
                target=10.0,  # type: ignore[call-arg]
            )

    def test_bull_case_confidence_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            BullCase(argument="a", catalysts="c", confidence=1.5)
        with pytest.raises(ValidationError):
            BearCase(argument="a", risks="r", confidence=-0.1)


class TestImmutability:
    def test_bull_case_is_frozen(self) -> None:
        case = BullCase(argument="a", catalysts="c", confidence=0.5)
        with pytest.raises(ValidationError):
            case.confidence = 0.9  # type: ignore[misc]

    def test_turn_is_frozen(self) -> None:
        turn = DebateTurn(role=DebateRole.BULL, turn_number=0, argument="x")
        with pytest.raises(ValidationError):
            turn.argument = "y"  # type: ignore[misc]


class TestTranscript:
    def test_transcript_preserves_order(self) -> None:
        turns = (
            DebateTurn(role=DebateRole.BULL, turn_number=0, argument="bull-0"),
            DebateTurn(role=DebateRole.BEAR, turn_number=1, argument="bear-1"),
        )
        transcript = DebateTranscript(ticker="SYNTH", turns=turns, ruling=_ruling(), max_rounds=1)
        assert [t.role for t in transcript.turns] == [DebateRole.BULL, DebateRole.BEAR]
        assert [t.turn_number for t in transcript.turns] == [0, 1]

    def test_transcript_requires_positive_max_rounds(self) -> None:
        with pytest.raises(ValidationError):
            DebateTranscript(ticker="SYNTH", turns=(), ruling=_ruling(), max_rounds=0)
