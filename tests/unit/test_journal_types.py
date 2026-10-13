"""Unit tests for the journal domain types."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from alphalab.domain.journal import (
    Direction,
    Lesson,
    OutcomeGrade,
    Prediction,
    ProcessGrade,
    Settlement,
)
from alphalab.domain.journal.errors import LessonTooLongError


def _prediction(**overrides: object) -> Prediction:
    defaults: dict[str, object] = dict(
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
        ticker="AAPL",
        direction=Direction.UP,
        probability=0.7,
        horizon=date(2026, 2, 1),
        entry_price=150.0,
        knowledge_date=date(2026, 1, 1),
    )
    defaults.update(overrides)
    return Prediction(**defaults)  # type: ignore[arg-type]


class TestPrediction:
    def test_minimal_prediction_builds(self) -> None:
        # Arrange / Act
        pred = _prediction()

        # Assert
        assert 0.0 <= pred.probability <= 1.0
        assert pred.prediction_id

    def test_probability_out_of_range_rejected(self) -> None:
        # Act / Assert
        with pytest.raises(ValidationError):
            _prediction(probability=1.5)

    def test_horizon_before_knowledge_rejected(self) -> None:
        # Act / Assert
        with pytest.raises(ValidationError):
            _prediction(horizon=date(2025, 12, 1), knowledge_date=date(2026, 1, 1))

    def test_is_frozen(self) -> None:
        # Arrange
        pred = _prediction()

        # Act / Assert
        with pytest.raises(ValidationError):
            pred.probability = 0.1  # type: ignore[misc]

    def test_llm_metadata_recorded(self) -> None:
        # Arrange / Act
        pred = _prediction(llm_version="frontier-2026-09", llm_cutoff=date(2026, 9, 1))

        # Assert
        assert pred.llm_version == "frontier-2026-09"
        assert pred.llm_cutoff == date(2026, 9, 1)


class TestDirection:
    @pytest.mark.parametrize(
        ("direction", "realized", "expected"),
        [
            (Direction.UP, 0.05, True),
            (Direction.UP, -0.05, False),
            (Direction.DOWN, -0.05, True),
            (Direction.DOWN, 0.05, False),
            (Direction.UP, 0.0, False),
            (Direction.DOWN, 0.0, False),
        ],
    )
    def test_hits(self, direction: Direction, realized: float, expected: bool) -> None:
        # Act / Assert
        assert direction.hits(realized) is expected


class TestSettlement:
    def test_process_outcome_maps_bool(self) -> None:
        # Arrange
        won = Settlement(
            prediction_id="p",
            settled_at=datetime(2026, 2, 1, tzinfo=UTC),
            realized_return=0.1,
            benchmark_return=0.02,
            alpha_gross=0.08,
            alpha_net=0.07,
            outcome=True,
        )
        lost = won.model_copy(update={"outcome": False})

        # Act / Assert
        assert won.process_outcome is OutcomeGrade.WIN
        assert lost.process_outcome is OutcomeGrade.LOSS


class TestLesson:
    def test_two_sentence_lesson_accepted(self) -> None:
        # Arrange / Act
        lesson = Lesson(
            prediction_id="p",
            text="The call was right. Keep the sizing.",
            process_grade=ProcessGrade.SOUND,
            outcome_grade=OutcomeGrade.WIN,
        )

        # Assert
        assert lesson.text

    def test_five_sentence_lesson_rejected(self) -> None:
        # Arrange
        too_long = "One. Two. Three. Four. Five."

        # Act / Assert
        with pytest.raises(LessonTooLongError):
            Lesson(
                prediction_id="p",
                text=too_long,
                process_grade=ProcessGrade.SOUND,
                outcome_grade=OutcomeGrade.WIN,
            )

    def test_single_sentence_lesson_rejected(self) -> None:
        # Act / Assert
        with pytest.raises(LessonTooLongError):
            Lesson(
                prediction_id="p",
                text="Only one sentence here.",
                process_grade=ProcessGrade.FLAWED,
                outcome_grade=OutcomeGrade.LOSS,
            )

    def test_overlong_characters_rejected(self) -> None:
        # Arrange: two sentences but a wall of characters.
        wall = "a" * 400 + ". " + "b" * 400 + "."

        # Act / Assert
        with pytest.raises(LessonTooLongError):
            Lesson(
                prediction_id="p",
                text=wall,
                process_grade=ProcessGrade.SOUND,
                outcome_grade=OutcomeGrade.WIN,
            )
