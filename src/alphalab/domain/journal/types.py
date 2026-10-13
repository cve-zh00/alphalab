"""Immutable domain types for the prediction journal.

The journal is what turns a generator of opinions into a system with a
measurable track record. Four frozen value types model the settle-then-reflect
loop and the separate grading of process from outcome:

* :class:`Prediction` - a thesis emitted by a run, carrying the **declared
  probability** that is later calibrated, plus the ``llm_version`` /
  ``llm_cutoff`` metadata needed to attribute a hit to the model that made it.
* :class:`Settlement` - the deterministic result at the horizon: realized and
  benchmark return, **alpha gross and net**, and whether the direction hit.
* :class:`Lesson` - a short prose reflection with a **hard, validated** length
  budget (2-4 sentences).
* :class:`ProcessGrade` / :class:`OutcomeGrade` - graded **separately** so a
  well-executed trade that lost and a lucky trade that won are never confused.

As with :class:`alphalab.domain.ledger.TrialRecord`, the LLM metadata fields
are named ``llm_*`` rather than ``model_*`` to sidestep Pydantic v2's protected
``model_`` namespace without disabling the guard.

Pure module: stdlib + pydantic only. No I/O.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alphalab.domain.journal._text import count_sentences
from alphalab.domain.journal.errors import LessonTooLongError

__all__ = [
    "LESSON_MAX_CHARS",
    "LESSON_MAX_SENTENCES",
    "LESSON_MIN_SENTENCES",
    "Direction",
    "Lesson",
    "OutcomeGrade",
    "Prediction",
    "ProcessGrade",
    "Settlement",
    "new_prediction_id",
]

#: Hard length budget for a reflection lesson. 2-4 sentences keeps the text
#: short enough that a downstream model reading it is not degraded by bloat.
LESSON_MIN_SENTENCES = 2
LESSON_MAX_SENTENCES = 4
#: Character ceiling as a second, independent guard against a single
#: run-on "sentence" smuggling in a wall of text.
LESSON_MAX_CHARS = 600


def new_prediction_id() -> str:
    """Generate a fresh, globally-unique prediction id (UUID4 hex)."""
    return uuid.uuid4().hex


class Direction(Enum):
    """The directional verdict of a thesis."""

    UP = "up"
    DOWN = "down"

    def hits(self, realized_return: float) -> bool:
        """Whether this direction was correct given a realized return.

        A flat realized return (exactly zero) is treated as a miss for both
        directions: the thesis predicted a move and none happened.
        """
        if realized_return > 0.0:
            return self is Direction.UP
        if realized_return < 0.0:
            return self is Direction.DOWN
        return False


class ProcessGrade(Enum):
    """Quality of the *reasoning* behind a prediction, independent of result.

    HyperGrok's lesson: process and outcome must be judged separately. A trade
    can be well-reasoned and still lose, or sloppy and still win. Grading them
    together is how a system learns the wrong lesson.
    """

    SOUND = "sound"
    FLAWED = "flawed"


class OutcomeGrade(Enum):
    """Quality of the *result* of a prediction, independent of reasoning."""

    WIN = "win"
    LOSS = "loss"


class Prediction(BaseModel):
    """An immutable thesis emitted by a run.

    The ``probability`` field is the declared confidence in ``[0, 1]`` - the
    quantity the calibration machinery scores. ``horizon`` is the target
    settlement date; ``knowledge_date`` is the point-in-time cutoff of the run
    that produced the thesis, kept distinct from the model's own
    ``llm_cutoff`` so temporal hygiene stays auditable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_id: str = Field(default_factory=new_prediction_id)
    recorded_at: datetime
    ticker: str = Field(min_length=1)
    direction: Direction
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Declared confidence in the direction; the calibrated quantity.",
    )
    horizon: date = Field(description="Target settlement date for the thesis.")
    entry_price: float = Field(gt=0.0, description="Reference entry price per share.")
    knowledge_date: date = Field(description="Point-in-time cutoff of the producing run.")
    llm_version: str | None = Field(
        default=None,
        description="Version of the LLM that produced the thesis, if any.",
    )
    llm_cutoff: date | None = Field(
        default=None,
        description="Knowledge cutoff of that LLM; guards temporal hygiene.",
    )

    @model_validator(mode="after")
    def _horizon_after_knowledge(self) -> Prediction:
        if self.horizon < self.knowledge_date:
            raise ValueError("horizon must not precede knowledge_date")
        return self


class Settlement(BaseModel):
    """The immutable, deterministic result of a prediction at its horizon.

    Alpha is reported **both gross and net** of costs - reporting only gross
    would flatter the track record exactly where it matters. ``outcome`` is the
    boolean hit: did the declared direction match the realized move.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_id: str = Field(min_length=1)
    settled_at: datetime
    realized_return: float = Field(description="Realized return of the position.")
    benchmark_return: float = Field(description="Return of the benchmark over the same window.")
    alpha_gross: float = Field(description="Realized return minus benchmark, before costs.")
    alpha_net: float = Field(description="Alpha after subtracting cost drag.")
    outcome: bool = Field(description="True iff the declared direction was correct.")

    @property
    def process_outcome(self) -> OutcomeGrade:
        """Map the boolean hit onto an :class:`OutcomeGrade`."""
        return OutcomeGrade.WIN if self.outcome else OutcomeGrade.LOSS


class Lesson(BaseModel):
    """A short prose reflection tied to a settled prediction.

    The text is validated against a **hard budget** (2-4 sentences and a
    character ceiling). Anything longer raises
    :class:`~alphalab.domain.journal.errors.LessonTooLongError` - the budget is
    enforced, not advisory, because unbounded reflections degrade the context
    a downstream model consumes.

    Process and outcome grades are carried **separately and are both required**
    so the type itself prevents conflating "well reasoned" with "it won".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    process_grade: ProcessGrade
    outcome_grade: OutcomeGrade

    @field_validator("text")
    @classmethod
    def _within_budget(cls, value: str) -> str:
        if len(value) > LESSON_MAX_CHARS:
            raise LessonTooLongError(f"lesson exceeds {LESSON_MAX_CHARS} characters ({len(value)})")
        sentences = count_sentences(value)
        if sentences < LESSON_MIN_SENTENCES or sentences > LESSON_MAX_SENTENCES:
            raise LessonTooLongError(
                f"lesson must be {LESSON_MIN_SENTENCES}-{LESSON_MAX_SENTENCES} "
                f"sentences, got {sentences}"
            )
        return value
