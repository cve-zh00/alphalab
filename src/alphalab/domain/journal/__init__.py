"""Prediction journal domain: settle-then-reflect and calibration.

This package is what gives the system a *measurable* track record. It stays in
the pure domain layer (stdlib + pydantic + numpy/scipy only); SQLite
persistence lives in :mod:`alphalab.adapters.journal_sqlite`.

Two mechanisms:

* **Settle-then-reflect** - each run records a :class:`Prediction`; a later
  run liquidates due predictions into a deterministic :class:`Settlement`
  (:func:`settle_prediction`) and a :class:`Reflector` distils a short prose
  :class:`Lesson`.
* **Calibration** - :func:`calibration_report` scores whether declared
  probabilities match realized accuracy (Brier, ECE, reliability diagram,
  signed overconfidence).
"""

from __future__ import annotations

from alphalab.domain.journal.calibration import (
    DEFAULT_N_BINS,
    MIN_SAMPLE_SIZE,
    BrierDecomposition,
    CalibrationReport,
    ReliabilityBin,
    brier_decomposition,
    brier_score,
    calibration_report,
    expected_calibration_error,
    overconfidence,
    reliability_diagram,
    winkler_score,
)
from alphalab.domain.journal.errors import (
    JournalError,
    LessonTooLongError,
    NotYetDueError,
)
from alphalab.domain.journal.reflect import (
    Reflector,
    TemplateReflector,
    grade_process,
)
from alphalab.domain.journal.settle import due_predictions, settle_prediction
from alphalab.domain.journal.types import (
    LESSON_MAX_CHARS,
    LESSON_MAX_SENTENCES,
    LESSON_MIN_SENTENCES,
    Direction,
    Lesson,
    OutcomeGrade,
    Prediction,
    ProcessGrade,
    Settlement,
    new_prediction_id,
)

__all__ = [
    "DEFAULT_N_BINS",
    "LESSON_MAX_CHARS",
    "LESSON_MAX_SENTENCES",
    "LESSON_MIN_SENTENCES",
    "MIN_SAMPLE_SIZE",
    "BrierDecomposition",
    "CalibrationReport",
    "Direction",
    "JournalError",
    "Lesson",
    "LessonTooLongError",
    "NotYetDueError",
    "OutcomeGrade",
    "Prediction",
    "ProcessGrade",
    "Reflector",
    "ReliabilityBin",
    "Settlement",
    "TemplateReflector",
    "brier_decomposition",
    "brier_score",
    "calibration_report",
    "due_predictions",
    "expected_calibration_error",
    "grade_process",
    "new_prediction_id",
    "overconfidence",
    "reliability_diagram",
    "settle_prediction",
    "winkler_score",
]
