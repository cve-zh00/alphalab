"""The *reflect* half of settle-then-reflect: the reflector interface.

Once a prediction is settled, a reflector turns the (prediction, settlement)
pair into a short prose :class:`~alphalab.domain.journal.types.Lesson`. In
Phase 3 an LLM will implement this; here we define the **interface** as a
``Protocol`` and ship a **deterministic, template-based reference** so the
whole settle-then-reflect cycle is testable and runs today without any model.

The reflector also assigns the two grades. Process and outcome are graded
**separately**: the template grades process from whether realized alpha and
direction agree in sign (a coherent, well-reasoned call), and outcome purely
from the boolean hit. This keeps "well reasoned" and "it won" from collapsing
into one judgement.

Pure module: stdlib only. No I/O.
"""

from __future__ import annotations

from typing import Protocol

from alphalab.domain.journal.types import (
    Lesson,
    Prediction,
    ProcessGrade,
    Settlement,
)

__all__ = ["Reflector", "TemplateReflector", "grade_process"]


class Reflector(Protocol):
    """Turns a settled prediction into a bounded prose lesson.

    Phase 3 supplies an LLM-backed implementation. Any implementation MUST
    return a :class:`Lesson` whose text respects the hard length budget
    enforced by the type (2-4 sentences); over-budget text is rejected at
    construction, so a conforming reflector keeps its prose short.
    """

    def reflect(self, prediction: Prediction, settlement: Settlement) -> Lesson:
        """Produce a :class:`Lesson` for the given settled prediction."""
        ...


def grade_process(prediction: Prediction, settlement: Settlement) -> ProcessGrade:
    """Grade the *reasoning* independently of the result.

    The process is judged SOUND when the declared direction is coherent with
    the realized alpha's sign - i.e. the call pointed the right way on a
    risk-adjusted basis - regardless of whether the raw outcome was a win.
    This deliberately differs from the outcome grade: a sound process can post
    a loss, and a flawed one can get lucky.
    """
    directional_alpha = settlement.alpha_net
    if directional_alpha == 0.0:
        return ProcessGrade.FLAWED
    aligned = (directional_alpha > 0.0) == settlement.outcome
    return ProcessGrade.SOUND if aligned else ProcessGrade.FLAWED


class TemplateReflector:
    """Deterministic, LLM-free reference implementation of :class:`Reflector`.

    Emits a fixed, bounded two-sentence lesson built from the settled facts.
    Useful as a baseline and to exercise the cycle in tests; Phase 3 swaps in
    an LLM-backed reflector behind the same protocol.
    """

    def reflect(self, prediction: Prediction, settlement: Settlement) -> Lesson:
        """Build a two-sentence lesson from the settled facts."""
        outcome_grade = settlement.process_outcome
        process_grade = grade_process(prediction, settlement)
        verdict = "correct" if settlement.outcome else "wrong"
        text = (
            f"Called {prediction.ticker} {prediction.direction.value} at "
            f"{prediction.probability:.0%} confidence and was {verdict}, with "
            f"net alpha {settlement.alpha_net:+.4f}. "
            f"Process was {process_grade.value} and the outcome a "
            f"{outcome_grade.value}; weight the declared confidence accordingly."
        )
        return Lesson(
            prediction_id=prediction.prediction_id,
            text=text,
            process_grade=process_grade,
            outcome_grade=outcome_grade,
        )
