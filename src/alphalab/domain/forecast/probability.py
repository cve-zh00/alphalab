"""Turning research evidence into a scoreable probability — in code, never by the LLM.

Design constraints this module answers
--------------------------------------
A prediction is only worth logging if it can be scored, which means it needs a
number. But the project's standing invariant is that the language model never
emits a figure: narratives are LLM-assisted, numbers are code-calculated. A
confidence level is a number, so it is computed here from evidence the
valuation layer produced, not asked of the model.

The second constraint is humility. A discounted cash flow plus a language
model's narrative does not justify claiming 95% confidence about a single
stock over one quarter. The mapping is therefore **bounded**: however
compelling the evidence, confidence saturates well short of certainty. An
unbounded mapping would mint overconfidence, and the journal would spend months
measuring a flaw that could have been excluded by construction.

The base rate is deliberately set at even odds. Over a quarter, whether an
individual US equity beats a broad index is close to a coin flip — slightly
worse than even, in fact, since index returns are driven by a minority of
names. Starting anywhere above 0.5 would smuggle in an edge that has not been
demonstrated.

These constants are hypotheses, not truths. The journal's calibration report
exists to falsify them: persistent signed overconfidence means ``MAX_CONFIDENCE``
is too high or the evidence ratio is not informative.
"""

from __future__ import annotations

import math

__all__ = [
    "BASE_RATE",
    "EVIDENCE_SCALE",
    "MAX_CONFIDENCE",
    "forecast_probability",
]

#: Probability assigned when the evidence carries no information. Even odds.
BASE_RATE = 0.50

#: Hard ceiling on confidence. A valuation model plus a narrative does not
#: license more than this about one name over one horizon.
MAX_CONFIDENCE = 0.70

#: Evidence ratio that maps to roughly three quarters of the available
#: confidence range. An upside equal to the consensus dispersion is decent but
#: not overwhelming support, and lands near 0.62.
EVIDENCE_SCALE = 1.0


def forecast_probability(
    *,
    implied_upside: float | None,
    relative_span: float | None,
    valuation_withheld: bool,
    overconfidence_penalty: float = 0.0,
) -> float:
    """Confidence that a directional call will be right, in ``[BASE_RATE, MAX_CONFIDENCE]``.

    Evidence strength is the implied upside measured against the disagreement
    between valuation methods: a large expected move means little if the
    methods that produced it are far apart. Dividing by the dispersion makes
    the two cases distinguishable, which a raw upside cannot do.

    Returns :data:`BASE_RATE` exactly when the valuation was withheld by the
    contract gate, or when either input is missing. That is the honest encoding
    of "the verdict stands but its quantitative basis did not survive review":
    a directional opinion with no measurable support deserves no more than even
    odds, and must not be silently dropped either.

    Args:
        implied_upside: Expected move as a fraction of the current price;
            sign is ignored, since direction is carried separately.
        relative_span: Dispersion across valuation methods, as a fraction.
        valuation_withheld: Whether the contract gate suppressed the figure.
        overconfidence_penalty: Optional positive shrinkage derived from
            observed journal overconfidence ``mean(p) - mean(outcome)``.
    """
    if valuation_withheld or implied_upside is None or relative_span is None:
        return BASE_RATE

    evidence = _evidence_ratio(abs(implied_upside), relative_span)
    headroom = MAX_CONFIDENCE - BASE_RATE
    raw = BASE_RATE + headroom * math.tanh(evidence / EVIDENCE_SCALE)
    if overconfidence_penalty > 0.0:
        raw = max(BASE_RATE, raw - overconfidence_penalty)
    return min(MAX_CONFIDENCE, raw)


def _evidence_ratio(upside: float, span: float) -> float:
    """Upside per unit of method disagreement.

    A zero span would mean the methods agreed perfectly, which in practice
    signals a single surviving method rather than genuine consensus. It is
    floored so that case cannot divide its way to maximum confidence.
    """
    floor = 0.05
    return upside / max(span, floor)
