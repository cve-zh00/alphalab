"""Bridging research output to the forward-evaluation journal.

An agentic research system cannot be backtested: the language model's training
data already contains the outcome of any historical date, so a "prediction"
about the past is an oracle, not a forecast. The only honest evaluation is
forward — dated claims, settled when their horizon arrives, scored for
calibration.

This package is the bridge that starts that clock: it converts a research
verdict into a scoreable :class:`~alphalab.domain.journal.Prediction`, or
abstains with a stated reason.
"""

from __future__ import annotations

from alphalab.domain.forecast.builder import Abstention, ForecastInputs, build_forecast
from alphalab.domain.forecast.integrity import (
    FORWARD_TOLERANCE,
    is_forward,
    partition_by_integrity,
)
from alphalab.domain.forecast.probability import (
    BASE_RATE,
    EVIDENCE_SCALE,
    MAX_CONFIDENCE,
    forecast_probability,
)

__all__ = [
    "BASE_RATE",
    "EVIDENCE_SCALE",
    "FORWARD_TOLERANCE",
    "MAX_CONFIDENCE",
    "Abstention",
    "ForecastInputs",
    "build_forecast",
    "forecast_probability",
    "is_forward",
    "partition_by_integrity",
]
