"""Directional sense of a qualitative finding.

Kept separate from the artifact's :class:`~alphalab.artifact.models.Verdict`
on purpose: a *finding* has a direction (does this fact help or hurt the
thesis?), which is a weaker, orthogonal signal to the final BUY/HOLD/SELL
call. The LLM may assign a direction to a fact it extracted; it may never
assign a magnitude.
"""

from __future__ import annotations

from enum import StrEnum


class Direction(StrEnum):
    """Qualitative sense of an extracted fact relative to the thesis."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
