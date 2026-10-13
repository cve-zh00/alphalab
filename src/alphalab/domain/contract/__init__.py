"""Output contract gate for the research artifact.

Pure, immutable clauses plus a gate that enforces them at the single
persistence boundary with a *value-withhold* discipline: a corrupt or
out-of-band number is withheld, the directional verdict and narrative are
preserved, and the gate never raises.
"""

from __future__ import annotations

from alphalab.domain.contract.gate import GateResult, enforce_contract
from alphalab.domain.contract.thresholds import (
    METHOD_CORROBORATION_SPAN_K,
    MULTI_METHOD_RATIO_K,
    NARRATIVE_DRIFT_TOLERANCE,
    SINGLE_METHOD_RATIO_K,
)

__all__ = [
    "METHOD_CORROBORATION_SPAN_K",
    "MULTI_METHOD_RATIO_K",
    "NARRATIVE_DRIFT_TOLERANCE",
    "SINGLE_METHOD_RATIO_K",
    "GateResult",
    "enforce_contract",
]
