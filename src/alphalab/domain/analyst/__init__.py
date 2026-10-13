"""Pure contracts for the equity-research analyst agent (Phase 2).

This package holds *only* immutable Pydantic models and typing ``Protocol``s —
the structured shapes the LLM must produce and the ports the adapter layer
implements. It is deliberately pure: no PydanticAI, no I/O, no ``adapters``/
``app`` imports (enforced by ``tests/unit/test_architecture.py``).

The design encodes the inviolable rule of this phase (dossier §0, FinRobot
"numbers are code-calculated, narratives are LLM-assisted"):

* The LLM **extracts** qualitative facts from text into
  :class:`QualitativeFinding` — no free numeric valuation fields, only a
  bounded confidence in ``[0, 1]``.
* The LLM **narrates** into :class:`ThesisNarrative` — pure text slots plus a
  directional :class:`~alphalab.artifact.models.Verdict`. There is **no price
  target field**: the number is injected downstream from ``domain/compute``,
  so a valuation number is *structurally* impossible to emit here.
* Every run carries mandatory :class:`RunMetadata` (model name/tier,
  ``llm_version``, ``llm_cutoff``, ``knowledge_date``, dataset hash) — first
  class, never optional (dossier §6).

The ports (:class:`ExtractionProvider`, :class:`NarrationProvider`) are typed
only against these pure contracts so the domain never learns about PydanticAI
and the pipeline is testable with deterministic doubles.
"""

from __future__ import annotations

from alphalab.domain.analyst.contracts import (
    ExtractionRequest,
    NarrationRequest,
    QualitativeFinding,
    RunMetadata,
    ThesisNarrative,
)
from alphalab.domain.analyst.direction import Direction
from alphalab.domain.analyst.ports import ExtractionProvider, NarrationProvider
from alphalab.domain.analyst.tiers import ModelTier

__all__ = [
    "Direction",
    "ExtractionProvider",
    "ExtractionRequest",
    "ModelTier",
    "NarrationProvider",
    "NarrationRequest",
    "QualitativeFinding",
    "RunMetadata",
    "ThesisNarrative",
]
