"""The analyst use case: assemble a gated research artifact from numbers + text.

This is the Phase-2 baseline agent (the one Phase 3's debate variants must
beat). It wires the pure pieces together under one inviolable rule: **every
number comes from ``domain/compute``; the LLM only extracts facts and narrates.**

Flow (dossier §0/§4, FinRobot typed pipeline):

1. Resolve features **only** through ``resolve_feature`` (structural
   anti-leakage; raises :class:`~alphalab.domain.pit.LeakageError` on
   look-ahead).
2. Compute every number with ``domain/compute`` (DCF / DDM / multiples →
   ``consensus``).
3. Ask the LLM for the qualitative extraction and the narrative **only**.
4. Assemble the :class:`~alphalab.artifact.models.EquityResearchArtifact`,
   injecting the **computed** consensus target — never a number from the model.
5. Persist with ``store.save``, which runs the contract gate.

The target injected into the artifact is the ``ConsensusRange.point`` — a
``ProvenancedValue`` — so the headline is traceable to its leaves via
``tree()``/``leaves()``.
"""

from __future__ import annotations

from alphalab.app.inputs import AnalystRequest, ValuationBundle
from alphalab.app.pipeline import AnalystPipeline, AnalystResult

__all__ = [
    "AnalystPipeline",
    "AnalystRequest",
    "AnalystResult",
    "ValuationBundle",
]
