"""Ports (typing ``Protocol``s) for the LLM providers.

Defined in the pure domain and typed *only* against the pure contracts, so the
domain never imports PydanticAI. The adapter layer (``adapters/llm.py``)
implements these; tests implement them with deterministic doubles. This is the
hexagonal seam that keeps the pipeline testable without the network.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from alphalab.domain.analyst.contracts import (
    ExtractionRequest,
    NarrationRequest,
    QualitativeFinding,
    ThesisNarrative,
)


@runtime_checkable
class ExtractionProvider(Protocol):
    """Extracts structured qualitative findings from source text (quick-think)."""

    def extract(self, request: ExtractionRequest) -> tuple[QualitativeFinding, ...]:
        """Return the facts extracted from ``request.documents``."""
        ...


@runtime_checkable
class NarrationProvider(Protocol):
    """Synthesises a thesis narrative from findings + numeric context (deep-think)."""

    def narrate(self, request: NarrationRequest) -> ThesisNarrative:
        """Return the synthesised narrative for ``request``."""
        ...
