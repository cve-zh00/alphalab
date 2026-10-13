"""Deterministic test doubles implementing the analyst domain ports.

These doubles let the pipeline tests run with zero network and zero LLM
latency while exercising the full assembly and gating logic. One double can be
told to *attempt* to smuggle a fabricated price target into the narrative prose
so the "numbers come from code, never the model" rule can be proven.
"""

from __future__ import annotations

from alphalab.artifact.models import Verdict
from alphalab.domain.analyst.contracts import (
    ExtractionRequest,
    NarrationRequest,
    QualitativeFinding,
    ThesisNarrative,
)
from alphalab.domain.analyst.direction import Direction


class StubExtractor:
    """Returns a fixed, deterministic set of qualitative findings."""

    def __init__(self, findings: tuple[QualitativeFinding, ...] | None = None) -> None:
        self._findings = findings or (
            QualitativeFinding(
                category="management_guidance",
                direction=Direction.POSITIVE,
                evidence_quote="Management reaffirmed full-year guidance.",
                source="Q3 call",
                confidence=0.7,
            ),
        )
        self.last_request: ExtractionRequest | None = None

    def extract(self, request: ExtractionRequest) -> tuple[QualitativeFinding, ...]:
        self.last_request = request
        return self._findings


class StubNarrator:
    """Returns a fixed narrative; can be told to inject a fake number in prose.

    When ``injected_price`` is set, the double writes that fabricated figure
    into the thesis prose — simulating a model that tries to smuggle a
    valuation number. The pipeline/gate must ensure that number never becomes
    the artifact's headline target.
    """

    def __init__(
        self,
        *,
        verdict: Verdict = Verdict.BUY,
        injected_price: float | None = None,
    ) -> None:
        self._verdict = verdict
        self._injected_price = injected_price
        self.last_request: NarrationRequest | None = None

    def narrate(self, request: NarrationRequest) -> ThesisNarrative:
        self.last_request = request
        thesis = "Durable franchise with expanding margins."
        if self._injected_price is not None:
            thesis = (
                f"We see fair value of ${self._injected_price:.2f} per share "
                "based on our own model."
            )
        return ThesisNarrative(
            thesis=thesis,
            risks="Regulatory tightening could compress returns.",
            catalysts="Buyback and new product cycle.",
            verdict=self._verdict,
        )
