"""The analyst pipeline: numbers from compute, narrative from the LLM, gated.

The pipeline is a small, typed state machine, not an open conversation
(FinRobot pattern). Each step has a single responsibility and the number flows
one way only: ``consensus`` → artifact ``price_target``. The LLM output is
consumed solely for text (findings + narrative verdict/prose); it is never read
for a magnitude.
"""

from __future__ import annotations

from alphalab.app.inputs import AnalystRequest, ValuationBundle
from alphalab.artifact.models import EquityResearchArtifact
from alphalab.artifact.store import InMemoryArtifactStore
from alphalab.domain.analyst.contracts import (
    ExtractionRequest,
    NarrationRequest,
    QualitativeFinding,
    RunMetadata,
    ThesisNarrative,
)
from alphalab.domain.analyst.ports import ExtractionProvider, NarrationProvider
from alphalab.domain.compute import ConsensusRange, ProvenancedValue, consensus


class AnalystResult:
    """Immutable outcome of one pipeline run.

    Attributes:
        artifact: The gated artifact as persisted (gate output).
        narrative: The LLM narrative that fed the artifact.
        findings: The qualitative facts the LLM extracted.
        consensus: The aggregated valuation whose point was injected.
        run_metadata: The run's first-class provenance.
    """

    __slots__ = ("artifact", "consensus", "findings", "narrative", "run_metadata")

    def __init__(
        self,
        *,
        artifact: EquityResearchArtifact,
        narrative: ThesisNarrative,
        findings: tuple[QualitativeFinding, ...],
        consensus: ConsensusRange,
        run_metadata: RunMetadata,
    ) -> None:
        self.artifact = artifact
        self.narrative = narrative
        self.findings = findings
        self.consensus = consensus
        self.run_metadata = run_metadata


class AnalystPipeline:
    """Assemble a gated equity-research artifact from numbers and LLM text.

    The pipeline depends only on the pure domain ports, so it is fully testable
    with deterministic doubles. The concrete LLM adapter is injected from the
    composition root.
    """

    __slots__ = ("_extractor", "_narrator", "_store")

    def __init__(
        self,
        *,
        extractor: ExtractionProvider,
        narrator: NarrationProvider,
        store: InMemoryArtifactStore,
    ) -> None:
        self._extractor = extractor
        self._narrator = narrator
        self._store = store

    def run(self, request: AnalystRequest) -> AnalystResult:
        """Execute the full pipeline and return the gated result."""
        consensus_range = self._aggregate(request.valuation)
        findings = self._extract(request)
        narrative = self._narrate(request, findings, consensus_range)
        artifact = self._assemble(request, consensus_range, narrative)
        gated = self._store.save(artifact)
        return AnalystResult(
            artifact=gated,
            narrative=narrative,
            findings=findings,
            consensus=consensus_range,
            run_metadata=request.run_metadata,
        )

    @staticmethod
    def _aggregate(valuation: ValuationBundle) -> ConsensusRange:
        """Aggregate provenanced method estimates into a consensus."""
        return consensus(estimates=valuation.estimates, unit=valuation.unit)

    def _extract(self, request: AnalystRequest) -> tuple[QualitativeFinding, ...]:
        """Ask the quick-think provider for qualitative facts only."""
        return self._extractor.extract(
            ExtractionRequest(ticker=request.ticker, documents=request.documents)
        )

    def _narrate(
        self,
        request: AnalystRequest,
        findings: tuple[QualitativeFinding, ...],
        consensus_range: ConsensusRange,
    ) -> ThesisNarrative:
        """Ask the deep-think provider to narrate around computed numbers."""
        summary = (
            ("consensus_mid", consensus_range.mid),
            ("consensus_low", consensus_range.low),
            ("consensus_high", consensus_range.high),
            ("current_price", request.current_price),
        )
        return self._narrator.narrate(
            NarrationRequest(
                ticker=request.ticker,
                findings=findings,
                valuation_summary=summary,
            )
        )

    @staticmethod
    def _assemble(
        request: AnalystRequest,
        consensus_range: ConsensusRange,
        narrative: ThesisNarrative,
    ) -> EquityResearchArtifact:
        """Build the artifact, injecting the *computed* consensus point.

        ``price_target`` is the ``consensus`` point — a ``ProvenancedValue`` —
        so the headline is the code-computed number and is traceable to its
        leaves. The LLM's narrative text goes into ``narrative_slots``; its
        verdict into ``verdict``. No number from the model touches the target.
        """
        point: ProvenancedValue = consensus_range.point
        method_values = {est.operator: est.value for est in request.valuation.estimates}
        return EquityResearchArtifact(
            ticker=request.ticker,
            knowledge_date=request.knowledge_date,
            verdict=narrative.verdict,
            price_target=point,
            current_price=request.current_price,
            methods_used=tuple(method_values.keys()),
            method_values=method_values,
            consensus_relative_span=consensus_range.relative_span,
            narrative_slots={
                "thesis": narrative.thesis,
                "risks": narrative.risks,
                "catalysts": narrative.catalysts,
            },
        )
