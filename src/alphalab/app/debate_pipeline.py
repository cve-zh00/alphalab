"""The debate pipeline: numbers from compute, verdict from a bounded debate.

This is the Phase-3 debate variant of :class:`~alphalab.app.pipeline.
AnalystPipeline`. It is **interchangeable** with the baseline for the Task 14
A/B: it takes the same :class:`~alphalab.app.inputs.AnalystRequest` and returns
a result whose ``.artifact`` is a gated
:class:`~alphalab.artifact.models.EquityResearchArtifact`, so the experiment
can swap one for the other.

The inviolable rule is unchanged (dossier §0, FinRobot): **every number comes
from ``domain/compute``; the LLM debate only argues and judges.** The headline
``price_target`` is the computed ``consensus`` point; the debate supplies only
the directional verdict and prose. The full :class:`DebateTranscript` is
attached to the result for audit and persisted-prose summary.

Flow:

1. Aggregate provenanced estimates with ``consensus`` (all numbers here).
2. Run the bounded debate via the pure conductor (router hard-caps turns).
3. The judge resolves to a directional verdict.
4. Assemble the artifact with the **computed** target and the debate prose.
5. Persist with ``store.save`` (the contract gate runs, mandatory).
"""

from __future__ import annotations

from alphalab.app.inputs import AnalystRequest, ValuationBundle
from alphalab.artifact.models import EquityResearchArtifact
from alphalab.artifact.store import InMemoryArtifactStore
from alphalab.domain.analyst.contracts import RunMetadata
from alphalab.domain.compute import ConsensusRange, ProvenancedValue, consensus
from alphalab.domain.debate.conductor import conduct_debate
from alphalab.domain.debate.contracts import DebateTranscript
from alphalab.domain.debate.ports import BearProvider, BullProvider, JudgeProvider


class DebateResult:
    """Immutable outcome of one debate-pipeline run.

    Attributes:
        artifact: The gated artifact as persisted (gate output).
        transcript: The full auditable debate transcript (turns + ruling).
        consensus: The aggregated valuation whose point was injected.
        run_metadata: The run's first-class provenance.
    """

    __slots__ = ("artifact", "consensus", "run_metadata", "transcript")

    def __init__(
        self,
        *,
        artifact: EquityResearchArtifact,
        transcript: DebateTranscript,
        consensus: ConsensusRange,
        run_metadata: RunMetadata,
    ) -> None:
        self.artifact = artifact
        self.transcript = transcript
        self.consensus = consensus
        self.run_metadata = run_metadata


class DebatePipeline:
    """Assemble a gated equity-research artifact from numbers and a debate.

    Depends only on the pure debate ports, so it is fully testable with
    deterministic doubles. The concrete LLM adapter is injected from the
    composition root. Interchangeable with ``AnalystPipeline`` for the A/B.
    """

    __slots__ = ("_bear", "_bull", "_judge", "_max_rounds", "_store")

    def __init__(
        self,
        *,
        bull: BullProvider,
        bear: BearProvider,
        judge: JudgeProvider,
        store: InMemoryArtifactStore,
        max_rounds: int = 1,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self._bull = bull
        self._bear = bear
        self._judge = judge
        self._store = store
        self._max_rounds = max_rounds

    def run(self, request: AnalystRequest) -> DebateResult:
        """Execute the full debate pipeline and return the gated result."""
        consensus_range = self._aggregate(request.valuation)
        transcript = self._debate(request, consensus_range)
        artifact = self._assemble(request, consensus_range, transcript)
        gated = self._store.save(artifact)
        return DebateResult(
            artifact=gated,
            transcript=transcript,
            consensus=consensus_range,
            run_metadata=request.run_metadata,
        )

    @staticmethod
    def _aggregate(valuation: ValuationBundle) -> ConsensusRange:
        """Aggregate provenanced method estimates into a consensus."""
        return consensus(estimates=valuation.estimates, unit=valuation.unit)

    def _debate(
        self,
        request: AnalystRequest,
        consensus_range: ConsensusRange,
    ) -> DebateTranscript:
        """Run the bounded debate around the computed numbers."""
        summary = _valuation_summary(request, consensus_range)
        return conduct_debate(
            ticker=request.ticker,
            evidence=request.documents,
            valuation_summary=summary,
            max_rounds=self._max_rounds,
            bull=self._bull,
            bear=self._bear,
            judge=self._judge,
        )

    @staticmethod
    def _assemble(
        request: AnalystRequest,
        consensus_range: ConsensusRange,
        transcript: DebateTranscript,
    ) -> EquityResearchArtifact:
        """Build the artifact, injecting the *computed* consensus point.

        ``price_target`` is the ``consensus`` point — a ``ProvenancedValue`` —
        so the headline is code-computed and traceable to its leaves. The
        verdict is the judge's; the debate prose goes into ``narrative_slots``.
        No number from any debater touches the target.
        """
        point: ProvenancedValue = consensus_range.point
        method_values = {est.operator: est.value for est in request.valuation.estimates}
        ruling = transcript.ruling
        return EquityResearchArtifact(
            ticker=request.ticker,
            knowledge_date=request.knowledge_date,
            verdict=ruling.verdict,
            price_target=point,
            current_price=request.current_price,
            methods_used=tuple(method_values.keys()),
            method_values=method_values,
            consensus_relative_span=consensus_range.relative_span,
            narrative_slots={
                "thesis": ruling.rationale,
                "bull_points": " | ".join(ruling.bull_points_weighed),
                "bear_points": " | ".join(ruling.bear_points_weighed),
            },
        )


def _valuation_summary(
    request: AnalystRequest,
    consensus_range: ConsensusRange,
) -> tuple[tuple[str, float], ...]:
    """Pre-computed numeric context handed to the debaters and the judge."""
    return (
        ("consensus_mid", consensus_range.mid),
        ("consensus_low", consensus_range.low),
        ("consensus_high", consensus_range.high),
        ("current_price", request.current_price),
    )
