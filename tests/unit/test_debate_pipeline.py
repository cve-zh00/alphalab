"""Unit tests for the debate pipeline, run entirely with doubles (no LLM).

Covers the full assembly: consensus aggregation, the bounded debate, the
injection of the *computed* target, the judge's verdict flowing to the
artifact, persistence through the gate, and the demonstration of the inviolable
rule — a fabricated number from the judge never becomes the headline target.
"""

from __future__ import annotations

from datetime import date

from alphalab.app.debate_pipeline import DebatePipeline
from alphalab.app.inputs import AnalystRequest, ValuationBundle
from alphalab.artifact.models import Verdict
from alphalab.artifact.store import InMemoryArtifactStore
from alphalab.domain.analyst import ModelTier, RunMetadata
from alphalab.domain.compute import ProvenancedValue, consensus, leaf
from tests.unit._debate_doubles import StubBear, StubBull, StubJudge


def _estimate(name: str, value: float) -> ProvenancedValue:
    base = leaf(value=value, name=f"{name}_input", unit="USD_per_share")
    return ProvenancedValue(
        value=value,
        operator=name,
        operator_version="1.0.0",
        inputs=(("base", value),),
        unit="USD_per_share",
        parents=(base,),
    )


def _bundle() -> ValuationBundle:
    return ValuationBundle(
        estimates=(
            _estimate("dcf_value_per_share", 40.0),
            _estimate("ddm_value_per_share", 42.0),
            _estimate("pe_implied_value", 44.0),
        ),
        unit="USD_per_share",
    )


def _metadata() -> RunMetadata:
    return RunMetadata(
        model_name="qwen3.8:27b-mlx",
        tier=ModelTier.DEEP,
        llm_version="qwen3.8-27b-mlx",
        llm_cutoff=date(2024, 10, 1),
        knowledge_date=date(2025, 1, 15),
        dataset_hash="deadbeef",
    )


def _request(*, current_price: float = 40.0) -> AnalystRequest:
    return AnalystRequest(
        ticker="SYNTH",
        knowledge_date=date(2025, 1, 15),
        current_price=current_price,
        documents=("Some filing text.",),
        valuation=_bundle(),
        run_metadata=_metadata(),
    )


def _pipeline(judge: StubJudge, *, max_rounds: int = 1) -> DebatePipeline:
    return DebatePipeline(
        bull=StubBull(),
        bear=StubBear(),
        judge=judge,
        store=InMemoryArtifactStore(),
        max_rounds=max_rounds,
    )


class TestAssembly:
    def test_target_equals_computed_consensus(self) -> None:
        request = _request()
        expected = consensus(estimates=request.valuation.estimates, unit="USD_per_share")
        result = _pipeline(StubJudge(verdict=Verdict.BUY)).run(request)
        assert result.artifact.price_target is not None
        assert result.artifact.price_target.value == expected.mid
        assert result.consensus.mid == expected.mid

    def test_verdict_comes_from_judge(self) -> None:
        result = _pipeline(StubJudge(verdict=Verdict.SELL)).run(_request())
        assert result.artifact.verdict is Verdict.SELL

    def test_judge_rationale_in_narrative_slots(self) -> None:
        result = _pipeline(StubJudge(verdict=Verdict.HOLD)).run(_request())
        assert result.artifact.narrative_slots["thesis"]

    def test_transcript_attached_to_result(self) -> None:
        result = _pipeline(StubJudge(), max_rounds=2).run(_request())
        assert len(result.transcript.turns) == 4
        assert result.transcript.ruling.verdict is Verdict.BUY

    def test_numbers_handed_to_judge_are_computed(self) -> None:
        judge = StubJudge()
        _pipeline(judge).run(_request())
        summary = dict(judge.requests[0].valuation_summary)
        assert summary["consensus_mid"] == 42.0


class TestNoNumberInjection:
    """The inviolable rule: a number from the model never becomes the target."""

    def test_fabricated_price_never_becomes_target(self) -> None:
        judge = StubJudge(injected_price=999.0, verdict=Verdict.BUY)
        request = _request()
        expected = consensus(estimates=request.valuation.estimates, unit="USD_per_share")
        result = _pipeline(judge).run(request)
        assert result.artifact.price_target is not None
        assert result.artifact.price_target.value == expected.mid
        assert result.artifact.price_target.value != 999.0

    def test_absurd_number_scrubbed_by_gate(self) -> None:
        # current_price tiny => huge implied upside => hard clause fires and the
        # gate withholds the price and scrubs the headline number from prose.
        judge = StubJudge(verdict=Verdict.BUY)
        result = _pipeline(judge).run(_request(current_price=0.01))
        assert result.artifact.valuation_withheld is True
        assert result.artifact.price_target is None
        assert any(s.startswith("[CONTRACT/") for s in result.artifact.contract_stamps)


class TestDemo:
    """End-to-end demo with doubles proving the full chain of custody."""

    def test_full_debate_demo(self) -> None:
        # Arrange
        request = _request(current_price=41.0)
        pipeline = _pipeline(StubJudge(verdict=Verdict.BUY), max_rounds=2)

        # Act
        result = pipeline.run(request)
        artifact = result.artifact
        transcript = result.transcript

        # (a) the turn sequence respected the hard cap (2 * max_rounds).
        assert len(transcript.turns) == 4
        assert transcript.max_rounds == 2

        # (b) the judge issued a directional verdict citing both sides.
        assert artifact.verdict is Verdict.BUY
        assert transcript.ruling.bull_points_weighed
        assert transcript.ruling.bear_points_weighed

        # (c) the artifact target is the consensus point and is traceable.
        expected = consensus(estimates=request.valuation.estimates, unit="USD_per_share")
        assert artifact.price_target is not None
        assert artifact.price_target.value == expected.mid == 42.0
        point = result.consensus.point
        leaves = point.leaves()
        assert len(leaves) == 3
        leaf_names = {pair[0] for lf in leaves for pair in lf.inputs}
        assert leaf_names == {
            "dcf_value_per_share_input",
            "ddm_value_per_share_input",
            "pe_implied_value_input",
        }
        tree = point.tree()
        assert tree["operator"] == "consensus"
        assert len(tree["parents"]) == 3

        # (d) the full transcript is attached and auditable in order.
        assert [t.turn_number for t in transcript.turns] == [0, 1, 2, 3]

        # (e) the artifact passed through the gate (accepted, not withheld).
        assert artifact.valuation_withheld is False
        assert pipeline._store.findings_for(artifact) is not None
