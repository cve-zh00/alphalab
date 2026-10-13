"""Unit tests for the analyst pipeline, run entirely with doubles (no LLM).

Covers the full assembly: consensus aggregation, LLM text consumption, the
injection of the *computed* target, persistence through the gate, and — the
demonstration of the inviolable rule — that a fabricated number produced by the
model never becomes the artifact's headline target.
"""

from __future__ import annotations

from datetime import date

from alphalab.app.inputs import AnalystRequest, ValuationBundle
from alphalab.app.pipeline import AnalystPipeline
from alphalab.artifact.models import Verdict
from alphalab.artifact.store import InMemoryArtifactStore
from alphalab.domain.analyst import ModelTier, RunMetadata
from alphalab.domain.compute import ProvenancedValue, consensus, leaf
from tests.unit._analyst_doubles import StubExtractor, StubNarrator


def _estimate(name: str, value: float) -> ProvenancedValue:
    """A tiny provenanced estimate built from a traceable leaf input."""
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


def _pipeline(narrator: StubNarrator) -> AnalystPipeline:
    return AnalystPipeline(
        extractor=StubExtractor(),
        narrator=narrator,
        store=InMemoryArtifactStore(),
    )


class TestAssembly:
    def test_target_equals_computed_consensus(self) -> None:
        # Arrange
        request = _request()
        expected = consensus(estimates=request.valuation.estimates, unit="USD_per_share")
        pipeline = _pipeline(StubNarrator(verdict=Verdict.BUY))

        # Act
        result = pipeline.run(request)

        # Assert: the artifact target is exactly the computed consensus point.
        assert result.artifact.price_target is not None
        assert result.artifact.price_target.value == expected.mid
        assert result.consensus.mid == expected.mid

    def test_verdict_comes_from_narrative(self) -> None:
        result = _pipeline(StubNarrator(verdict=Verdict.SELL)).run(_request())
        assert result.artifact.verdict is Verdict.SELL

    def test_narrative_prose_is_present(self) -> None:
        result = _pipeline(StubNarrator(verdict=Verdict.HOLD)).run(_request())
        slots = result.artifact.narrative_slots
        assert slots["thesis"]
        assert slots["risks"]
        assert slots["catalysts"]

    def test_numbers_passed_to_narrator_are_computed(self) -> None:
        narrator = StubNarrator()
        _pipeline(narrator).run(_request())
        assert narrator.last_request is not None
        summary = dict(narrator.last_request.valuation_summary)
        assert "consensus_mid" in summary
        # The number handed to the model is the code-computed median.
        assert summary["consensus_mid"] == 42.0


class TestNoNumberInjection:
    """The inviolable rule: a number from the model never becomes the target."""

    def test_fabricated_price_never_becomes_target(self) -> None:
        # Arrange: a narrator that writes a bogus $999 fair value into prose.
        narrator = StubNarrator(injected_price=999.0, verdict=Verdict.BUY)
        request = _request()
        expected = consensus(estimates=request.valuation.estimates, unit="USD_per_share")

        # Act
        result = _pipeline(narrator).run(request)

        # Assert: target is the computed consensus, never the model's $999.
        assert result.artifact.price_target is not None
        assert result.artifact.price_target.value == expected.mid
        assert result.artifact.price_target.value != 999.0

    def test_absurd_narrative_number_is_scrubbed_by_gate(self) -> None:
        # A current price far below the target trips the C1 upside-band hard
        # clause, so the gate withholds the price and scrubs the headline
        # number from the narrative prose. We prove the gate cleans an
        # out-of-band number embedded in text.
        narrator = StubNarrator(verdict=Verdict.BUY)
        # current_price tiny => huge implied upside => hard clause fires.
        request = _request(current_price=0.01)
        result = _pipeline(narrator).run(request)

        assert result.artifact.valuation_withheld is True
        assert result.artifact.price_target is None
        assert any(stamp.startswith("[CONTRACT/") for stamp in result.artifact.contract_stamps)


class TestDemo:
    """End-to-end demo with doubles proving the full chain of custody."""

    def test_full_artifact_demo(self) -> None:
        # Arrange
        request = _request(current_price=41.0)
        pipeline = _pipeline(StubNarrator(verdict=Verdict.BUY))

        # Act
        result = pipeline.run(request)
        artifact = result.artifact

        # (a) target == consensus and traceable to leaf inputs.
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

        # (b) narrative present.
        assert artifact.narrative_slots["thesis"]

        # (c) RunMetadata carries model + cutoff.
        assert result.run_metadata.model_name == "qwen3.8:27b-mlx"
        assert result.run_metadata.llm_cutoff == date(2024, 10, 1)

        # (d) artifact went through the gate and was accepted (not withheld).
        assert artifact.valuation_withheld is False
        assert pipeline._store.findings_for(artifact) is not None
