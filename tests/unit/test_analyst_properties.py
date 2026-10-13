"""Property-based tests for the analyst contracts and pipeline assembly.

Two invariants:

* ``QualitativeFinding`` is immutable for any valid confidence in ``[0, 1]``.
* The pipeline assembly is deterministic: given the same inputs and the same
  (deterministic) doubles, the injected target is always exactly the computed
  consensus median, regardless of the estimate values.
"""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from alphalab.app.inputs import AnalystRequest, ValuationBundle
from alphalab.app.pipeline import AnalystPipeline
from alphalab.artifact.models import Verdict
from alphalab.artifact.store import InMemoryArtifactStore
from alphalab.domain.analyst import Direction, ModelTier, QualitativeFinding, RunMetadata
from alphalab.domain.compute import ProvenancedValue, consensus, leaf
from tests.unit._analyst_doubles import StubExtractor, StubNarrator


@given(confidence=st.floats(min_value=0.0, max_value=1.0))
def test_finding_is_immutable(confidence: float) -> None:
    finding = QualitativeFinding(
        category="c",
        direction=Direction.NEUTRAL,
        evidence_quote="q",
        source="s",
        confidence=confidence,
    )
    with pytest.raises(ValidationError):
        finding.confidence = 0.0  # type: ignore[misc]


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


@given(
    v1=st.floats(min_value=1.0, max_value=500.0),
    v2=st.floats(min_value=1.0, max_value=500.0),
    v3=st.floats(min_value=1.0, max_value=500.0),
)
def test_target_always_equals_computed_consensus(v1: float, v2: float, v3: float) -> None:
    bundle = ValuationBundle(
        estimates=(
            _estimate("dcf_value_per_share", v1),
            _estimate("ddm_value_per_share", v2),
            _estimate("pe_implied_value", v3),
        ),
        unit="USD_per_share",
    )
    # Use the median as the reference price so the C1 upside band never fires;
    # this isolates the assembly-determinism property from the gate.
    median = sorted((v1, v2, v3))[1]
    request = AnalystRequest(
        ticker="SYNTH",
        knowledge_date=date(2025, 1, 15),
        current_price=median,
        documents=("doc",),
        valuation=bundle,
        run_metadata=RunMetadata(
            model_name="m",
            tier=ModelTier.DEEP,
            llm_version="v",
            llm_cutoff=date(2024, 1, 1),
            knowledge_date=date(2025, 1, 15),
            dataset_hash="h",
        ),
    )
    pipeline = AnalystPipeline(
        extractor=StubExtractor(),
        narrator=StubNarrator(verdict=Verdict.HOLD),
        store=InMemoryArtifactStore(),
    )
    expected = consensus(estimates=bundle.estimates, unit="USD_per_share")
    result = pipeline.run(request)
    assert result.consensus.mid == expected.mid
    if result.artifact.price_target is not None:
        assert result.artifact.price_target.value == expected.mid
