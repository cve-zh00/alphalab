"""Unit tests for the gated artifact store (the single persistence boundary)."""

from __future__ import annotations

from alphalab.artifact.models import (
    EquityResearchArtifact,
    PriceTarget,
    Verdict,
)
from alphalab.artifact.store import InMemoryArtifactStore


def _bad_artifact() -> EquityResearchArtifact:
    return EquityResearchArtifact(
        ticker="NVDA",
        verdict=Verdict.BUY,
        current_price=100.0,
        price_target=PriceTarget(value=1000.0, unit="USD_per_share"),
        methods_used=("dcf",),
        single_method_disclosed=True,
    )


def _good_artifact() -> EquityResearchArtifact:
    return EquityResearchArtifact(
        ticker="AAPL",
        verdict=Verdict.HOLD,
        current_price=100.0,
        price_target=PriceTarget(value=150.0, unit="USD_per_share"),
        methods_used=("dcf", "ddm"),
        consensus_relative_span=0.1,
        single_method_disclosed=True,
    )


class TestSaveRunsGate:
    def test_save_enforces_contract_on_bad_artifact(self) -> None:
        store = InMemoryArtifactStore()
        out = store.save(_bad_artifact())
        assert out.valuation_withheld is True
        assert out.price_target is None

    def test_save_returns_enforced_and_stores_enforced(self) -> None:
        store = InMemoryArtifactStore()
        out = store.save(_bad_artifact())
        fetched = store.get("NVDA")
        assert fetched is out  # exactly the enforced artifact was stored.
        assert fetched is not None and fetched.valuation_withheld is True

    def test_good_artifact_persists_with_target(self) -> None:
        store = InMemoryArtifactStore()
        out = store.save(_good_artifact())
        assert out.price_target is not None
        assert out.valuation_withheld is False


class TestNoBypass:
    def test_no_public_raw_save_method(self) -> None:
        # The only persistence method is `save`, which always runs the gate.
        store = InMemoryArtifactStore()
        public = {n for n in dir(store) if not n.startswith("_")}
        assert "save" in public
        # No alternative write path is exposed.
        assert not any(name in public for name in ("save_raw", "put", "insert", "write", "store"))

    def test_internal_buffer_is_private(self) -> None:
        store = InMemoryArtifactStore()
        # __slots__ keeps state private; no public buffer attribute exists.
        assert not hasattr(store, "buffer")
        assert "_buffer" in InMemoryArtifactStore.__slots__

    def test_findings_recorded_for_saved_artifact(self) -> None:
        store = InMemoryArtifactStore()
        out = store.save(_bad_artifact())
        result = store.findings_for(out)
        assert result is not None
        assert any(not f.passed for f in result.findings)
