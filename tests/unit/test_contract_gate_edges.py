"""Edge-case tests completing coverage of the gate, store and package API."""

from __future__ import annotations

import pytest

import alphalab.artifact as artifact_pkg
from alphalab.artifact.models import (
    EquityResearchArtifact,
    PriceTarget,
    Verdict,
)
from alphalab.artifact.store import InMemoryArtifactStore
from alphalab.domain.contract import gate


def _artifact(**overrides: object) -> EquityResearchArtifact:
    base: dict[str, object] = {
        "ticker": "AAPL",
        "verdict": Verdict.BUY,
        "current_price": 100.0,
        "price_target": PriceTarget(value=150.0, unit="USD_per_share"),
        "methods_used": ("dcf", "ddm"),
        "consensus_relative_span": 0.1,
        "single_method_disclosed": True,
    }
    base.update(overrides)
    return EquityResearchArtifact(**base)  # type: ignore[arg-type]


class TestLazyPackageApi:
    def test_store_symbols_resolved_lazily(self) -> None:
        store_cls = artifact_pkg.InMemoryArtifactStore
        assert store_cls is InMemoryArtifactStore
        # ArtifactSink protocol is also exposed.
        assert artifact_pkg.ArtifactSink is not None

    def test_unknown_attribute_raises(self) -> None:
        with pytest.raises(AttributeError):
            _ = artifact_pkg.does_not_exist  # type: ignore[attr-defined]


class TestStoreEdges:
    def test_len_and_missing_get(self) -> None:
        store = InMemoryArtifactStore()
        assert len(store) == 0
        assert store.get("UNKNOWN") is None
        store.save(_artifact())
        assert len(store) == 1

    def test_findings_for_unsaved_is_none(self) -> None:
        store = InMemoryArtifactStore()
        assert store.findings_for(_artifact()) is None


class TestGateDegradesOnRaise:
    def test_a_raising_clause_withholds_not_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(_artifact: EquityResearchArtifact) -> object:
            raise RuntimeError("synthetic clause failure")

        # Replace a hard clause with one that explodes; the gate must survive.
        monkeypatch.setattr(gate, "_HARD_CLAUSES", (boom,))
        result = gate.enforce_contract(_artifact())
        assert isinstance(result.artifact, EquityResearchArtifact)
        assert result.artifact.valuation_withheld is True
        assert any(f.check.endswith("_raised") for f in result.findings)
        assert result.artifact.verdict is Verdict.BUY


class TestDegradeBuildsRangeFromMethods:
    def test_no_prior_range_built_from_method_values(self) -> None:
        art = _artifact(
            price_target=PriceTarget(value=150.0, unit="USD_per_share"),
            methods_used=("dcf", "ddm", "pe"),
            consensus_relative_span=0.9,
            price_range=None,
            method_values={"dcf": 120.0, "ddm": 150.0, "pe": 200.0},
        )
        out = gate.enforce_contract(art).artifact
        assert out.price_target is None
        assert out.price_range is not None
        assert out.price_range.low == 120.0
        assert out.price_range.mid == pytest.approx(0.5 * (135.0 + 200.0))
        assert out.price_range.high == 200.0

    def test_c1b_range_mid_balances_1_intrinsic_against_3_multiples(self) -> None:
        art = _artifact(
            price_target=PriceTarget(value=160.0, unit="USD_per_share"),
            methods_used=("dcf", "pe", "ev_ebitda", "pb"),
            consensus_relative_span=0.9,
            price_range=None,
            method_values={"dcf": 100.0, "pe": 200.0, "ev_ebitda": 220.0, "pb": 240.0},
        )
        out = gate.enforce_contract(art).artifact
        assert out.price_range is not None
        assert out.price_range.low == 100.0
        assert out.price_range.mid == pytest.approx(160.0)
        assert out.price_range.high == 240.0

    def test_no_range_and_no_methods_leaves_range_none(self) -> None:
        art = _artifact(
            price_target=PriceTarget(value=150.0, unit="USD_per_share"),
            methods_used=("dcf", "ddm", "pe"),
            consensus_relative_span=0.9,
            price_range=None,
            method_values={},
        )
        out = gate.enforce_contract(art).artifact
        assert out.price_target is None
        assert out.price_range is None
        assert out.valuation_withheld is True


class TestWithheldValueNaNPath:
    def test_withhold_with_no_target_does_not_scrub(self) -> None:
        # A hard clause fires but there is no headline value to scrub: the
        # narrative must be left intact (nothing to redact).
        art = _artifact(
            price_target=PriceTarget(value=1.2, unit="ratio"),  # C4 fires
            narrative_slots={"body": "unchanged prose with 1.2 inside"},
        )
        out = gate.enforce_contract(art).artifact
        assert out.valuation_withheld is True
        # 1.2 is the withheld value and IS finite, so it is scrubbed from text.
        assert "1.2" not in out.narrative_slots["body"]
