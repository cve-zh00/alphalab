"""Unit tests for the output contract gate and the required demo.

Covers the central value-withhold contract, no-resurrection, degradation to a
range, graceful never-raise behaviour, and the literal "withholds the price,
never the judgment" demonstration.
"""

from __future__ import annotations

import math

import pytest

from alphalab.artifact.models import (
    EquityResearchArtifact,
    PriceRange,
    PriceTarget,
    Verdict,
)
from alphalab.domain.contract import enforce_contract


def _good_artifact(**overrides: object) -> EquityResearchArtifact:
    base: dict[str, object] = {
        "ticker": "AAPL",
        "verdict": Verdict.BUY,
        "current_price": 100.0,
        "price_target": PriceTarget(value=150.0, unit="USD_per_share"),
        "methods_used": ("dcf", "ddm", "pe"),
        "consensus_relative_span": 0.10,
        "narrative_slots": {"conclusion": "Our target is $150.00 per share."},
        "single_method_disclosed": True,
    }
    base.update(overrides)
    return EquityResearchArtifact(**base)  # type: ignore[arg-type]


class TestCleanArtifact:
    def test_clean_artifact_passes_untouched(self) -> None:
        art = _good_artifact()
        result = enforce_contract(art)
        assert result.artifact.price_target is not None
        assert result.artifact.price_target.value == 150.0
        assert result.artifact.valuation_withheld is False
        assert result.artifact.contract_stamps == ()


class TestValueWithhold:
    def test_out_of_band_withholds_but_keeps_judgment(self) -> None:
        # Arrange: single-method target 10x the price, present in prose.
        art = EquityResearchArtifact(
            ticker="NVDA",
            verdict=Verdict.BUY,
            current_price=100.0,
            price_target=PriceTarget(value=1000.0, unit="USD_per_share"),
            methods_used=("dcf",),
            single_method_disclosed=True,
            narrative_slots={
                "conclusion": "We rate NVDA a BUY with a target of $1000.00.",
                "thesis": "The investment case remains compelling.",
            },
        )

        # Act
        result = enforce_contract(art)
        out = result.artifact

        # Assert: the number is gone everywhere, the judgment survives.
        assert out.valuation_withheld is True
        assert out.price_target is None
        assert "1000" not in out.narrative_slots["conclusion"]
        assert out.verdict is Verdict.BUY
        assert out.narrative_slots["thesis"] == "The investment case remains compelling."
        assert any(s.startswith("[CONTRACT/") for s in out.contract_stamps)

    def test_original_artifact_not_mutated(self) -> None:
        art = _good_artifact(price_target=PriceTarget(value=1000.0, unit="USD_per_share"))
        _ = enforce_contract(art)
        assert art.price_target is not None
        assert art.price_target.value == 1000.0
        assert art.valuation_withheld is False


class TestNoResurrection:
    def test_value_scrubbed_from_all_three_places(self) -> None:
        # Arrange: bad value in headline, a fallback slot, and narrative prose.
        art = EquityResearchArtifact(
            ticker="TSLA",
            verdict=Verdict.HOLD,
            current_price=100.0,
            price_target=PriceTarget(value=1000.0, unit="USD_per_share"),
            methods_used=("dcf",),
            single_method_disclosed=True,
            narrative_slots={"body": "Our model implies 1000 per share, a strong move."},
            fallback_slots={"cache": "prior_target=1000.00"},
        )

        # Act
        out = enforce_contract(art).artifact

        # Assert: not in headline, not in fallback, not in narrative.
        assert out.price_target is None
        assert "1000" not in out.narrative_slots["body"]
        assert "1000" not in out.fallback_slots["cache"]
        c7 = [f for f in enforce_contract(art).findings if f.check.startswith("contract_c7")]
        # C7 confirms no resurrection: it should NOT be blocking after scrub.
        assert all(not f.is_blocking() for f in c7)


class TestDegradeToRange:
    def test_divergent_methods_drop_point_keep_range(self) -> None:
        # Arrange: methods corroborate poorly but target is otherwise in-band.
        art = EquityResearchArtifact(
            ticker="AMD",
            verdict=Verdict.BUY,
            current_price=100.0,
            price_target=PriceTarget(value=150.0, unit="USD_per_share"),
            methods_used=("dcf", "ddm", "pe"),
            consensus_relative_span=0.90,
            price_range=PriceRange(low=90.0, mid=150.0, high=260.0),
            single_method_disclosed=True,
        )

        # Act
        out = enforce_contract(art).artifact

        # Assert: point retired, range preserved, verdict intact.
        assert out.price_target is None
        assert out.price_range is not None
        assert out.price_range.low == 90.0
        assert out.price_range.high == 260.0
        assert out.verdict is Verdict.BUY
        assert out.valuation_withheld is True


class TestNeverRaises:
    def test_degenerate_inputs_return_artifact(self) -> None:
        for bad in (math.nan, math.inf, -math.inf, -50.0):
            art = EquityResearchArtifact(
                ticker="X",
                verdict=Verdict.SELL,
                current_price=100.0,
                price_target=PriceTarget(value=bad, unit="ratio"),
                methods_used=("dcf",),
                narrative_slots={"body": "weird"},
            )
            out = enforce_contract(art).artifact
            assert isinstance(out, EquityResearchArtifact)
            assert out.verdict is Verdict.SELL


class TestDemo:
    def test_withholds_the_price_never_the_judgment(self) -> None:
        """Literal demo: absurd target in three places, judgment survives."""
        # Arrange: single-method target 10x price, in headline + fallback +
        # embedded in a narrative paragraph.
        bad = 1000.0
        art = EquityResearchArtifact(
            ticker="GME",
            knowledge_date=None,
            verdict=Verdict.BUY,
            current_price=100.0,
            price_target=PriceTarget(value=bad, unit="USD_per_share"),
            methods_used=("dcf",),
            single_method_disclosed=True,
            narrative_slots={
                "headline": "GME price target: $1000.00",
                "analysis": (
                    "After reviewing the fundamentals we arrive at a 1000 per-share "
                    "valuation; the thesis is that demand stays resilient and the "
                    "turnaround executes on schedule."
                ),
            },
            fallback_slots={"last_run": "target=1000"},
        )

        # Act: persistence is where the gate runs.
        from alphalab.artifact.store import InMemoryArtifactStore

        store = InMemoryArtifactStore()
        out = store.save(art)

        # Assert: the number is gone from all three slots.
        assert out.price_target is None
        assert "1000" not in out.narrative_slots["headline"]
        assert "1000" not in out.narrative_slots["analysis"]
        assert "1000" not in out.fallback_slots["last_run"]
        # The contract stamp is present and visible.
        assert any(s.startswith("[CONTRACT/") for s in out.contract_stamps)
        assert out.valuation_withheld is True
        # The judgment and the surrounding analysis survive.
        assert out.verdict is Verdict.BUY
        assert "demand stays resilient" in out.narrative_slots["analysis"]
        assert "turnaround executes" in out.narrative_slots["analysis"]


class TestC1bConservativeTarget:
    def _dispersed(self, **kw: object) -> EquityResearchArtifact:
        return _good_artifact(
            consensus_relative_span=5.0,
            narrative_slots={},
            price_range=PriceRange(
                low=kw.pop("low", 120.0), mid=kw.pop("mid", 150.0), high=kw.pop("high", 200.0)
            ),  # type: ignore[arg-type]
            **kw,
        )

    def test_bullish_range_yields_conservative_target(self) -> None:
        from alphalab.domain.contract.gate import conservative_c1b_target, conservative_c1b_upside

        out = enforce_contract(self._dispersed()).artifact
        assert out.price_target is None
        assert out.price_range is not None
        assert "[CONTRACT/contract_c1b_corroboration]" in out.contract_stamps
        assert conservative_c1b_target(out) == 135.0
        assert conservative_c1b_upside(out) == pytest.approx(0.35)

    def test_bearish_range(self) -> None:
        from alphalab.domain.contract.gate import conservative_c1b_target

        out = enforce_contract(self._dispersed(low=20.0, mid=40.0, high=60.0)).artifact
        assert conservative_c1b_target(out) == 50.0

    def test_straddling_range_has_no_target(self) -> None:
        from alphalab.domain.contract.gate import conservative_c1b_target

        out = enforce_contract(self._dispersed(low=50.0, high=200.0)).artifact
        assert conservative_c1b_target(out) is None

    def test_idempotent(self) -> None:
        once = enforce_contract(self._dispersed()).artifact
        twice = enforce_contract(once).artifact
        assert twice == once

    def test_hard_withhold_gives_no_target(self) -> None:
        from alphalab.domain.contract.gate import conservative_c1b_target

        art = self._dispersed(price_target=PriceTarget(value=1000.0, unit="USD_per_share"))
        out = enforce_contract(art).artifact
        assert out.price_range is None
        assert conservative_c1b_target(out) is None
