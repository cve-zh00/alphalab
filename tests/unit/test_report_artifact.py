"""Unit tests for the artifact report formatter.

Covers the three required cases — a normal artifact, a withheld-price artifact
(must still show the verdict and the contract stamps), and an artifact with no
price target — plus the provenance view and the code-calculated/LLM-assisted
labelling.
"""

from __future__ import annotations

from datetime import date

from alphalab.artifact.models import (
    EquityResearchArtifact,
    PriceRange,
    PriceTarget,
    Verdict,
)
from alphalab.domain.analyst.contracts import RunMetadata
from alphalab.domain.analyst.tiers import ModelTier
from alphalab.domain.compute import combine, leaf
from alphalab.report import Theme, format_artifact

_PLAIN = Theme(enabled=False)


def _run_metadata() -> RunMetadata:
    return RunMetadata(
        model_name="qwen3.8:27b-mlx",
        tier=ModelTier.DEEP,
        llm_version="qwen3.8-27b-mlx",
        llm_cutoff=date(2024, 10, 1),
        knowledge_date=date(2024, 6, 1),
        dataset_hash="sha256:abc123",
    )


def _provenanced_target() -> object:
    eps = leaf(value=15.0, name="eps", unit="USD")
    return combine(
        value=150.0,
        operator="consensus",
        version="1.0.0",
        unit="USD_per_share",
        parents=(eps,),
        extra_inputs={"n_methods": 2.0},
    )


def _normal_artifact() -> EquityResearchArtifact:
    return EquityResearchArtifact(
        ticker="AAPL",
        knowledge_date=date(2024, 6, 1),
        verdict=Verdict.BUY,
        price_target=PriceTarget(value=150.0, unit="USD_per_share"),
        current_price=120.0,
        methods_used=("dcf", "pe"),
        method_values={"dcf": 148.0, "pe": 152.0},
        consensus_relative_span=0.03,
        narrative_slots={"thesis": "Strong franchise.", "risks": "Regulatory."},
    )


class TestNormalArtifact:
    def test_shows_verdict_target_and_upside(self) -> None:
        text = format_artifact(_normal_artifact(), theme=_PLAIN)
        assert "VERDICT:" in text
        assert "BUY" in text
        assert "150.00 USD_per_share" in text
        assert "Current price: 120.00" in text
        # (150 - 120) / 120 = +25.0%
        assert "+25.0%" in text
        assert "Implied upside" in text

    def test_shows_methods_and_values(self) -> None:
        text = format_artifact(_normal_artifact(), theme=_PLAIN)
        assert "dcf: 148.00" in text
        assert "pe: 152.00" in text
        assert "relative span: 0.030" in text

    def test_separates_code_numbers_from_llm_narrative(self) -> None:
        text = format_artifact(_normal_artifact(), theme=_PLAIN)
        assert "[code-calculated]" in text
        assert "[LLM-assisted]" in text
        assert "Strong franchise." in text

    def test_includes_disclaimer(self) -> None:
        text = format_artifact(_normal_artifact(), theme=_PLAIN)
        assert "NOT financial advice" in text

    def test_shows_run_metadata_when_provided(self) -> None:
        text = format_artifact(_normal_artifact(), theme=_PLAIN, run_metadata=_run_metadata())
        assert "qwen3.8:27b-mlx" in text
        assert "sha256:abc123" in text
        assert "2024-10-01" in text  # cutoff

    def test_shows_data_source_banner(self) -> None:
        text = format_artifact(_normal_artifact(), theme=_PLAIN, data_source="LOCAL FIXTURE (demo)")
        assert "DATA SOURCE: LOCAL FIXTURE (demo)" in text


class TestWithheldArtifact:
    def _withheld(self) -> EquityResearchArtifact:
        return EquityResearchArtifact(
            ticker="XYZ",
            knowledge_date=date(2024, 6, 1),
            verdict=Verdict.SELL,
            price_target=None,
            price_range=PriceRange(low=10.0, mid=20.0, high=90.0),
            current_price=25.0,
            methods_used=("dcf", "pe"),
            method_values={"dcf": 10.0, "pe": 90.0},
            narrative_slots={"thesis": "Thesis survives."},
            contract_stamps=("[CONTRACT/contract_c1b_corroboration]",),
            valuation_withheld=True,
        )

    def test_shows_verdict_despite_withheld_price(self) -> None:
        """'withholds the price, never the judgment' — the verdict stays."""
        text = format_artifact(self._withheld(), theme=_PLAIN)
        assert "SELL" in text
        assert "VERDICT:" in text

    def test_shows_withheld_banner_and_stamps(self) -> None:
        text = format_artifact(self._withheld(), theme=_PLAIN)
        assert "PRICE WITHHELD" in text
        assert "[CONTRACT/contract_c1b_corroboration]" in text

    def test_shows_analysis_and_retained_range(self) -> None:
        text = format_artifact(self._withheld(), theme=_PLAIN)
        assert "Thesis survives." in text
        assert "Retained range" in text
        assert "90.00" in text

    def test_does_not_show_a_point_target(self) -> None:
        text = format_artifact(self._withheld(), theme=_PLAIN)
        assert "Price target:" not in text


class TestNoPriceTarget:
    def test_handles_missing_target_without_withholding(self) -> None:
        art = EquityResearchArtifact(
            ticker="AAPL",
            verdict=Verdict.HOLD,
            price_target=None,
            current_price=100.0,
            narrative_slots={"thesis": "Neutral."},
        )
        text = format_artifact(art, theme=_PLAIN)
        assert "(none reported)" in text
        assert "HOLD" in text
        assert "Neutral." in text


class TestProvenanceView:
    def test_provenance_shows_derivation_tree_and_leaves(self) -> None:
        art = EquityResearchArtifact(
            ticker="AAPL",
            verdict=Verdict.BUY,
            price_target=_provenanced_target(),
            current_price=120.0,
        )
        text = format_artifact(art, theme=_PLAIN, show_provenance=True)
        assert "PROVENANCE" in text
        assert "consensus = 150.0" in text
        assert "eps = 15.0" in text

    def test_provenance_absent_by_default(self) -> None:
        text = format_artifact(_normal_artifact(), theme=_PLAIN)
        assert "PROVENANCE" not in text

    def test_provenance_handles_non_traced_target(self) -> None:
        art = EquityResearchArtifact(
            ticker="AAPL",
            verdict=Verdict.BUY,
            price_target=PriceTarget(value=150.0, unit="USD_per_share"),
            current_price=120.0,
        )
        text = format_artifact(art, theme=_PLAIN, show_provenance=True)
        assert "no provenance tree" in text


class TestColorGating:
    def test_colored_theme_emits_ansi_but_plain_does_not(self) -> None:
        colored = format_artifact(_normal_artifact(), theme=Theme(enabled=True))
        plain = format_artifact(_normal_artifact(), theme=_PLAIN)
        assert "\x1b[" in colored
        assert "\x1b[" not in plain


class TestZeroCurrentPrice:
    def test_upside_na_when_current_price_not_positive(self) -> None:
        art = EquityResearchArtifact(
            ticker="AAPL",
            verdict=Verdict.BUY,
            price_target=PriceTarget(value=150.0, unit="USD_per_share"),
            current_price=0.0,
        )
        text = format_artifact(art, theme=_PLAIN)
        assert "n/a" in text
