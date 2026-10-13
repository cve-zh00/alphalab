"""Unit tests for the individual contract clauses.

Each clause is checked in *both* directions: the bad artifact produces a
finding, the good artifact produces ``None``. AAA structure throughout.
"""

from __future__ import annotations

from alphalab.artifact.models import (
    EquityResearchArtifact,
    PriceTarget,
    Verdict,
)
from alphalab.domain.contract import clauses


def _artifact(**overrides: object) -> EquityResearchArtifact:
    base: dict[str, object] = {
        "ticker": "AAPL",
        "verdict": Verdict.BUY,
        "current_price": 100.0,
        "price_target": PriceTarget(value=120.0, unit="USD_per_share"),
        "methods_used": ("dcf",),
        "single_method_disclosed": True,
    }
    base.update(overrides)
    return EquityResearchArtifact(**base)  # type: ignore[arg-type]


class TestC1UpsideBand:
    def test_single_method_in_band_passes(self) -> None:
        art = _artifact(price_target=PriceTarget(value=180.0, unit="USD_per_share"))
        assert clauses.check_c1_upside_band(art) is None

    def test_single_method_above_2x_fails(self) -> None:
        art = _artifact(price_target=PriceTarget(value=250.0, unit="USD_per_share"))
        finding = clauses.check_c1_upside_band(art)
        assert finding is not None
        assert finding.is_blocking()

    def test_single_method_below_half_fails(self) -> None:
        art = _artifact(price_target=PriceTarget(value=40.0, unit="USD_per_share"))
        finding = clauses.check_c1_upside_band(art)
        assert finding is not None and finding.is_blocking()

    def test_multi_method_wider_band_allows_3x(self) -> None:
        art = _artifact(
            price_target=PriceTarget(value=300.0, unit="USD_per_share"),
            methods_used=("dcf", "ddm", "pe"),
        )
        assert clauses.check_c1_upside_band(art) is None

    def test_multi_method_above_4x_fails(self) -> None:
        art = _artifact(
            price_target=PriceTarget(value=450.0, unit="USD_per_share"),
            methods_used=("dcf", "ddm", "pe"),
        )
        finding = clauses.check_c1_upside_band(art)
        assert finding is not None and finding.is_blocking()

    def test_no_target_passes(self) -> None:
        art = _artifact(price_target=None)
        assert clauses.check_c1_upside_band(art) is None

    def test_non_positive_price_fails(self) -> None:
        art = _artifact(current_price=0.0)
        finding = clauses.check_c1_upside_band(art)
        assert finding is not None and finding.is_blocking()


class TestC1bCorroboration:
    def test_tight_methods_pass(self) -> None:
        art = _artifact(
            methods_used=("dcf", "ddm", "pe"),
            consensus_relative_span=0.10,
        )
        assert clauses.check_c1b_corroboration(art) is None

    def test_wide_methods_fail(self) -> None:
        art = _artifact(
            methods_used=("dcf", "ddm", "pe"),
            consensus_relative_span=0.80,
        )
        finding = clauses.check_c1b_corroboration(art)
        assert finding is not None and finding.is_blocking()

    def test_single_method_not_evaluated(self) -> None:
        art = _artifact(methods_used=("dcf",), consensus_relative_span=0.90)
        assert clauses.check_c1b_corroboration(art) is None

    def test_missing_span_passes(self) -> None:
        art = _artifact(methods_used=("dcf", "ddm"), consensus_relative_span=None)
        assert clauses.check_c1b_corroboration(art) is None


class TestC2NumericConcatenation:
    def test_double_decimal_fails(self) -> None:
        art = _artifact(narrative_slots={"body": "Fair value is $2172.062.06 per share."})
        finding = clauses.check_c2_numeric_concatenation(art)
        assert finding is not None and finding.is_blocking()

    def test_glued_digits_fail(self) -> None:
        art = _artifact(fallback_slots={"raw": "value 123456789012"})
        finding = clauses.check_c2_numeric_concatenation(art)
        assert finding is not None and finding.is_blocking()

    def test_clean_amounts_pass(self) -> None:
        art = _artifact(narrative_slots={"body": "Fair value is $172.06 per share, up 1,250 bps."})
        assert clauses.check_c2_numeric_concatenation(art) is None


class TestC3ConclusionHeadline:
    def test_matching_conclusion_passes(self) -> None:
        art = _artifact(
            price_target=PriceTarget(value=150.0, unit="USD_per_share"),
            narrative_slots={"conclusion": "Our target is $150.00."},
        )
        assert clauses.check_c3_conclusion_headline(art) is None

    def test_disagreeing_conclusion_fails(self) -> None:
        art = _artifact(
            price_target=PriceTarget(value=150.0, unit="USD_per_share"),
            narrative_slots={"conclusion": "Our target is $175.00."},
        )
        finding = clauses.check_c3_conclusion_headline(art)
        assert finding is not None and finding.is_blocking()

    def test_conclusion_without_amount_passes(self) -> None:
        art = _artifact(
            price_target=PriceTarget(value=150.0, unit="USD_per_share"),
            narrative_slots={"conclusion": "We remain constructive on the name."},
        )
        assert clauses.check_c3_conclusion_headline(art) is None

    def test_no_conclusion_slot_passes(self) -> None:
        art = _artifact(price_target=PriceTarget(value=150.0, unit="USD_per_share"))
        assert clauses.check_c3_conclusion_headline(art) is None


class TestC4UnitCaliber:
    def test_per_share_unit_passes(self) -> None:
        art = _artifact(price_target=PriceTarget(value=120.0, unit="USD_per_share"))
        assert clauses.check_c4_unit_caliber(art) is None

    def test_ratio_unit_fails(self) -> None:
        art = _artifact(price_target=PriceTarget(value=1.2, unit="ratio"))
        finding = clauses.check_c4_unit_caliber(art)
        assert finding is not None and finding.is_blocking()

    def test_no_target_passes(self) -> None:
        art = _artifact(price_target=None)
        assert clauses.check_c4_unit_caliber(art) is None


class TestC6SingleMethodDisclosure:
    def test_disclosed_single_method_passes(self) -> None:
        art = _artifact(methods_used=("dcf",), single_method_disclosed=True)
        assert clauses.check_c6_single_method_disclosure(art) is None

    def test_undisclosed_single_method_soft_fails(self) -> None:
        art = _artifact(methods_used=("dcf",), single_method_disclosed=False)
        finding = clauses.check_c6_single_method_disclosure(art)
        assert finding is not None
        assert not finding.passed
        assert not finding.is_blocking()  # SOFT: never blocks.

    def test_multi_method_not_required_to_disclose(self) -> None:
        art = _artifact(methods_used=("dcf", "ddm"), single_method_disclosed=False)
        assert clauses.check_c6_single_method_disclosure(art) is None


class TestC7NoResurrection:
    def test_value_absent_passes(self) -> None:
        art = _artifact(
            price_target=None,
            narrative_slots={"body": "No price is published."},
            fallback_slots={},
        )
        assert clauses.check_c7_no_resurrection(art, 999.0) is None

    def test_value_in_narrative_fails(self) -> None:
        art = _artifact(
            price_target=None,
            narrative_slots={"body": "Prior target was 999."},
        )
        finding = clauses.check_c7_no_resurrection(art, 999.0)
        assert finding is not None and finding.is_blocking()

    def test_value_in_fallback_fails(self) -> None:
        art = _artifact(
            price_target=None,
            fallback_slots={"cache": "999.00"},
        )
        finding = clauses.check_c7_no_resurrection(art, 999.0)
        assert finding is not None and finding.is_blocking()

    def test_value_in_price_target_fails(self) -> None:
        art = _artifact(price_target=PriceTarget(value=999.0, unit="USD_per_share"))
        finding = clauses.check_c7_no_resurrection(art, 999.0)
        assert finding is not None and finding.is_blocking()
