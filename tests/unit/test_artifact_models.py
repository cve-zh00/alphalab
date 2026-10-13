"""Unit tests for the artifact data model and its structural protocol."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alphalab.artifact.models import (
    EquityResearchArtifact,
    HasValueAndUnit,
    PriceTarget,
    Verdict,
)
from alphalab.domain.compute.provenance import leaf


def _artifact() -> EquityResearchArtifact:
    return EquityResearchArtifact(
        ticker="AAPL",
        verdict=Verdict.BUY,
        current_price=100.0,
        price_target=PriceTarget(value=150.0, unit="USD_per_share"),
    )


class TestImmutability:
    def test_artifact_is_frozen(self) -> None:
        art = _artifact()
        with pytest.raises(ValidationError):
            art.ticker = "MSFT"  # type: ignore[misc]

    def test_model_copy_does_not_mutate_original(self) -> None:
        art = _artifact()
        copy = art.model_copy(update={"valuation_withheld": True})
        assert copy.valuation_withheld is True
        assert art.valuation_withheld is False


class TestProtocolConformance:
    def test_price_target_satisfies_protocol(self) -> None:
        pt = PriceTarget(value=10.0, unit="USD_per_share")
        assert isinstance(pt, HasValueAndUnit)

    def test_provenanced_value_satisfies_protocol(self) -> None:
        pv = leaf(value=10.0, name="x", unit="USD_per_share")
        assert isinstance(pv, HasValueAndUnit)

    def test_artifact_accepts_provenanced_value_as_target(self) -> None:
        pv = leaf(value=120.0, name="dcf", unit="USD_per_share")
        art = EquityResearchArtifact(
            ticker="AAPL", verdict=Verdict.HOLD, current_price=100.0, price_target=pv
        )
        assert art.price_target is not None
        assert art.price_target.value == 120.0
        assert art.price_target.unit == "USD_per_share"


class TestVerdict:
    def test_verdict_members(self) -> None:
        assert {v.value for v in Verdict} == {"BUY", "HOLD", "SELL"}
