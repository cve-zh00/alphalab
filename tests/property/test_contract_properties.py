"""Property-based tests for the contract gate (Hypothesis).

The most valuable guarantee: fed *any* degenerate artifact — NaN/inf targets,
weird strings, negative prices, missing ranges — the gate always returns an
artifact and never raises. Also proven: idempotence and non-mutation of the
input.
"""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from alphalab.artifact.models import (
    EquityResearchArtifact,
    PriceRange,
    PriceTarget,
    Verdict,
)
from alphalab.domain.contract import enforce_contract

# Strategies -----------------------------------------------------------------

_floats = st.floats(allow_nan=True, allow_infinity=True, width=64)
_finite = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9)
_units = st.sampled_from(["USD_per_share", "ratio", "x", "USD", "price_usd", "multiple", "percent"])
_weird_text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=0, max_size=40
)


def _slot_dict() -> st.SearchStrategy[dict[str, str]]:
    return st.dictionaries(
        keys=st.sampled_from(["conclusion", "body", "headline", "thesis", "cache"]),
        values=_weird_text,
        max_size=4,
    )


@st.composite
def artifacts(draw: st.DrawFn) -> EquityResearchArtifact:
    has_target = draw(st.booleans())
    target: PriceTarget | None = None
    if has_target:
        target = PriceTarget(value=draw(_floats), unit=draw(_units))

    has_range = draw(st.booleans())
    price_range: PriceRange | None = None
    if has_range:
        price_range = PriceRange(low=draw(_finite), mid=draw(_finite), high=draw(_finite))

    methods = draw(st.lists(st.sampled_from(["dcf", "ddm", "pe", "pb"]), max_size=4, unique=True))
    span = draw(st.one_of(st.none(), _floats))

    return EquityResearchArtifact(
        ticker=draw(
            st.text(
                min_size=1, max_size=6, alphabet=st.characters(min_codepoint=65, max_codepoint=90)
            )
        ),
        verdict=draw(st.sampled_from(list(Verdict))),
        current_price=draw(_floats),
        price_target=target,
        price_range=price_range,
        methods_used=tuple(methods),
        method_values=draw(
            st.dictionaries(st.sampled_from(["dcf", "ddm", "pe"]), _finite, max_size=3)
        ),
        consensus_relative_span=span,
        narrative_slots=draw(_slot_dict()),
        fallback_slots=draw(_slot_dict()),
        single_method_disclosed=draw(st.booleans()),
    )


# Properties -----------------------------------------------------------------


class TestGateNeverRaises:
    @given(artifacts())
    def test_always_returns_an_artifact(self, art: EquityResearchArtifact) -> None:
        result = enforce_contract(art)
        assert isinstance(result.artifact, EquityResearchArtifact)
        # The directional verdict always survives.
        assert result.artifact.verdict is art.verdict


class TestGateIdempotent:
    @given(artifacts())
    def test_applying_twice_is_stable(self, art: EquityResearchArtifact) -> None:
        once = enforce_contract(art).artifact
        twice = enforce_contract(once).artifact
        assert twice.price_target == once.price_target
        assert twice.valuation_withheld == once.valuation_withheld
        assert twice.verdict == once.verdict
        # No new stamps appear beyond those already present (idempotent set).
        assert set(twice.contract_stamps) == set(once.contract_stamps)


class TestGateDoesNotMutateInput:
    @given(artifacts())
    def test_input_unchanged(self, art: EquityResearchArtifact) -> None:
        before = art.model_dump()
        _ = enforce_contract(art)
        after = art.model_dump()
        # model_dump of float NaN compares unequal to itself, so compare keys
        # structurally with NaN-awareness.
        assert _dump_equal(before, after)


def _dump_equal(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_dump_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_dump_equal(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)
