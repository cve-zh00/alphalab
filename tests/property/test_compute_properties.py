"""Property-based tests (Hypothesis) for the compute layer.

Properties verified:

* **Provenance preserved under composition**: every derived value enumerates
  its ancestors down to the leaves.
* **Determinism**: identical inputs produce an identical ``value`` *and* an
  identical derivation tree.
* **Monotonicity**: DCF rises with the cash flows and falls with the WACC;
  DDM falls with the required return.
* **Immutability**: mutating any provenanced value raises.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import ValidationError

from alphalab.domain.compute.operators.dcf import dcf_value_per_share, wacc
from alphalab.domain.compute.operators.ddm import ddm_value_per_share
from alphalab.domain.compute.provenance import leaf, provenanced

_finite = {"allow_nan": False, "allow_infinity": False}
_cf = st.floats(min_value=1.0, max_value=1e6, **_finite)
_cf_list = st.lists(_cf, min_size=1, max_size=8)


def _rate_node(r: float) -> object:
    """A debt-free WACC node equal to ``r`` exactly."""
    return wacc(
        equity_value=100.0,
        debt_value=0.0,
        cost_of_equity=r,
        cost_of_debt=0.0,
        tax_rate=0.0,
    )


@provenanced(operator="sum3", version="1", unit="USD")
def _sum3(*, a: float, b: float, c: float) -> float:
    return a + b + c


class TestProvenancePreserved:
    @given(
        a=st.floats(min_value=-1e6, max_value=1e6, **_finite),
        b=st.floats(min_value=-1e6, max_value=1e6, **_finite),
        c=st.floats(min_value=-1e6, max_value=1e6, **_finite),
    )
    def test_derived_value_enumerates_all_leaves(self, a: float, b: float, c: float) -> None:
        # Arrange
        la = leaf(value=a, name="a", unit="USD")
        lb = leaf(value=b, name="b", unit="USD")
        lc = leaf(value=c, name="c", unit="USD")

        # Act: compose two levels.
        inner = _sum3(a=la, b=lb, c=lc)
        outer = _sum3(a=inner, b=la, c=lc)

        # Assert: every leaf is reachable; all are genuine leaves.
        leaves = outer.leaves()
        assert all(node.is_leaf() for node in leaves)
        names = {node.inputs[0][0] for node in leaves}
        assert names == {"a", "b", "c"}


class TestDeterminism:
    @given(
        d0=st.floats(min_value=0.0, max_value=1e4, **_finite),
        r=st.floats(min_value=0.02, max_value=0.5, **_finite),
        g=st.floats(min_value=0.0, max_value=0.5, **_finite),
    )
    def test_ddm_same_inputs_same_value_and_tree(self, d0: float, r: float, g: float) -> None:
        # Arrange: keep the g < r precondition satisfied.
        assume(r - g >= 1e-2)

        # Act
        first = ddm_value_per_share(dividend_per_share=d0, required_return=r, dividend_growth=g)
        second = ddm_value_per_share(dividend_per_share=d0, required_return=r, dividend_growth=g)

        # Assert
        assert first.value == second.value
        assert first.tree() == second.tree()


class TestMonotonicity:
    @given(
        flows=_cf_list,
        bump=st.floats(min_value=1.0, max_value=1e5, **_finite),
    )
    def test_dcf_increasing_in_cash_flows(self, flows: list[float], bump: float) -> None:
        # Arrange: raise every cash flow by a positive bump.
        rate = _rate_node(0.10)
        higher = [cf + bump for cf in flows]

        # Act
        base = dcf_value_per_share(
            cash_flows=flows,
            discount_rate=rate,  # type: ignore[arg-type]
            terminal_growth=0.0,
            shares_outstanding=10.0,
        )
        raised = dcf_value_per_share(
            cash_flows=higher,
            discount_rate=rate,  # type: ignore[arg-type]
            terminal_growth=0.0,
            shares_outstanding=10.0,
        )

        # Assert: more cash flow never lowers the value.
        assert raised.value >= base.value - 1e-6

    @given(
        flows=_cf_list,
        r_low=st.floats(min_value=0.05, max_value=0.20, **_finite),
        extra=st.floats(min_value=0.01, max_value=0.30, **_finite),
    )
    def test_dcf_decreasing_in_wacc(self, flows: list[float], r_low: float, extra: float) -> None:
        # Arrange: a strictly higher discount rate.
        r_high = r_low + extra
        low_rate = _rate_node(r_low)
        high_rate = _rate_node(r_high)

        # Act
        v_low = dcf_value_per_share(
            cash_flows=flows,
            discount_rate=low_rate,  # type: ignore[arg-type]
            terminal_growth=0.0,
            shares_outstanding=10.0,
        )
        v_high = dcf_value_per_share(
            cash_flows=flows,
            discount_rate=high_rate,  # type: ignore[arg-type]
            terminal_growth=0.0,
            shares_outstanding=10.0,
        )

        # Assert: a higher WACC never raises the value.
        assert v_high.value <= v_low.value + 1e-6

    @given(
        d0=st.floats(min_value=1.0, max_value=1e3, **_finite),
        r_low=st.floats(min_value=0.05, max_value=0.20, **_finite),
        extra=st.floats(min_value=0.01, max_value=0.30, **_finite),
        g=st.floats(min_value=0.0, max_value=0.03, **_finite),
    )
    def test_ddm_decreasing_in_required_return(
        self, d0: float, r_low: float, extra: float, g: float
    ) -> None:
        # Arrange: keep g < r for both rates.
        r_high = r_low + extra
        assume(r_low - g >= 1e-2)

        # Act
        v_low = ddm_value_per_share(dividend_per_share=d0, required_return=r_low, dividend_growth=g)
        v_high = ddm_value_per_share(
            dividend_per_share=d0, required_return=r_high, dividend_growth=g
        )

        # Assert: a higher required return never raises the DDM value.
        assert v_high.value <= v_low.value + 1e-6


class TestImmutability:
    @given(v=st.floats(min_value=-1e6, max_value=1e6, **_finite))
    def test_leaf_cannot_be_mutated(self, v: float) -> None:
        node = leaf(value=v, name="x", unit="USD")
        with pytest.raises(ValidationError):
            node.value = v + 1.0  # type: ignore[misc]
