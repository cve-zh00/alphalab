"""End-to-end demo: value a synthetic company by three methods, aggregate with
``consensus``, and trace every final number back to its leaf inputs.

This is the proof that "no number without an origin" is real, not aspirational.
The synthetic company's numbers are chosen so each method's result is a clean
hand-computable figure.
"""

from __future__ import annotations

import pytest

from alphalab.domain.compute.operators.consensus import consensus
from alphalab.domain.compute.operators.dcf import dcf_value_per_share, wacc
from alphalab.domain.compute.operators.ddm import ddm_value_per_share
from alphalab.domain.compute.operators.multiples import ev_ebitda_implied_value
from alphalab.domain.compute.provenance import ProvenancedValue


def _dcf_estimate() -> ProvenancedValue:
    # Debt-free WACC of exactly 0.10, cfs=[100,110,121], g=0.03, shares=50
    # => 32.207792... per share (hand-computed in test_compute_dcf).
    rate = wacc(
        equity_value=100.0,
        debt_value=0.0,
        cost_of_equity=0.10,
        cost_of_debt=0.0,
        tax_rate=0.0,
    )
    return dcf_value_per_share(
        cash_flows=[100.0, 110.0, 121.0],
        discount_rate=rate,
        terminal_growth=0.03,
        shares_outstanding=50.0,
    )


def _ddm_estimate() -> ProvenancedValue:
    # D0=1.5, r=0.09, g=0.04 => 1.5*1.04/0.05 = 31.2 per share.
    return ddm_value_per_share(
        dividend_per_share=1.5,
        required_return=0.09,
        dividend_growth=0.04,
    )


def _multiples_estimate() -> ProvenancedValue:
    # median([8,10,12])=10 ; EV=10*170=1700 ; equity=1700-200=1500 ;
    # per_share = 1500/50 = 30.0.
    return ev_ebitda_implied_value(
        peer_ev_ebitda=[8.0, 10.0, 12.0],
        ebitda=170.0,
        net_debt=200.0,
        shares_outstanding=50.0,
    )


class TestThreeMethodConsensusDemo:
    def test_consensus_point_and_full_derivation_trace(self) -> None:
        # Arrange: three independent method estimates (hand-computed values).
        dcf = _dcf_estimate()
        ddm = _ddm_estimate()
        mult = _multiples_estimate()

        assert dcf.value == pytest.approx(32.2077922077922)
        assert ddm.value == pytest.approx(31.2)
        assert mult.value == pytest.approx(30.0)

        # Act: aggregate the three methods.
        result = consensus(
            estimates=[dcf, ddm, mult],
            unit="USD_per_share",
        )

        # Assert: point = median(30.0, 31.2, 32.2078) = 31.2.
        assert result.mid == pytest.approx(31.2)
        assert result.low == pytest.approx(30.0)
        assert result.high == pytest.approx(32.2077922077922)
        # span = (32.2078 - 30.0) / 31.2 = 0.070762...
        assert result.relative_span == pytest.approx((32.2077922077922 - 30.0) / 31.2)
        assert result.is_wide is False

        # Assert: the consensus point traces all the way to its leaves, and the
        # three methods are its direct parents.
        parent_ops = {p.operator for p in result.point.parents}
        assert parent_ops == {
            "dcf_value_per_share",
            "ddm_value_per_share",
            "ev_ebitda_implied_value",
        }

        # Every final number resolves to raw-input leaves (no orphan numbers).
        leaves = result.point.leaves()
        assert len(leaves) > 0
        for node in leaves:
            assert node.is_leaf()
            assert node.parents == ()

        # The readable derivation tree is fully populated and nested.
        tree = result.point.tree()
        assert tree["operator"] == "consensus"
        assert len(tree["parents"]) == 3

        # Spot-check a specific chain: the DCF branch must contain its WACC.
        dcf_branch = next(p for p in tree["parents"] if p["operator"] == "dcf_value_per_share")
        dcf_sub_ops = {child["operator"] for child in dcf_branch["parents"]}
        assert "wacc" in dcf_sub_ops

    def test_determinism_same_inputs_same_value_and_trace(self) -> None:
        # Arrange / Act: build the consensus twice.
        first = consensus(
            estimates=[_dcf_estimate(), _ddm_estimate(), _multiples_estimate()],
            unit="USD_per_share",
        )
        second = consensus(
            estimates=[_dcf_estimate(), _ddm_estimate(), _multiples_estimate()],
            unit="USD_per_share",
        )

        # Assert: identical value AND identical derivation tree.
        assert first.point.value == second.point.value
        assert first.point.tree() == second.point.tree()
