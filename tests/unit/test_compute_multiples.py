"""Unit tests for :mod:`alphalab.domain.compute.operators.multiples`.

Expected values are computed by hand using the median of each peer set.
"""

from __future__ import annotations

import pytest

from alphalab.domain.compute.errors import InvalidValuationInputError
from alphalab.domain.compute.operators.multiples import (
    ev_bridge,
    ev_ebitda_implied_value,
    pb_implied_value,
    pe_implied_value,
)


class TestEvBridge:
    def test_equity_equals_ev_minus_net_debt(self) -> None:
        # EV=2000, net_debt=400 => equity=1600.
        node = ev_bridge(enterprise_value=2000.0, net_debt=400.0)
        assert node.value == pytest.approx(1600.0)
        assert node.unit == "USD"

    def test_net_cash_increases_equity(self) -> None:
        # Negative net debt (net cash) => equity > EV.
        node = ev_bridge(enterprise_value=1000.0, net_debt=-200.0)
        assert node.value == pytest.approx(1200.0)


class TestPeImpliedValue:
    def test_hand_computed(self) -> None:
        # median([15,18,21]) = 18 ; 18 * eps(5) = 90.
        node = pe_implied_value(peer_pe=[15.0, 18.0, 21.0], eps=5.0)
        assert node.value == pytest.approx(90.0)
        assert node.unit == "USD_per_share"

    def test_empty_peers_rejected(self) -> None:
        with pytest.raises(InvalidValuationInputError, match="P/E"):
            pe_implied_value(peer_pe=[], eps=5.0)


class TestEvEbitdaImpliedValue:
    def test_hand_computed_with_bridge(self) -> None:
        # median([8,10,12])=10 ; EV=10*200=2000 ; equity=2000-400=1600 ;
        # per_share = 1600/20 = 80.
        node = ev_ebitda_implied_value(
            peer_ev_ebitda=[8.0, 10.0, 12.0],
            ebitda=200.0,
            net_debt=400.0,
            shares_outstanding=20.0,
        )
        assert node.value == pytest.approx(80.0)
        assert node.unit == "USD_per_share"
        # The EV bridge is a traceable parent.
        assert any(p.operator == "ev_bridge" for p in node.parents)

    def test_non_positive_shares_rejected(self) -> None:
        with pytest.raises(InvalidValuationInputError, match="shares_outstanding"):
            ev_ebitda_implied_value(
                peer_ev_ebitda=[10.0],
                ebitda=200.0,
                net_debt=0.0,
                shares_outstanding=0.0,
            )

    def test_empty_peers_rejected(self) -> None:
        with pytest.raises(InvalidValuationInputError, match="EV/EBITDA"):
            ev_ebitda_implied_value(
                peer_ev_ebitda=[],
                ebitda=200.0,
                net_debt=0.0,
                shares_outstanding=20.0,
            )


class TestPbImpliedValue:
    def test_hand_computed(self) -> None:
        # median([1.2,1.5,1.8])=1.5 ; 1.5 * 40 = 60.
        node = pb_implied_value(peer_pb=[1.2, 1.5, 1.8], book_value_per_share=40.0)
        assert node.value == pytest.approx(60.0)
        assert node.unit == "USD_per_share"

    def test_empty_peers_rejected(self) -> None:
        with pytest.raises(InvalidValuationInputError, match="P/B"):
            pb_implied_value(peer_pb=[], book_value_per_share=40.0)
