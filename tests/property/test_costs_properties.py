"""Property-based tests (Hypothesis) for the transaction-cost model.

Properties verified:

* **Non-negativity**: no cost component is ever negative for valid inputs.
* **Monotonicity**: larger size never lowers total cost.
* **Super-linearity**: for ``alpha < 1`` the per-share impact grows with size,
  so doubling ``Q`` more than doubles the total impact cost.
* **net <= gross**: a non-negative cost drag can never raise the net Sharpe.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from alphalab.domain.costs.components import (
    borrow_cost,
    commission_cost,
    market_impact_bps,
    market_impact_cost,
    slippage_cost,
    spread_cost,
)
from alphalab.domain.costs.result import sharpe_gross_net
from alphalab.domain.types import Side

_finite = dict(allow_nan=False, allow_infinity=False)

_notional = st.floats(min_value=0.0, max_value=1e9, **_finite)
_bps = st.floats(min_value=0.0, max_value=1e4, **_finite)
_shares = st.floats(min_value=0.0, max_value=1e8, **_finite)
_positive_adv = st.floats(min_value=1.0, max_value=1e10, **_finite)
_sigma = st.floats(min_value=0.0, max_value=1e4, **_finite)
_alpha = st.floats(min_value=0.1, max_value=1.0, **_finite)
_eta = st.floats(min_value=0.0, max_value=10.0, **_finite)
_rate = st.floats(min_value=0.0, max_value=1.0, **_finite)
_days = st.floats(min_value=0.0, max_value=3650.0, **_finite)


class TestNonNegativity:
    @given(notional=_notional, bps=_bps)
    def test_spread_non_negative(self, notional: float, bps: float) -> None:
        assert spread_cost(notional, bps) >= 0.0

    @given(notional=_notional, bps=_bps)
    def test_slippage_non_negative(self, notional: float, bps: float) -> None:
        assert slippage_cost(notional, bps) >= 0.0

    @given(q=_shares, adv=_positive_adv, sigma=_sigma, eta=_eta, alpha=_alpha)
    def test_impact_bps_non_negative(
        self, q: float, adv: float, sigma: float, eta: float, alpha: float
    ) -> None:
        assert market_impact_bps(q, adv, sigma, eta=eta, alpha=alpha) >= 0.0

    @given(
        shares=_shares,
        notional=_notional,
        per_share=st.floats(min_value=0.0, max_value=10.0, **_finite),
        min_commission=st.floats(min_value=0.0, max_value=100.0, **_finite),
        ad_valorem=_bps,
    )
    def test_commission_non_negative(
        self,
        shares: float,
        notional: float,
        per_share: float,
        min_commission: float,
        ad_valorem: float,
    ) -> None:
        result = commission_cost(
            shares,
            notional,
            per_share=per_share,
            min_commission=min_commission,
            ad_valorem_bps=ad_valorem,
        )
        assert result >= 0.0

    @given(
        side=st.sampled_from(list(Side)),
        notional=_notional,
        days=_days,
        rate=_rate,
    )
    def test_borrow_non_negative(
        self, side: Side, notional: float, days: float, rate: float
    ) -> None:
        assert borrow_cost(side, notional, days, rate) >= 0.0


class TestMonotonicity:
    @given(
        q=st.floats(min_value=0.0, max_value=1e7, **_finite),
        extra=st.floats(min_value=0.0, max_value=1e7, **_finite),
        adv=_positive_adv,
        sigma=_sigma,
        eta=_eta,
        alpha=_alpha,
    )
    def test_impact_bps_monotone_in_size(
        self, q: float, extra: float, adv: float, sigma: float, eta: float, alpha: float
    ) -> None:
        # Arrange: a larger quantity q + extra.
        small = market_impact_bps(q, adv, sigma, eta=eta, alpha=alpha)
        large = market_impact_bps(q + extra, adv, sigma, eta=eta, alpha=alpha)

        # Assert: more size never lowers the impact.
        assert large >= small - 1e-9

    @given(
        price=st.floats(min_value=0.0, max_value=1e5, **_finite),
        q=st.floats(min_value=0.0, max_value=1e6, **_finite),
        extra=st.floats(min_value=0.0, max_value=1e6, **_finite),
        adv=_positive_adv,
        sigma=_sigma,
    )
    def test_total_impact_cost_monotone_in_size(
        self, price: float, q: float, extra: float, adv: float, sigma: float
    ) -> None:
        small = market_impact_cost(price, q, adv, sigma)
        large = market_impact_cost(price, q + extra, adv, sigma)
        assert large >= small - 1e-6


class TestSuperlinearity:
    @given(
        q=st.floats(min_value=1.0, max_value=1e6, **_finite),
        adv=_positive_adv,
        sigma=st.floats(min_value=1.0, max_value=1e4, **_finite),
        eta=st.floats(min_value=0.1, max_value=10.0, **_finite),
        alpha=st.floats(min_value=0.1, max_value=0.999, **_finite),
    )
    def test_doubling_more_than_doubles_total_impact(
        self, q: float, adv: float, sigma: float, eta: float, alpha: float
    ) -> None:
        # Arrange: total impact cost ~ Q * impact_bps(Q) (price factored out).
        single = q * market_impact_bps(q, adv, sigma, eta=eta, alpha=alpha)
        double = (2.0 * q) * market_impact_bps(2.0 * q, adv, sigma, eta=eta, alpha=alpha)

        # Assert: with alpha < 1, doubling Q MORE than doubles total impact.
        assert double > 2.0 * single

    @given(
        q=st.floats(min_value=1.0, max_value=1e6, **_finite),
        adv=_positive_adv,
        sigma=st.floats(min_value=1.0, max_value=1e4, **_finite),
        eta=st.floats(min_value=0.1, max_value=10.0, **_finite),
        alpha=st.floats(min_value=0.1, max_value=0.999, **_finite),
    )
    def test_per_share_cost_grows_with_size(
        self, q: float, adv: float, sigma: float, eta: float, alpha: float
    ) -> None:
        # Arrange: per-share impact (in bps) is exactly market_impact_bps.
        small = market_impact_bps(q, adv, sigma, eta=eta, alpha=alpha)
        large = market_impact_bps(2.0 * q, adv, sigma, eta=eta, alpha=alpha)

        # Assert: per-share cost strictly increases with size for alpha < 1.
        assert large > small


class TestNetLeqGross:
    @given(
        gross=st.lists(
            st.floats(min_value=-0.5, max_value=0.5, **_finite),
            min_size=2,
            max_size=50,
        ),
        drag=st.floats(min_value=0.0, max_value=0.05, **_finite),
    )
    def test_net_sharpe_never_exceeds_gross(self, gross: list[float], drag: float) -> None:
        # Arrange: a constant non-negative cost drag applied every period.
        cost_fractions = [drag] * len(gross)

        # Act
        result = sharpe_gross_net(gross, cost_fractions)

        # Assert: positive cost drag can never raise the Sharpe ratio.
        assert result.net <= result.gross + 1e-9
