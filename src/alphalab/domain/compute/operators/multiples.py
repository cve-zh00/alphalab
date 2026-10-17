"""Relative (multiples) valuation operators.

Exposes:

* :func:`ev_bridge` — convert enterprise value to equity value (or inspect the
  reverse) through net debt. This bridge is a classic source of valuation
  error, so it is its own traceable operator.
* :func:`pe_implied_value` — equity value per share from a peer P/E and EPS.
* :func:`ev_ebitda_implied_value` — equity value per share from a peer
  EV/EBITDA, bridged back to equity via net debt.
* :func:`pb_implied_value` — equity value per share from a peer P/B and book
  value per share.
* :func:`residual_income_value_per_share` — Ohlson residual-income value for
  financials (book value plus PV of fading excess returns).
* :func:`ffo_implied_value` — equity value per share from a peer P/FFO (REITs).

Each applies the *median* of the supplied peer multiples (robust to outliers)
and returns a provenanced ``USD_per_share`` value.

Pure: numpy only, no I/O, no state.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alphalab.domain.compute.errors import InvalidValuationInputError
from alphalab.domain.compute.provenance import ProvenancedValue, combine

_EV_BRIDGE_VERSION = "1.0.0"
_PE_VERSION = "1.0.0"
_EV_EBITDA_VERSION = "1.0.0"
_PB_VERSION = "1.0.0"
_RI_VERSION = "1.0.0"
_FFO_VERSION = "1.0.0"


def ev_bridge(
    *,
    enterprise_value: float,
    net_debt: float,
) -> ProvenancedValue:
    """Equity value implied by an enterprise value and net debt.

    ``equity_value = enterprise_value - net_debt``. ``net_debt`` is total debt
    minus cash and may be negative for a net-cash company.

    Returns:
        A provenanced equity value in ``USD``.
    """
    equity_value = enterprise_value - net_debt
    return combine(
        value=equity_value,
        operator="ev_bridge",
        version=_EV_BRIDGE_VERSION,
        unit="USD",
        parents=(),
        extra_inputs={
            "enterprise_value": enterprise_value,
            "net_debt": net_debt,
        },
    )


def _median_multiple(peers: Sequence[float], *, label: str) -> float:
    if len(peers) == 0:
        raise InvalidValuationInputError(f"{label} peer set must be non-empty")
    return float(np.median(np.asarray(peers, dtype=float)))


def pe_implied_value(
    *,
    peer_pe: Sequence[float],
    eps: float,
) -> ProvenancedValue:
    """Equity value per share from peer P/E multiples and EPS.

    ``value_per_share = median(peer_pe) * eps``.

    Returns:
        A provenanced value per share in ``USD_per_share``.
    """
    multiple = _median_multiple(peer_pe, label="P/E")
    value_per_share = multiple * eps
    extra = {f"pe_{i}": m for i, m in enumerate(peer_pe, start=1)}
    extra["eps"] = eps
    extra["median_pe"] = multiple
    return combine(
        value=value_per_share,
        operator="pe_implied_value",
        version=_PE_VERSION,
        unit="USD_per_share",
        parents=(),
        extra_inputs=extra,
    )


def ev_ebitda_implied_value(
    *,
    peer_ev_ebitda: Sequence[float],
    ebitda: float,
    net_debt: float,
    shares_outstanding: float,
) -> ProvenancedValue:
    """Equity value per share from peer EV/EBITDA multiples.

    Applies ``median(peer_ev_ebitda) * ebitda`` to get an enterprise value,
    bridges to equity via :func:`ev_bridge`, then divides by shares. The EV
    bridge node is a parent, keeping the net-debt adjustment traceable.

    Returns:
        A provenanced value per share in ``USD_per_share`` whose parents
        include the ``ev_bridge`` node.
    """
    if shares_outstanding <= 0.0:
        raise InvalidValuationInputError("shares_outstanding must be positive")
    multiple = _median_multiple(peer_ev_ebitda, label="EV/EBITDA")
    enterprise_value = multiple * ebitda
    equity = ev_bridge(enterprise_value=enterprise_value, net_debt=net_debt)
    value_per_share = equity.value / shares_outstanding
    extra = {f"ev_ebitda_{i}": m for i, m in enumerate(peer_ev_ebitda, start=1)}
    extra["ebitda"] = ebitda
    extra["median_ev_ebitda"] = multiple
    extra["shares_outstanding"] = shares_outstanding
    return combine(
        value=value_per_share,
        operator="ev_ebitda_implied_value",
        version=_EV_EBITDA_VERSION,
        unit="USD_per_share",
        parents=(equity,),
        extra_inputs=extra,
    )


def pb_implied_value(
    *,
    peer_pb: Sequence[float],
    book_value_per_share: float,
) -> ProvenancedValue:
    """Equity value per share from peer P/B multiples and book value.

    ``value_per_share = median(peer_pb) * book_value_per_share``.

    Returns:
        A provenanced value per share in ``USD_per_share``.
    """
    multiple = _median_multiple(peer_pb, label="P/B")
    value_per_share = multiple * book_value_per_share
    extra = {f"pb_{i}": m for i, m in enumerate(peer_pb, start=1)}
    extra["book_value_per_share"] = book_value_per_share
    extra["median_pb"] = multiple
    return combine(
        value=value_per_share,
        operator="pb_implied_value",
        version=_PB_VERSION,
        unit="USD_per_share",
        parents=(),
        extra_inputs=extra,
    )


def residual_income_value_per_share(
    *,
    book_value_per_share: float,
    eps: float,
    cost_of_equity: float,
    terminal_growth: float,
    projection_years: int = 5,
    fade_factor: float = 0.85,
) -> ProvenancedValue:
    """Residual-income (Ohlson) value per share.

    ``V = B0 + sum_t RI_t / (1+ke)^t + TV / (1+ke)^N`` where
    ``RI_t = (ROE_t - ke) * B_{t-1}``. ``ROE_t`` starts at ``eps / B0`` and
    its spread over ``ke`` decays by ``fade_factor`` each year. Book value
    compounds at ``terminal_growth``. The terminal value capitalises the next
    (faded) residual income at ``ke - terminal_growth``.

    Raises:
        InvalidValuationInputError: non-positive book value / horizon,
            ``cost_of_equity <= terminal_growth`` or ``fade_factor`` outside
            ``[0, 1]``.
    """
    if book_value_per_share <= 0.0:
        raise InvalidValuationInputError("book_value_per_share must be positive")
    if projection_years < 1:
        raise InvalidValuationInputError("projection_years must be at least 1")
    if cost_of_equity <= 0.0 or cost_of_equity <= terminal_growth:
        raise InvalidValuationInputError("cost_of_equity must exceed terminal_growth and be > 0")
    if not 0.0 <= fade_factor <= 1.0:
        raise InvalidValuationInputError("fade_factor must be within [0, 1]")

    spread = eps / book_value_per_share - cost_of_equity
    book = book_value_per_share
    pv_ri = 0.0
    ri = 0.0
    for year in range(1, projection_years + 1):
        spread *= fade_factor
        ri = spread * book
        pv_ri += ri / (1.0 + cost_of_equity) ** year
        book *= 1.0 + terminal_growth
    terminal_ri = ri * fade_factor * (1.0 + terminal_growth)
    terminal_value = terminal_ri / (cost_of_equity - terminal_growth)
    pv_terminal = terminal_value / (1.0 + cost_of_equity) ** projection_years
    value = book_value_per_share + pv_ri + pv_terminal
    return combine(
        value=value,
        operator="residual_income_value_per_share",
        version=_RI_VERSION,
        unit="USD_per_share",
        parents=(),
        extra_inputs={
            "book_value_per_share": book_value_per_share,
            "eps": eps,
            "cost_of_equity": cost_of_equity,
            "terminal_growth": terminal_growth,
            "projection_years": float(projection_years),
            "fade_factor": fade_factor,
        },
    )


def ffo_implied_value(
    *,
    peer_ffo: Sequence[float],
    ffo_per_share: float,
) -> ProvenancedValue:
    """Equity value per share from peer P/FFO multiples (``median * FFO/share``)."""
    multiple = _median_multiple(peer_ffo, label="P/FFO")
    extra = {f"ffo_{i}": m for i, m in enumerate(peer_ffo, start=1)}
    extra["ffo_per_share"] = ffo_per_share
    extra["median_ffo"] = multiple
    return combine(
        value=multiple * ffo_per_share,
        operator="ffo_implied_value",
        version=_FFO_VERSION,
        unit="USD_per_share",
        parents=(),
        extra_inputs=extra,
    )
