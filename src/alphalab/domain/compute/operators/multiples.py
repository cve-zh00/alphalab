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
