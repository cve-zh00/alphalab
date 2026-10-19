"""Point-in-time current-price lookup over a :class:`PriceSeriesSource`.

The analyst request needs a ``current_price``. The only safe reference at a
given knowledge date is the last close observed **on or before** that date;
reading any later close would be look-ahead. This helper enforces that cut-off
against whatever source sits behind the existing
:class:`~alphalab.adapters.prices.PriceSeriesSource` protocol — by default the
local CSV fixtures, so no network is required.
"""

from __future__ import annotations

from alphalab.adapters.prices import PriceSeriesSource
from alphalab.domain.pit import KnowledgeDate
from alphalab.domain.types import Ticker


class NoPriceAvailableError(Exception):
    """Raised when no close exists on or before the knowledge date."""


def current_price_at(
    source: PriceSeriesSource,
    ticker: Ticker,
    knowledge_date: KnowledgeDate,
) -> float:
    """Return the last close on or before ``knowledge_date`` for ``ticker``.

    Raises:
        NoPriceAvailableError: if the series is empty or every point is after
            the knowledge date (reading a later close would be look-ahead).
    """
    cutoff = knowledge_date.value
    eligible = [p for p in source.series(ticker) if p.on <= cutoff]
    if not eligible:
        raise NoPriceAvailableError(
            f"no close on or before {cutoff!r} for {ticker!r}: refusing to read a future price"
        )
    latest = max(eligible, key=lambda p: p.on)
    return latest.close


def current_valuation_price_at(
    source: PriceSeriesSource,
    ticker: Ticker,
    knowledge_date: KnowledgeDate,
) -> float:
    """Return the as-quoted close for valuation against as-reported share counts.

    Falls back to the adjusted close when the source carries no raw close.
    EDGAR share counts are not split-adjusted, so pairing them with an
    ``adjustment=all`` price understates market cap after a later split.
    """
    cutoff = knowledge_date.value
    eligible = [p for p in source.series(ticker) if p.on <= cutoff]
    if not eligible:
        raise NoPriceAvailableError(
            f"no close on or before {cutoff!r} for {ticker!r}: refusing to read a future price"
        )
    latest = max(eligible, key=lambda p: p.on)
    return latest.raw_close if latest.raw_close is not None else latest.close
