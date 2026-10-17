"""Live price provider backed by Alpaca market data.

Why this exists alongside the CSV provider
------------------------------------------
A system meant to be *left running* cannot decide from a snapshot. The CSV
provider reads files downloaded at some past moment, so every cycle after that
download reuses stale closes and the portfolio drifts away from the market
without anything reporting a fault.

This provider fetches bars on each cycle from Alpaca, using the SIP feed where
the account allows it (verified available here, with history back to 2016;
the free IEX feed only reaches 2020-07-27 and would silently truncate the
250-day lookback for recently listed names).

The benchmark symbol is always included in the request, whatever the strategy
universe. The shadow buy-and-hold is what makes a forward return interpretable,
and quietly omitting its price series would leave the comparison blank exactly
when it starts to matter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta

from alphalab.adapters.alpaca_bars import AlpacaBarsClient, FeedSubscriptionError
from alphalab.domain.backtest import PriceBar
from alphalab.domain.types import Ticker

__all__ = ["BENCHMARK_SYMBOL", "AlpacaPriceProvider"]

#: Yardstick for the shadow buy-and-hold. Always fetched.
BENCHMARK_SYMBOL = Ticker("SPY")

#: Calendar days of history requested. The signal's longest lookback is 250
#: *trading* days, so a calendar window must be materially wider; 600 leaves
#: room for weekends, holidays and the liquidity screen's own window.
_HISTORY_DAYS = 600

#: Symbols per request. Alpaca accepts multi-symbol queries, and batching keeps
#: the cycle inside the rate limit instead of issuing hundreds of calls.
_BATCH = 100

#: Days back from today for the newest bar requested.
#:
#: Two independent reasons, both pointing the same way. Methodologically,
#: today's daily bar is still forming, and deciding on a partial bar is using
#: information that does not exist yet in final form. Practically, this
#: account's SIP entitlement refuses recent data outright — measured:
#: ``end=today`` returns HTTP 403 "subscription does not permit querying
#: recent SIP data", while ``end=today-1`` returns bars normally.
_END_LAG_DAYS = 1


class AlpacaPriceProvider:
    """Fetches recent daily bars for a universe plus the benchmark."""

    def __init__(
        self,
        client: AlpacaBarsClient,
        *,
        tickers: Sequence[Ticker],
        today: date | None = None,
        history_days: int = _HISTORY_DAYS,
    ) -> None:
        self._client = client
        self._today = today
        self._history_days = history_days
        wanted = dict.fromkeys(tickers)
        wanted[BENCHMARK_SYMBOL] = None
        self._tickers = tuple(wanted)

    def recent_bars(self) -> Mapping[Ticker, Sequence[PriceBar]]:
        """Daily adjusted bars per symbol, omitting symbols with no data.

        A symbol that returns nothing is left out rather than represented by an
        empty series, so downstream code cannot mistake "no data" for "no
        movement". Batches are fetched sequentially; a failing batch propagates
        rather than being swallowed, because a partially loaded panel would
        silently change which names the strategy can even see.
        """
        today = self._today or datetime.now(UTC).date()
        end = today - timedelta(days=_END_LAG_DAYS)
        try:
            return self._fetch_for_end(end)
        except FeedSubscriptionError:
            # When UTC has rolled past midnight while US Eastern has not,
            # ``today - 1 day`` in UTC is still today's US session and SIP
            # rejects the query as "recent SIP data". Step back one day.
            return self._fetch_for_end(end - timedelta(days=1))

    def _fetch_for_end(self, end: date) -> dict[Ticker, Sequence[PriceBar]]:
        start = end - timedelta(days=self._history_days)
        out: dict[Ticker, Sequence[PriceBar]] = {}
        for batch in self._batches():
            fetched = self._client.daily_bars(batch, start=start, end=end)
            for ticker, bars in fetched.items():
                if bars:
                    out[ticker] = bars
        return out

    def _batches(self) -> list[list[Ticker]]:
        return [list(self._tickers[i : i + _BATCH]) for i in range(0, len(self._tickers), _BATCH)]
