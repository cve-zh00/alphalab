"""The liquidity-only candidate universe for the agentic layer.

Why this module exists
----------------------
The agentic pipeline must decide *which* names it is even allowed to consider
before any language model sees a single filing. That gate has to be built from
exactly one thing — **liquidity** — and from nothing that predicts returns.

The reason is not fastidiousness. If the candidate set were narrowed by
momentum (or any return signal), momentum would still be making the investment
decision and the model would only be decorating it; the experiment would be
indistinguishable from what this account already did, and its whole point —
does an LLM add anything over price momentum? — would evaporate. Liquidity is
a *different and legitimate* axis: it guarantees the names are tradable and
that the backtest's few-basis-point cost assumption is defensible. Measured
proof that the screen is mandatory, not cosmetic: without it, a mechanically
sampled, equal-weighted universe compounded at an impossible **+430%/year**,
pure bid-ask bounce on sub-dollar stocks.

What it produces
----------------
An immutable :class:`CandidateUniverse`: the surviving tickers (each with its
resolved SEC CIK, because without a CIK there are no EDGAR fundamentals) plus a
*separated* tally of every name that fell out and why — illiquid, no price
data, or no CIK. A counted omission is information; a silent one is an
invisible bias, so the three reasons never merge into a single "excluded"
number.

The cap and why it does not bias
--------------------------------
The screen can leave more names than the pipeline can afford: each LLM call
costs ~8.7s, so 40 candidates is already ~20 minutes per cycle. When the screen
overflows, the universe is trimmed to the ``cap`` **most liquid** names, ranked
by trailing-window *median dollar volume* (close x volume), highest first, ties
broken alphabetically for determinism.

That ranking key is a pure liquidity statistic — the very same quantity the
screen itself thresholds on — and it reads only past traded value, never a
return. Ranking by it therefore cannot smuggle a return forecast into the
selection: it prefers the names that are *easiest to trade*, which is the only
thing the cap is allowed to care about. The same input yields the same output,
byte for byte.

The one-day lag
---------------
Bars are requested with ``end = as_of - 1`` for two independent, aligned
reasons documented in :mod:`alphalab.cli.alpaca_prices`: today's daily bar is
still forming (deciding on a partial bar uses data that does not exist yet in
final form), and this account's SIP entitlement returns HTTP 403 for recent
data at ``end = today`` while ``end = today - 1`` returns 200.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.backtest.context import SignalContext
from alphalab.domain.backtest.signals import liquidity_screened
from alphalab.domain.backtest.types import PriceBar
from alphalab.domain.types import Ticker

__all__ = [
    "DEFAULT_CANDIDATE_CAP",
    "LIQUIDITY_MIN_DOLLAR_VOLUME",
    "LIQUIDITY_MIN_PRICE",
    "LIQUIDITY_WINDOW",
    "BarsPort",
    "CandidateUniverse",
    "CikPort",
    "build_candidate_universe",
]

#: Hard ceiling on candidates per cycle. Each LLM research call costs ~8.7s, so
#: 40 names is already ~20 minutes per cycle; more would make a cycle unusable.
DEFAULT_CANDIDATE_CAP = 40

#: Liquidity thresholds — identical to ``liquidity_screened``'s own defaults, so
#: there is a single source of truth for "tradable". Named here only so the
#: audit record and the demo can report the exact floors that were applied.
LIQUIDITY_MIN_PRICE = 5.0
LIQUIDITY_MIN_DOLLAR_VOLUME = 5_000_000.0
LIQUIDITY_WINDOW = 60

#: Days back from ``as_of`` for the newest bar requested. See module docstring:
#: today's bar is incomplete and recent SIP data returns HTTP 403.
_END_LAG_DAYS = 1

#: Calendar days of history requested. The liquidity window is 60 *trading*
#: days, so a calendar span must be materially wider to cover weekends and
#: holidays; 120 leaves comfortable room.
_HISTORY_DAYS = 120


class BarsPort(Protocol):
    """The slice of ``AlpacaBarsClient`` this module needs."""

    def daily_bars(
        self,
        symbols: tuple[Ticker, ...] | list[Ticker],
        *,
        start: date,
        end: date,
    ) -> dict[Ticker, tuple[PriceBar, ...]]: ...


class CikPort(Protocol):
    """The slice of ``CikResolver`` this module needs.

    ``resolve`` raises ``KeyError`` when the SEC map has no entry, matching the
    real resolver; this module turns that into a counted exclusion rather than
    letting it abort the cycle.
    """

    def resolve(self, ticker: Ticker) -> str: ...


class CandidateUniverse(BaseModel):
    """The names the agentic layer may consider, plus why the rest fell out.

    Immutable by construction: once built, the record is an auditable snapshot
    that downstream code reads but cannot edit. Every requested name lands in
    exactly one place — ``candidates``, one of the three exclusion buckets, or
    ``capped_out`` — so the counts always reconcile against ``requested_count``.
    """

    model_config = ConfigDict(frozen=True)

    as_of: date
    cap: int = Field(gt=0)
    requested_count: int = Field(ge=0)

    candidates: tuple[Ticker, ...]
    ciks: Mapping[Ticker, str]

    excluded_illiquid: tuple[Ticker, ...]
    excluded_no_price: tuple[Ticker, ...]
    excluded_no_cik: tuple[Ticker, ...]
    capped_out: tuple[Ticker, ...]

    def cik_for(self, ticker: Ticker) -> str:
        """The resolved SEC CIK for a candidate.

        Raises:
            KeyError: if ``ticker`` is not a candidate (defensive: only
                candidates have a resolved CIK on record).
        """
        return self.ciks[ticker]

    @property
    def excluded_illiquid_count(self) -> int:
        """Names dropped because they failed the liquidity screen."""
        return len(self.excluded_illiquid)

    @property
    def excluded_no_price_count(self) -> int:
        """Names dropped because Alpaca returned no bars for them."""
        return len(self.excluded_no_price)

    @property
    def excluded_no_cik_count(self) -> int:
        """Liquid names dropped because the SEC map has no CIK for them."""
        return len(self.excluded_no_cik)

    @property
    def capped_out_count(self) -> int:
        """Liquid, CIK-resolved names trimmed by the cost cap (not a failure)."""
        return len(self.capped_out)


def build_candidate_universe(
    *,
    requested: Sequence[Ticker],
    bars: BarsPort,
    cik: CikPort,
    as_of: date,
    cap: int = DEFAULT_CANDIDATE_CAP,
) -> CandidateUniverse:
    """Narrow ``requested`` to a tradable, CIK-resolved candidate set.

    The pipeline is: fetch bars (lagged one day), keep names that clear the
    liquidity screen, resolve each survivor's CIK, then — if more survive than
    the cost cap allows — keep the ``cap`` most liquid by trailing median dollar
    volume. Every drop is recorded in its own bucket.

    Raises:
        ValueError: if ``cap`` is not positive.
    """
    if cap <= 0:
        raise ValueError("cap must be a positive integer")

    unique = tuple(dict.fromkeys(requested))
    panel = _fetch_panel(bars, unique, as_of)

    with_price = tuple(t for t in unique if panel.get(t))
    no_price = tuple(t for t in unique if not panel.get(t))

    context = SignalContext(as_of=as_of, prices=panel)
    liquid = _liquid_names(context, with_price)
    illiquid = tuple(t for t in with_price if t not in set(liquid))

    resolved, no_cik = _resolve_ciks(liquid, cik)

    ranked = _rank_by_liquidity(resolved, context)
    kept = ranked[:cap]
    capped_out = ranked[cap:]

    return CandidateUniverse(
        as_of=as_of,
        cap=cap,
        requested_count=len(unique),
        candidates=tuple(sorted(kept)),
        ciks={t: resolved[t] for t in kept},
        excluded_illiquid=illiquid,
        excluded_no_price=no_price,
        excluded_no_cik=no_cik,
        capped_out=tuple(capped_out),
    )


def _fetch_panel(
    bars: BarsPort, symbols: tuple[Ticker, ...], as_of: date
) -> dict[Ticker, tuple[PriceBar, ...]]:
    """Fetch daily bars for ``symbols`` with the measured one-day lag.

    An empty ``symbols`` short-circuits so no empty request is issued. Symbols
    the source omits simply do not appear in the result and are handled by the
    caller as the "no price" bucket.
    """
    if not symbols:
        return {}
    end = as_of - timedelta(days=_END_LAG_DAYS)
    start = end - timedelta(days=_HISTORY_DAYS)
    fetched = bars.daily_bars(symbols, start=start, end=end)
    return {t: tuple(series) for t, series in fetched.items() if series}


def _liquid_names(context: SignalContext, candidates: tuple[Ticker, ...]) -> tuple[Ticker, ...]:
    """Names that clear the shared liquidity screen, in input order.

    The screen is reused verbatim via ``liquidity_screened`` wrapped around a
    constant ``inner`` signal that merely nominates every priced name. This
    keeps one definition of "tradable": the same code that protects the
    backtest decides the agentic universe. The constant score carries no return
    information — it is the same for every name — so nothing predictive leaks.
    """
    wanted = set(candidates)

    def nominate(context: SignalContext) -> Mapping[Ticker, float]:
        return {t: 0.0 for t in wanted}

    screen = liquidity_screened(
        nominate,
        min_price=LIQUIDITY_MIN_PRICE,
        min_dollar_volume=LIQUIDITY_MIN_DOLLAR_VOLUME,
        window=LIQUIDITY_WINDOW,
    )
    survivors = set(screen(context))
    return tuple(t for t in candidates if t in survivors)


def _resolve_ciks(
    liquid: tuple[Ticker, ...], cik: CikPort
) -> tuple[dict[Ticker, str], tuple[Ticker, ...]]:
    """Resolve each liquid name's CIK, bucketing the unresolvable ones.

    A ``KeyError`` from the resolver means the SEC map has no entry; such a name
    cannot yield EDGAR fundamentals, so it is excluded and counted rather than
    carried forward to fail silently later.
    """
    resolved: dict[Ticker, str] = {}
    missing: list[Ticker] = []
    for ticker in liquid:
        try:
            resolved[ticker] = cik.resolve(ticker)
        except KeyError:
            missing.append(ticker)
    return resolved, tuple(missing)


def _rank_by_liquidity(resolved: dict[Ticker, str], context: SignalContext) -> list[Ticker]:
    """Order names by trailing median dollar volume, descending.

    This is the cap's tie-free, return-blind ranking key. ``close x volume`` is
    traded notional — a measure of how easy a name is to trade, never a measure
    of how it is expected to perform — so preferring the most liquid names does
    not express any view on returns. Ties break alphabetically so the order is
    fully determined by the inputs and reproduces byte for byte.
    """
    return sorted(
        resolved,
        key=lambda t: (-_median_dollar_volume(context, t), str(t)),
    )


def _median_dollar_volume(context: SignalContext, ticker: Ticker) -> float:
    """Median of ``close x volume`` over the trailing liquidity window.

    Returns ``0.0`` if any bar in the window lacks volume. A name here has
    already passed the screen (which rejects absent volume), so this is a
    defensive floor rather than an expected path.
    """
    bars = context.history(ticker)[-LIQUIDITY_WINDOW:]
    traded = [bar.close * bar.volume for bar in bars if bar.volume is not None]
    if not traded or len(traded) < len(bars):
        return 0.0
    ordered = sorted(traded)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
