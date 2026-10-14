"""Price-based signal constructors.

Each factory returns a :class:`~alphalab.domain.backtest.context.SignalFunction`
closed over its parameters. Signals read only from the context they are handed,
which is truncated at the decision date, so none of them can reach the future
even by accident.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from alphalab.domain.backtest.context import SignalContext, SignalFunction
from alphalab.domain.types import Ticker

__all__ = [
    "equal_weight_signal",
    "liquidity_screened",
    "momentum_signal",
    "rank_ensemble_signal",
]


def momentum_signal(*, lookback: int, skip: int = 0) -> SignalFunction:
    """Score each name by its total return over the trailing ``lookback`` bars.

    ``skip`` omits the most recent ``skip`` bars from the measurement window, a
    standard guard against short-term reversal contaminating the momentum
    estimate. Names with insufficient history are omitted rather than scored
    from a shorter window, which would make their scores incomparable.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if skip < 0:
        raise ValueError("skip must be non-negative")

    needed = lookback + skip + 1

    def score(context: SignalContext) -> Mapping[Ticker, float]:
        out: dict[Ticker, float] = {}
        for ticker in context.tickers():
            closes = context.closes(ticker)
            if len(closes) < needed:
                continue
            end = len(closes) - skip - 1
            start = end - lookback
            past, recent = closes[start], closes[end]
            if past > 0.0:
                out[ticker] = recent / past - 1.0
        return out

    return score


def equal_weight_signal() -> SignalFunction:
    """Score every available name identically.

    With equal scores the engine's deterministic tie-break selects the first
    ``top_n`` tickers alphabetically, giving a naive diversified portfolio that
    is useful as a reference point against any ranked strategy.
    """

    def score(context: SignalContext) -> Mapping[Ticker, float]:
        return dict.fromkeys(context.tickers(), 0.0)

    return score


def rank_ensemble_signal(*, lookbacks: Sequence[int], skip: int = 0) -> SignalFunction:
    """Average each name's cross-sectional rank across several lookbacks.

    Why ensemble instead of choosing the best lookback
    --------------------------------------------------
    Measured on this project's universe, picking the best-performing lookback
    in-sample was actively harmful: the top in-sample configuration ranked
    *last* out-of-sample, and the rank correlation between the two windows was
    negative (-0.257). The hardened PBO gate independently flagged the same
    thing — with configurations whose Sharpe ratios differ only by noise,
    "best" is not a property that persists.

    Averaging ranks sidesteps the problem by making no choice at all. It is one
    strategy with one specification, so there is no in-sample selection step
    left to overfit.

    Ranks are used rather than raw returns so a single volatile name cannot
    dominate the blend, and are normalised to ``[0, 1]`` per lookback so that
    lookbacks covering different universe sizes remain comparable.
    """
    if not lookbacks:
        raise ValueError("lookbacks must not be empty")

    components = [momentum_signal(lookback=lb, skip=skip) for lb in lookbacks]

    def score(context: SignalContext) -> Mapping[Ticker, float]:
        totals: dict[Ticker, float] = {}
        counts: dict[Ticker, int] = {}
        for component in components:
            for ticker, rank in _normalised_ranks(component(context)).items():
                totals[ticker] = totals.get(ticker, 0.0) + rank
                counts[ticker] = counts.get(ticker, 0) + 1
        return {t: totals[t] / counts[t] for t in totals}

    return score


def _normalised_ranks(scores: Mapping[Ticker, float]) -> dict[Ticker, float]:
    """Map raw scores to ranks in ``[0, 1]``, highest score to 1.0."""
    if not scores:
        return {}
    ordered = sorted(scores.items(), key=lambda pair: (pair[1], pair[0]))
    last = len(ordered) - 1
    if last == 0:
        return {ordered[0][0]: 1.0}
    return {ticker: i / last for i, (ticker, _) in enumerate(ordered)}


def liquidity_screened(
    inner: SignalFunction,
    *,
    min_price: float = 5.0,
    min_dollar_volume: float = 5_000_000.0,
    window: int = 60,
) -> SignalFunction:
    """Restrict ``inner``'s opinions to names that are actually tradable.

    Why a screen is mandatory, not cosmetic
    ---------------------------------------
    Run without one on a mechanically sampled universe, this project measured
    an equal-weight portfolio compounding at an impossible **+414% a year**.
    That figure is an artefact, not an edge: on a sub-dollar stock a one-cent
    tick is a double-digit percentage return, and equal-weighting hundreds of
    such names harvests bid-ask bounce rather than value. Any cost assumption
    calibrated for liquid large caps is pure fiction on those names.

    Screening on price and traded value removes the artefact and makes the cost
    model honest, because the surviving names are ones where a few basis points
    of friction is a defensible estimate.

    The screen is point-in-time by construction: it reads only the trailing
    ``window`` bars of the context, which is itself truncated at the decision
    date. Names whose bars carry no volume are excluded rather than assumed
    liquid — absent data is not evidence of tradability.
    """
    if min_price < 0.0 or min_dollar_volume < 0.0:
        raise ValueError("thresholds must be non-negative")
    if window <= 0:
        raise ValueError("window must be positive")

    def score(context: SignalContext) -> Mapping[Ticker, float]:
        return {
            ticker: value
            for ticker, value in inner(context).items()
            if _is_liquid(context, ticker, min_price, min_dollar_volume, window)
        }

    return score


def _is_liquid(
    context: SignalContext,
    ticker: Ticker,
    min_price: float,
    min_dollar_volume: float,
    window: int,
) -> bool:
    """Whether a name cleared both thresholds over the trailing window."""
    bars = context.history(ticker)[-window:]
    if not bars:
        return False
    if bars[-1].close < min_price:
        return False
    traded = [bar.close * bar.volume for bar in bars if bar.volume is not None]
    if len(traded) < len(bars):
        return False
    return _median(traded) >= min_dollar_volume


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
