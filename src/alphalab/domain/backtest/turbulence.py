"""Turbulence-based risk overlay, adapted from FinRL.

Provenance
----------
The construction is taken from ``AI4Finance-Foundation/FinRL`` (commit
``8511e7c``, ``finrl/meta/preprocessor/preprocessors.py``), MIT licensed. It is
the one part of that repository with genuinely correct temporal discipline: the
covariance is estimated on a strictly trailing window, so the index at a given
date uses no future information.

Turbulence is the Mahalanobis distance between the current cross-section of
returns and the trailing distribution of returns. It is large when assets move
together in unusual magnitudes or in unusual directions relative to their recent
covariance — the statistical signature of a stress episode.

Two deliberate improvements over the source
-------------------------------------------
1. **Point-in-time threshold.** FinRL compares turbulence against a threshold
   derived from the sample. Here the threshold is a percentile of turbulence
   values observed *within the trailing window only*, so the decision to
   de-risk never consults a distribution that includes the future.
2. **No same-bar execution.** FinRL's index at date ``i`` uses the return at
   ``i``, which invites trading on the same close. This overlay returns scores
   to an engine that fills at the next bar, so the signal cannot be acted on
   before it exists.

One tractability compromise, stated rather than hidden: a single covariance
matrix estimated at the decision date is reused to score every day in the
trailing window, instead of re-estimating a rolling covariance per day. A fully
rolling estimate would require hundreds of matrix inversions per rebalance. The
compromise is conservative in the sense that it uses *less* information, and it
remains strictly point-in-time.

Measured verdict: do not treat this as an edge
----------------------------------------------
This overlay was incorporated and then measured on the de-biased universe
(345 names sampled mechanically from the SEC list, liquidity screened),
in-sample 2010-2017 and out-of-sample 2018-2024. Two results matter:

**A real bug was found and fixed.** Applied to the full universe the index
degenerated: ``p/n = 1.333`` made the sample covariance singular, the
pseudo-inverse inflated the current reading to **6.9x** its own trailing 90th
percentile, and the overlay sat in cash on 86% of rebalances *identically for
every percentile threshold*. It was measuring dimensionality, not stress. The
``max_assets`` cap restores ``p/n = 0.119`` and the index becomes responsive
(11%, 21% and 39% cash at the 95th, 90th and 80th percentiles).

**It still does not beat the obvious control.** Out-of-sample the repaired
overlay improves net Sharpe from 0.171 to between 0.241 and 0.266 and cuts the
worst drawdown from -72.1% to as little as -54.3%. Tempting. But compared
against simply holding cash *at random* on the same fraction of rebalances
(40 seeds), turbulence lands at the 82nd, 78th and 72nd percentile of that
control. **None reaches 95%.** Most of the apparent benefit is explained by
reduced market exposure, not by timing when to reduce it.

So this module is kept for three reasons — the dimensionality diagnosis is a
genuine finding about the method not transferring beyond a narrow index, the
negative result is worth preserving, and the overlay is a correct
implementation should a future universe or threshold change alter the verdict.
It is **not** endorsed as a source of edge. Any exposure-reducing overlay must
clear the random-abstention control before it may be described as skill.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from alphalab.domain.backtest.context import SignalContext, SignalFunction
from alphalab.domain.types import Ticker

__all__ = [
    "DEFAULT_LOOKBACK",
    "DEFAULT_PERCENTILE",
    "TurbulenceReading",
    "turbulence_reading",
    "turbulence_screened",
]

#: Trailing window for the covariance estimate, one trading year.
DEFAULT_LOOKBACK = 252

#: Percentile of trailing turbulence above which exposure is cut.
DEFAULT_PERCENTILE = 90.0

#: Cap on assets entering the covariance estimate.
#:
#: This is not a tuning knob, it is a correctness requirement. Mahalanobis
#: distance degenerates as the asset count ``p`` approaches the observation
#: count ``n``: the sample covariance becomes singular and the pseudo-inverse
#: inflates the distance of any point outside the estimation sample.
#:
#: Measured on this project's 345-name universe, the uncapped index gave
#: ``p/n = 1.333`` (336 assets, 252 days), a current reading **6.9x** its own
#: trailing 90th percentile, and therefore fired on 86% of rebalances
#: regardless of the percentile chosen. It was measuring dimensionality, not
#: market stress. FinRL avoids this only incidentally, by applying the index to
#: the Dow 30 (``p/n = 0.119``).
#:
#: Thirty keeps ``p/n`` near FinRL's well-conditioned ratio.
DEFAULT_MAX_ASSETS = 30

#: Minimum assets required for a covariance estimate worth inverting.
_MIN_ASSETS = 5


class TurbulenceReading:
    """Current turbulence alongside the trailing threshold it is judged against."""

    __slots__ = ("current", "threshold")

    def __init__(self, *, current: float, threshold: float) -> None:
        self.current = current
        self.threshold = threshold

    @property
    def is_turbulent(self) -> bool:
        """Whether the current reading exceeds its trailing threshold."""
        return self.current > self.threshold


def turbulence_reading(
    context: SignalContext,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    percentile: float = DEFAULT_PERCENTILE,
    max_assets: int = DEFAULT_MAX_ASSETS,
) -> TurbulenceReading | None:
    """Compute turbulence at the context's decision date, or ``None``.

    Returns ``None`` when the trailing window holds too few complete series to
    estimate a covariance. Absent data yields no reading rather than a
    fabricated calm one, so the caller must decide what to do with ignorance
    instead of being handed a false reassurance.
    """
    matrix, _ = _return_matrix(context, lookback, max_assets)
    if matrix is None or matrix.shape[0] < _MIN_ASSETS + 1:
        return None

    history = matrix[:-1]
    current = matrix[-1]
    if history.shape[0] < _MIN_ASSETS:
        return None

    mean = history.mean(axis=0)
    inverse = np.linalg.pinv(np.cov(history, rowvar=False))

    distances = _mahalanobis(history, mean, inverse)
    if distances.size == 0:
        return None

    return TurbulenceReading(
        current=float(_mahalanobis(current.reshape(1, -1), mean, inverse)[0]),
        threshold=float(np.percentile(distances, percentile)),
    )


def turbulence_screened(
    inner: SignalFunction,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    percentile: float = DEFAULT_PERCENTILE,
    max_assets: int = DEFAULT_MAX_ASSETS,
    flat_when_unknown: bool = False,
) -> SignalFunction:
    """Wrap ``inner`` so it expresses no opinion while markets are turbulent.

    Returning an empty mapping makes the engine select nothing, which closes
    positions and holds cash. No engine change is needed: "no opinion" already
    means "no exposure".

    ``flat_when_unknown`` decides the behaviour when turbulence cannot be
    computed. The default is to stay invested, on the grounds that early in a
    backtest the window is simply not full yet and sitting out would confuse a
    data limitation with a risk signal.
    """

    def score(context: SignalContext) -> Mapping[Ticker, float]:
        reading = turbulence_reading(
            context, lookback=lookback, percentile=percentile, max_assets=max_assets
        )
        if reading is None:
            return {} if flat_when_unknown else inner(context)
        if reading.is_turbulent:
            return {}
        return inner(context)

    return score


def _return_matrix(
    context: SignalContext, lookback: int, max_assets: int
) -> tuple[np.ndarray | None, tuple[Ticker, ...]]:
    """Daily returns over the trailing window for names with complete history.

    Incomplete series are dropped rather than imputed: filling a gap would
    invent co-movement, which is precisely the quantity being measured.

    At most ``max_assets`` names are kept, chosen by largest trailing traded
    value, so the covariance stays well conditioned. Selecting on liquidity
    uses only data already inside the truncated context, so the choice remains
    point-in-time; it also keeps the index anchored to names that actually
    transmit market stress rather than to thinly traded ones.
    """
    needed = lookback + 1
    columns: list[np.ndarray] = []
    kept: list[Ticker] = []
    weights: list[float] = []
    for ticker in context.tickers():
        closes = context.closes(ticker)[-needed:]
        if len(closes) < needed:
            continue
        prices = np.asarray(closes, dtype=float)
        if np.any(prices <= 0.0):
            continue
        columns.append(prices[1:] / prices[:-1] - 1.0)
        kept.append(ticker)
        weights.append(_traded_value(context, ticker, needed))

    if len(kept) < _MIN_ASSETS:
        return None, ()
    order = sorted(range(len(kept)), key=lambda i: -weights[i])[:max_assets]
    order.sort()
    return np.column_stack([columns[i] for i in order]), tuple(kept[i] for i in order)


def _traded_value(context: SignalContext, ticker: Ticker, window: int) -> float:
    """Median trailing traded value, or zero when volume is unavailable."""
    bars = context.history(ticker)[-window:]
    values = [b.close * b.volume for b in bars if b.volume is not None]
    if not values:
        return 0.0
    return float(np.median(values))


def _mahalanobis(rows: np.ndarray, mean: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of each row from ``mean`` under ``inverse``."""
    centred = rows - mean
    quadratic = np.einsum("ij,jk,ik->i", centred, inverse, centred)
    clipped: np.ndarray = np.maximum(quadratic, 0.0)
    return clipped
