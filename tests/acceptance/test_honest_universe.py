"""The honest-universe verdict: momentum's edge does not survive de-biasing.

This module pins the most consequential measurement in the project. Run on the
48 names the author picked from memory, a momentum ensemble looked excellent —
information ratio +0.97 out-of-sample. Run on a universe the author did not
choose, with an added tradability screen, the same strategy produced a
*negative* out-of-sample excess return.

The difference was never the strategy. It was who picked the names.

Two independent corrections were required to see it:

1. **Mechanical universe.** Tickers drawn by a seeded sample over the SEC's
   full company list (``scripts/fetch_universe.py``, seed 42), so selection is
   reproducible but uncurated.
2. **Point-in-time liquidity screen.** Without it the de-biased universe
   equal-weighted compounded at an impossible +430% a year — bid-ask bounce on
   sub-dollar stocks, not return. The screen also makes the few-basis-point
   cost assumption defensible.

A useful sanity check falls out: after both corrections, a random sample of US
equities slightly *underperforms* SPY. That is the expected shape of equity
returns, where a minority of names drive the index, and it is evidence the
de-biased universe is behaving honestly.

The data lives outside the repository because it is tens of megabytes. These
tests skip when it is absent and the script regenerates it deterministically.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from alphalab.domain.backtest import (
    PriceBar,
    ProportionalCostModel,
    buy_and_hold,
    equal_weight_signal,
    liquidity_screened,
    rank_ensemble_signal,
    relative_performance,
    run_backtest,
)
from alphalab.domain.backtest.types import BacktestResult
from alphalab.domain.costs import CostSpec
from alphalab.domain.types import Ticker

_ROOT = Path(__file__).resolve().parents[2]
_PRICES = _ROOT / "data" / "prices"
_SPY = _ROOT / "tests" / "fixtures" / "SPY.csv"

_COSTS = CostSpec(half_spread_bps=2.0, slippage_bps=3.0)
_LOOKBACKS = (60, 120, 250)
_SCREEN = {"min_price": 5.0, "min_dollar_volume": 5_000_000.0}

_IS = (date(2010, 1, 1), date(2017, 12, 31))
_OOS = (date(2018, 1, 1), date(2024, 12, 31))

pytestmark = pytest.mark.skipif(
    not _PRICES.is_dir() or len(list(_PRICES.glob("*.csv"))) < 200,
    reason=(
        "mechanically sampled universe absent; regenerate with "
        "ALPHALAB_EDGAR_USER_AGENT=... uv run python scripts/fetch_universe.py"
    ),
)


def _load(path: Path, lo: date, hi: date) -> list[PriceBar]:
    bars: list[PriceBar] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                on = date.fromisoformat(row["date"][:10])
                close = float(row["close"])
                volume = float(row["volume"]) if row.get("volume") else None
            except (ValueError, KeyError):
                continue
            if close > 0.0 and lo <= on <= hi:
                bars.append(PriceBar(on=on, close=close, volume=volume))
    return bars


def _universe(lo: date, hi: date) -> dict[Ticker, list[PriceBar]]:
    out: dict[Ticker, list[PriceBar]] = {}
    for path in sorted(_PRICES.glob("*.csv")):
        bars = _load(path, lo, hi)
        if len(bars) > 300:
            out[Ticker(path.stem)] = bars
    return out


def _quarterly(universe: dict[Ticker, list[PriceBar]]) -> list[date]:
    calendar = sorted({bar.on for bars in universe.values() for bar in bars})
    picked: list[date] = []
    seen: set[tuple[int, int]] = set()
    for day in calendar:
        key = (day.year, (day.month - 1) // 3)
        if key not in seen:
            seen.add(key)
            picked.append(day)
    return picked


def _run(window: tuple[date, date], *, screened: bool, ensemble: bool) -> BacktestResult:
    universe = _universe(*window)
    inner = rank_ensemble_signal(lookbacks=_LOOKBACKS) if ensemble else equal_weight_signal()
    signal = liquidity_screened(inner, **_SCREEN) if screened else inner  # type: ignore[arg-type]
    return run_backtest(
        prices=universe,
        rebalance_dates=_quarterly(universe),
        signal=signal,
        cost_model=ProportionalCostModel(_COSTS),
        top_n=5 if ensemble else len(universe),
    )


def test_liquidity_screen_removes_the_microcap_artefact() -> None:
    """Unscreened, the universe compounds impossibly; screened, plausibly."""
    # Act
    unscreened = _run(_OOS, screened=False, ensemble=False)
    screened = _run(_OOS, screened=True, ensemble=False)

    # Assert — the artefact is enormous and the screen eliminates it.
    assert unscreened.net.cagr > 1.0, "expected the unscreened artefact to be present"
    assert screened.net.cagr < 0.30


def test_mechanical_universe_does_not_beat_the_index() -> None:
    """A universe the author did not pick shows no free outperformance.

    Contrast with the hand-picked fixture, where equal-weighting beat SPY by
    over seven points a year with no signal. That gap was the author's
    hindsight, and here it is gone.
    """
    # Act
    naive = _run(_OOS, screened=True, ensemble=False)
    spy = buy_and_hold(
        ticker=Ticker("SPY"),
        bars=_load(_SPY, *_OOS),
        cost_model=ProportionalCostModel(_COSTS),
    )

    # Assert
    assert naive.net.cagr - spy.net.cagr < 0.02


def test_momentum_edge_does_not_survive_on_an_honest_universe() -> None:
    """The headline negative result, pinned so it cannot be quietly lost.

    In-sample the ensemble still looks additive; out-of-sample it does not.
    Should a future change make this pass, that is not cause for celebration —
    it means either the universe or the measurement changed, and the cause must
    be found before any claim of edge is made.
    """
    # Act
    in_sample = relative_performance(
        _run(_IS, screened=True, ensemble=True).net_returns,
        _run(_IS, screened=True, ensemble=False).net_returns,
    )
    out_of_sample = relative_performance(
        _run(_OOS, screened=True, ensemble=True).net_returns,
        _run(_OOS, screened=True, ensemble=False).net_returns,
    )

    # Assert
    assert in_sample.information_ratio > 0.0, "in-sample edge expected to appear"
    assert out_of_sample.information_ratio < 0.1, (
        f"OOS information ratio {out_of_sample.information_ratio:.4f} is no longer "
        "negligible; investigate before claiming an edge"
    )
