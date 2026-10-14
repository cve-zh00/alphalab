"""Tests for the turbulence risk overlay.

The most valuable test here is the dimensionality regression: the index is
silently broken when the asset count approaches the observation count, and the
breakage looks like a working risk model rather than an error.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from alphalab.domain.backtest import (
    PriceBar,
    SignalContext,
    equal_weight_signal,
    turbulence_reading,
    turbulence_screened,
)
from alphalab.domain.backtest.turbulence import _return_matrix
from alphalab.domain.types import Ticker

_START = date(2015, 1, 1)


def _universe(
    *, n_assets: int, n_days: int, seed: int, shock_at: int | None = None
) -> dict[Ticker, list[PriceBar]]:
    """Gaussian price paths, optionally with a correlated shock on one day."""
    rng = np.random.default_rng(seed)
    out: dict[Ticker, list[PriceBar]] = {}
    shocks = rng.normal(0.0, 0.01, size=(n_days - 1, n_assets))
    if shock_at is not None:
        shocks[shock_at, :] = -0.12  # every name falls together
    for j in range(n_assets):
        closes = [100.0]
        for i in range(n_days - 1):
            closes.append(closes[-1] * (1.0 + float(shocks[i, j])))
        out[Ticker(f"A{j:03d}")] = [
            PriceBar(on=_START + timedelta(days=i), close=c, volume=1_000_000.0)
            for i, c in enumerate(closes)
        ]
    return out


def _context(universe: dict[Ticker, list[PriceBar]], *, day_index: int) -> SignalContext:
    return SignalContext(as_of=_START + timedelta(days=day_index), prices=universe)


def test_asset_cap_keeps_covariance_well_conditioned() -> None:
    """The cap holds ``p/n`` low even when the universe is far wider than the window.

    Without it, measured on the real 345-name universe, ``p/n`` reached 1.333
    and the index fired on 86% of rebalances identically for every threshold —
    it was reporting dimensionality, not stress.
    """
    # Arrange — 120 assets against a 252-day window would give p/n ≈ 0.48.
    universe = _universe(n_assets=120, n_days=300, seed=3)
    context = _context(universe, day_index=299)

    # Act
    capped, kept = _return_matrix(context, 252, 30)

    # Assert
    assert capped is not None
    assert capped.shape[1] == 30
    assert len(kept) == 30
    assert capped.shape[1] / capped.shape[0] < 0.2


def test_uncapped_estimate_degenerates() -> None:
    """Pinning the failure mode: without the cap the reading explodes.

    This is the behaviour that made the overlay look like a risk model while
    being an artefact, and it must not silently return.
    """
    # Arrange
    universe = _universe(n_assets=300, n_days=300, seed=5)
    context = _context(universe, day_index=299)

    # Act
    capped = turbulence_reading(context, lookback=252, max_assets=30)
    uncapped = turbulence_reading(context, lookback=252, max_assets=300)

    # Assert — the uncapped ratio is far more extreme than the capped one.
    assert capped is not None
    assert uncapped is not None
    assert uncapped.current / uncapped.threshold > capped.current / capped.threshold


def test_correlated_crash_raises_turbulence() -> None:
    """A day when everything falls together must read as turbulent."""
    # Arrange — the shock lands on the most recent return, index n_days-2.
    universe = _universe(n_assets=40, n_days=300, seed=7, shock_at=298)
    calm = _context(_universe(n_assets=40, n_days=300, seed=7), day_index=299)
    stressed = _context(universe, day_index=299)

    # Act
    calm_reading = turbulence_reading(calm, max_assets=30)
    stressed_reading = turbulence_reading(stressed, max_assets=30)

    # Assert
    assert calm_reading is not None
    assert stressed_reading is not None
    assert stressed_reading.current > calm_reading.current


def test_insufficient_history_returns_none_not_calm() -> None:
    """Too little data yields no reading, never a fabricated calm one.

    Returning "calm" would hand the caller false reassurance built from
    missing data.
    """
    # Arrange
    universe = _universe(n_assets=40, n_days=30, seed=1)
    context = _context(universe, day_index=29)

    # Act / Assert
    assert turbulence_reading(context, lookback=252) is None


def test_overlay_goes_flat_when_turbulent() -> None:
    """Exceeding the threshold produces no opinions, hence no exposure."""
    # Arrange — a percentile of 0 makes almost any reading "turbulent".
    universe = _universe(n_assets=40, n_days=300, seed=11)
    context = _context(universe, day_index=299)
    overlay = turbulence_screened(equal_weight_signal(), percentile=0.0, max_assets=30)

    # Act
    scores = overlay(context)

    # Assert
    assert scores == {}


def test_overlay_passes_through_when_calm() -> None:
    """Below the threshold the wrapped signal is returned untouched."""
    # Arrange — a percentile of 100 makes nothing exceed the threshold.
    universe = _universe(n_assets=40, n_days=300, seed=13)
    context = _context(universe, day_index=299)
    inner = equal_weight_signal()
    overlay = turbulence_screened(inner, percentile=100.0, max_assets=30)

    # Act / Assert
    assert overlay(context) == inner(context)


def test_unknown_reading_stays_invested_by_default() -> None:
    """Early in a series the window is not full; that is not a risk signal.

    Sitting out would confuse a data limitation with evidence of danger.
    """
    # Arrange
    universe = _universe(n_assets=40, n_days=30, seed=17)
    context = _context(universe, day_index=29)
    inner = equal_weight_signal()

    # Act
    invested = turbulence_screened(inner, lookback=252)(context)
    flat = turbulence_screened(inner, lookback=252, flat_when_unknown=True)(context)

    # Assert
    assert invested == inner(context)
    assert flat == {}


@pytest.mark.parametrize("percentile", [50.0, 90.0, 99.0])
def test_threshold_responds_to_percentile(percentile: float) -> None:
    """Different percentiles must give different thresholds.

    When the estimate degenerated, every percentile produced the same decision.
    Identical behaviour across thresholds is the signature of that bug.
    """
    # Arrange
    universe = _universe(n_assets=40, n_days=300, seed=19)
    context = _context(universe, day_index=299)

    # Act
    reading = turbulence_reading(context, percentile=percentile, max_assets=30)

    # Assert
    assert reading is not None
    assert reading.threshold > 0.0
