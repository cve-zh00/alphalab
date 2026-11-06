"""Unit tests for pure Variance-Targeted GARCH(1,1) forward volatility."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from alphalab.app.autopilot_runner import _volatilities_60d
from alphalab.domain.backtest.types import PriceBar
from alphalab.domain.compute import (
    GarchVolatilityForecast,
    InvalidValuationInputError,
    forecast_garch11_volatility,
)
from alphalab.domain.types import Ticker


def test_garch11_reacts_to_recent_shock_and_mean_reverts_across_horizons() -> None:
    rng = np.random.default_rng(42)
    calm = list(rng.normal(0.0, 0.008, size=110))
    burst = list(rng.normal(0.0, 0.035, size=16))
    rets = calm + burst

    f_short = forecast_garch11_volatility(rets, horizon_days=2, as_of=date(2026, 11, 3))
    f_long = forecast_garch11_volatility(rets, horizon_days=60, as_of=date(2026, 11, 3))

    assert isinstance(f_short, GarchVolatilityForecast)
    assert f_short.spot_volatility > f_short.unconditional_volatility
    assert f_short.annualized_volatility > f_long.annualized_volatility
    assert f_long.annualized_volatility > f_short.unconditional_volatility
    assert f_short.ci_low_annualized <= f_short.annualized_volatility <= f_short.ci_high_annualized
    assert f_short.provenanced.operator == "garch11_variance_targeted"
    assert f_short.provenanced.knowledge_date == date(2026, 11, 3)


def test_garch11_clips_split_outliers_without_exploding() -> None:
    rng = np.random.default_rng(7)
    clean_rets = list(rng.normal(0.0, 0.012, size=120))
    contaminated = list(clean_rets)
    contaminated[50] = -2.30  # Simulated unadjusted 10-for-1 stock split

    f_clean = forecast_garch11_volatility(clean_rets, horizon_days=10)
    f_contam = forecast_garch11_volatility(contaminated, horizon_days=10)

    assert f_contam.annualized_volatility < 0.45
    assert abs(f_contam.annualized_volatility - f_clean.annualized_volatility) < 0.15


def test_garch11_split_outlier_also_bypasses_parkinson_channel() -> None:
    rng = np.random.default_rng(7)
    clean_rets = list(rng.normal(0.0, 0.010, size=120))
    clean_hl: list[float | None] = [1.012] * 120

    contam_rets = list(clean_rets)
    contam_hl = list(clean_hl)
    # Unadjusted 10-for-1 split on the final bar in both close-to-close and high/low
    contam_rets[-1] = -0.90
    contam_hl[-1] = 10.0

    f_clean = forecast_garch11_volatility(clean_rets, high_low_ratios=clean_hl, horizon_days=10)
    f_contam = forecast_garch11_volatility(contam_rets, high_low_ratios=contam_hl, horizon_days=10)

    assert abs(f_contam.spot_volatility - f_clean.spot_volatility) < 0.02
    assert abs(f_contam.annualized_volatility - f_clean.annualized_volatility) < 0.02


def test_garch11_provenance_omega_drives_exact_multi_horizon_recursion() -> None:
    rng = np.random.default_rng(21)
    # Earlier high-vol regime + recent calm 126d regime tests single unconditional target
    early = list(rng.normal(0.0, 0.030, size=126))
    recent = list(rng.normal(0.0, 0.010, size=126))
    rets = early + recent

    fc = forecast_garch11_volatility(rets, horizon_days=10, target_window=126)
    inputs = dict(fc.provenanced.inputs)

    omega = inputs["omega"]
    alpha = inputs["alpha"]
    beta = inputs["beta"]
    v_target = (fc.unconditional_volatility**2) / 252.0
    s2_spot = (fc.spot_volatility**2) / 252.0

    # 1. Provenance omega matches the variance-targeting identity on v_target
    assert omega == pytest.approx(v_target * (1.0 - alpha - beta), rel=1e-9)

    # 2. Iterating s2_{t+h} = omega + (alpha + beta) * s2_{t+h-1} reproduces annualized_volatility
    h_vars: list[float] = []
    cur = s2_spot
    for _ in range(10):
        h_vars.append(cur)
        cur = omega + (alpha + beta) * cur
    iterated_ann_vol = float(np.sqrt(np.mean(h_vars) * 252.0))
    assert fc.annualized_volatility == pytest.approx(iterated_ann_vol, rel=1e-9)


def test_volatilities_60d_uses_spot_volatility_for_stop_loss_scaling() -> None:
    rng = np.random.default_rng(123)
    # 110 days of elevated vol followed by 16 days of very low vol so spot < H=10 forward
    rets = list(rng.normal(0.0, 0.028, size=110)) + list(rng.normal(0.0, 0.005, size=16))
    px = 100.0
    start = date(2026, 1, 1)
    series = [PriceBar(on=start, close=px, volume=1_000_000)]
    for idx, r in enumerate(rets, start=1):
        px = px * (1.0 + float(r))
        series.append(PriceBar(on=start + timedelta(days=idx), close=px, volume=1_000_000))

    as_of = series[-1].on
    bars = {Ticker("AAPL"): tuple(series)}
    spot_map = _volatilities_60d(bars, as_of)
    fwd10_map = _volatilities_60d(bars, as_of, horizon_days=10)
    expected_fc = forecast_garch11_volatility(rets, horizon_days=10, min_garch_obs=60, as_of=as_of)

    assert spot_map[Ticker("AAPL")] == pytest.approx(expected_fc.spot_volatility, rel=1e-9)
    assert fwd10_map[Ticker("AAPL")] == pytest.approx(expected_fc.annualized_volatility, rel=1e-9)
    assert spot_map[Ticker("AAPL")] < fwd10_map[Ticker("AAPL")]


def test_garch11_short_series_falls_back_to_realized_stdev() -> None:
    rets = [0.01, -0.015, 0.008, -0.005, 0.012, -0.009, 0.004, 0.006, -0.007, 0.005]
    fc = forecast_garch11_volatility(rets, horizon_days=10, min_garch_obs=30)

    assert fc.provenanced.operator == "realized_volatility_fallback"
    assert fc.annualized_volatility > 0.0
    assert fc.ci_low_annualized <= fc.annualized_volatility <= fc.ci_high_annualized


def test_garch11_zero_variance_series_returns_zero_forecast() -> None:
    rets = [0.0] * 60
    fc = forecast_garch11_volatility(rets, horizon_days=10)
    assert fc.annualized_volatility == 0.0
    assert fc.spot_volatility == 0.0
    assert fc.unconditional_volatility == 0.0


def test_garch11_incorporates_parkinson_high_low_intraday_range() -> None:
    rng = np.random.default_rng(99)
    rets = list(rng.normal(0.0, 0.005, size=80))
    wide_intraday = [1.04] * 80  # 4% intraday high/low range despite small close-to-close moves

    fc_close = forecast_garch11_volatility(rets, horizon_days=10)
    fc_range = forecast_garch11_volatility(rets, high_low_ratios=wide_intraday, horizon_days=10)

    assert fc_range.annualized_volatility > fc_close.annualized_volatility * 1.5


def test_garch11_rejects_invalid_inputs() -> None:
    with pytest.raises(InvalidValuationInputError):
        forecast_garch11_volatility([0.01])
    with pytest.raises(InvalidValuationInputError):
        forecast_garch11_volatility([float("nan"), 0.01])
    with pytest.raises(InvalidValuationInputError):
        forecast_garch11_volatility([0.01, -0.01, 0.02], horizon_days=0)
