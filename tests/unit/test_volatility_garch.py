"""Unit tests for pure Variance-Targeted GARCH(1,1) forward volatility."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from alphalab.domain.compute import (
    GarchVolatilityForecast,
    InvalidValuationInputError,
    forecast_garch11_volatility,
)


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


def test_garch11_rejects_invalid_inputs() -> None:
    with pytest.raises(InvalidValuationInputError):
        forecast_garch11_volatility([0.01])
    with pytest.raises(InvalidValuationInputError):
        forecast_garch11_volatility([float("nan"), 0.01])
    with pytest.raises(InvalidValuationInputError):
        forecast_garch11_volatility([0.01, -0.01, 0.02], horizon_days=0)
