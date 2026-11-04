"""Pure Variance-Targeted GARCH(1,1) forward volatility estimator.

Why GARCH(1,1) over foundation models (Kronos) and trailing sample stdev
------------------------------------------------------------------------
1. **Empirical dominance on US equities:** In Table 18 of the Kronos paper
   (*arXiv:2508.02739*, AAAI 2026), classical GARCH (Bollerslev 1986) beats
   every variant of Kronos (``Kronos-small``, ``Kronos-base``, ``Kronos-large``)
   on NASDAQ (``XNAS``) realized volatility forecasting (``MAE = 0.1259`` vs.
   ``0.1407``, ``R² = 0.2271`` vs. ``0.0961``).
2. **Mean reversion + volatility clustering:** Trailing sample volatility
   (:math:`\\sigma_{20\\text{d}}` or :math:`\\sigma_{60\\text{d}}`) weights all
   observations equally inside a flat boxcar window and assumes zero mean
   reversion. Variance-targeted :math:`\\text{GARCH}(1,1)` (Engle & Mezrich
   1996) anchors long-run variance to the medium-term structural variance
   :math:`\\bar{\\sigma}^2`, reacts immediately to recent shocks via
   :math:`\\alpha r_t^2`, and integrates expected variance analytically over
   the forward holding horizon :math:`H`:

   .. math::

      \\mathbb{E}_t\\!\\left[\\frac{1}{H}\\sum_{h=1}^{H}\\sigma_{t+h}^2\\right]
      = \\bar{\\sigma}^2
      + \\frac{1 - (\\alpha + \\beta)^H}{H(1 - \\alpha - \\beta)}
        \\left(\\sigma_{t+1}^2 - \\bar{\\sigma}^2\\right)

3. **Pure domain & C1 Provenance:** Uses only :mod:`math`, :mod:`numpy`, and
   :func:`scipy.optimize.minimize` (already in the core dependency set), is
   100% deterministic, passes ``mypy --strict``, and emits a full
   :class:`~alphalab.domain.compute.provenance.ProvenancedValue` derivation node.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from datetime import date

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import minimize

from alphalab.domain.compute.errors import InvalidValuationInputError
from alphalab.domain.compute.provenance import ProvenancedValue

__all__ = [
    "DEFAULT_GARCH_ALPHA",
    "DEFAULT_GARCH_BETA",
    "DEFAULT_GARCH_HORIZON_DAYS",
    "DEFAULT_GARCH_MIN_OBS",
    "DEFAULT_GARCH_TARGET_WINDOW",
    "GarchVolatilityForecast",
    "forecast_garch11_volatility",
]

_OPERATOR_VERSION = "1.0.0"
_TRADING_DAYS_PER_YEAR = 252.0
DEFAULT_GARCH_ALPHA = 0.07
DEFAULT_GARCH_BETA = 0.89
DEFAULT_GARCH_HORIZON_DAYS = 10
DEFAULT_GARCH_TARGET_WINDOW = 126
DEFAULT_GARCH_MIN_OBS = 30
_DEFAULT_MAX_ABS_DAILY_RETURN = 0.20
_DEFAULT_SPLIT_JUMP_THRESHOLD = 0.35


class GarchVolatilityForecast(BaseModel):
    """Immutable forward volatility forecast and C1 provenance lineage."""

    model_config = ConfigDict(frozen=True)

    annualized_volatility: float = Field(ge=0.0)
    ci_low_annualized: float = Field(ge=0.0)
    ci_high_annualized: float = Field(ge=0.0)
    spot_volatility: float = Field(ge=0.0)
    unconditional_volatility: float = Field(ge=0.0)
    alpha: float = Field(ge=0.0, le=1.0)
    beta: float = Field(ge=0.0, le=1.0)
    horizon_days: int = Field(ge=1)
    provenanced: ProvenancedValue


def _fit_qml_alpha_beta(r2: NDArray[np.float64], v_uncond: float) -> tuple[float, float]:
    """Fit GARCH(1,1) (alpha, beta) by variance-targeted Quasi-Maximum Likelihood."""

    def _nll(params: NDArray[np.float64]) -> float:
        alpha = float(params[0])
        beta = float(params[1])
        if alpha < 0.01 or beta < 0.60 or (alpha + beta) >= 0.992:
            return 1e12
        omega = v_uncond * (1.0 - alpha - beta)
        s2 = v_uncond
        total = 0.0
        for x in r2:
            total += math.log(s2) + float(x) / s2
            s2 = omega + alpha * float(x) + beta * s2
        return 0.5 * total

    res = minimize(
        _nll,
        np.array([DEFAULT_GARCH_ALPHA, DEFAULT_GARCH_BETA], dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": 60, "xatol": 5e-4, "fatol": 1e-3},
    )
    alpha = float(res.x[0])
    beta = float(res.x[1])
    if alpha < 0.01 or beta < 0.60 or (alpha + beta) >= 0.992:
        return DEFAULT_GARCH_ALPHA, DEFAULT_GARCH_BETA
    return alpha, beta


def forecast_garch11_volatility(
    returns: Sequence[float],
    *,
    high_low_ratios: Sequence[float | None] | None = None,
    horizon_days: int = DEFAULT_GARCH_HORIZON_DAYS,
    target_window: int = DEFAULT_GARCH_TARGET_WINDOW,
    min_garch_obs: int = DEFAULT_GARCH_MIN_OBS,
    max_abs_daily_return: float = _DEFAULT_MAX_ABS_DAILY_RETURN,
    as_of: date | None = None,
) -> GarchVolatilityForecast:
    """Forecast annualized forward volatility over ``horizon_days`` via GARCH(1,1).

    When ``high_low_ratios`` (intraday ``high / low`` per bar) are supplied,
    blends squared close-to-close returns with Parkinson (1980) intraday range
    variance :math:`(\\ln(H_t/L_t))^2 / (4\\ln 2)` (Range-GARCH, Brandt & Jones
    2006). When ``len(returns) < min_garch_obs``, gracefully falls back to
    sample realized standard deviation.
    """
    if len(returns) < 2:
        raise InvalidValuationInputError("forecast_garch11_volatility requires >= 2 daily returns")
    if horizon_days < 1:
        raise InvalidValuationInputError("horizon_days must be >= 1")

    clean = [
        0.0
        if abs(float(r)) > _DEFAULT_SPLIT_JUMP_THRESHOLD
        else max(-max_abs_daily_return, min(max_abs_daily_return, float(r)))
        for r in returns
        if math.isfinite(r)
    ]
    n_obs = len(clean)
    if n_obs < 2:
        raise InvalidValuationInputError("forecast_garch11_volatility requires >= 2 finite returns")

    if n_obs < min_garch_obs:
        sample_vol = statistics.stdev(clean) * math.sqrt(_TRADING_DAYS_PER_YEAR)
        se_factor = 1.0 / math.sqrt(2.0 * max(1, n_obs - 1))
        ci_low = max(0.0, sample_vol * max(0.25, 1.0 - 1.96 * se_factor))
        ci_high = sample_vol * (1.0 + 1.96 * se_factor)
        node = ProvenancedValue(
            value=sample_vol,
            operator="realized_volatility_fallback",
            operator_version=_OPERATOR_VERSION,
            inputs=(
                ("n_returns", float(n_obs)),
                ("horizon_days", float(horizon_days)),
                ("sample_vol_annualized", sample_vol),
            ),
            unit="ratio",
            knowledge_date=as_of,
        )
        return GarchVolatilityForecast(
            annualized_volatility=sample_vol,
            ci_low_annualized=ci_low,
            ci_high_annualized=ci_high,
            spot_volatility=sample_vol,
            unconditional_volatility=sample_vol,
            alpha=0.0,
            beta=1.0,
            horizon_days=horizon_days,
            provenanced=node,
        )

    arr = np.asarray(clean, dtype=np.float64)
    r2 = arr * arr
    if high_low_ratios is not None and len(high_low_ratios) == len(returns):
        inv_4ln2 = 1.0 / (4.0 * math.log(2.0))
        blended: list[float] = []
        for idx, r_val in enumerate(returns):
            if not math.isfinite(r_val):
                continue
            c_r2 = float(r2[len(blended)])
            hl = high_low_ratios[idx]
            if hl is not None and math.isfinite(hl) and hl > 1.0:
                log_hl = min(max_abs_daily_return * 1.5, math.log(hl))
                park_var = (log_hl * log_hl) * inv_4ln2
                blended.append(0.5 * c_r2 + 0.5 * park_var)
            else:
                blended.append(c_r2)
        if len(blended) == n_obs:
            r2 = np.asarray(blended, dtype=np.float64)
    v_full = float(np.mean(r2))
    v_target = float(np.mean(r2[-min(n_obs, max(20, target_window)) :]))
    if v_full <= 1e-12 or v_target <= 1e-12:
        node = ProvenancedValue(
            value=0.0,
            operator="garch11_variance_targeted",
            operator_version=_OPERATOR_VERSION,
            inputs=(("n_returns", float(n_obs)), ("horizon_days", float(horizon_days))),
            unit="ratio",
            knowledge_date=as_of,
        )
        return GarchVolatilityForecast(
            annualized_volatility=0.0,
            ci_low_annualized=0.0,
            ci_high_annualized=0.0,
            spot_volatility=0.0,
            unconditional_volatility=0.0,
            alpha=DEFAULT_GARCH_ALPHA,
            beta=DEFAULT_GARCH_BETA,
            horizon_days=horizon_days,
            provenanced=node,
        )

    alpha, beta = _fit_qml_alpha_beta(r2, v_full)
    persistence = alpha + beta

    # Strictly causal expanding/rolling variance targeting inside the recursion:
    # at each bar tau, omega_tau uses only data observed up to tau (never future bars).
    warmup = min(20, n_obs)
    s2 = float(np.mean(r2[:warmup]))
    w_len = max(20, target_window)
    for tau in range(warmup, n_obs):
        start_tau = max(0, tau - w_len)
        local_uncond = float(np.mean(r2[start_tau:tau]))
        omega_tau = local_uncond * (1.0 - persistence)
        s2 = omega_tau + alpha * float(r2[tau]) + beta * s2
    omega = v_target * (1.0 - persistence)

    scale = (
        (1.0 - persistence**horizon_days) / (horizon_days * (1.0 - persistence))
        if abs(1.0 - persistence) > 1e-6
        else 1.0
    )
    expected_daily_var = max(1e-12, v_target + scale * (s2 - v_target))

    ann_vol = math.sqrt(expected_daily_var * _TRADING_DAYS_PER_YEAR)
    spot_vol = math.sqrt(max(1e-12, s2) * _TRADING_DAYS_PER_YEAR)
    uncond_vol = math.sqrt(v_target * _TRADING_DAYS_PER_YEAR)

    # Asymptotic 95% interval for H-day realized volatility around conditional forecast
    se_ratio = 1.0 / math.sqrt(2.0 * max(1, horizon_days))
    ci_low = max(0.0, ann_vol * max(0.25, 1.0 - 1.96 * se_ratio))
    ci_high = ann_vol * (1.0 + 1.96 * se_ratio)

    node = ProvenancedValue(
        value=ann_vol,
        operator="garch11_variance_targeted",
        operator_version=_OPERATOR_VERSION,
        inputs=(
            ("n_returns", float(n_obs)),
            ("horizon_days", float(horizon_days)),
            ("alpha", alpha),
            ("beta", beta),
            ("omega", omega),
            ("spot_volatility", spot_vol),
            ("unconditional_volatility", uncond_vol),
        ),
        unit="ratio",
        knowledge_date=as_of,
    )
    return GarchVolatilityForecast(
        annualized_volatility=ann_vol,
        ci_low_annualized=ci_low,
        ci_high_annualized=ci_high,
        spot_volatility=spot_vol,
        unconditional_volatility=uncond_vol,
        alpha=alpha,
        beta=beta,
        horizon_days=horizon_days,
        provenanced=node,
    )
