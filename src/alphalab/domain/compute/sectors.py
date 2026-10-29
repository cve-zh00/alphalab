"""GICS sector multiple bands and CAPM cost-of-equity calibration.

Replaces the single market-wide P/E band ``(15, 18, 20)`` and fixed ``9%``
cost of equity with sector-aware peer multiples (P/E, EV/EBITDA, P/B) and a
Blume-adjusted CAPM cost of equity derived from trailing price history (or
sector beta when price history is short).

Pure: numpy only, no I/O, no global mutable state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

__all__ = [
    "SECTOR_PROFILES",
    "SPY_SECTOR_WEIGHTS",
    "TICKER_TO_SECTOR",
    "GicsSector",
    "SectorProfile",
    "compute_capm_cost_of_equity",
    "compute_rolling_beta",
    "estimate_beta_from_prices",
    "estimate_ke_beta_from_prices",
    "estimate_welch_beta",
    "resolve_sector_profile",
    "shrink_upside_empirical_bayes",
]


class GicsSector(StrEnum):
    """Canonical GICS sector classification."""

    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    HEALTH_CARE = "health_care"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    FINANCIALS = "financials"
    INDUSTRIALS = "industrials"
    ENERGY = "energy"
    MATERIALS = "materials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    MARKET_DEFAULT = "market_default"


@dataclass(frozen=True)
class SectorProfile:
    """Calibrated peer multiples and baseline beta for one GICS sector."""

    sector: GicsSector
    peer_pe: tuple[float, ...]
    peer_ev_ebitda: tuple[float, ...]
    peer_pb: tuple[float, ...]
    beta: float
    peer_ffo: tuple[float, ...] = ()


SECTOR_PROFILES: dict[GicsSector, SectorProfile] = {
    GicsSector.INFORMATION_TECHNOLOGY: SectorProfile(
        sector=GicsSector.INFORMATION_TECHNOLOGY,
        peer_pe=(22.0, 25.0, 28.0),
        peer_ev_ebitda=(16.0, 18.5, 21.0),
        peer_pb=(4.5, 6.0, 7.5),
        beta=1.18,
    ),
    GicsSector.COMMUNICATION_SERVICES: SectorProfile(
        sector=GicsSector.COMMUNICATION_SERVICES,
        peer_pe=(16.0, 19.0, 22.0),
        peer_ev_ebitda=(10.5, 12.5, 14.5),
        peer_pb=(2.8, 3.6, 4.5),
        beta=1.05,
    ),
    GicsSector.HEALTH_CARE: SectorProfile(
        sector=GicsSector.HEALTH_CARE,
        peer_pe=(17.0, 20.0, 23.0),
        peer_ev_ebitda=(12.5, 14.5, 16.5),
        peer_pb=(3.2, 4.2, 5.2),
        beta=0.82,
    ),
    GicsSector.CONSUMER_DISCRETIONARY: SectorProfile(
        sector=GicsSector.CONSUMER_DISCRETIONARY,
        peer_pe=(17.0, 20.0, 23.0),
        peer_ev_ebitda=(11.5, 13.5, 15.5),
        peer_pb=(3.2, 4.5, 6.0),
        beta=1.15,
    ),
    GicsSector.CONSUMER_STAPLES: SectorProfile(
        sector=GicsSector.CONSUMER_STAPLES,
        peer_pe=(17.0, 19.0, 21.0),
        peer_ev_ebitda=(12.0, 13.5, 15.0),
        peer_pb=(3.0, 4.0, 5.0),
        beta=0.68,
    ),
    GicsSector.FINANCIALS: SectorProfile(
        sector=GicsSector.FINANCIALS,
        peer_pe=(11.0, 13.0, 15.0),
        peer_ev_ebitda=(),  # EV/EBITDA is not meaningful for banks/insurers
        peer_pb=(1.2, 1.5, 1.9),
        beta=1.08,
    ),
    GicsSector.INDUSTRIALS: SectorProfile(
        sector=GicsSector.INDUSTRIALS,
        peer_pe=(17.0, 19.5, 22.0),
        peer_ev_ebitda=(11.5, 13.5, 15.5),
        peer_pb=(2.8, 3.6, 4.5),
        beta=1.05,
    ),
    GicsSector.ENERGY: SectorProfile(
        sector=GicsSector.ENERGY,
        peer_pe=(10.0, 12.0, 14.0),
        peer_ev_ebitda=(5.5, 6.5, 7.5),
        peer_pb=(1.5, 2.0, 2.5),
        beta=1.10,
    ),
    GicsSector.MATERIALS: SectorProfile(
        sector=GicsSector.MATERIALS,
        peer_pe=(14.0, 16.5, 19.0),
        peer_ev_ebitda=(8.5, 10.0, 11.5),
        peer_pb=(1.8, 2.4, 3.0),
        beta=1.05,
    ),
    GicsSector.UTILITIES: SectorProfile(
        sector=GicsSector.UTILITIES,
        peer_pe=(15.0, 17.0, 19.0),
        peer_ev_ebitda=(10.0, 11.5, 13.0),
        peer_pb=(1.5, 1.9, 2.3),
        beta=0.62,
    ),
    GicsSector.REAL_ESTATE: SectorProfile(
        sector=GicsSector.REAL_ESTATE,
        peer_pe=(24.0, 28.0, 32.0),
        peer_ev_ebitda=(14.0, 16.0, 18.0),
        peer_pb=(1.8, 2.3, 2.8),
        beta=0.88,
        peer_ffo=(14.0, 16.5, 19.0),
    ),
    GicsSector.MARKET_DEFAULT: SectorProfile(
        sector=GicsSector.MARKET_DEFAULT,
        peer_pe=(15.0, 18.0, 20.0),
        peer_ev_ebitda=(11.0, 13.0, 15.0),
        peer_pb=(2.5, 3.2, 4.0),
        beta=1.00,
    ),
}

#: Benchmark (SPY / S&P 500) sector weights used for active-exposure measurement
#: and benchmark-relative sector bands.
SPY_SECTOR_WEIGHTS: dict[GicsSector, float] = {
    GicsSector.INFORMATION_TECHNOLOGY: 0.310,
    GicsSector.FINANCIALS: 0.135,
    GicsSector.HEALTH_CARE: 0.115,
    GicsSector.CONSUMER_DISCRETIONARY: 0.105,
    GicsSector.COMMUNICATION_SERVICES: 0.090,
    GicsSector.INDUSTRIALS: 0.085,
    GicsSector.CONSUMER_STAPLES: 0.058,
    GicsSector.ENERGY: 0.035,
    GicsSector.UTILITIES: 0.025,
    GicsSector.REAL_ESTATE: 0.023,
    GicsSector.MATERIALS: 0.019,
    GicsSector.MARKET_DEFAULT: 0.050,
}

_SECTOR_TICKERS: dict[GicsSector, tuple[str, ...]] = {
    GicsSector.INFORMATION_TECHNOLOGY: (
        "AAPL",
        "ACN",
        "ADBE",
        "ADI",
        "ADSK",
        "AKAM",
        "AMAT",
        "AMD",
        "ANET",
        "APH",
        "APP",
        "AVGO",
        "CDNS",
        "CDW",
        "CIEN",
        "CRM",
        "CRWD",
        "CSCO",
        "CTSH",
        "DDOG",
        "DELL",
        "EPAM",
        "FFIV",
        "FICO",
        "FSLR",
        "FTNT",
        "GDDY",
        "GEN",
        "GLW",
        "GRMN",
        "HPE",
        "HPQ",
        "IBM",
        "INTC",
        "INTU",
        "IT",
        "JBL",
        "KEYS",
        "KLAC",
        "LRCX",
        "MCHP",
        "MPWR",
        "MRVL",
        "MSFT",
        "MSI",
        "MU",
        "NOW",
        "NTAP",
        "NVDA",
        "NXPI",
        "ON",
        "ORCL",
        "PANW",
        "PLTR",
        "PTC",
        "QCOM",
        "ROP",
        "SMCI",
        "SNPS",
        "STX",
        "SWKS",
        "TDY",
        "TEL",
        "TER",
        "TRMB",
        "TXN",
        "TYL",
        "VRSN",
        "WDC",
        "WDAY",
        "ZBRA",
    ),
    GicsSector.COMMUNICATION_SERVICES: (
        "CHTR",
        "CMCSA",
        "DIS",
        "EA",
        "FOX",
        "FOXA",
        "GOOG",
        "GOOGL",
        "IPG",
        "LYV",
        "META",
        "MTCH",
        "NFLX",
        "NWS",
        "NWSA",
        "OMC",
        "PARA",
        "RDDT",
        "T",
        "TKO",
        "TMUS",
        "TTWO",
        "VZ",
        "WBD",
    ),
    GicsSector.HEALTH_CARE: (
        "A",
        "ABBV",
        "ABT",
        "ALGN",
        "AMGN",
        "BAX",
        "BDX",
        "BIIB",
        "BMY",
        "BSX",
        "CAH",
        "CI",
        "CNC",
        "COO",
        "COR",
        "CRL",
        "CVS",
        "DGX",
        "DHR",
        "DVA",
        "DXCM",
        "ELV",
        "EW",
        "GEHC",
        "GILD",
        "HCA",
        "HOLX",
        "HSIC",
        "HUM",
        "IDXX",
        "ILMN",
        "INCY",
        "IQV",
        "ISRG",
        "JNJ",
        "LH",
        "LLY",
        "MCK",
        "MDT",
        "MOH",
        "MRK",
        "MRNA",
        "MTD",
        "PFE",
        "PODD",
        "REGN",
        "RMD",
        "RVTY",
        "SOLV",
        "STE",
        "SYK",
        "TECH",
        "TMO",
        "UHS",
        "UNH",
        "VEEV",
        "VRTX",
        "VTRS",
        "WAT",
        "WST",
        "ZBH",
        "ZTS",
    ),
    GicsSector.FINANCIALS: (
        "ACGL",
        "AFL",
        "AIG",
        "AIZ",
        "AJG",
        "ALL",
        "AMP",
        "AON",
        "APO",
        "ARES",
        "AXP",
        "BAC",
        "BEN",
        "BK",
        "BLK",
        "BNY",
        "BRK.B",
        "BRO",
        "BX",
        "C",
        "CB",
        "CBOE",
        "CFG",
        "CINF",
        "CME",
        "COF",
        "COIN",
        "CPAY",
        "EG",
        "ERIE",
        "FDS",
        "FI",
        "FIS",
        "FISV",
        "FITB",
        "GPN",
        "GL",
        "GS",
        "HBAN",
        "HIG",
        "HOOD",
        "IBKR",
        "ICE",
        "IVZ",
        "JKHY",
        "JPM",
        "KEY",
        "KKR",
        "L",
        "MA",
        "MCO",
        "MET",
        "MMC",
        "MRSH",
        "MS",
        "MSCI",
        "MTB",
        "NDAQ",
        "NTRS",
        "PFG",
        "PGR",
        "PNC",
        "PRU",
        "PYPL",
        "RF",
        "RJF",
        "SCHW",
        "SPGI",
        "STT",
        "SYF",
        "TFC",
        "TROW",
        "TRV",
        "USB",
        "V",
        "WFC",
        "WRB",
        "WTW",
        "XYZ",
    ),
    GicsSector.CONSUMER_DISCRETIONARY: (
        "ABNB",
        "AMZN",
        "APTV",
        "AZO",
        "BBY",
        "BKNG",
        "CCL",
        "CMG",
        "CVNA",
        "DASH",
        "DECK",
        "DHI",
        "DLTR",
        "DPZ",
        "DRI",
        "EBAY",
        "ETSY",
        "EXPE",
        "F",
        "GM",
        "GPC",
        "GRBK",
        "HAS",
        "HD",
        "HLT",
        "LEN",
        "LKQ",
        "LOW",
        "LULU",
        "LVS",
        "MAR",
        "MCD",
        "MGM",
        "MHK",
        "NCLH",
        "NKE",
        "NVR",
        "ORLY",
        "PHM",
        "POOL",
        "RCL",
        "RL",
        "ROST",
        "SBUX",
        "TJX",
        "TPR",
        "TSCO",
        "TSLA",
        "ULTA",
        "WSM",
        "WYNN",
        "YUM",
    ),
    GicsSector.CONSUMER_STAPLES: (
        "ADM",
        "BF.B",
        "BG",
        "CAG",
        "CASY",
        "CHD",
        "CL",
        "CLX",
        "COST",
        "CPB",
        "DG",
        "EL",
        "GIS",
        "HRL",
        "HSY",
        "K",
        "KDP",
        "KHC",
        "KMB",
        "KO",
        "KR",
        "KVUE",
        "LW",
        "MDLZ",
        "MKC",
        "MNST",
        "MO",
        "PEP",
        "PG",
        "PM",
        "SJM",
        "STZ",
        "SYY",
        "TAP",
        "TGT",
        "TSN",
        "UL",
        "WBA",
        "WMT",
    ),
    GicsSector.INDUSTRIALS: (
        "ALLE",
        "AME",
        "AOS",
        "AXON",
        "BA",
        "BR",
        "BLDR",
        "CARR",
        "CAT",
        "CHRW",
        "CMI",
        "CPRT",
        "CSX",
        "CTAS",
        "DAL",
        "DE",
        "DOV",
        "EFX",
        "EME",
        "EMR",
        "ETN",
        "EXPD",
        "FAST",
        "FDX",
        "FERG",
        "FIX",
        "FTV",
        "GD",
        "GE",
        "GEV",
        "GNRC",
        "GWW",
        "HII",
        "HON",
        "HUBB",
        "HWM",
        "IEX",
        "IR",
        "ITW",
        "J",
        "JBHT",
        "JCI",
        "LDOS",
        "LHX",
        "LII",
        "LMT",
        "LUV",
        "MAS",
        "MMM",
        "NDSN",
        "NOC",
        "NSC",
        "ODFL",
        "OTIS",
        "PAYC",
        "PAYX",
        "PCAR",
        "PH",
        "PNR",
        "PWR",
        "ROK",
        "ROL",
        "RSG",
        "RTX",
        "SNA",
        "SWK",
        "TDG",
        "TT",
        "TXT",
        "UAL",
        "UBER",
        "UNP",
        "UPS",
        "URI",
        "VLTO",
        "VRSK",
        "VRT",
        "WAB",
        "WM",
        "XYL",
    ),
    GicsSector.ENERGY: (
        "APA",
        "BKR",
        "COP",
        "CTRA",
        "CVX",
        "DVN",
        "EOG",
        "EQT",
        "EXE",
        "FANG",
        "HAL",
        "HES",
        "KMI",
        "MPC",
        "OKE",
        "OXY",
        "PSX",
        "SLB",
        "TPL",
        "TRGP",
        "VLO",
        "WMB",
        "XOM",
    ),
    GicsSector.MATERIALS: (
        "ALB",
        "AMCR",
        "APD",
        "AVY",
        "BALL",
        "CE",
        "CF",
        "CRH",
        "CTVA",
        "DD",
        "DOW",
        "ECL",
        "FCX",
        "FMC",
        "IFF",
        "IP",
        "LIN",
        "LYB",
        "MLM",
        "MOS",
        "NEM",
        "NUE",
        "PKG",
        "PPG",
        "SHW",
        "SMG",
        "STLD",
        "SW",
        "VMC",
    ),
    GicsSector.UTILITIES: (
        "AEE",
        "AEP",
        "AES",
        "ATO",
        "AWK",
        "CEG",
        "CMS",
        "CNP",
        "D",
        "DTE",
        "DUK",
        "ED",
        "EIX",
        "ES",
        "ETR",
        "EVRG",
        "EXC",
        "FE",
        "LNT",
        "NEE",
        "NI",
        "NRG",
        "PCG",
        "PEG",
        "PNW",
        "PPL",
        "SO",
        "SRE",
        "VST",
        "WEC",
        "XEL",
    ),
    GicsSector.REAL_ESTATE: (
        "AMT",
        "ARE",
        "AVB",
        "BXP",
        "CBRE",
        "CCI",
        "CPT",
        "CSGP",
        "DLR",
        "DOC",
        "EQIX",
        "EQR",
        "ESS",
        "EXR",
        "FRT",
        "HST",
        "INVH",
        "IRM",
        "KIM",
        "MAA",
        "O",
        "PLD",
        "PSA",
        "REG",
        "SBAC",
        "SPG",
        "UDR",
        "VICI",
        "VTR",
        "WELL",
        "WY",
    ),
}

TICKER_TO_SECTOR: dict[str, GicsSector] = {
    ticker: sector for sector, tickers in _SECTOR_TICKERS.items() for ticker in tickers
}


def resolve_sector_profile(ticker: str) -> SectorProfile:
    """Return the calibrated :class:`SectorProfile` for ``ticker``."""
    sector = TICKER_TO_SECTOR.get(ticker.strip().upper(), GicsSector.MARKET_DEFAULT)
    return SECTOR_PROFILES[sector]


def estimate_beta_from_prices(
    asset_closes: Sequence[float],
    benchmark_closes: Sequence[float] | None = None,
    *,
    default_beta: float = 1.0,
    min_observations: int = 30,
    lookback: int = 252,
) -> float:
    """Estimate Vasicek-shrunk equity beta from trailing daily closes.

    When ``benchmark_closes`` is supplied with at least ``min_observations``
    overlapping returns, computes OLS beta ``Cov(r_i, r_m) / Var(r_m)`` and
    applies Vasicek (1973) precision-weighted Bayesian shrinkage toward the
    GICS sector prior ``default_beta``. When no benchmark is available, blends
    ``default_beta`` with the asset's annualized volatility relative to a 16%
    market baseline. Clamped to ``[0.45, 2.10]``.
    """
    arr = np.asarray(asset_closes[-lookback - 1 :], dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size < min_observations + 1:
        return float(np.clip(default_beta, 0.45, 2.10))

    asset_ret = np.diff(arr) / arr[:-1]
    if benchmark_closes is not None:
        bench = np.asarray(benchmark_closes[-lookback - 1 :], dtype=float)
        bench = bench[np.isfinite(bench) & (bench > 0.0)]
        n = min(asset_ret.size, bench.size - 1)
        if n >= min_observations:
            bench_ret = np.diff(bench[-(n + 1) :]) / bench[-(n + 1) : -1]
            a_ret = asset_ret[-n:]
            var_m = float(np.var(bench_ret, ddof=1))
            if var_m > 1e-10:
                cov_im = float(np.cov(a_ret, bench_ret, ddof=1)[0, 1])
                raw_beta = cov_im / var_m
                var_i = float(np.var(a_ret, ddof=1))
                resid_var = max(0.0, var_i - (raw_beta**2) * var_m)
                s_beta_sq = resid_var / (max(1, n - 2) * var_m)
                tau_sector_sq = 0.09
                w_ols = tau_sector_sq / (tau_sector_sq + s_beta_sq)
                prior_beta = float(np.clip(default_beta, 0.45, 2.10))
                shrunk = w_ols * raw_beta + (1.0 - w_ols) * prior_beta
                return float(np.clip(shrunk, 0.45, 2.10))

    ann_vol = float(np.std(asset_ret, ddof=1)) * math.sqrt(252.0)
    vol_beta = ann_vol / 0.16
    blended = 0.60 * default_beta + 0.40 * vol_beta
    return float(np.clip(blended, 0.45, 2.10))


compute_rolling_beta = estimate_beta_from_prices


def estimate_ke_beta_from_prices(
    asset_closes: Sequence[float],
    benchmark_closes: Sequence[float] | None = None,
    *,
    default_beta: float = 1.0,
    double_shrink_s: float = 0.35,
    min_observations: int = 30,
    lookback: int = 252,
) -> float:
    """Estimate double-shrunk cost-of-equity beta (Levi & Welch 2017; Blume 1975).

    Applies a second shrinkage pass toward ``1.0`` on top of the Vasicek (1973)
    estimate so that short-window low-beta regimes do not mechanically inflate
    perpetual DCF/DDM terminal values via the ``6.5%`` CAPM floor:
    ``beta_ke = 1.0 + s * (beta_vasicek - 1.0)`` with ``s in [0.0, 1.0]``.
    When ``double_shrink_s == 0.0``, returns ``1.0`` (common market discount rate).
    """
    s = float(np.clip(double_shrink_s, 0.0, 1.0))
    if s <= 0.0:
        return 1.0
    beta_v = estimate_beta_from_prices(
        asset_closes,
        benchmark_closes,
        default_beta=default_beta,
        min_observations=min_observations,
        lookback=lookback,
    )
    return float(np.clip(1.0 + s * (beta_v - 1.0), 0.45, 2.10))


def estimate_welch_beta(
    asset_closes: Sequence[float],
    benchmark_closes: Sequence[float] | None = None,
    *,
    default_beta: float = 1.0,
    halflife_days: float = 84.0,
    min_observations: int = 30,
    lookback: int = 252,
) -> float:
    """Estimate Welch (2022, CFR) slope-winsorized decay-weighted market beta.

    Winsorizes daily asset excess returns inside the slope interval
    ``[-2 * r_m, +4 * r_m]`` and estimates WLS with exponential decay weights
    (half-life ~4 months = 84 trading days). Deliberately **unclamped** from the
    ``[0.45, 2.10]`` valuation floor so risk reporting and portfolio construction
    see the true market sensitivity of defensive/bond-proxy stocks.
    """
    arr = np.asarray(asset_closes[-lookback - 1 :], dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size < min_observations + 1 or benchmark_closes is None:
        return float(default_beta)

    bench = np.asarray(benchmark_closes[-lookback - 1 :], dtype=float)
    bench = bench[np.isfinite(bench) & (bench > 0.0)]
    asset_ret = np.diff(arr) / arr[:-1]
    n = min(asset_ret.size, bench.size - 1)
    if n < min_observations:
        return float(default_beta)

    bench_ret = np.diff(bench[-(n + 1) :]) / bench[-(n + 1) : -1]
    a_ret = asset_ret[-n:]

    # Welch (2022) Eq. (3): winsorize r_i at [-2 * r_m, +4 * r_m]
    bound_a = -2.0 * bench_ret
    bound_b = 4.0 * bench_ret
    lo = np.minimum(bound_a, bound_b)
    hi = np.maximum(bound_a, bound_b)
    a_win = np.clip(a_ret, lo, hi)

    hl = max(1.0, float(halflife_days))
    ages = np.arange(n - 1, -1, -1, dtype=float)
    weights = np.exp(-math.log(2.0) * ages / hl)
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-12:
        return float(default_beta)
    weights = weights / w_sum

    mean_m = float(np.sum(weights * bench_ret))
    mean_i = float(np.sum(weights * a_win))
    dm = bench_ret - mean_m
    di = a_win - mean_i
    var_m = float(np.sum(weights * dm * dm))
    if var_m <= 1e-12:
        return float(default_beta)
    cov_im = float(np.sum(weights * di * dm))
    return float(np.clip(cov_im / var_m, -2.0, 4.0))


def shrink_upside_empirical_bayes(
    upside: float,
    *,
    relative_span: float | None = None,
    methods_count: int = 2,
    prior_mean: float = 0.10,
    prior_std: float = 0.25,
) -> float:
    """Apply Smith & Winkler (2006) Empirical Bayes shrinkage to raw upside.

    Dampens extreme valuation outliers (Optimizer's Curse) proportionally to
    cross-method dispersion and single-method uncertainty.
    """
    if upside <= 0.0:
        return upside
    span = max(float(relative_span) if relative_span is not None else 0.15, 0.15)
    obs_var = (0.5 * span) ** 2 + (0.25**2 if methods_count < 2 else 0.0)
    prior_var = max(1e-6, prior_std**2)
    shrinkage = prior_var / (prior_var + obs_var)
    return float(prior_mean + shrinkage * (upside - prior_mean))


def compute_capm_cost_of_equity(
    beta: float,
    *,
    risk_free_rate: float = 0.042,
    equity_risk_premium: float = 0.048,
) -> float:
    """Compute CAPM required return on equity bounded to ``[6.5%, 14.5%]``."""
    raw = risk_free_rate + beta * equity_risk_premium
    return round(float(np.clip(raw, 0.065, 0.145)), 4)
