"""Backtest engine: scheduling, sizing, cost accounting and metrics.

The engine never computes a signal; it receives one. That separation lets the
same machinery evaluate the real strategy, a synthetic positive control and a
buy-and-hold benchmark, so their results are directly comparable.
"""

from __future__ import annotations

from alphalab.domain.backtest.benchmark import buy_and_hold
from alphalab.domain.backtest.context import SignalContext, SignalFunction
from alphalab.domain.backtest.costs_model import ProportionalCostModel, TurnoverCostModel
from alphalab.domain.backtest.engine import BacktestError, run_backtest
from alphalab.domain.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    max_drawdown,
    metrics_from_returns,
)
from alphalab.domain.backtest.relative import RelativePerformance, relative_performance
from alphalab.domain.backtest.signals import (
    equal_weight_signal,
    liquidity_screened,
    momentum_signal,
    rank_ensemble_signal,
)
from alphalab.domain.backtest.turbulence import (
    DEFAULT_LOOKBACK,
    DEFAULT_MAX_ASSETS,
    DEFAULT_PERCENTILE,
    TurbulenceReading,
    turbulence_reading,
    turbulence_screened,
)
from alphalab.domain.backtest.types import (
    BacktestResult,
    PerformanceMetrics,
    PriceBar,
    RebalanceRecord,
)

__all__ = [
    "DEFAULT_LOOKBACK",
    "DEFAULT_MAX_ASSETS",
    "DEFAULT_PERCENTILE",
    "TRADING_DAYS_PER_YEAR",
    "BacktestError",
    "BacktestResult",
    "PerformanceMetrics",
    "PriceBar",
    "ProportionalCostModel",
    "RebalanceRecord",
    "RelativePerformance",
    "SignalContext",
    "SignalFunction",
    "TurbulenceReading",
    "TurnoverCostModel",
    "buy_and_hold",
    "equal_weight_signal",
    "liquidity_screened",
    "max_drawdown",
    "metrics_from_returns",
    "momentum_signal",
    "rank_ensemble_signal",
    "relative_performance",
    "run_backtest",
    "turbulence_reading",
    "turbulence_screened",
]
