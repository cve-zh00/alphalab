"""Synthetic return generators with known parameters (pure, seeded).

These generators exist to manufacture series with a *known* data-generating
process so the certification harness can be stress-tested. In particular, a
driftless Geometric Brownian Motion (GBM) produces returns with **no edge by
construction**: if the harness ever certifies such a series it is a
demonstrable false positive.

Randomness is always supplied through an explicit ``numpy.random.Generator``
so every series is reproducible.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["gbm_log_returns", "gbm_return_matrix"]


def gbm_log_returns(
    *,
    rng: np.random.Generator,
    n_periods: int,
    mu: float = 0.0,
    sigma: float = 0.01,
    dt: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Return ``n_periods`` i.i.d. GBM log-returns with known parameters.

    The per-period log-return is drawn from
    ``Normal((mu - sigma^2/2) * dt, sigma^2 * dt)``. With ``mu = 0`` the series
    has no expected drift — i.e. no edge by construction.

    Args:
        rng: Explicit seeded generator (reproducibility is mandatory).
        n_periods: Number of return observations to produce.
        mu: Annualized-style drift of the underlying process.
        sigma: Volatility (standard deviation per unit time).
        dt: Time step per period.

    Raises:
        ValueError: If ``n_periods < 1``, ``sigma < 0`` or ``dt <= 0``.
    """
    if n_periods < 1:
        raise ValueError("n_periods must be at least 1")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    draws = rng.normal(loc=drift, scale=vol, size=n_periods)
    return np.asarray(draws, dtype=np.float64)


def gbm_return_matrix(
    *,
    rng: np.random.Generator,
    n_periods: int,
    n_configs: int,
    mu: float = 0.0,
    sigma: float = 0.01,
    dt: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Return a ``T x N`` matrix of independent GBM return columns.

    Each of the ``n_configs`` columns is an independent driftless-by-default
    GBM path, suitable as the strategy-configuration matrix consumed by the
    PBO/CSCV routine when demonstrating overfitting on edgeless data.

    Raises:
        ValueError: If ``n_periods < 1`` or ``n_configs < 1`` or invalid
            ``sigma``/``dt``.
    """
    if n_configs < 1:
        raise ValueError("n_configs must be at least 1")
    columns = [
        gbm_log_returns(rng=rng, n_periods=n_periods, mu=mu, sigma=sigma, dt=dt)
        for _ in range(n_configs)
    ]
    return np.column_stack(columns).astype(np.float64)
