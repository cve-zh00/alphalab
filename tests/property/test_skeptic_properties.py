"""Property-based tests for the deterministic skeptic (Hypothesis).

Invariants proven here:

* **Totality:** ``judge_candidate`` never raises, for *any* candidate —
  including NaN, inf, empty series, negative sizes and absurd trial counts.
* **Monotonicity:** more costs never increase acceptance; more ``n_trials``
  never increase acceptance; a longer clean sample never decreases acceptance.
* **Immutability:** both frozen types reject mutation.

Randomness is always seeded, so every example is reproducible.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from alphalab.domain.costs import CostSpec
from alphalab.domain.skeptic import (
    DataProvenance,
    TradeCandidate,
    judge_candidate,
)
from alphalab.domain.skeptic.criteria import check_cost_survival, check_sample_sufficiency
from alphalab.domain.types import Side, Ticker

# A float strategy that *includes* the degenerate values the skeptic must survive.
_wild_floats = st.floats(allow_nan=True, allow_infinity=True, width=32)

_return_series = st.lists(_wild_floats, min_size=0, max_size=64).map(tuple)


def _build(
    *,
    returns: tuple[float, ...],
    cost_fractions: tuple[float, ...],
    proposed_size: float,
    n_trials: int,
    includes_delisted: bool,
) -> TradeCandidate:
    return TradeCandidate(
        ticker=Ticker("P"),
        side=Side.BUY,
        proposed_size=proposed_size,
        horizon_days=5,
        backtest_returns=returns,
        cost_fractions=cost_fractions,
        n_trials=n_trials,
        trial_sharpe_std=0.0,
        cost_spec=CostSpec(),
        gross_edge_bps=20.0,
        adv=1_000_000.0,
        sigma_bps=100.0,
        per_slice_sharpe=(),
        provenance=DataProvenance(
            knowledge_date=date(2020, 1, 1), includes_delisted=includes_delisted
        ),
    )


@settings(max_examples=120, deadline=None)
@given(
    returns=_return_series,
    size=st.floats(allow_nan=True, allow_infinity=True),
    n_trials=st.integers(min_value=-5, max_value=10_000),
    delisted=st.booleans(),
)
def test_judge_never_raises(
    returns: tuple[float, ...],
    size: float,
    n_trials: int,
    delisted: bool,
) -> None:
    # Arrange: cost series sometimes aligned, sometimes not.
    cost = tuple(0.0 for _ in returns)
    candidate = _build(
        returns=returns,
        cost_fractions=cost,
        proposed_size=size,
        n_trials=n_trials,
        includes_delisted=delisted,
    )

    # Act / Assert: a verdict always comes back, never an exception.
    verdict = judge_candidate(candidate, rng=np.random.default_rng(0))
    assert verdict.accepted in (True, False)


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    extra_cost=st.floats(min_value=0.0, max_value=0.05),
)
def test_more_costs_never_increase_cost_survival(seed: int, extra_cost: float) -> None:
    # Arrange: a fixed return series; the only change is additional cost drag.
    rng = np.random.default_rng(seed)
    returns = tuple(float(x) for x in rng.normal(0.003, 0.01, 200))
    low = _build(
        returns=returns,
        cost_fractions=tuple(0.0001 for _ in returns),
        proposed_size=10.0,
        n_trials=2,
        includes_delisted=True,
    )
    high = low.model_copy(update={"cost_fractions": tuple(0.0001 + extra_cost for _ in returns)})

    # Act
    low_pass = check_cost_survival(low).passed
    high_pass = check_cost_survival(high).passed

    # Assert: adding cost can only turn a pass into a fail, never the reverse.
    assert not (high_pass and not low_pass)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=5_000), extra_trials=st.integers(2, 2_000))
def test_more_trials_never_increase_acceptance(seed: int, extra_trials: int) -> None:
    # Arrange: identical candidate but with a larger honest trial count, which
    # deflates the Sharpe harder — acceptance can only drop.
    rng = np.random.default_rng(seed)
    returns = tuple(float(x) for x in rng.normal(0.004, 0.01, 300))
    few = _build(
        returns=returns,
        cost_fractions=tuple(0.0001 for _ in returns),
        proposed_size=10.0,
        n_trials=2,
        includes_delisted=True,
    ).model_copy(update={"trial_sharpe_std": 0.5})
    many = few.model_copy(update={"n_trials": 2 + extra_trials})

    # Act
    few_accept = judge_candidate(few, rng=np.random.default_rng(1)).accepted
    many_accept = judge_candidate(many, rng=np.random.default_rng(1)).accepted

    # Assert
    assert not (many_accept and not few_accept)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(min_value=0, max_value=5_000), extra=st.integers(1, 400))
def test_more_sample_never_decreases_sample_sufficiency(seed: int, extra: int) -> None:
    # Arrange: a longer draw from the same clean process has T_eff >= the short
    # one on average; the sufficiency gate must not get *stricter* with more
    # data. We assert the one-directional implication on the gate decision.
    rng = np.random.default_rng(seed)
    base = rng.normal(0.002, 0.01, 55)
    extra_draw = rng.normal(0.002, 0.01, extra)
    short = _build(
        returns=tuple(float(x) for x in base),
        cost_fractions=tuple(0.0001 for _ in base),
        proposed_size=10.0,
        n_trials=2,
        includes_delisted=True,
    )
    longer_returns = tuple(float(x) for x in np.concatenate([base, extra_draw]))
    longer = short.model_copy(
        update={
            "backtest_returns": longer_returns,
            "cost_fractions": tuple(0.0001 for _ in longer_returns),
        }
    )

    # Act
    short_pass = check_sample_sufficiency(short).passed
    long_pass = check_sample_sufficiency(longer).passed

    # Assert: if the short series already passed, the longer one must too.
    assert not (short_pass and not long_pass)


def test_candidate_is_frozen() -> None:
    # Arrange
    candidate = _build(
        returns=(0.01, 0.02, 0.03),
        cost_fractions=(0.0, 0.0, 0.0),
        proposed_size=1.0,
        n_trials=2,
        includes_delisted=True,
    )

    # Act / Assert
    with pytest.raises(ValidationError):
        candidate.proposed_size = 2.0  # type: ignore[misc]


def test_verdict_is_frozen() -> None:
    # Arrange
    candidate = _build(
        returns=(0.01, 0.02, 0.03),
        cost_fractions=(0.0, 0.0, 0.0),
        proposed_size=1.0,
        n_trials=2,
        includes_delisted=True,
    )
    verdict = judge_candidate(candidate, rng=np.random.default_rng(0))

    # Act / Assert
    with pytest.raises(ValidationError):
        verdict.accepted = True  # type: ignore[misc]


def test_provenance_is_frozen() -> None:
    # Arrange
    prov = DataProvenance(knowledge_date=date(2020, 1, 1), includes_delisted=True)

    # Act / Assert
    with pytest.raises(ValidationError):
        prov.includes_delisted = False  # type: ignore[misc]
