"""ACID TEST — the honest-evaluation harness, end to end.

This is the gate the whole of Phase 0 exists to pass. Reference:
arXiv:2608.27734 showed that an oracle with deliberate leakage reached Sharpe
34.7 and *passed* Deflated Sharpe (DSR=1.00) and PBO cleanly. The reason is
structural: statistical deflation corrects **multiple selection**, not
**contamination of the information set**. So the anti-leakage defence must be
*structural* (by construction), never statistical (after the fact).

Three criteria, all mandatory:

1. The harness **certifies** a real passive benchmark (SPY buy-and-hold). A
   harness that rejects a genuine passive is broken on the conservative side
   and equally useless.
2. The harness makes the leakage oracle **inexpressible** through the
   sanctioned data path (Part A, :class:`LeakageError`), while documenting that
   the statistical gates are **blind** to the same leakage when the data path
   is bypassed (Part B — the paper's finding, an expected result).
3. The harness **rejects** the best of ``N`` noise strategies selected on
   edgeless synthetic data (a demonstrable false discovery), and names the
   failing gate.

SPY window used for criterion 1: **2010-01-01 .. 2024-12-31**, 3773 daily
return observations, annualized Sharpe **0.8374** (per-period 0.052749), which
matches the paper's SPY evaluation reference (SR 0.85, IC [+0.24, +1.53]).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from alphalab.adapters.ledger_sqlite import SqliteTrialLedger
from alphalab.adapters.prices import CsvPriceSeriesSource
from alphalab.domain.ledger import TrialMetrics, TrialRecord
from alphalab.domain.pit import (
    FeatureRegistry,
    FeatureSpec,
    KnowledgeDate,
    LeakageError,
    PITRecord,
    resolve_feature,
)
from alphalab.domain.stats._moments import sharpe_ratio
from alphalab.domain.stats.certification import certify_strategy
from alphalab.domain.stats.synthetic import gbm_return_matrix
from alphalab.domain.types import Ticker

pytestmark = pytest.mark.acceptance

# Annualization factor for daily returns (trading days per year). The
# certification machinery works in the per-period Sharpe convention, so we
# never feed an annualized Sharpe into it; the factor is used only to relate
# the per-period result back to the documented annualized reference.
_TRADING_DAYS = 252

# SPY evaluation window (criterion 1).
_SPY_START = date(2010, 1, 1)
_SPY_END = date(2024, 12, 31)
_SPY_SHARPE_ANN_REF = 0.8374


def _spy_window_returns() -> np.ndarray:
    """Return daily simple returns of SPY over the documented eval window."""
    source = CsvPriceSeriesSource(root=Path(__file__).parent.parent / "fixtures")
    points = [p for p in source.series(Ticker("SPY")) if _SPY_START <= p.on <= _SPY_END]
    closes = np.array([p.close for p in points], dtype=np.float64)
    return closes[1:] / closes[:-1] - 1.0


# ---------------------------------------------------------------------------
# CRITERION 1 — the harness certifies a real passive benchmark.
# ---------------------------------------------------------------------------


def test_criterion_1_certifies_spy_buy_and_hold() -> None:
    """SPY buy-and-hold over 2010-2024 must certify cleanly.

    Design decisions, documented as the plan requires:

    * ``n_trials``: buy-and-hold is an **a-priori hypothesis**, not the
      survivor of a search. The honest no-selection statement is "SR0 = 0".
      The DSR implementation floors ``n_trials`` at 2, so we pass
      ``n_trials=2`` with ``trial_sharpe_std=0.0``; because SR0 is
      ``std(SR) * E[max z]``, a zero std makes SR0 exactly 0 — mathematically
      identical to a true a-priori (N=1) test, with no selection deflation.
    * ``configuration_matrix``: a single column (the benchmark itself). With
      ``N=1`` the CSCV PBO is 0 by construction (no selection => no
      selection-overfitting); this is the documented N=1 branch in ``pbo.py``.
    * Sharpe convention: ``certify_strategy`` uses the **per-period** Sharpe
      internally, so we feed daily returns directly. The per-period Sharpe is
      0.052749; multiplying by ``sqrt(252)`` gives the documented annualized
      0.8374, which matches the paper's SPY reference (SR 0.85).
    """
    # Arrange
    returns = _spy_window_returns()
    assert returns.size == 3773  # pin the documented window length
    config_matrix = returns.reshape(-1, 1)

    # Act
    result = certify_strategy(
        returns,
        config_matrix,
        n_trials=2,
        trial_sharpe_std=0.0,
        rng=np.random.default_rng(42),
        n_resamples=10_000,
        n_strips=16,
    )

    # Assert — the passive benchmark certifies.
    assert result.certified is True
    assert result.rejection_reasons == ()
    assert result.pbo == 0.0  # N=1 config matrix => no selection overfitting
    assert result.bootstrap_ci.excludes_zero
    assert result.bootstrap_ci.lower > 0.0

    # Assert — the per-period Sharpe annualizes to the documented reference.
    sharpe_ann = result.observed_sharpe * np.sqrt(_TRADING_DAYS)
    assert sharpe_ann == pytest.approx(_SPY_SHARPE_ANN_REF, abs=1e-3)


# ---------------------------------------------------------------------------
# CRITERION 2 — the leakage oracle. Part A: inexpressible. Part B: blind gates.
# ---------------------------------------------------------------------------

# A quarterly fundamental with a realistic 45-day filing lag. The oracle wants
# to use a datum "from the future"; the registry fixes the earliest date any
# datum for a given period could be known, which is what makes the look-ahead
# structurally illegal.
_ORACLE_FEATURE = FeatureSpec(name="next_day_return", availability_lag_days=1)
_AS_OF = date(2015, 6, 30)


def _oracle_leakage_record() -> PITRecord:
    """A datum that is only knowable *after* the decision date (look-ahead)."""
    # The signal for the decision made on ``_AS_OF`` is tomorrow's return,
    # which cannot be observed until the next day — by construction later than
    # the knowledge date a strategy would hold on ``_AS_OF``.
    return PITRecord(
        ticker=Ticker("SPY"),
        field="next_day_return",
        value=0.0123,
        as_of=_AS_OF,
        observed_at=_AS_OF + timedelta(days=1),
        source="ORACLE",
    )


def test_criterion_2a_leakage_oracle_is_structurally_inexpressible() -> None:
    """Part A — the oracle cannot be built through the sanctioned data path.

    Expressing "use tomorrow's return as today's signal" means asking the PIT
    resolver for a datum whose ``observed_at`` is after the knowledge date the
    strategy holds. The structural gate :func:`resolve_feature` refuses, so the
    look-ahead is *inexpressible*, not merely detectable after the fact.
    """
    # Arrange — a strategy deciding on _AS_OF knows only what is knowable then.
    registry = FeatureRegistry([_ORACLE_FEATURE])
    knowledge_on_decision_day = KnowledgeDate(_AS_OF)
    future_datum = _oracle_leakage_record()

    # Act / Assert — the only sanctioned read raises; the oracle cannot form.
    with pytest.raises(LeakageError, match="look-ahead"):
        resolve_feature(
            future_datum,
            registry=registry,
            knowledge_date=knowledge_on_decision_day,
        )


def test_statistical_gates_are_blind_to_leakage() -> None:
    """Part B — EXPECTED RESULT: DSR/PBO certify a known-fraudulent oracle.

    This is **not a bug**. It replicates the central finding of
    arXiv:2608.27734: if you bypass the point-in-time data path and hand the
    oracle's returns straight to :func:`certify_strategy`, every statistical
    gate passes (DSR approx 1.0, PBO 0, astronomical Sharpe). Statistical
    deflation corrects selection, not information-set contamination, so it is
    structurally blind to leakage. The ONLY real defence is the structural
    gate exercised in :func:`test_criterion_2a_leakage_oracle_is_structurally_inexpressible`.

    Observed values with this seed: per-period Sharpe approx 1.27
    (annualized approx 20.2), DSR = 1.000000, PBO = 0.0 — certified True.
    """
    # Arrange — a sign-perfect oracle: it knows the next return's sign, so its
    # realized return is always the positive magnitude of the true return.
    rng = np.random.default_rng(7)
    true_returns = rng.normal(0.0005, 0.01, size=1000)
    oracle_returns = np.sign(true_returns) * true_returns  # leakage: future sign
    config_matrix = oracle_returns.reshape(-1, 1)

    # Act
    result = certify_strategy(
        oracle_returns,
        config_matrix,
        n_trials=2,
        trial_sharpe_std=0.0,
        rng=np.random.default_rng(1),
        n_resamples=10_000,
        n_strips=16,
    )

    # Assert — the fraudulent strategy is certified. This documents the
    # statistical blindness; it must stay True so a regression that "fixes" it
    # statistically (which is impossible) is caught as a false sense of safety.
    assert result.certified is True
    assert result.dsr == pytest.approx(1.0, abs=1e-6)
    assert result.pbo == 0.0
    assert result.observed_sharpe * np.sqrt(_TRADING_DAYS) > 10.0


# ---------------------------------------------------------------------------
# CRITERION 3 — reject the best of N noise strategies (false discovery).
# ---------------------------------------------------------------------------


def _record_noise_trials(ledger: SqliteTrialLedger, matrix: np.ndarray) -> None:
    """Append one ledger trial per noise configuration (full ledger circuit)."""
    recorded_at = datetime(2024, 1, 1, 12, 0, 0)
    for j in range(matrix.shape[1]):
        column = matrix[:, j]
        sr = sharpe_ratio(column)
        ledger.append(
            TrialRecord.create(
                spec={"family": "noise", "config": j},
                dataset_hash="gbm-edgeless-v1",
                family="noise",
                window_start=date(2022, 1, 1),
                window_end=date(2023, 12, 31),
                params={"config": j},
                metrics=TrialMetrics(
                    sharpe_gross=sr,
                    sharpe_net=sr,
                    return_gross=float(column.sum()),
                    return_net=float(column.sum()),
                    max_drawdown=0.0,
                ),
                recorded_at=recorded_at,
            )
        )


def test_criterion_3_rejects_best_of_n_noise(tmp_path: Path) -> None:
    """Best-of-100 on edgeless GBM must be rejected; we control the truth.

    We generate ``N=100`` driftless GBM strategies (no edge by construction),
    keep the best in-sample one — the exact procedure that manufactures false
    discoveries — and feed the real trial count from a :class:`SqliteTrialLedger`
    into the DSR. Because the data has no edge, any certification would be a
    demonstrable false positive, so the harness must reject it and name the
    gate that caught it.

    Observed values with this seed: best config has annualized Sharpe approx
    1.85 in-sample, yet DSR approx 0.55 (< 0.95) rejects it once the 100-trial
    selection is deflated. PBO approx 0.42 and the naive bootstrap CI both
    "pass" — only the DSR, fed the honest trial count, catches the overfit.
    """
    # Arrange — 100 edgeless configs; record each as a real ledger trial.
    rng = np.random.default_rng(123)
    matrix = gbm_return_matrix(rng=rng, n_periods=504, n_configs=100, mu=0.0, sigma=0.01)
    with SqliteTrialLedger(tmp_path / "trials.sqlite3") as ledger:
        _record_noise_trials(ledger, matrix)
        n_trials = ledger.trial_count(family="noise")
        assert n_trials == 100  # ledger -> SR0 -> DSR circuit is live

    in_sample_sharpes = np.array([sharpe_ratio(matrix[:, j]) for j in range(100)])
    best = int(np.argmax(in_sample_sharpes))
    trial_sharpe_std = float(np.std(in_sample_sharpes, ddof=1))

    # Act — certify the cherry-picked winner with the honest trial count.
    result = certify_strategy(
        matrix[:, best],
        matrix,
        n_trials=n_trials,
        trial_sharpe_std=trial_sharpe_std,
        rng=np.random.default_rng(9),
        n_resamples=10_000,
        n_strips=16,
    )

    # Assert — a known false discovery must be rejected, with a named gate.
    assert result.certified is False
    assert result.rejection_reasons
    assert any("DSR" in reason for reason in result.rejection_reasons)
    assert result.dsr < 0.95
