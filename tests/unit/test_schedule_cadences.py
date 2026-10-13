"""Unit tests for the cadence use cases (``alphalab.app.cadences``).

Every dependency is injected and doubled: no real LLM, no network, no clock.
The daily cadence must not spend the trial budget (no deep LLM); the weekly
cadence drives the analyst pipeline; the monthly cadence closes the loop from
the journal to a real calibration report.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from alphalab.app.cadences import (
    CadenceOutcome,
    run_daily,
    run_event,
    run_monthly,
    run_weekly,
)
from alphalab.domain.journal import (
    Direction,
    Prediction,
    Settlement,
)
from alphalab.domain.schedule import Cadence


# ---------------------------------------------------------------------------
# Daily cadence — deterministic refresh, no trial-budget spend
# ---------------------------------------------------------------------------
def test_daily_refresh_runs_without_deep_llm() -> None:
    # Arrange
    refreshed: list[str] = []

    def refresh(tickers: tuple[str, ...]) -> int:
        refreshed.extend(tickers)
        return len(tickers)

    now = datetime(2025, 7, 7, 22, 0)
    # Act
    outcome = run_daily(tickers=("MSFT", "AAPL"), refresh=refresh, now=now)
    # Assert
    assert outcome.cadence is Cadence.DAILY
    assert outcome.spent_trial_budget is False
    assert refreshed == ["MSFT", "AAPL"]
    assert "2" in outcome.produced


# ---------------------------------------------------------------------------
# Weekly cadence — the deep-dive via the analyst pipeline
# ---------------------------------------------------------------------------
def test_weekly_deep_dive_invokes_pipeline_for_each_name() -> None:
    # Arrange: a pipeline double records the tickers it was asked to analyse.
    analysed: list[str] = []

    class PipelineDouble:
        def analyse(self, ticker: str) -> str:
            analysed.append(ticker)
            return f"verdict:{ticker}"

    double = PipelineDouble()
    now = datetime(2025, 7, 13, 10, 0)
    # Act
    outcome = run_weekly(
        tickers=("MSFT", "NVDA", "AAPL"),
        analyse=double.analyse,
        now=now,
    )
    # Assert
    assert outcome.cadence is Cadence.WEEKLY
    assert outcome.spent_trial_budget is True
    assert analysed == ["MSFT", "NVDA", "AAPL"]


# ---------------------------------------------------------------------------
# Monthly cadence — calibration, closing the journal loop
# ---------------------------------------------------------------------------
def test_monthly_calibration_settles_due_and_reports() -> None:
    # Arrange: 40 due predictions (> MIN_SAMPLE_SIZE) with known outcomes.
    as_of = date(2025, 7, 31)
    predictions = _due_predictions(count=40, horizon=date(2025, 6, 30))
    settled_ids: list[str] = []

    def realized_for(prediction: Prediction) -> tuple[float, float, float]:
        # Even indices "hit" an UP call; crude but deterministic.
        idx = int(prediction.ticker.split("-")[1])
        realized = 0.05 if idx % 2 == 0 else -0.05
        return realized, 0.0, 0.0

    recorded_settlements: list[Settlement] = []

    def persist(settlement: Settlement) -> None:
        recorded_settlements.append(settlement)
        settled_ids.append(settlement.prediction_id)

    now = datetime(2025, 7, 31, 18, 0)
    # Act
    outcome = run_monthly(
        predictions=predictions,
        settled_ids=(),
        realized_for=realized_for,
        persist_settlement=persist,
        as_of=as_of,
        now=now,
    )
    # Assert
    assert outcome.cadence is Cadence.MONTHLY
    assert outcome.calibration is not None
    assert outcome.calibration.sample_size == 40
    assert outcome.calibration.conclusive is True
    assert len(recorded_settlements) == 40


def test_monthly_with_no_due_predictions_reports_empty() -> None:
    # Arrange: horizon far in the future → nothing due.
    as_of = date(2025, 7, 31)
    predictions = _due_predictions(count=5, horizon=date(2026, 1, 1))
    now = datetime(2025, 7, 31, 18, 0)
    # Act
    outcome = run_monthly(
        predictions=predictions,
        settled_ids=(),
        realized_for=lambda _p: (0.0, 0.0, 0.0),
        persist_settlement=lambda _s: None,
        as_of=as_of,
        now=now,
    )
    # Assert
    assert outcome.calibration is None
    assert "0" in outcome.produced


# ---------------------------------------------------------------------------
# Event cadence — takes the event as input
# ---------------------------------------------------------------------------
def test_event_cadence_analyses_the_event_ticker() -> None:
    # Arrange
    analysed: list[str] = []
    now = datetime(2025, 7, 9, 14, 30)
    # Act
    outcome = run_event(
        ticker="NVDA",
        event_kind="earnings",
        analyse=lambda t: analysed.append(t) or f"verdict:{t}",  # type: ignore[func-returns-value]
        now=now,
    )
    # Assert
    assert outcome.cadence is Cadence.EVENT
    assert analysed == ["NVDA"]
    assert "earnings" in outcome.produced


# ---------------------------------------------------------------------------
# Shared outcome shape
# ---------------------------------------------------------------------------
def test_cadence_outcome_is_frozen() -> None:
    # Arrange
    outcome = CadenceOutcome(cadence=Cadence.DAILY, produced="x", spent_trial_budget=False)
    # Act / Assert
    with pytest.raises(Exception):  # noqa: B017
        outcome.produced = "y"  # type: ignore[misc]


def _due_predictions(*, count: int, horizon: date) -> tuple[Prediction, ...]:
    """Build ``count`` predictions on a shared horizon, tagged by index."""
    return tuple(
        Prediction(
            recorded_at=datetime(2025, 1, 1, 9, 0),
            ticker=f"T-{i}",
            direction=Direction.UP,
            probability=0.6,
            horizon=horizon,
            entry_price=100.0,
            knowledge_date=date(2025, 1, 1),
        )
        for i in range(count)
    )
