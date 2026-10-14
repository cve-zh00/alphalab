"""Tests for settling due predictions from real price series.

The sign of a bearish call is the single most dangerous thing in this module:
a flipped sign would invert every DOWN outcome and silently corrupt the whole
calibration record. So both directions are verified explicitly, with prices
chosen so the correct and incorrect answers are unambiguous.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from alphalab.adapters.journal_sqlite import SqliteJournal
from alphalab.adapters.prices import CsvPriceSeriesSource
from alphalab.app.forecast_runner import settle_due_predictions
from alphalab.domain.backtest import ProportionalCostModel
from alphalab.domain.costs.result import CostSpec
from alphalab.domain.journal import Direction, Prediction

_KD = date(2026, 1, 5)
_HORIZON = date(2026, 4, 6)
_RECORDED = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
_SETTLED_AT = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
_AS_OF = date(2026, 4, 7)
# A non-zero spread so the cost drag is observable in the net alpha.
_COST_MODEL = ProportionalCostModel(CostSpec(half_spread_bps=5.0))


@pytest.fixture
def journal(tmp_path: Path) -> Iterator[SqliteJournal]:
    with SqliteJournal(tmp_path / "journal.db") as j:
        yield j


def _write_series(root: Path, ticker: str, rows: list[tuple[str, float]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = ["date,close", *[f"{on},{close}" for on, close in rows]]
    (root / f"{ticker}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prices(
    tmp_path: Path, name_exit: float, bench_entry: float, bench_exit: float
) -> CsvPriceSeriesSource:
    """A price root where SPY and the name each have an entry and a horizon close.

    Entry prices for the *name* come from the stored prediction, not the CSV,
    so only the name's horizon close is needed here; SPY needs both ends.
    """
    root = tmp_path / "prices"
    _write_series(root, "SPY", [("2026-01-05", bench_entry), ("2026-04-06", bench_exit)])
    _write_series(root, "AAPL", [("2026-01-05", 100.0), ("2026-04-06", name_exit)])
    return CsvPriceSeriesSource(root)


def _prediction(direction: Direction, *, entry_price: float = 100.0, pid: str = "p1") -> Prediction:
    return Prediction(
        prediction_id=pid,
        recorded_at=_RECORDED,
        ticker="AAPL",
        direction=direction,
        probability=0.60,
        horizon=_HORIZON,
        entry_price=entry_price,
        knowledge_date=_KD,
        llm_version="qwen3",
        llm_cutoff=date(2025, 1, 1),
    )


class TestDirectionSign:
    def test_bullish_call_that_rose_is_a_hit(self, journal: SqliteJournal, tmp_path: Path) -> None:
        """UP wins when the realized move is positive."""
        journal.record(_prediction(Direction.UP))
        prices = _prices(tmp_path, name_exit=110.0, bench_entry=400.0, bench_exit=404.0)

        run = settle_due_predictions(
            journal=journal,
            prices=prices,
            as_of=_AS_OF,
            settled_at=_SETTLED_AT,
            cost_model=_COST_MODEL,
        )

        assert len(run.settled) == 1
        settlement = run.settled[0].settlement
        assert settlement.realized_return == pytest.approx(0.10)
        assert settlement.benchmark_return == pytest.approx(0.01)
        assert settlement.outcome is True  # rose -> bullish hit

    def test_bullish_call_that_fell_is_a_miss(self, journal: SqliteJournal, tmp_path: Path) -> None:
        journal.record(_prediction(Direction.UP))
        prices = _prices(tmp_path, name_exit=90.0, bench_entry=400.0, bench_exit=404.0)

        run = settle_due_predictions(
            journal=journal,
            prices=prices,
            as_of=_AS_OF,
            settled_at=_SETTLED_AT,
            cost_model=_COST_MODEL,
        )

        assert run.settled[0].settlement.outcome is False  # fell -> bullish miss

    def test_bearish_call_that_fell_is_a_hit(self, journal: SqliteJournal, tmp_path: Path) -> None:
        """DOWN wins when the realized move is negative — the sign that must not flip.

        The name falls 100 -> 85 (realized -15%). A bearish thesis is right.
        If the sign were inverted this would read as a miss, so this test is the
        guard against a catastrophic, hard-to-detect calibration bug.
        """
        journal.record(_prediction(Direction.DOWN))
        prices = _prices(tmp_path, name_exit=85.0, bench_entry=400.0, bench_exit=404.0)

        run = settle_due_predictions(
            journal=journal,
            prices=prices,
            as_of=_AS_OF,
            settled_at=_SETTLED_AT,
            cost_model=_COST_MODEL,
        )

        settlement = run.settled[0].settlement
        assert settlement.realized_return == pytest.approx(-0.15)
        assert settlement.outcome is True  # fell -> bearish hit

    def test_bearish_call_that_rose_is_a_miss(self, journal: SqliteJournal, tmp_path: Path) -> None:
        """DOWN loses when the realized move is positive."""
        journal.record(_prediction(Direction.DOWN))
        prices = _prices(tmp_path, name_exit=115.0, bench_entry=400.0, bench_exit=404.0)

        run = settle_due_predictions(
            journal=journal,
            prices=prices,
            as_of=_AS_OF,
            settled_at=_SETTLED_AT,
            cost_model=_COST_MODEL,
        )

        assert run.settled[0].settlement.outcome is False  # rose -> bearish miss

    def test_alpha_is_net_of_a_positive_cost_drag(
        self, journal: SqliteJournal, tmp_path: Path
    ) -> None:
        """The round-trip cost is deducted, so net alpha is below gross."""
        journal.record(_prediction(Direction.UP))
        prices = _prices(tmp_path, name_exit=110.0, bench_entry=400.0, bench_exit=404.0)

        run = settle_due_predictions(
            journal=journal,
            prices=prices,
            as_of=_AS_OF,
            settled_at=_SETTLED_AT,
            cost_model=_COST_MODEL,
        )

        settlement = run.settled[0].settlement
        assert settlement.alpha_gross == pytest.approx(0.10 - 0.01)
        assert settlement.alpha_net < settlement.alpha_gross


class TestDegradation:
    def test_not_yet_due_is_not_settled(self, journal: SqliteJournal, tmp_path: Path) -> None:
        """A prediction whose horizon has not elapsed is left untouched."""
        journal.record(_prediction(Direction.UP))
        prices = _prices(tmp_path, name_exit=110.0, bench_entry=400.0, bench_exit=404.0)

        run = settle_due_predictions(
            journal=journal,
            prices=prices,
            as_of=date(2026, 1, 10),  # before the horizon
            settled_at=_SETTLED_AT,
            cost_model=_COST_MODEL,
        )

        assert run.settled == ()
        assert run.skipped == ()
        assert journal.settled_ids() == []

    def test_missing_name_price_is_reported_not_invented(
        self, journal: SqliteJournal, tmp_path: Path
    ) -> None:
        """When the name has no close near the horizon, it is skipped with a reason."""
        journal.record(_prediction(Direction.UP))
        root = tmp_path / "prices"
        _write_series(root, "SPY", [("2026-01-05", 400.0), ("2026-04-06", 404.0)])
        # The name's only price is months stale relative to the horizon.
        _write_series(root, "AAPL", [("2026-01-05", 100.0)])
        prices = CsvPriceSeriesSource(root)

        run = settle_due_predictions(
            journal=journal,
            prices=prices,
            as_of=_AS_OF,
            settled_at=_SETTLED_AT,
            cost_model=_COST_MODEL,
        )

        assert run.settled == ()
        assert len(run.skipped) == 1
        assert "AAPL" in run.skipped[0].reason
        assert journal.settled_ids() == []

    def test_missing_benchmark_price_is_reported(
        self, journal: SqliteJournal, tmp_path: Path
    ) -> None:
        """A missing benchmark close also degrades honestly."""
        journal.record(_prediction(Direction.UP))
        root = tmp_path / "prices"
        _write_series(root, "AAPL", [("2026-01-05", 100.0), ("2026-04-06", 110.0)])
        # No SPY file at all.
        prices = CsvPriceSeriesSource(root)

        run = settle_due_predictions(
            journal=journal,
            prices=prices,
            as_of=_AS_OF,
            settled_at=_SETTLED_AT,
            cost_model=_COST_MODEL,
        )

        assert run.settled == ()
        assert len(run.skipped) == 1
        assert "SPY" in run.skipped[0].reason

    def test_stale_within_tolerance_is_accepted(
        self, journal: SqliteJournal, tmp_path: Path
    ) -> None:
        """A weekend gap (horizon on a Monday, last close the prior Friday) still settles."""
        journal.record(_prediction(Direction.UP))
        root = tmp_path / "prices"
        # Horizon is 2026-04-06 (Mon); nearest closes are the prior Fri, 3 days back.
        _write_series(root, "SPY", [("2026-01-05", 400.0), ("2026-04-03", 404.0)])
        _write_series(root, "AAPL", [("2026-01-05", 100.0), ("2026-04-03", 110.0)])
        prices = CsvPriceSeriesSource(root)

        run = settle_due_predictions(
            journal=journal,
            prices=prices,
            as_of=_AS_OF,
            settled_at=_SETTLED_AT,
            cost_model=_COST_MODEL,
            max_staleness_days=7,
        )

        assert len(run.settled) == 1
        assert run.settled[0].settlement.realized_return == pytest.approx(0.10)


class TestIdempotency:
    def test_second_pass_settles_nothing(self, journal: SqliteJournal, tmp_path: Path) -> None:
        """Re-running settlement is a no-op; already-settled ids are excluded."""
        journal.record(_prediction(Direction.UP))
        prices = _prices(tmp_path, name_exit=110.0, bench_entry=400.0, bench_exit=404.0)

        first = settle_due_predictions(
            journal=journal,
            prices=prices,
            as_of=_AS_OF,
            settled_at=_SETTLED_AT,
            cost_model=_COST_MODEL,
        )
        second = settle_due_predictions(
            journal=journal,
            prices=prices,
            as_of=_AS_OF,
            settled_at=_SETTLED_AT,
            cost_model=_COST_MODEL,
        )

        assert len(first.settled) == 1
        assert second.settled == ()
        assert len(journal.list_settlements()) == 1

    def test_settlement_does_not_trip_immutability_triggers(
        self, journal: SqliteJournal, tmp_path: Path
    ) -> None:
        """The runner never updates or deletes; a second pass raises no IntegrityError."""
        journal.record(_prediction(Direction.UP))
        prices = _prices(tmp_path, name_exit=110.0, bench_entry=400.0, bench_exit=404.0)

        settle_due_predictions(
            journal=journal,
            prices=prices,
            as_of=_AS_OF,
            settled_at=_SETTLED_AT,
            cost_model=_COST_MODEL,
        )
        try:
            settle_due_predictions(
                journal=journal,
                prices=prices,
                as_of=_AS_OF,
                settled_at=_SETTLED_AT,
                cost_model=_COST_MODEL,
            )
        except sqlite3.IntegrityError as exc:  # pragma: no cover - guard assertion
            pytest.fail(f"settlement collided with immutability trigger: {exc}")
