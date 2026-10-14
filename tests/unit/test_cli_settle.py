"""Unit tests for the ``settle`` command: liquidate due predictions honestly.

The settler prices realized and benchmark moves from a local CSV price source;
these tests point it at a tiny temp directory of fixtures so there is no
network. The three reportable outcomes — settled, still-pending, and
could-not-price (with a reason) — must all be visible, and a run with nothing
due must succeed and report zero rather than fail.
"""

from __future__ import annotations

import io
import json
from datetime import date, datetime
from pathlib import Path

from alphalab.adapters.journal_sqlite import SqliteJournal
from alphalab.cli import main
from alphalab.domain.journal import Direction, Prediction


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def _write_series(prices_dir: Path, ticker: str, rows: list[tuple[str, float]]) -> None:
    prices_dir.mkdir(parents=True, exist_ok=True)
    lines = ["date,close"]
    lines += [f"{d},{c}" for d, c in rows]
    (prices_dir / f"{ticker}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record(journal_db: Path, prediction: Prediction) -> None:
    journal = SqliteJournal(journal_db)
    try:
        journal.record(prediction)
    finally:
        journal.close()


def _a_prediction(
    *, ticker: str = "AAPL", knowledge: date, horizon: date, entry: float = 100.0
) -> Prediction:
    return Prediction(
        prediction_id=f"{ticker}-{horizon.isoformat()}",
        recorded_at=datetime(2024, 1, 1, 12, 0, 0),
        ticker=ticker,
        direction=Direction.UP,
        probability=0.65,
        horizon=horizon,
        entry_price=entry,
        knowledge_date=knowledge,
        llm_version="qwen3",
        llm_cutoff=date(2023, 1, 1),
    )


class TestSettleNothingDue:
    def test_empty_journal_reports_zero_and_exits_zero(self, tmp_path: Path) -> None:
        db = tmp_path / "journal.db"
        SqliteJournal(db).close()
        prices = tmp_path / "prices"
        _write_series(prices, "SPY", [("2024-01-01", 400.0)])
        code, out, err = _run(
            ["settle", "--db", str(db), "--prices-dir", str(prices), "--as-of", "2024-06-01"]
        )
        assert code == 0, err
        assert "0" in out
        assert "settled" in out.lower()

    def test_prediction_not_yet_due_stays_pending(self, tmp_path: Path) -> None:
        db = tmp_path / "journal.db"
        _record(db, _a_prediction(knowledge=date(2024, 1, 1), horizon=date(2024, 12, 31)))
        prices = tmp_path / "prices"
        _write_series(prices, "SPY", [("2024-01-01", 400.0)])
        _write_series(prices, "AAPL", [("2024-01-01", 100.0)])
        code, out, _ = _run(
            ["settle", "--db", str(db), "--prices-dir", str(prices), "--as-of", "2024-02-01"]
        )
        assert code == 0
        assert "pending" in out.lower()


class TestSettleLiquidates:
    def test_due_prediction_with_prices_is_settled(self, tmp_path: Path) -> None:
        db = tmp_path / "journal.db"
        _record(db, _a_prediction(knowledge=date(2024, 1, 1), horizon=date(2024, 3, 1)))
        prices = tmp_path / "prices"
        _write_series(prices, "SPY", [("2024-01-01", 400.0), ("2024-03-01", 404.0)])
        _write_series(prices, "AAPL", [("2024-01-01", 100.0), ("2024-03-01", 110.0)])
        code, out, err = _run(
            ["settle", "--db", str(db), "--prices-dir", str(prices), "--as-of", "2024-03-15"]
        )
        assert code == 0, err
        assert "1" in out
        journal = SqliteJournal(db)
        try:
            assert len(journal.settled_ids()) == 1
        finally:
            journal.close()

    def test_missing_price_is_reported_with_reason(self, tmp_path: Path) -> None:
        """A due prediction with no horizon price is skipped, not invented."""
        db = tmp_path / "journal.db"
        _record(db, _a_prediction(knowledge=date(2024, 1, 1), horizon=date(2024, 3, 1)))
        prices = tmp_path / "prices"
        _write_series(prices, "SPY", [("2024-01-01", 400.0), ("2024-03-01", 404.0)])
        _write_series(prices, "AAPL", [("2024-01-01", 100.0)])
        code, out, _ = _run(
            ["settle", "--db", str(db), "--prices-dir", str(prices), "--as-of", "2024-06-01"]
        )
        assert code == 0
        assert "AAPL" in out
        assert "no " in out.lower() or "could not" in out.lower()
        journal = SqliteJournal(db)
        try:
            assert journal.settled_ids() == []
        finally:
            journal.close()


class TestSettleJson:
    def test_json_is_parseable(self, tmp_path: Path) -> None:
        db = tmp_path / "journal.db"
        _record(db, _a_prediction(knowledge=date(2024, 1, 1), horizon=date(2024, 3, 1)))
        prices = tmp_path / "prices"
        _write_series(prices, "SPY", [("2024-01-01", 400.0), ("2024-03-01", 404.0)])
        _write_series(prices, "AAPL", [("2024-01-01", 100.0), ("2024-03-01", 110.0)])
        code, out, _ = _run(
            [
                "settle",
                "--db",
                str(db),
                "--prices-dir",
                str(prices),
                "--as-of",
                "2024-03-15",
                "--json",
            ]
        )
        assert code == 0
        payload = json.loads(out)
        assert payload["settled"] == 1
        assert payload["skipped"] == 0
        assert "pending" in payload

    def test_missing_db_is_actionable_error(self, tmp_path: Path) -> None:
        code, _out, err = _run(["settle", "--db", str(tmp_path / "absent.db")])
        assert code == 2
        assert "no journal" in err.lower()
