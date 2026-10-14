"""Unit tests for the ``journal`` command: list open and settled predictions.

The command is read-only: it opens the journal (and the abstention store) and
prints the accumulating history so the user can watch the forward record grow.
Open predictions, settled ones with their outcome, and the count of recorded
abstentions must all be visible, and an empty/missing journal must be reported
plainly rather than crashing.
"""

from __future__ import annotations

import io
import json
from datetime import date, datetime
from pathlib import Path

from alphalab.adapters.journal_sqlite import SqliteJournal
from alphalab.cli import main
from alphalab.domain.journal import Direction, Prediction, Settlement


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def _seed(db: Path) -> None:
    journal = SqliteJournal(db)
    try:
        open_pred = Prediction(
            prediction_id="open-1",
            recorded_at=datetime(2024, 1, 1, 12, 0, 0),
            ticker="AAPL",
            direction=Direction.UP,
            probability=0.7,
            horizon=date(2024, 12, 31),
            entry_price=100.0,
            knowledge_date=date(2024, 1, 1),
            llm_version="qwen3",
            llm_cutoff=date(2023, 1, 1),
        )
        settled_pred = Prediction(
            prediction_id="settled-1",
            recorded_at=datetime(2024, 1, 1, 12, 0, 0),
            ticker="MSFT",
            direction=Direction.DOWN,
            probability=0.6,
            horizon=date(2024, 3, 1),
            entry_price=200.0,
            knowledge_date=date(2024, 1, 1),
            llm_version="qwen3",
            llm_cutoff=date(2023, 1, 1),
        )
        journal.record(open_pred)
        journal.record(settled_pred)
        journal.settle(
            Settlement(
                prediction_id="settled-1",
                settled_at=datetime(2024, 3, 1, 12, 0, 0),
                realized_return=-0.05,
                benchmark_return=0.0,
                alpha_gross=-0.05,
                alpha_net=-0.06,
                outcome=True,
            )
        )
    finally:
        journal.close()


class TestJournalListing:
    def test_empty_journal_reports_plainly(self, tmp_path: Path) -> None:
        db = tmp_path / "journal.db"
        SqliteJournal(db).close()
        code, out, _ = _run(["journal", "--db", str(db)])
        assert code == 0
        assert "no predictions" in out.lower() or "empty" in out.lower()

    def test_missing_journal_is_actionable_error(self, tmp_path: Path) -> None:
        code, _out, err = _run(["journal", "--db", str(tmp_path / "absent.db")])
        assert code == 2
        assert "no journal" in err.lower()

    def test_lists_open_and_settled(self, tmp_path: Path) -> None:
        db = tmp_path / "journal.db"
        _seed(db)
        code, out, _ = _run(["journal", "--db", str(db)])
        assert code == 0
        assert "AAPL" in out
        assert "MSFT" in out
        assert "open" in out.lower()
        assert "settled" in out.lower()

    def test_json_is_parseable(self, tmp_path: Path) -> None:
        db = tmp_path / "journal.db"
        _seed(db)
        code, out, _ = _run(["journal", "--db", str(db), "--json"])
        assert code == 0
        payload = json.loads(out)
        assert payload["open_count"] == 1
        assert payload["settled_count"] == 1
        tickers = {row["ticker"] for row in payload["predictions"]}
        assert {"AAPL", "MSFT"} <= tickers
