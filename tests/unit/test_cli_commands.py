"""Unit tests for the doctor, schedule, and calibration commands.

Guarantees asserted here:
* ``doctor`` is read-only — it creates/modifies nothing on disk.
* ``schedule generate`` writes plists but NEVER runs ``launchctl``.
* ``calibration`` reads a journal and reports; empty/missing is handled.
"""

from __future__ import annotations

import io
import json
from datetime import date, datetime
from pathlib import Path

import pytest

from alphalab.cli import main
from alphalab.cli.commands.doctor import DoctorOptions, run_doctor


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


class TestDoctorReadOnly:
    def test_doctor_exits_zero_and_reports(self, tmp_path: Path) -> None:
        code, out, _ = _run(["doctor", "--db", str(tmp_path / "none.db")])
        assert code == 0
        assert "DOCTOR" in out
        assert "fixtures" in out
        assert "modified nothing" in out

    def test_doctor_creates_nothing_on_disk(self, tmp_path: Path) -> None:
        """A read-only check must not touch the filesystem."""
        before = set(tmp_path.iterdir())
        db = tmp_path / "journal.db"
        out, err = io.StringIO(), io.StringIO()
        run_doctor(
            DoctorOptions(db_path=db, as_json=False, no_color=True),
            out=out,
            err=err,
        )
        after = set(tmp_path.iterdir())
        assert before == after
        assert not db.exists()

    def test_doctor_json_is_parseable(self, tmp_path: Path) -> None:
        code, out, _ = _run(["doctor", "--db", str(tmp_path / "none.db"), "--json"])
        assert code == 0
        payload = json.loads(out)
        names = {item["name"] for item in payload}
        assert {"ollama", "fixtures", "journal", "edgar_user_agent"} <= names

    def test_doctor_reports_missing_journal(self, tmp_path: Path) -> None:
        _, out, _ = _run(["doctor", "--db", str(tmp_path / "absent.db")])
        assert "no journal DB" in out


class TestScheduleGenerate:
    def test_writes_plists_and_prints_install_command(self, tmp_path: Path) -> None:
        code, out, _ = _run(["schedule", "generate", "--out-dir", str(tmp_path)])
        assert code == 0
        written = list(tmp_path.glob("*.plist"))
        assert written, "expected plists to be written"
        assert "install with: launchctl bootstrap" in out
        assert "did NOT modify your system" in out

    def test_never_invokes_launchctl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The command must not shell out to launchctl under any circumstance."""
        calls: list[tuple[object, ...]] = []

        import subprocess

        def _spy(*args: object, **kwargs: object) -> None:
            calls.append(args)
            raise AssertionError("launchctl must never be executed")

        monkeypatch.setattr(subprocess, "run", _spy)
        monkeypatch.setattr(subprocess, "Popen", _spy)
        monkeypatch.setattr(subprocess, "call", _spy)

        code, out, _ = _run(["schedule", "generate", "--out-dir", str(tmp_path)])
        assert code == 0
        assert calls == []
        assert "launchctl" in out  # printed as instruction, not executed

    def test_schedule_without_subcommand_is_usage_error(self) -> None:
        code, _out, err = _run(["schedule"])
        assert code == 2
        assert "generate" in err


class TestCalibration:
    def test_missing_db_is_actionable_error(self, tmp_path: Path) -> None:
        code, _out, err = _run(["calibration", "--db", str(tmp_path / "absent.db")])
        assert code == 2
        assert "no journal database" in err

    def test_empty_journal_reports_nothing_to_calibrate(self, tmp_path: Path) -> None:
        from alphalab.adapters.journal_sqlite import SqliteJournal

        db = tmp_path / "journal.db"
        SqliteJournal(db).close()  # create empty schema
        code, out, _ = _run(["calibration", "--db", str(db)])
        assert code == 0
        assert "nothing to calibrate" in out

    def test_reports_calibration_from_settled_predictions(self, tmp_path: Path) -> None:
        db = _journal_with_settlements(tmp_path)
        code, out, _ = _run(["calibration", "--db", str(db)])
        assert code == 0
        assert "CALIBRATION REPORT" in out
        assert "Brier score" in out
        assert "Reliability diagram" in out

    def test_calibration_json(self, tmp_path: Path) -> None:
        db = _journal_with_settlements(tmp_path)
        code, out, _ = _run(["calibration", "--db", str(db), "--json"])
        assert code == 0
        payload = json.loads(out)
        assert "brier" in payload
        assert payload["sample_size"] >= 1


def _journal_with_settlements(tmp_path: Path) -> Path:
    """Build a small journal with two settled predictions for calibration."""
    from alphalab.adapters.journal_sqlite import SqliteJournal
    from alphalab.domain.journal import Direction, Prediction, Settlement

    db = tmp_path / "journal.db"
    journal = SqliteJournal(db)
    try:
        for i, (prob, outcome) in enumerate([(0.7, True), (0.4, False)]):
            pred = Prediction(
                prediction_id=f"p{i}",
                recorded_at=datetime(2024, 1, 1, 12, 0, 0),
                ticker="AAPL",
                direction=Direction.UP,
                probability=prob,
                horizon=date(2024, 2, 1),
                entry_price=100.0,
                knowledge_date=date(2024, 1, 1),
                llm_version="v1",
                llm_cutoff=date(2023, 12, 1),
            )
            journal.record(pred)
            journal.settle(
                Settlement(
                    prediction_id=f"p{i}",
                    settled_at=datetime(2024, 2, 1, 12, 0, 0),
                    realized_return=0.05 if outcome else -0.05,
                    benchmark_return=0.0,
                    alpha_gross=0.05 if outcome else -0.05,
                    alpha_net=0.04 if outcome else -0.06,
                    outcome=outcome,
                )
            )
    finally:
        journal.close()
    return db
