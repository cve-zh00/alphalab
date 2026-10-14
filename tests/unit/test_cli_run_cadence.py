"""Unit tests for the launchd-fired ``_run-cadence`` dispatch and plist wiring.

``schedule generate`` writes plists that invoke ``alphalab _run-cadence
<cadence>``; this is the receiving end that maps a cadence onto the new
commands (daily→settle, weekly→predict, monthly→calibration). The dispatch runs
the real runners against temp DBs with no network and no LLM, and must never
shell out to ``launchctl`` or ``subprocess`` under any circumstance.
"""

from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path

import pytest

from alphalab.adapters.journal_sqlite import SqliteJournal
from alphalab.cli import main
from alphalab.cli.commands import run_cadence as run_cadence_mod
from alphalab.domain.journal import Direction, Prediction


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


class TestRunCadenceDispatch:
    def test_daily_cadence_settles(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """daily → settle: a due prediction with prices is liquidated."""
        journal_db = tmp_path / "journal.db"
        prices = tmp_path / "prices"
        prices.mkdir()
        (prices / "SPY.csv").write_text(
            "date,close\n2024-01-01,400.0\n2024-03-01,404.0\n", encoding="utf-8"
        )
        (prices / "AAPL.csv").write_text(
            "date,close\n2024-01-01,100.0\n2024-03-01,110.0\n", encoding="utf-8"
        )
        journal = SqliteJournal(journal_db)
        try:
            journal.record(
                Prediction(
                    prediction_id="p1",
                    recorded_at=datetime(2024, 1, 1, 12, 0, 0),
                    ticker="AAPL",
                    direction=Direction.UP,
                    probability=0.65,
                    horizon=date(2024, 3, 1),
                    entry_price=100.0,
                    knowledge_date=date(2024, 1, 1),
                    llm_version="qwen3",
                    llm_cutoff=date(2023, 1, 1),
                )
            )
        finally:
            journal.close()

        monkeypatch.setattr(run_cadence_mod, "default_journal_db", lambda: journal_db)
        monkeypatch.setattr(run_cadence_mod, "default_prices_dir", lambda: prices)

        out, err = io.StringIO(), io.StringIO()
        code = run_cadence_mod.run_cadence("daily", out=out, err=err, no_color=True)
        assert code == 0, err.getvalue()
        assert "SETTLEMENT PASS" in out.getvalue()

    def test_unknown_cadence_is_error(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        code = run_cadence_mod.run_cadence("quarterly", out=out, err=err, no_color=True)
        assert code == 2
        assert "unknown cadence" in err.getvalue().lower()

    def test_cli_run_cadence_rejects_bad_cadence(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["_run-cadence", "quarterly"], out=io.StringIO(), err=io.StringIO())
        assert exc.value.code == 2


class TestScheduleGenerateWiresNewCommands:
    def test_plists_invoke_run_cadence(self, tmp_path: Path) -> None:
        """Each generated plist must invoke the new `_run-cadence` entrypoint."""
        import plistlib

        code, _out, _ = _run(["schedule", "generate", "--out-dir", str(tmp_path)])
        assert code == 0
        plists = sorted(tmp_path.glob("*.plist"))
        assert len(plists) == 3
        for path in plists:
            parsed = plistlib.loads(path.read_bytes())
            argv = parsed["ProgramArguments"]
            assert "_run-cadence" in argv
            assert argv[-1] in {"daily", "weekly", "monthly"}

    def test_generate_never_invokes_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The command must never shell out (no launchctl, no subprocess)."""
        import subprocess

        def _spy(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("subprocess must never be invoked by schedule generate")

        monkeypatch.setattr(subprocess, "run", _spy)
        monkeypatch.setattr(subprocess, "Popen", _spy)
        monkeypatch.setattr(subprocess, "call", _spy)
        monkeypatch.setattr(subprocess, "check_call", _spy)
        monkeypatch.setattr(subprocess, "check_output", _spy)

        code, out, _ = _run(["schedule", "generate", "--out-dir", str(tmp_path)])
        assert code == 0
        assert "launchctl" in out  # printed as an instruction, never executed
