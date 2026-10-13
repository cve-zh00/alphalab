"""The required scheduling demo — closes the loop end to end.

Three acts, all with injected time and test doubles (no clock, no LLM, no
network, no ``launchctl``):

(a) **Missed job on wake.** The Mac slept 3 days; prove the daily cadence is
    still due on wake (launchd enqueues; cron would have discarded it).
(b) **Plist generation.** Generate the three calendar plists and validate each
    with ``plistlib`` — without installing anything.
(c) **Monthly calibration.** Run the monthly cadence over doubles and produce a
    real ``CalibrationReport``, closing journal → calibration → observability.
"""

from __future__ import annotations

import plistlib
from datetime import date, datetime
from pathlib import Path

from structlog.testing import capture_logs

from alphalab.adapters.scheduler import LaunchdJobSpec, write_plists
from alphalab.app.cadences import run_monthly
from alphalab.domain.journal import Direction, Prediction, Settlement
from alphalab.domain.schedule import Cadence, ScheduleSpec, due_runs, run_key
from alphalab.obs import RunRecord, RunStatus, configure_logging, log_run_record


def test_demo_a_missed_daily_job_runs_on_wake() -> None:
    # Arrange: the daily cadence last ran 3 days ago; the Mac was asleep since.
    spec = ScheduleSpec(cadence=Cadence.DAILY)
    last_run = datetime(2025, 7, 3, 22, 0)  # Thursday
    wake = datetime(2025, 7, 7, 8, 0)  # Monday morning — a trading day
    # Act
    is_due = due_runs(spec, last_run=last_run, now=wake)
    # Assert: the queued job runs; it was not discarded (the launchd property).
    assert is_due is True
    # And the run key for the wake day differs from the last run's key, which
    # is exactly why it is due.
    assert run_key(spec, last_run) != run_key(spec, wake)


def test_demo_b_generate_and_validate_three_plists(tmp_path: Path) -> None:
    # Arrange
    spec = LaunchdJobSpec(
        label_prefix="com.alphalab",
        program_arguments=("python", "scripts/schedule_runner.py"),
        working_directory=tmp_path / "project",
        log_directory=tmp_path / "logs",
    )
    out_dir = tmp_path / "LaunchAgents"
    # Act
    written = write_plists(spec, out_dir)
    # Assert: three valid plists, one per calendar cadence.
    assert len(written) == 3
    labels = set()
    for path in written:
        parsed = plistlib.loads(path.read_bytes())
        labels.add(parsed["Label"])
        assert "StartCalendarInterval" in parsed
        assert parsed["ProgramArguments"][-1] in {"daily", "weekly", "monthly"}
    assert labels == {
        "com.alphalab.daily",
        "com.alphalab.weekly",
        "com.alphalab.monthly",
    }


def test_demo_c_monthly_calibration_closes_the_loop() -> None:
    # Arrange: 50 due predictions with a realistic, slightly-overconfident mix.
    as_of = date(2025, 7, 31)
    predictions = _demo_predictions(count=50, horizon=date(2025, 6, 30))
    persisted: list[Settlement] = []

    def realized_for(prediction: Prediction) -> tuple[float, float, float]:
        # Deterministic: exactly 32 of 50 UP calls actually rise (hit_rate
        # 0.64), a mild overconfidence relative to the declared 0.70.
        idx = int(prediction.ticker.split("-")[1])
        rises = idx < 32
        realized = 0.04 if rises else -0.03
        return realized, 0.0, 0.005  # small cost drag

    now = datetime(2025, 7, 31, 18, 0)

    # Act: run the monthly cadence, and log its RunRecord through observability.
    configure_logging()
    started = datetime(2025, 7, 31, 18, 0, 0)
    with capture_logs() as entries:
        outcome = run_monthly(
            predictions=predictions,
            settled_ids=(),
            realized_for=realized_for,
            persist_settlement=persisted.append,
            as_of=as_of,
            now=now,
        )
        finished = datetime(2025, 7, 31, 18, 0, 12)
        log_run_record(
            RunRecord(
                run_id="demo-monthly",
                run_key=run_key(ScheduleSpec(cadence=Cadence.MONTHLY), now),
                cadence=Cadence.MONTHLY,
                started_at=started,
                finished_at=finished,
                status=RunStatus.OK,
                produced=outcome.produced,
            )
        )

    # Assert: a real, conclusive calibration report came out of the loop.
    report = outcome.calibration
    assert report is not None
    assert report.sample_size == 50
    assert report.conclusive is True
    assert 0.0 <= report.brier <= 1.0
    assert 0.0 <= report.ece <= 1.0
    assert report.hit_rate == 0.64  # 32 of 50 hit by construction
    # The system declared 0.70 but was right 64% of the time → overconfident.
    assert report.overconfidence > 0.0
    # Every due prediction was settled exactly once and persisted.
    assert len(persisted) == 50
    # The run was observed: one structured record entry carrying the duration.
    record_entries = [e for e in entries if e.get("event") == "run_completed"]
    assert len(record_entries) == 1
    assert record_entries[0]["duration_seconds"] == 12.0


def _demo_predictions(*, count: int, horizon: date) -> tuple[Prediction, ...]:
    """Build predictions each declaring probability 0.70 on an UP call."""
    return tuple(
        Prediction(
            recorded_at=datetime(2025, 1, 1, 9, 0),
            ticker=f"D-{i}",
            direction=Direction.UP,
            probability=0.70,
            horizon=horizon,
            entry_price=100.0,
            knowledge_date=date(2025, 1, 1),
            llm_version="qwen3.8-27b-mlx",
            llm_cutoff=date(2024, 10, 1),
        )
        for i in range(count)
    )
