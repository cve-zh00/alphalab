"""Unit tests for the launchd plist generator (``alphalab.adapters.scheduler``).

The generator must produce **valid** plist XML (parseable by ``plistlib``) with
the right ``StartCalendarInterval`` for each cadence. It must never touch the
user's system — no ``launchctl`` is invoked anywhere. Writing files is tested
against pytest's ``tmp_path`` so nothing leaks onto the real disk.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from alphalab.adapters.scheduler import (
    LaunchdJobSpec,
    build_plist,
    install_command,
    write_plists,
)
from alphalab.domain.schedule import Cadence


@pytest.fixture
def job_spec(tmp_path: Path) -> LaunchdJobSpec:
    """A representative job spec rooted under tmp_path."""
    return LaunchdJobSpec(
        label_prefix="com.alphalab",
        program_arguments=("/usr/bin/env", "python", "-m", "alphalab_runner"),
        working_directory=tmp_path / "project",
        log_directory=tmp_path / "logs",
    )


# ---------------------------------------------------------------------------
# Plist structure
# ---------------------------------------------------------------------------
def test_build_plist_is_valid_and_parseable(job_spec: LaunchdJobSpec) -> None:
    # Arrange / Act
    xml = build_plist(job_spec, Cadence.DAILY)
    parsed = plistlib.loads(xml)
    # Assert
    assert parsed["Label"] == "com.alphalab.daily"
    assert parsed["ProgramArguments"][-1] == "daily"
    assert parsed["RunAtLoad"] is False


def test_daily_plist_runs_once_a_day(job_spec: LaunchdJobSpec) -> None:
    # Arrange / Act
    parsed = plistlib.loads(build_plist(job_spec, Cadence.DAILY))
    interval = parsed["StartCalendarInterval"]
    # Assert: a daily interval pins Hour+Minute but no weekday/day.
    assert "Hour" in interval and "Minute" in interval
    assert "Weekday" not in interval
    assert "Day" not in interval


def test_weekly_plist_pins_a_weekday(job_spec: LaunchdJobSpec) -> None:
    # Arrange / Act
    parsed = plistlib.loads(build_plist(job_spec, Cadence.WEEKLY))
    interval = parsed["StartCalendarInterval"]
    # Assert: launchd Weekday uses Sunday=0; the weekly default is Sunday.
    assert interval["Weekday"] == 0


def test_monthly_plist_pins_a_day_of_month(job_spec: LaunchdJobSpec) -> None:
    # Arrange / Act
    parsed = plistlib.loads(build_plist(job_spec, Cadence.MONTHLY))
    interval = parsed["StartCalendarInterval"]
    # Assert
    assert interval["Day"] == 1


def test_event_cadence_has_no_plist(job_spec: LaunchdJobSpec) -> None:
    # Arrange / Act / Assert: event runs are not calendar-scheduled.
    with pytest.raises(ValueError, match="event"):
        build_plist(job_spec, Cadence.EVENT)


def test_plist_sets_log_paths_and_working_directory(job_spec: LaunchdJobSpec) -> None:
    # Arrange / Act
    parsed = plistlib.loads(build_plist(job_spec, Cadence.DAILY))
    # Assert
    assert parsed["WorkingDirectory"] == str(job_spec.working_directory)
    assert parsed["StandardOutPath"].endswith("daily.out.log")
    assert parsed["StandardErrorPath"].endswith("daily.err.log")


def test_plist_never_contains_a_secret(job_spec: LaunchdJobSpec) -> None:
    # Arrange / Act: render all calendar cadences and scan the bytes.
    blob = b"".join(
        build_plist(job_spec, c) for c in (Cadence.DAILY, Cadence.WEEKLY, Cadence.MONTHLY)
    )
    # Assert
    lowered = blob.lower()
    assert b"api_key" not in lowered
    assert b"password" not in lowered
    assert b"secret" not in lowered


# ---------------------------------------------------------------------------
# Writing to disk (tmp_path only)
# ---------------------------------------------------------------------------
def test_write_plists_creates_three_files(job_spec: LaunchdJobSpec, tmp_path: Path) -> None:
    # Arrange
    out_dir = tmp_path / "LaunchAgents"
    # Act
    written = write_plists(job_spec, out_dir)
    # Assert: one file per calendar cadence (daily, weekly, monthly).
    assert len(written) == 3
    for path in written:
        assert path.exists()
        parsed = plistlib.loads(path.read_bytes())
        assert parsed["Label"].startswith("com.alphalab.")


def test_install_command_references_launchctl_but_does_not_run_it(
    job_spec: LaunchdJobSpec, tmp_path: Path
) -> None:
    # Arrange
    plist_path = tmp_path / "com.alphalab.daily.plist"
    # Act
    command = install_command(plist_path)
    # Assert: it is a *string the user runs*, not something we execute.
    assert "launchctl" in command
    assert str(plist_path) in command
