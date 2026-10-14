"""Command runners for the alphalab CLI.

Each module here owns one subcommand's orchestration: it takes a resolved,
typed options object plus output/error streams and returns a process exit code.
Argument parsing lives in :mod:`alphalab.cli.app`; these runners never parse
``argv`` themselves, which keeps them trivially unit-testable.
"""

from __future__ import annotations

from alphalab.cli.commands.calibration import CalibrationOptions, run_calibration
from alphalab.cli.commands.doctor import DoctorOptions, run_doctor
from alphalab.cli.commands.research import ResearchOptions, run_research
from alphalab.cli.commands.schedule import ScheduleOptions, run_schedule_generate

__all__ = [
    "CalibrationOptions",
    "DoctorOptions",
    "ResearchOptions",
    "ScheduleOptions",
    "run_calibration",
    "run_doctor",
    "run_research",
    "run_schedule_generate",
]
