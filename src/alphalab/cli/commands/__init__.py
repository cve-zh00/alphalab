"""Command runners for the alphalab CLI.

Each module here owns one subcommand's orchestration: it takes a resolved,
typed options object plus output/error streams and returns a process exit code.
Argument parsing lives in :mod:`alphalab.cli.app`; these runners never parse
``argv`` themselves, which keeps them trivially unit-testable.
"""

from __future__ import annotations

from alphalab.cli.commands.autopilot import (
    AutopilotCycleOptions,
    AutopilotPreflightOptions,
    run_autopilot_loop,
    run_autopilot_preflight,
    run_autopilot_run,
    run_autopilot_status,
)
from alphalab.cli.commands.calibration import CalibrationOptions, run_calibration
from alphalab.cli.commands.doctor import DoctorOptions, run_doctor
from alphalab.cli.commands.journal import JournalOptions, run_journal
from alphalab.cli.commands.paper import (
    PaperLogOptions,
    PaperRunOptions,
    PaperStatusOptions,
    run_paper_log,
    run_paper_run,
    run_paper_status,
)
from alphalab.cli.commands.predict import PredictOptions, run_predict
from alphalab.cli.commands.research import ResearchOptions, run_research
from alphalab.cli.commands.schedule import ScheduleOptions, run_schedule_generate
from alphalab.cli.commands.settle import SettleOptions, run_settle

__all__ = [
    "AutopilotCycleOptions",
    "AutopilotPreflightOptions",
    "CalibrationOptions",
    "DoctorOptions",
    "JournalOptions",
    "PaperLogOptions",
    "PaperRunOptions",
    "PaperStatusOptions",
    "PredictOptions",
    "ResearchOptions",
    "ScheduleOptions",
    "SettleOptions",
    "run_autopilot_loop",
    "run_autopilot_preflight",
    "run_autopilot_run",
    "run_autopilot_status",
    "run_calibration",
    "run_doctor",
    "run_journal",
    "run_paper_log",
    "run_paper_run",
    "run_paper_status",
    "run_predict",
    "run_research",
    "run_schedule_generate",
    "run_settle",
]
