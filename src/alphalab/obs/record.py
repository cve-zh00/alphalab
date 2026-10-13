"""The persistable record of a single scheduled run.

A :class:`RunRecord` is the answer to "did it run last night, and what
happened?" without guessing: it names which cadence ran, when it started and
finished (hence how long it took), what it produced, and — if it failed — why.
It is a frozen pydantic model so a run's history is immutable once written, and
it serialises to JSON for a simple append-only log the user can grep.

This module is pure data: it performs no I/O. Persisting a record is the
caller's concern (``obs.logging.log_run_record`` emits it to the structured
log; a future sink could append JSON lines to a file).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalab.domain.schedule import Cadence

__all__ = ["RunRecord", "RunStatus"]


class RunStatus(StrEnum):
    """Terminal status of a scheduled run."""

    OK = "ok"
    FAILED = "failed"


class RunRecord(BaseModel):
    """Immutable record of one scheduled run.

    Attributes:
        run_id: Unique id of this run (distinct from the idempotency run key).
        run_key: The cadence+period idempotency key the run executed under.
        cadence: Which cadence ran.
        started_at: When the run began.
        finished_at: When the run ended; must not precede ``started_at``.
        status: Terminal status (OK / FAILED).
        produced: Short human description of what the run produced.
        error: Failure reason; required when ``status`` is FAILED, else None.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    run_key: str = Field(min_length=1)
    cadence: Cadence
    started_at: datetime
    finished_at: datetime
    status: RunStatus
    produced: str = ""
    error: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> RunRecord:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.status is RunStatus.FAILED and not self.error:
            raise ValueError("a FAILED run must carry an error reason")
        return self

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of the run, in seconds."""
        return (self.finished_at - self.started_at).total_seconds()
