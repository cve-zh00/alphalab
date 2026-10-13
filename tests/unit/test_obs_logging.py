"""Unit tests for the observability layer (``alphalab.obs``).

Covers the persistable :class:`RunRecord` and the structured-logging helpers.
Logging is captured with structlog's ``capture_logs`` (event dicts, pre-render)
so assertions are exact and no real sink is touched. The secret-safety test
pins the promise that sensitive configuration never reaches a log line.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import structlog
from structlog.testing import capture_logs

from alphalab.domain.schedule import Cadence
from alphalab.obs import (
    RunRecord,
    RunStatus,
    bind_run_context,
    configure_logging,
    log_run_record,
)


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Ensure a clean structlog pipeline per test."""
    configure_logging()


# ---------------------------------------------------------------------------
# RunRecord
# ---------------------------------------------------------------------------
def test_run_record_is_frozen() -> None:
    # Arrange
    record = _sample_record()
    # Act / Assert
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        record.cadence = Cadence.WEEKLY  # type: ignore[misc]


def test_run_record_duration_seconds_is_derived() -> None:
    # Arrange
    started = datetime(2025, 7, 7, 22, 0, 0)
    finished = started + timedelta(seconds=42)
    # Act
    record = _sample_record(started_at=started, finished_at=finished)
    # Assert
    assert record.duration_seconds == pytest.approx(42.0)


def test_run_record_rejects_finish_before_start() -> None:
    # Arrange
    started = datetime(2025, 7, 7, 22, 0, 0)
    finished = started - timedelta(seconds=1)
    # Act / Assert
    with pytest.raises(ValueError, match="finished_at"):
        _sample_record(started_at=started, finished_at=finished)


def test_run_record_failed_requires_error() -> None:
    # Arrange / Act / Assert: a FAILED record must carry a reason.
    with pytest.raises(ValueError, match="error"):
        _sample_record(status=RunStatus.FAILED, error=None)


def test_run_record_ok_round_trips_through_json() -> None:
    # Arrange
    record = _sample_record()
    # Act
    restored = RunRecord.model_validate_json(record.model_dump_json())
    # Assert
    assert restored == record


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
def test_bind_run_context_attaches_metadata() -> None:
    # Arrange / Act
    with capture_logs() as entries:
        log = bind_run_context(
            run_id="run-123",
            cadence=Cadence.DAILY,
            ticker="MSFT",
            llm_version="qwen3-vl",
            llm_cutoff="2024-10-01",
        )
        log.info("refresh_started")
    # Assert
    (entry,) = entries
    assert entry["run_id"] == "run-123"
    assert entry["cadence"] == "daily"
    assert entry["ticker"] == "MSFT"
    assert entry["llm_version"] == "qwen3-vl"
    assert entry["event"] == "refresh_started"


def test_log_run_record_emits_structured_entry() -> None:
    # Arrange
    record = _sample_record()
    # Act
    with capture_logs() as entries:
        log_run_record(record)
    # Assert
    (entry,) = entries
    assert entry["run_id"] == record.run_id
    assert entry["status"] == "ok"
    assert entry["duration_seconds"] == pytest.approx(record.duration_seconds)
    assert entry["event"] == "run_completed"


def test_log_run_record_failed_logs_error_level() -> None:
    # Arrange
    record = _sample_record(status=RunStatus.FAILED, error="boom")
    # Act
    with capture_logs() as entries:
        log_run_record(record)
    # Assert
    (entry,) = entries
    assert entry["log_level"] == "error"
    assert entry["error"] == "boom"


def test_logging_never_emits_api_key() -> None:
    # Arrange: exercise the *real* processor chain (capture_logs bypasses it),
    # so this proves redaction happens end-to-end, not just in isolation.
    import json

    factory = structlog.testing.CapturingLoggerFactory()
    configure_logging(logger_factory=factory)
    # Act
    log = bind_run_context(run_id="run-1", cadence=Cadence.DAILY)
    log.info("started", base_url="http://localhost:11434/v1", api_key="ollama")
    # Assert: the rendered JSON line carries config but never the secret key.
    (call,) = factory.logger.calls
    rendered = json.loads(call.args[0])
    assert rendered["base_url"] == "http://localhost:11434/v1"
    assert "api_key" not in rendered
    assert "password" not in rendered
    assert "secret" not in rendered
    # Clean up: restore the default pipeline for later tests.
    configure_logging()


def test_redaction_processor_drops_secret_keys_directly() -> None:
    # Arrange: capture_logs bypasses the configured processor chain, so assert
    # the redaction processor itself here.
    from alphalab.obs.logging import _redact_secrets

    event_dict = {"api_key": "ollama", "token": "abc", "ticker": "MSFT"}
    # Act
    cleaned = _redact_secrets(None, "info", dict(event_dict))
    # Assert
    assert cleaned == {"ticker": "MSFT"}


def _sample_record(
    *,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    status: RunStatus = RunStatus.OK,
    error: str | None = "sentinel",
) -> RunRecord:
    """Build a RunRecord with sensible defaults for the tests."""
    started = started_at or datetime(2025, 7, 7, 22, 0, 0)
    finished = finished_at or (started + timedelta(seconds=1))
    return RunRecord(
        run_id="run-abc",
        run_key="daily:2025-07-07",
        cadence=Cadence.DAILY,
        started_at=started,
        finished_at=finished,
        status=status,
        produced="refreshed 42 tickers",
        error=None if status is RunStatus.OK else error,
    )


_ = structlog  # keep the import meaningful for type-checkers reading fixtures
