"""Structured logging for scheduled runs (structlog).

Every run gets a bound logger carrying its identity (run id, cadence, ticker)
and the model provenance (``llm_version`` / ``llm_cutoff``) so a single log
line answers *what ran, when, and under which model cutoff* — the temporal
hygiene the dossier (§6) makes load-bearing.

Secret safety is enforced, not hoped for: a processor drops any key whose name
looks sensitive (``api_key``, ``password``, ``token``, ``secret``) before the
entry is rendered. The Ollama ``api_key`` is a placeholder, but we refuse to
establish the habit of logging credential-shaped configuration.

No server is started (Langfuse is deferred). The instrumentation is structlog
only; the extension point for an OTel/Langfuse processor is a single entry in
the processor chain built by :func:`configure_logging`.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import structlog

from alphalab.domain.schedule import Cadence
from alphalab.obs.record import RunRecord, RunStatus

__all__ = [
    "SENSITIVE_KEYS",
    "bind_run_context",
    "configure_logging",
    "log_run_record",
]

#: Substrings that mark a log key as sensitive. Any event-dict key containing
#: one of these (case-insensitive) is dropped before rendering.
SENSITIVE_KEYS = frozenset({"api_key", "apikey", "password", "secret", "token"})


def _redact_secrets(
    _logger: object, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Drop any sensitive-looking key from the event dict.

    This is a defence-in-depth guard: callers should never bind a secret in
    the first place, but if one slips through it never reaches a sink.
    """
    for key in list(event_dict):
        lowered = key.lower()
        if any(marker in lowered for marker in SENSITIVE_KEYS):
            del event_dict[key]
    return event_dict


def configure_logging(
    *,
    logger_factory: structlog.typing.WrappedLogger | Any | None = None,
) -> None:
    """Configure structlog with a JSON-oriented, secret-redacting pipeline.

    Args:
        logger_factory: Optional factory override. Tests pass a
            ``CapturingLoggerFactory`` to assert on emitted entries; production
            leaves it ``None`` to use structlog's default stdlib logger.
    """
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact_secrets,
        structlog.processors.JSONRenderer(),
    ]
    structlog.configure(
        processors=processors,
        logger_factory=logger_factory or structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def bind_run_context(
    *,
    run_id: str,
    cadence: Cadence,
    ticker: str | None = None,
    llm_version: str | None = None,
    llm_cutoff: str | None = None,
) -> structlog.stdlib.BoundLogger:
    """Return a logger bound with the per-run context.

    Only non-None fields are bound, so an entry never carries empty metadata.
    The returned logger is the one a cadence should use for the whole run so
    every line shares the same run id and model provenance.
    """
    context: dict[str, Any] = {"run_id": run_id, "cadence": cadence.value}
    if ticker is not None:
        context["ticker"] = ticker
    if llm_version is not None:
        context["llm_version"] = llm_version
    if llm_cutoff is not None:
        context["llm_cutoff"] = llm_cutoff
    logger: structlog.stdlib.BoundLogger = structlog.get_logger()
    return logger.bind(**context)


def log_run_record(record: RunRecord) -> None:
    """Emit a :class:`RunRecord` as one structured log entry.

    OK runs log at info level; failures at error level with the reason. The
    duration is included so "how long did last night's run take?" is answerable
    from the log alone.
    """
    logger = structlog.get_logger().bind(
        run_id=record.run_id,
        run_key=record.run_key,
        cadence=record.cadence.value,
        status=record.status.value,
        duration_seconds=record.duration_seconds,
        produced=record.produced,
    )
    if record.status is RunStatus.OK:
        logger.info("run_completed")
    else:
        logger.error("run_failed", error=record.error)
