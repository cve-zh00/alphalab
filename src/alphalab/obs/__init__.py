"""Observability for scheduled runs: structured logging + a run record.

Deliberately minimal (dossier §6 defers Langfuse): structlog is the only
instrumentation. The pieces:

* :class:`RunRecord` / :class:`RunStatus` — the persistable "what ran and what
  happened" record.
* :func:`configure_logging` — build the structlog pipeline (JSON, timestamps,
  secret redaction). The single extension point for a future OTel/Langfuse
  processor is one entry in that pipeline.
* :func:`bind_run_context` — a per-run bound logger carrying run id, cadence,
  ticker and model provenance.
* :func:`log_run_record` — emit a :class:`RunRecord` as one structured entry.

Secret safety is enforced by a redaction processor: credential-shaped keys
never reach a sink.
"""

from __future__ import annotations

from alphalab.obs.logging import (
    SENSITIVE_KEYS,
    bind_run_context,
    configure_logging,
    log_run_record,
)
from alphalab.obs.record import RunRecord, RunStatus

__all__ = [
    "SENSITIVE_KEYS",
    "RunRecord",
    "RunStatus",
    "bind_run_context",
    "configure_logging",
    "log_run_record",
]
