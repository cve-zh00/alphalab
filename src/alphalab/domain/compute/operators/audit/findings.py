"""Structured findings produced by the number-vs-number audit checks.

An audit check never returns a naked ``bool``: a bare true/false is unusable
downstream because it carries no reason, no severity and no evidence. Instead
every check returns a frozen :class:`AuditFinding` describing *what* was
checked, *how bad* it is, the concrete values involved and a human-readable
message.

Severity contract (consumed by the Task 7 contract gate)
---------------------------------------------------------
* :attr:`Severity.HARD` — a structural impossibility or an irreconcilable
  mismatch. The gate must block / refuse to emit a point number.
* :attr:`Severity.SOFT` — a suspicious-but-survivable signal. The gate may
  warn, widen a range, or down-weight, but need not block.

A check that finds nothing wrong returns a finding with :attr:`passed` set to
``True`` and :attr:`severity` ``SOFT`` (an informational all-clear), so callers
can always introspect what ran. Only ``passed is False`` findings represent a
problem; use :meth:`AuditFinding.is_blocking` to detect a hard failure.

Pure module: no I/O, no mutation, no global state.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Severity(StrEnum):
    """How serious an audit finding is for the downstream gate."""

    HARD = "HARD"
    SOFT = "SOFT"


class AuditFinding(BaseModel):
    """Immutable result of a single number-vs-number coherence check.

    Attributes:
        check: Stable identifier of the check that produced this finding
            (e.g. ``"ev_bridge_check"``).
        passed: ``True`` when the check found no problem.
        severity: ``HARD`` or ``SOFT``; meaningful only when ``passed`` is
            ``False`` but always populated for uniform handling.
        message: Human-readable explanation of the outcome.
        values: The concrete numbers involved, keyed by role, for traceability
            (an immutable mapping rendered as a tuple of pairs).
    """

    model_config = ConfigDict(frozen=True)

    check: str = Field(min_length=1)
    passed: bool
    severity: Severity
    message: str = Field(min_length=1)
    values: tuple[tuple[str, float], ...] = ()

    def is_blocking(self) -> bool:
        """True iff this is a failed HARD finding (the gate must block)."""
        return (not self.passed) and self.severity is Severity.HARD


def ok(check: str, message: str, values: tuple[tuple[str, float], ...] = ()) -> AuditFinding:
    """Build an informational all-clear finding."""
    return AuditFinding(
        check=check,
        passed=True,
        severity=Severity.SOFT,
        message=message,
        values=values,
    )


def fail(
    check: str,
    severity: Severity,
    message: str,
    values: tuple[tuple[str, float], ...] = (),
) -> AuditFinding:
    """Build a failed finding with an explicit severity."""
    return AuditFinding(
        check=check,
        passed=False,
        severity=severity,
        message=message,
        values=values,
    )
