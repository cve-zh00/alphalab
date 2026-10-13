"""Domain exceptions for the trial ledger."""

from __future__ import annotations

__all__ = ["ImmutableLedgerError", "LedgerError"]


class LedgerError(Exception):
    """Base class for all trial-ledger domain errors."""


class ImmutableLedgerError(LedgerError):
    """Raised when something attempts to mutate or delete a trial record.

    The ledger is append-only by design: every backtest trial, including the
    ones an autonomous LLM agent spawns in a loop, must stay on the record so
    the Deflated Sharpe Ratio sees an honest trial count. Mutation or deletion
    would let the trial count drift below the truth and certify noise.
    """
