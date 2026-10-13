"""Trial ledger domain: pure types, canonical hashing, and errors.

This package stays inside the pure domain layer (stdlib + pydantic only); the
SQLite persistence lives in :mod:`alphalab.adapters.ledger_sqlite`.
"""

from __future__ import annotations

from alphalab.domain.ledger.errors import ImmutableLedgerError, LedgerError
from alphalab.domain.ledger.hashing import canonicalize, spec_hash
from alphalab.domain.ledger.record import TrialMetrics, TrialRecord, new_trial_id

__all__ = [
    "ImmutableLedgerError",
    "LedgerError",
    "TrialMetrics",
    "TrialRecord",
    "canonicalize",
    "new_trial_id",
    "spec_hash",
]
