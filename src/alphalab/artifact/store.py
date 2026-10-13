"""The single persistence boundary for research artifacts.

Every artifact must pass the output contract gate before it is stored. The API
is designed so that persisting a *raw* artifact is impossible: there is no
public method that writes without first calling
:func:`alphalab.domain.contract.enforce_contract`. The only way in is
:meth:`ArtifactStore.save`, which enforces the contract, stores the enforced
(possibly degraded) artifact, and returns it. The internal buffer is private
and never exposed for direct mutation.

This module lives under ``alphalab.artifact`` (outside the pure ``domain``
layer), so it *could* perform I/O — but an in-memory buffer is sufficient and
more testable for this phase, so it does none.
"""

from __future__ import annotations

from typing import Protocol

from alphalab.artifact.models import EquityResearchArtifact
from alphalab.domain.contract import GateResult, enforce_contract


class ArtifactSink(Protocol):
    """The only persistence contract exposed to callers: a gated ``save``."""

    def save(self, artifact: EquityResearchArtifact) -> EquityResearchArtifact: ...


class InMemoryArtifactStore:
    """An in-memory artifact store whose ``save`` always runs the gate.

    There is deliberately no ``save_raw`` or public buffer: the gate cannot be
    bypassed by a well- or ill-intentioned caller. ``save`` returns the
    enforced artifact so callers persist exactly what was certified.
    """

    __slots__ = ("_buffer", "_findings")

    def __init__(self) -> None:
        # Private buffer; no public accessor returns a mutable handle to it.
        self._buffer: dict[str, EquityResearchArtifact] = {}
        self._findings: dict[str, GateResult] = {}

    def save(self, artifact: EquityResearchArtifact) -> EquityResearchArtifact:
        """Enforce the contract, store the enforced artifact, and return it.

        This is the ONLY way to persist. The raw ``artifact`` is never stored;
        only the gate's output reaches the buffer.
        """
        result = enforce_contract(artifact)
        enforced = result.artifact
        key = self._key(enforced)
        self._buffer[key] = enforced
        self._findings[key] = result
        return enforced

    def get(self, ticker: str, *, knowledge_date: object = None) -> EquityResearchArtifact | None:
        """Fetch a previously saved (already-enforced) artifact, or None."""
        return self._buffer.get(self._compose_key(ticker, knowledge_date))

    def findings_for(self, artifact: EquityResearchArtifact) -> GateResult | None:
        """Return the gate result recorded for a stored artifact, if any."""
        return self._findings.get(self._key(artifact))

    def __len__(self) -> int:
        return len(self._buffer)

    @staticmethod
    def _key(artifact: EquityResearchArtifact) -> str:
        return InMemoryArtifactStore._compose_key(artifact.ticker, artifact.knowledge_date)

    @staticmethod
    def _compose_key(ticker: str, knowledge_date: object) -> str:
        return f"{ticker}@{knowledge_date}"
