"""Equity-research artifact model and its gated persistence boundary.

``store`` is exposed lazily (PEP 562 ``__getattr__``) so importing the pure
``models`` — which the domain contract layer depends on — never eagerly drags
in ``store`` and the contract package, which in turn imports the models. That
indirect cycle is avoided without weakening the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from alphalab.artifact.models import (
    EquityResearchArtifact,
    HasValueAndUnit,
    PriceRange,
    PriceTarget,
    Verdict,
)

if TYPE_CHECKING:
    from alphalab.artifact.store import ArtifactSink, InMemoryArtifactStore

__all__ = [
    "ArtifactSink",
    "EquityResearchArtifact",
    "HasValueAndUnit",
    "InMemoryArtifactStore",
    "PriceRange",
    "PriceTarget",
    "Verdict",
]

_LAZY = {"ArtifactSink", "InMemoryArtifactStore"}


def __getattr__(name: str) -> Any:
    """Lazily resolve the store symbols to avoid an import cycle."""
    if name in _LAZY:
        from alphalab.artifact import store

        return getattr(store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
