"""Canonical, order-independent hashing of strategy specifications.

The trial count that feeds the Deflated Sharpe Ratio's ``SR0`` depends on
grouping *equivalent* specifications under a stable identity. Two specs that
differ only in the ordering of their parameter keys describe the *same*
strategy and must therefore hash identically; genuinely different specs must
not collide.

The subtlety lives in numeric normalisation. In Python ``True == 1`` and
``1 == 1.0``, and :func:`json.dumps` renders ``True`` as ``true`` and ``1.0``
as ``1.0``. If we serialised naively, ``{"a": 1}`` and ``{"a": True}`` could
collapse together while ``{"a": 1}`` and ``{"a": 1.0}`` would diverge in ways
that do not reflect strategy equivalence. To make the identity explicit and
testable we *type-tag* every scalar before serialising, so ``bool``, ``int``,
``float`` and ``str`` occupy disjoint value spaces.

Only the standard library is used (``hashlib`` and ``json``), keeping this
module inside the pure domain layer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["canonicalize", "spec_hash"]


def _tag_scalar(value: object) -> list[Any]:
    """Return a ``[type_tag, payload]`` pair for a scalar leaf.

    ``bool`` is checked before ``int`` because ``bool`` is a subclass of
    ``int`` in Python; without this order ``True`` would be tagged as an
    integer and collide with ``1``.
    """
    if value is None:
        return ["null", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        # ``repr`` round-trips floats exactly and normalises e.g. ``1e3`` to a
        # single canonical string, so equal floats always tag identically.
        # ``+ 0.0`` collapses the negative zero (``-0.0``) to ``0.0`` so two
        # values that compare equal (``-0.0 == 0.0``) also hash equal.
        return ["float", repr(value + 0.0)]
    if isinstance(value, str):
        return ["str", value]
    raise TypeError(f"unsupported scalar type for canonicalization: {type(value)!r}")


def _normalize(value: object) -> Any:
    """Recursively convert ``value`` into a canonical, JSON-safe structure.

    Mappings become ``["mapping", [[key, norm_value], ...]]`` with keys
    sorted, so key ordering cannot affect the output. Sequences keep their
    order (order is semantically meaningful) but are tagged so a list and a
    mapping never alias.
    """
    if isinstance(value, Mapping):
        keys = sorted(str(k) for k in value)
        return ["mapping", [[k, _normalize(value[k])] for k in keys]]
    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        seq: Sequence[Any] = value
        return ["sequence", [_normalize(item) for item in seq]]
    return _tag_scalar(value)


def canonicalize(spec: Mapping[str, Any]) -> str:
    """Serialise ``spec`` to a canonical JSON string (sorted, type-tagged)."""
    normalized = _normalize(spec)
    return json.dumps(normalized, separators=(",", ":"), ensure_ascii=True)


def spec_hash(spec: Mapping[str, Any]) -> str:
    """Return the SHA-256 hex digest of the canonical form of ``spec``."""
    canonical = canonicalize(spec)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
