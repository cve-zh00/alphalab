"""The single sanctioned path for turning a stored datum into a feature value.

This module is the *structural* anti-leakage gate. Every feature value that a
strategy consumes must pass through :func:`resolve_feature`, and that function
raises :class:`LeakageError` the instant a datum is requested that was not yet
publicly knowable at the active :class:`KnowledgeDate`.

Why a raising gate and not just the silent :func:`_visible_records` filter?
Because the acid test needs look-ahead to be **inexpressible**, not merely
filtered away. A silent filter lets a careless caller *believe* it read a
value when in fact the row was withheld; a raising gate makes the illegal read
a loud, structural failure. The oracle strategy — "use tomorrow's return as
today's signal" — cannot be built through this path: the only way to obtain
tomorrow's datum is to ask for it at a knowledge date that predates its
``observed_at``, and that request raises.

The gate is pure (no I/O) and lives in the domain layer so it is independent of
any storage adapter. The availability lag is cross-checked against the feature
registry, so a feature with no audited lag cannot be resolved at all
(:class:`UnregisteredFeatureError`).
"""

from __future__ import annotations

from alphalab.domain.pit.errors import LeakageError
from alphalab.domain.pit.features import FeatureRegistry
from alphalab.domain.pit.records import KnowledgeDate, PITRecord

__all__ = ["resolve_feature"]


def resolve_feature(
    record: PITRecord,
    *,
    registry: FeatureRegistry,
    knowledge_date: KnowledgeDate,
) -> float:
    """Return ``record.value`` iff it was knowable at ``knowledge_date``.

    This is the only sanctioned way to turn a stored :class:`PITRecord` into a
    feature value a strategy may use. It enforces two invariants, in order:

    1. **The feature must be registered.** Its availability lag has to be
       declared in ``registry`` or the resolution raises
       :class:`UnregisteredFeatureError`. No feature can be used without an
       audited lag.
    2. **The datum must have been knowable.** ``record.observed_at`` must be on
       or before ``knowledge_date``. If it is later, surfacing the value would
       be look-ahead and the resolution raises :class:`LeakageError`.

    Args:
        record: The stored point-in-time observation being requested.
        registry: Feature registry carrying each feature's availability lag.
        knowledge_date: What is knowable "now"; the temporal cut-off.

    Returns:
        The as-filed numeric value, safe to feed to a strategy.

    Raises:
        UnregisteredFeatureError: If ``record.field`` is not registered.
        LeakageError: If ``record.observed_at`` is after ``knowledge_date``.
    """
    # Invariant 1: an unaudited feature is unusable (raises if missing).
    spec = registry.get(record.field)

    # Invariant 2: the datum must have been publicly knowable by now. The
    # registry's declared lag is the earliest a datum for this ``as_of`` could
    # ever be known; a record claiming to be knowable earlier is malformed.
    earliest = spec.earliest_knowable(record.as_of)
    if record.observed_at < earliest:
        raise LeakageError(
            f"feature {record.field!r} for as_of={record.as_of!r} claims "
            f"observed_at={record.observed_at!r}, before the registered "
            f"earliest-knowable date {earliest!r}: availability lag violated"
        )

    if record.observed_at > knowledge_date.value:
        raise LeakageError(
            f"feature {record.field!r} became knowable on "
            f"{record.observed_at!r}, after the knowledge date "
            f"{knowledge_date.value!r}: refusing to surface look-ahead data"
        )
    return record.value
