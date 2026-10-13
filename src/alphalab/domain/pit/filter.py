"""The reference point-in-time filter — intentionally private.

This pure function is the semantic contract every storage adapter must honour:
given records and a knowledge date, only records knowable at that date are
visible. The adapter pushes the same predicate into SQL, but this is the
authority the adapter is tested against.

It is named ``_visible_records`` (private) and is **not** re-exported from the
package on purpose. A silent filter is dangerous as a feature-access path: it
*drops* unknowable rows without a sound, so a careless caller can read a value
and believe it honoured the temporal cut when in fact a row was merely
withheld — and worse, building a feature straight from these rows skips the
two invariants that :func:`alphalab.domain.pit.resolve_feature` enforces
(the feature must be registered with an audited lag, and look-ahead raises
:class:`LeakageError` instead of vanishing silently). The sanctioned way to
obtain a feature value is ``resolve_feature``; this filter exists only to back
the adapter and its own tests. The architecture guard in
``tests/unit/test_architecture.py`` enforces this mechanically.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from alphalab.domain.pit.records import KnowledgeDate, PITRecord


def _visible_records(
    records: Iterable[PITRecord], knowledge_date: KnowledgeDate
) -> Iterator[PITRecord]:
    """Yield only records knowable on or before ``knowledge_date``.

    A record is visible when ``observed_at <= knowledge_date``. Anything
    observed later would be look-ahead and is withheld. This is deliberately
    private: feature values must be obtained through
    :func:`alphalab.domain.pit.resolve_feature`, which raises on look-ahead
    rather than silently dropping it.
    """
    cutoff = knowledge_date.value
    for record in records:
        if record.observed_at <= cutoff:
            yield record
