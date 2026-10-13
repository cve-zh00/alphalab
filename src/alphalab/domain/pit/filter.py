"""The reference point-in-time filter.

This pure function is the semantic contract every storage adapter must honour:
given records and a knowledge date, only records knowable at that date are
visible. The adapter pushes the same predicate into SQL, but this is the
authority the adapter is tested against.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from alphalab.domain.pit.records import KnowledgeDate, PITRecord


def visible_records(
    records: Iterable[PITRecord], knowledge_date: KnowledgeDate
) -> Iterator[PITRecord]:
    """Yield only records knowable on or before ``knowledge_date``.

    A record is visible when ``observed_at <= knowledge_date``. Anything
    observed later would be look-ahead and is withheld.
    """
    cutoff = knowledge_date.value
    for record in records:
        if record.observed_at <= cutoff:
            yield record
