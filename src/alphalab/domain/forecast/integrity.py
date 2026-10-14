"""Separating genuine forecasts from backfilled ones.

The hole this closes
--------------------
Forward evaluation only means something if nobody knew the answer when the
claim was made. But nothing physically stops a prediction from being recorded
*today* about a knowledge date years in the past — and such a claim is not a
forecast, because its outcome was already sitting in the price history.

If those entries land in the same pool as genuine forecasts, the Brier score
and calibration curve stop measuring foresight and start measuring a mixture of
foresight and hindsight, with no way to tell the proportions. The whole point
of maintaining a dated journal would be lost, silently.

The distinction is recoverable from data already stored: a
:class:`~alphalab.domain.journal.Prediction` carries both ``knowledge_date``
(the information cutoff the research used) and ``recorded_at`` (when the claim
was committed). A genuine forecast has them close together. A backfilled one
has a long gap.

Backfilled predictions are not deleted — the journal is append-only by design,
and erasing them would be its own kind of dishonesty. They are *excluded from
scoring* and counted, so a reader can see exactly how many were set aside.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta

from alphalab.domain.journal import Prediction

__all__ = [
    "FORWARD_TOLERANCE",
    "is_forward",
    "partition_by_integrity",
]

#: How stale the information behind a claim may be and still count as a
#: forecast. Research legitimately runs on filings and prices that are a few
#: days old; a gap of weeks or more indicates the claim was reconstructed after
#: the fact.
FORWARD_TOLERANCE = timedelta(days=7)


def is_forward(prediction: Prediction, *, tolerance: timedelta = FORWARD_TOLERANCE) -> bool:
    """Whether ``prediction`` was committed close enough to its information cutoff.

    A claim recorded *before* its own knowledge date is also rejected: that
    ordering is impossible for honest research and signals a clock problem or
    a hand-edited record, either of which disqualifies it from scoring.
    """
    gap = prediction.recorded_at.date() - prediction.knowledge_date
    if gap < timedelta(0):
        return False
    return gap <= tolerance


def partition_by_integrity(
    predictions: Iterable[Prediction],
    *,
    tolerance: timedelta = FORWARD_TOLERANCE,
) -> tuple[tuple[Prediction, ...], tuple[Prediction, ...]]:
    """Split predictions into ``(forward, backfilled)``.

    Returning both halves rather than filtering in place is deliberate: the
    count of excluded entries belongs in the calibration report. A report that
    quietly dropped them would overstate how much evidence it rests on.
    """
    forward: list[Prediction] = []
    backfilled: list[Prediction] = []
    for prediction in predictions:
        target = forward if is_forward(prediction, tolerance=tolerance) else backfilled
        target.append(prediction)
    return tuple(forward), tuple(backfilled)
