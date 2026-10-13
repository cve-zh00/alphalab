"""The deterministic *settle* half of settle-then-reflect.

A later run liquidates the predictions whose horizon has elapsed: it looks up
the realized return and the benchmark return over the holding window and
computes the realized **alpha, gross and net** of costs, plus the boolean hit.
No LLM is involved - this step is pure arithmetic and fully reproducible.

The *reflect* half (turning a settled prediction into a prose
:class:`~alphalab.domain.journal.types.Lesson`) is defined by the
:class:`~alphalab.domain.journal.reflect.Reflector` protocol so a Phase-3 LLM
can supply the text; a deterministic template reflector ships for testing.

Pure module: stdlib + numpy only. No I/O.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime

from alphalab.domain.journal.errors import NotYetDueError
from alphalab.domain.journal.types import Prediction, Settlement

__all__ = ["due_predictions", "settle_prediction"]


def due_predictions(
    predictions: Iterable[Prediction],
    settled_ids: Sequence[str],
    *,
    as_of: date,
) -> tuple[Prediction, ...]:
    """Return predictions that are **due and not yet settled** at ``as_of``.

    A prediction is due when its ``horizon`` is on or before ``as_of``. Already
    settled ids are excluded so the same thesis is never liquidated twice. The
    result preserves input order and is deterministic.
    """
    already = frozenset(settled_ids)
    return tuple(p for p in predictions if p.horizon <= as_of and p.prediction_id not in already)


def settle_prediction(
    prediction: Prediction,
    *,
    realized_return: float,
    benchmark_return: float,
    cost_drag: float,
    settled_at: datetime,
    as_of: date | None = None,
) -> Settlement:
    """Settle one due prediction into an immutable :class:`Settlement`.

    Alpha is reported both gross and net::

        alpha_gross = realized_return - benchmark_return
        alpha_net   = alpha_gross - cost_drag

    ``cost_drag`` is the total return drag of the position (a non-negative
    fraction) so the net alpha can never flatter the gross figure. The boolean
    ``outcome`` is whether the declared direction matched the realized move.

    Args:
        prediction: The due prediction to liquidate.
        realized_return: Realized return of the position over the window.
        benchmark_return: Benchmark return over the same window.
        cost_drag: Non-negative total cost drag (fraction) to deduct for net.
        settled_at: Timestamp stamped on the settlement.
        as_of: Optional settlement date; when given, the prediction's horizon
            must be on or before it or :class:`NotYetDueError` is raised.

    Raises:
        ValueError: If ``cost_drag`` is negative.
        NotYetDueError: If ``as_of`` is given and the horizon has not elapsed.
    """
    if cost_drag < 0.0:
        raise ValueError("cost_drag must be non-negative")
    if as_of is not None and prediction.horizon > as_of:
        raise NotYetDueError(
            f"prediction {prediction.prediction_id} is not due until {prediction.horizon}"
        )
    alpha_gross = realized_return - benchmark_return
    alpha_net = alpha_gross - cost_drag
    outcome = prediction.direction.hits(realized_return)
    return Settlement(
        prediction_id=prediction.prediction_id,
        settled_at=settled_at,
        realized_return=realized_return,
        benchmark_return=benchmark_return,
        alpha_gross=alpha_gross,
        alpha_net=alpha_net,
        outcome=outcome,
    )
