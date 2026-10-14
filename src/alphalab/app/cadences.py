"""The four cadences as use cases, with dependencies injected (dossier §6/§9).

Each cadence is a plain function that takes its collaborators as arguments —
nothing is constructed inside. That keeps the layer testable with doubles (no
LLM, no network, no clock) and keeps the cost policy legible in the signature:

* :func:`run_daily` — deterministic refresh only. It takes a ``refresh``
  callable and **no deep-LLM collaborator at all**, so by construction it
  cannot spend the trial budget. ``spent_trial_budget`` is always ``False``.
  Daily LLM analysis is noise mining and the biggest inflator of the trial
  count, which the DSR penalises; the type signature enforces that it does not
  happen here.
* :func:`run_weekly` — the deep-dive. It takes an ``analyse`` callable that the
  composition root wires to :class:`~alphalab.app.pipeline.AnalystPipeline`.
  The Phase-3 debate pipeline (built in parallel by another agent) is **not**
  imported here: the extension point is simply "pass a different ``analyse``".
* :func:`run_monthly` — calibration. It settles every due prediction with the
  pure journal functions (:func:`due_predictions` / :func:`settle_prediction`),
  persists each settlement through an injected sink, and produces a real
  :class:`~alphalab.domain.journal.CalibrationReport`.
* :func:`run_event` — ad-hoc analysis of a single name, event supplied as input.

The ``now`` parameter is threaded through for provenance only (the *decision*
to run belongs to ``domain.schedule.due_runs``); these functions assume the
caller already decided it is time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.journal import (
    CalibrationReport,
    Prediction,
    Settlement,
    calibration_report,
    due_predictions,
    settle_prediction,
)
from alphalab.domain.schedule import Cadence

__all__ = [
    "CadenceOutcome",
    "SettleFn",
    "run_daily",
    "run_event",
    "run_monthly",
    "run_weekly",
]

#: A function that refreshes deterministic data for a set of tickers and
#: returns how many were refreshed. No LLM is involved.
RefreshFn = Callable[[tuple[str, ...]], int]

#: A function that runs the agentic analysis for one ticker and returns a short
#: textual result (e.g. the verdict). The composition root wires this to the
#: analyst pipeline (and, later, the debate pipeline).
AnalyseFn = Callable[[str], str]

#: Given a due prediction, return ``(realized_return, benchmark_return,
#: cost_drag)`` for its holding window. This is the market-data boundary; it is
#: injected so the use case stays pure of I/O.
RealizedFn = Callable[[Prediction], tuple[float, float, float]]

#: A sink that persists a settlement (e.g. the SQLite journal's ``settle``).
PersistSettlementFn = Callable[[Settlement], None]

#: Runs one settlement pass and returns ``(settled, pending, unpriced)`` counts.
#: Injected so the daily cadence can *advance the forward clock* — liquidating
#: due predictions — without the use case ever touching the journal or prices
#: itself. Settlement is pure arithmetic over stored prices and spends no trial
#: budget, which is why it belongs in the deterministic daily slot.
SettleFn = Callable[[], tuple[int, int, int]]


class CadenceOutcome(BaseModel):
    """Immutable summary of running one cadence.

    Attributes:
        cadence: Which cadence ran.
        produced: Short human description of what it produced.
        spent_trial_budget: True iff the run invoked the agentic/deep path that
            adds to the trial count. The daily refresh is always ``False``.
        calibration: The calibration report, present only for a monthly run
            that had due predictions to settle.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    cadence: Cadence
    produced: str = Field(min_length=1)
    spent_trial_budget: bool
    calibration: CalibrationReport | None = None


def run_daily(
    *,
    tickers: tuple[str, ...],
    refresh: RefreshFn,
    now: datetime,
    settle: SettleFn | None = None,
) -> CadenceOutcome:
    """Run the daily deterministic refresh and (optionally) settle due theses.

    There is deliberately no LLM collaborator in this signature: the daily
    cadence updates data, screens and signal states only.

    Settlement is what makes the forward clock *advance* — a dated thesis whose
    horizon has passed becomes a scored outcome — and it is pure arithmetic
    over already-stored prices, so it spends no trial budget and belongs in the
    deterministic daily slot. It is injected (and optional) to keep this use
    case free of the journal and price I/O; when wired, the settled/pending/
    unpriced counts are reported alongside the refresh.
    """
    count = refresh(tickers)
    _ = now  # provenance only; the run decision belongs to due_runs.
    produced = f"refreshed {count} tickers deterministically (no LLM)"
    if settle is not None:
        settled, pending, unpriced = settle()
        produced += f"; settled {settled} due predictions ({pending} pending, {unpriced} unpriced)"
    return CadenceOutcome(
        cadence=Cadence.DAILY,
        produced=produced,
        spent_trial_budget=False,
    )


def run_weekly(*, tickers: tuple[str, ...], analyse: AnalyseFn, now: datetime) -> CadenceOutcome:
    """Run the weekly deep-dive over a configurable list of names.

    ``analyse`` is wired at the composition root to *investigate and register a
    dated forecast* for each ticker — i.e. run the research pipeline and hand
    the artifact to ``register_forecast``, returning a short status (registered
    with its probability, or abstained with its reason). The debate pipeline is
    a drop-in replacement for this callable and is not imported here. The
    tickers are passed in, so the weekly watchlist is configurable by the
    caller rather than hard-coded in the use case.
    """
    results = [analyse(ticker) for ticker in tickers]
    _ = now
    return CadenceOutcome(
        cadence=Cadence.WEEKLY,
        produced=f"deep-dive on {len(results)} names: {', '.join(tickers)}",
        spent_trial_budget=True,
    )


def run_monthly(
    *,
    predictions: Sequence[Prediction],
    settled_ids: Sequence[str],
    realized_for: RealizedFn,
    persist_settlement: PersistSettlementFn,
    as_of: date,
    now: datetime,
) -> CadenceOutcome:
    """Run the monthly calibration: settle due theses, then score calibration.

    Liquidates every due, unsettled prediction with the pure journal functions,
    persists each settlement through the injected sink, and assembles a
    :class:`CalibrationReport` from the realized outcomes. When nothing is due,
    the report is ``None`` and the outcome says so.
    """
    due = due_predictions(predictions, settled_ids, as_of=as_of)
    if not due:
        return CadenceOutcome(
            cadence=Cadence.MONTHLY,
            produced="calibration: 0 predictions due to settle",
            spent_trial_budget=False,
        )

    probabilities: list[float] = []
    outcomes: list[int] = []
    for prediction in due:
        realized, benchmark, cost_drag = realized_for(prediction)
        settlement = settle_prediction(
            prediction,
            realized_return=realized,
            benchmark_return=benchmark,
            cost_drag=cost_drag,
            settled_at=now,
            as_of=as_of,
        )
        persist_settlement(settlement)
        probabilities.append(prediction.probability)
        outcomes.append(1 if settlement.outcome else 0)

    report = calibration_report(probabilities, outcomes)
    return CadenceOutcome(
        cadence=Cadence.MONTHLY,
        produced=(
            f"calibration: settled {len(due)} predictions; "
            f"brier={report.brier:.4f} ece={report.ece:.4f} "
            f"conclusive={report.conclusive}"
        ),
        spent_trial_budget=False,
        calibration=report,
    )


def run_event(*, ticker: str, event_kind: str, analyse: AnalyseFn, now: datetime) -> CadenceOutcome:
    """Run an ad-hoc, event-driven analysis of a single name.

    The event (its kind and target ticker) is supplied as input; this cadence
    is never calendar-driven.
    """
    result = analyse(ticker)
    _ = (now, result)
    return CadenceOutcome(
        cadence=Cadence.EVENT,
        produced=f"event-driven analysis of {ticker} ({event_kind})",
        spent_trial_budget=True,
    )
