"""The engine that starts the forward-evaluation clock.

The mathematics half of this project can be backtested; the agentic half
cannot. A local model's weights already contain the outcome of any historical
date, so a "prediction" about the past is an oracle, and scoring it would
contaminate the calibration record irreversibly — the journal is append-only
and trigger-guarded, so a poisoned row cannot be taken back. The only honest
evaluation is *forward*: dated claims recorded today, settled when their
horizon arrives, scored for calibration.

This module is that engine. It has two halves mirroring settle-then-reflect:

* :func:`register_forecast` turns a research artifact into a scoreable
  :class:`~alphalab.domain.journal.Prediction` recorded in the journal, or an
  :class:`~alphalab.domain.forecast.Abstention` recorded in the abstention
  store. It *refuses* to register a claim whose knowledge date the model could
  already have seen (the anti-leakage gate).
* :func:`settle_due_predictions` liquidates every prediction whose horizon has
  elapsed, pricing the realized and benchmark returns from real series and
  degrading honestly when a price is missing rather than inventing one.

Numbers come from code, prices from the series source; no LLM is consulted
here. This is an *app* layer use case: it composes the pure domain with the
I/O adapters, and all I/O stays behind those adapters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime

from alphalab.adapters.abstention_store import SqliteAbstentionStore
from alphalab.adapters.journal_sqlite import SqliteJournal
from alphalab.adapters.prices import PricePoint, PriceSeriesSource
from alphalab.artifact.models import EquityResearchArtifact, Verdict
from alphalab.domain.backtest import ProportionalCostModel
from alphalab.domain.contract.gate import conservative_c1b_upside
from alphalab.domain.forecast import Abstention, ForecastInputs, build_forecast
from alphalab.domain.journal import (
    Direction,
    Lesson,
    Prediction,
    Reflector,
    Settlement,
    TemplateReflector,
    due_predictions,
    overconfidence,
    settle_prediction,
)
from alphalab.domain.types import Ticker

__all__ = [
    "DEFAULT_BENCHMARK",
    "DEFAULT_PRICE_STALENESS_DAYS",
    "DEFAULT_ROUND_TRIP_TURNOVER",
    "LeakageError",
    "SettlementOutcome",
    "SettlementRun",
    "SkippedSettlement",
    "register_forecast",
    "settle_due_predictions",
]

#: Benchmark whose return is subtracted to form alpha. SPY tracks a broad US
#: index — the honest hurdle for a single-name directional call.
DEFAULT_BENCHMARK: Ticker = Ticker("SPY")

#: How many calendar days before the horizon a close price may be and still
#: count as "the price at the horizon". Markets close on weekends and holidays,
#: so the exact horizon date rarely has a print; a bounded look-back absorbs
#: that without silently reaching back weeks to whatever was last available.
#: A week comfortably spans any single market closure while refusing a stale
#: quote that would misstate the realized move.
DEFAULT_PRICE_STALENESS_DAYS = 7

#: Turnover charged on settlement. A directional thesis is entered and later
#: exited, so the position is traded twice — a full round trip is turnover 2.0
#: in fraction-of-equity space (one buy plus one sell), which the cost model
#: charges on both legs. Costing only one leg would understate the drag.
DEFAULT_ROUND_TRIP_TURNOVER = 2.0


class LeakageError(Exception):
    """Raised when a forecast's knowledge date is not strictly after the model cutoff.

    If the knowledge date is on or before the model's training cutoff, the
    weights may already encode what happened — the "forecast" is an oracle, not
    a prediction. Recording it would permanently contaminate the calibration
    history (the journal cannot be edited), so registration is refused before
    anything is written.
    """


def register_forecast(
    artifact: EquityResearchArtifact,
    *,
    horizon: date,
    recorded_at: datetime,
    journal: SqliteJournal,
    abstentions: SqliteAbstentionStore,
    llm_version: str | None = None,
    llm_cutoff: date | None = None,
    overconfidence_penalty: float | None = None,
) -> Prediction | Abstention:
    """Convert a research artifact into a persisted forecast or abstention.

    The verdict maps to a direction (``BUY → UP``, ``SELL → DOWN``,
    ``HOLD → None`` which abstains), the headline target becomes the implied
    upside, and the contract gate's dispersion and withholding flags carry
    through. :func:`~alphalab.domain.forecast.build_forecast` then either builds
    a scoreable :class:`Prediction` — recorded in the journal — or an
    :class:`Abstention` — recorded in the separate abstention store.

    Args:
        artifact: The gated research artifact to score forward.
        horizon: Settlement date; must fall after the knowledge date.
        recorded_at: Timestamp stamped on the prediction.
        journal: Append-only journal that receives any prediction.
        abstentions: Store that receives any abstention.
        llm_version: Version string of the model that produced the research.
        llm_cutoff: Training cutoff of that model; drives the anti-leakage gate.
        overconfidence_penalty: Optional shrinkage applied when calibration
            history shows positive overconfidence.

    Returns:
        The :class:`Prediction` or :class:`Abstention` that was persisted.

    Raises:
        LeakageError: If ``llm_cutoff`` is given and the artifact's knowledge
            date is on or before it (an oracle, not a forecast).
        ValueError: If the artifact has no knowledge date, or ``horizon`` is
            not after it (propagated from ``build_forecast``).
    """
    knowledge_date = _require_knowledge_date(artifact)
    _reject_leakage(knowledge_date, llm_cutoff)
    if horizon <= knowledge_date:
        raise ValueError("horizon must fall after the knowledge date")
    if not math.isfinite(artifact.current_price) or artifact.current_price <= 0.0:
        abstention = Abstention(
            ticker=artifact.ticker,
            knowledge_date=knowledge_date,
            reason="current price unavailable at knowledge date",
        )
        abstentions.record(abstention)
        return abstention

    effective_penalty = (
        overconfidence_penalty
        if overconfidence_penalty is not None
        else _derive_overconfidence_penalty(journal)
    )
    inputs = _artifact_to_inputs(artifact, knowledge_date)
    result = build_forecast(
        inputs,
        horizon=horizon,
        recorded_at=recorded_at,
        llm_version=llm_version,
        llm_cutoff=llm_cutoff,
        overconfidence_penalty=effective_penalty,
    )
    if isinstance(result, Abstention):
        abstentions.record(result)
    else:
        journal.record(result)
    return result


def _derive_overconfidence_penalty(journal: SqliteJournal) -> float:
    """Compute a damped overconfidence shrinkage from settled journal history."""
    settlements = {s.prediction_id: s for s in journal.list_settlements()}
    if len(settlements) < 10:
        return 0.0
    preds = {p.prediction_id: p for p in journal.list_predictions()}
    probs: list[float] = []
    outs: list[int] = []
    for pid, s in settlements.items():
        pred = preds.get(pid)
        if pred is not None:
            probs.append(pred.probability)
            outs.append(1 if s.outcome else 0)
    if len(probs) < 10:
        return 0.0
    signed_oc = overconfidence(probs, outs)
    if signed_oc <= 0.0:
        return 0.0
    damping = min(1.0, len(probs) / 30.0)
    return signed_oc * damping


def _require_knowledge_date(artifact: EquityResearchArtifact) -> date:
    """Return the artifact's knowledge date or fail loudly if absent.

    A forecast with no knowledge date cannot be checked for leakage nor dated,
    so it is rejected rather than defaulted — a silent default would be the
    exact temporal sloppiness this system exists to refuse.
    """
    if artifact.knowledge_date is None:
        raise ValueError("artifact has no knowledge_date; cannot date a forecast")
    return artifact.knowledge_date


def _reject_leakage(knowledge_date: date, llm_cutoff: date | None) -> None:
    """Refuse a knowledge date the model could already have seen in training."""
    if llm_cutoff is not None and knowledge_date <= llm_cutoff:
        raise LeakageError(
            f"knowledge_date {knowledge_date} is on or before the model cutoff "
            f"{llm_cutoff}: this is hindsight, not a forecast, and would "
            f"irreversibly contaminate the calibration record"
        )


def _artifact_to_inputs(artifact: EquityResearchArtifact, knowledge_date: date) -> ForecastInputs:
    """Project a gated artifact onto the primitive evidence the builder needs."""
    return ForecastInputs(
        ticker=artifact.ticker,
        knowledge_date=knowledge_date,
        current_price=artifact.current_price,
        direction=_verdict_to_direction(artifact.verdict),
        implied_upside=_implied_upside(artifact),
        relative_span=artifact.consensus_relative_span,
        valuation_withheld=artifact.valuation_withheld
        and conservative_c1b_upside(artifact) is None,
    )


def _verdict_to_direction(verdict: Verdict) -> Direction | None:
    """Map the directional verdict; ``HOLD`` has no claim and so abstains."""
    if verdict is Verdict.BUY:
        return Direction.UP
    if verdict is Verdict.SELL:
        return Direction.DOWN
    return None


def _implied_upside(artifact: EquityResearchArtifact) -> float | None:
    """Expected move as a fraction of price, or ``None`` when the target was withheld.

    A withheld valuation has no magnitude to form an upside from; returning
    ``None`` lets the probability mapping fall to the base rate rather than
    fabricating support the contract gate deliberately suppressed.
    """
    if artifact.valuation_withheld or artifact.price_target is None:
        # C1b-only degradation with directionally agreeing methods keeps a
        # conservative upside derived from the range; hard withholds do not.
        return conservative_c1b_upside(artifact)
    return artifact.price_target.value / artifact.current_price - 1.0


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettlementOutcome:
    """A prediction that was priced and settled, with its resulting record."""

    prediction: Prediction
    settlement: Settlement
    lesson: Lesson | None = None


@dataclass(frozen=True)
class SkippedSettlement:
    """A due prediction that could *not* be settled, with the honest reason."""

    prediction: Prediction
    reason: str


@dataclass(frozen=True)
class SettlementRun:
    """The deterministic result of one settlement pass.

    ``settled`` and ``skipped`` are reported side by side so a missing price is
    visible as a gap, never silently absorbed into the hit rate.
    """

    settled: tuple[SettlementOutcome, ...]
    skipped: tuple[SkippedSettlement, ...]
    lessons: tuple[Lesson, ...] = ()


def settle_due_predictions(
    *,
    journal: SqliteJournal,
    prices: PriceSeriesSource,
    as_of: date,
    settled_at: datetime,
    cost_model: ProportionalCostModel,
    benchmark: Ticker = DEFAULT_BENCHMARK,
    max_staleness_days: int = DEFAULT_PRICE_STALENESS_DAYS,
    reflector: Reflector | None = None,
) -> SettlementRun:
    """Settle every due, unsettled prediction from real price series.

    For each prediction whose horizon has elapsed, the realized return is taken
    from the stored entry price to the benchmark-aligned close at the horizon,
    and the benchmark return over the identical window. The boolean hit is
    decided by the domain (:func:`settle_prediction`): a bullish call wins on a
    positive realized move, a bearish call on a negative one.

    Honest degradation: if either the name or the benchmark lacks a close
    within ``max_staleness_days`` of the horizon, the prediction is *not*
    settled and is reported in ``skipped`` with the reason. No price is ever
    invented and no arbitrarily stale quote is used.

    Idempotency: already-settled ids are excluded by ``due_predictions``, so a
    second pass settles nothing and never collides with the journal's
    immutability triggers.
    """
    active_reflector = reflector if reflector is not None else TemplateReflector()
    benchmark_series = prices.series(benchmark)
    due = due_predictions(
        journal.list_predictions(), settled_ids=journal.settled_ids(), as_of=as_of
    )
    settled: list[SettlementOutcome] = []
    skipped: list[SkippedSettlement] = []
    lessons: list[Lesson] = []
    for prediction in due:
        outcome = _settle_one(
            prediction,
            prices=prices,
            benchmark_series=benchmark_series,
            benchmark=benchmark,
            settled_at=settled_at,
            as_of=as_of,
            cost_model=cost_model,
            max_staleness_days=max_staleness_days,
        )
        if isinstance(outcome, SkippedSettlement):
            skipped.append(outcome)
        else:
            journal.settle(outcome.settlement)
            lesson = active_reflector.reflect(
                outcome.prediction,
                outcome.settlement,
            )
            if journal.get_lesson(outcome.prediction.prediction_id) is None:
                journal.add_lesson(lesson)
            lessons.append(lesson)
            settled.append(
                SettlementOutcome(
                    prediction=outcome.prediction,
                    settlement=outcome.settlement,
                    lesson=lesson,
                )
            )
    return SettlementRun(
        settled=tuple(settled),
        skipped=tuple(skipped),
        lessons=tuple(lessons),
    )


def _settle_one(
    prediction: Prediction,
    *,
    prices: PriceSeriesSource,
    benchmark_series: list[PricePoint],
    benchmark: Ticker,
    settled_at: datetime,
    as_of: date,
    cost_model: ProportionalCostModel,
    max_staleness_days: int,
) -> SettlementOutcome | SkippedSettlement:
    """Price and settle a single prediction, or report why it could not be."""
    bench_entry = _close_on_or_before(
        benchmark_series, prediction.knowledge_date, max_staleness_days
    )
    bench_exit = _close_on_or_before(benchmark_series, prediction.horizon, max_staleness_days)
    if bench_entry is None or bench_exit is None:
        return SkippedSettlement(
            prediction=prediction,
            reason=f"no {benchmark} close within {max_staleness_days}d of the window",
        )

    name_series = prices.series(Ticker(prediction.ticker))
    name_exit = _close_on_or_before(name_series, prediction.horizon, max_staleness_days)
    if name_exit is None:
        return SkippedSettlement(
            prediction=prediction,
            reason=(
                f"no {prediction.ticker} close within {max_staleness_days}d of {prediction.horizon}"
            ),
        )

    realized_return = name_exit / prediction.entry_price - 1.0
    benchmark_return = bench_exit / bench_entry - 1.0
    cost_drag = cost_model.cost_fraction(turnover=DEFAULT_ROUND_TRIP_TURNOVER)
    settlement = settle_prediction(
        prediction,
        realized_return=realized_return,
        benchmark_return=benchmark_return,
        cost_drag=cost_drag,
        settled_at=settled_at,
        as_of=as_of,
    )
    return SettlementOutcome(prediction=prediction, settlement=settlement)


def _close_on_or_before(
    series: list[PricePoint], target: date, max_staleness_days: int
) -> float | None:
    """Return the close on ``target`` or the latest before it within tolerance.

    Picks the most recent point dated on or before ``target``; returns ``None``
    when none exists or the nearest is staler than ``max_staleness_days``. The
    tolerance is what distinguishes "the market was simply closed that day"
    from "we have no data anywhere near this date" — the latter must degrade,
    not guess.
    """
    candidates = [point for point in series if point.on <= target]
    if not candidates:
        return None
    nearest = max(candidates, key=lambda point: point.on)
    if (target - nearest.on).days > max_staleness_days:
        return None
    return nearest.close
