"""Building a scoreable prediction from a research verdict, or abstaining.

Why abstaining is a first-class outcome
---------------------------------------
Calibration is only meaningful over claims that were actually made. A system
that converts every verdict into a prediction — including ones it has no
directional view on, or where its own evidence contradicts its conclusion —
pollutes its track record with noise it was never confident about.

So this module returns either a :class:`~alphalab.domain.journal.Prediction` or
an :class:`Abstention` carrying the reason. Abstentions are recorded, not
discarded: "the system declined 40% of the time, for these reasons" is itself a
finding, and hiding it would make the hit rate look better than the process
deserves.

Two cases abstain:

* **No directional claim.** A HOLD is not a weak BUY. There is nothing to score.
* **Internal contradiction.** The verdict says buy while the valuation implies
  the name is expensive. That incoherence is a defect to surface, not an
  opinion to log, and a prediction built on it would score the wrong thing.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.forecast.probability import forecast_probability
from alphalab.domain.journal import Direction, Prediction, new_prediction_id

__all__ = ["Abstention", "ForecastInputs", "build_forecast"]


class ForecastInputs(BaseModel):
    """Evidence extracted from a research artifact, as primitives.

    Primitives rather than the artifact object keep this layer independent of
    the persistence and contract machinery, so the rule for turning research
    into a claim can be tested in isolation from how research is stored.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(min_length=1)
    knowledge_date: date
    current_price: float = Field(gt=0.0)
    direction: Direction | None
    implied_upside: float | None = None
    relative_span: float | None = Field(default=None, ge=0.0)
    valuation_withheld: bool = False


class Abstention(BaseModel):
    """A deliberate refusal to make a scoreable claim, with its reason."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    knowledge_date: date
    reason: str = Field(min_length=1)


def build_forecast(
    inputs: ForecastInputs,
    *,
    horizon: date,
    recorded_at: datetime,
    llm_version: str | None = None,
    llm_cutoff: date | None = None,
    overconfidence_penalty: float = 0.0,
) -> Prediction | Abstention:
    """Produce a dated, scoreable prediction, or an abstention explaining why not.

    ``llm_version`` and ``llm_cutoff`` are carried onto the prediction because
    a model's training cutoff determines whether a claim is a genuine forecast
    at all. A prediction dated before the cutoff is contaminated by hindsight
    the weights already contain, and recording the cutoff is what lets a later
    reader tell the two apart.

    Raises:
        ValueError: if ``horizon`` is not after ``knowledge_date``, which would
            mean settling a claim before it could resolve.
    """
    if horizon <= inputs.knowledge_date:
        raise ValueError("horizon must fall after the knowledge date")

    if inputs.direction is None:
        return Abstention(
            ticker=inputs.ticker,
            knowledge_date=inputs.knowledge_date,
            reason="verdict carries no directional claim",
        )

    conflict = _contradiction(inputs)
    if conflict is not None:
        return Abstention(
            ticker=inputs.ticker, knowledge_date=inputs.knowledge_date, reason=conflict
        )

    probability = forecast_probability(
        implied_upside=inputs.implied_upside,
        relative_span=inputs.relative_span,
        valuation_withheld=inputs.valuation_withheld,
        overconfidence_penalty=overconfidence_penalty,
    )
    return Prediction(
        prediction_id=new_prediction_id(),
        recorded_at=recorded_at,
        ticker=inputs.ticker,
        direction=inputs.direction,
        probability=probability,
        horizon=horizon,
        entry_price=inputs.current_price,
        knowledge_date=inputs.knowledge_date,
        llm_version=llm_version,
        llm_cutoff=llm_cutoff,
    )


def _contradiction(inputs: ForecastInputs) -> str | None:
    """Detect a verdict that disagrees with its own valuation evidence.

    Only checked when a figure survived the contract gate; a withheld
    valuation has no sign to contradict, and treating absent evidence as
    disagreement would abstain on every gated name.
    """
    if inputs.valuation_withheld or inputs.implied_upside is None:
        return None
    upside = inputs.implied_upside
    if inputs.direction is Direction.UP and upside < 0.0:
        return f"verdict is bullish but valuation implies {upside:.1%} downside"
    if inputs.direction is Direction.DOWN and upside > 0.0:
        return f"verdict is bearish but valuation implies {upside:.1%} upside"
    return None
