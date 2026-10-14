"""Tests for the research-to-prediction bridge."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from alphalab.domain.forecast import (
    BASE_RATE,
    MAX_CONFIDENCE,
    Abstention,
    ForecastInputs,
    build_forecast,
    forecast_probability,
)
from alphalab.domain.journal import Direction, Prediction

_KD = date(2026, 10, 13)
_HORIZON = date(2027, 1, 13)
_NOW = datetime(2026, 10, 13, 22, 0, tzinfo=UTC)


def _inputs(**overrides: object) -> ForecastInputs:
    base: dict[str, object] = {
        "ticker": "AAPL",
        "knowledge_date": _KD,
        "current_price": 200.0,
        "direction": Direction.UP,
        "implied_upside": 0.20,
        "relative_span": 0.20,
        "valuation_withheld": False,
    }
    base.update(overrides)
    return ForecastInputs.model_validate(base)


def test_probability_is_bounded_however_strong_the_evidence() -> None:
    """Extreme evidence saturates at the ceiling rather than approaching certainty.

    An unbounded mapping would mint overconfidence that the journal would then
    spend months measuring. Excluding it by construction is cheaper.
    """
    # Act — a 500% implied upside with near-perfect method agreement.
    extreme = forecast_probability(
        implied_upside=5.0, relative_span=0.001, valuation_withheld=False
    )

    # Assert
    assert extreme <= MAX_CONFIDENCE
    assert extreme == pytest.approx(MAX_CONFIDENCE, abs=1e-6)


def test_probability_never_falls_below_the_base_rate() -> None:
    """Weak evidence collapses to even odds, not to pessimism."""
    # Act
    weak = forecast_probability(implied_upside=0.0001, relative_span=2.0, valuation_withheld=False)

    # Assert
    assert weak >= BASE_RATE
    assert weak == pytest.approx(BASE_RATE, abs=1e-3)


def test_withheld_valuation_yields_exactly_the_base_rate() -> None:
    """A gated figure means no quantitative support, hence no confidence.

    The verdict still stands and is still recorded — the project withholds the
    price, never the judgment — but it earns no better than a coin flip.
    """
    # Act
    gated = forecast_probability(implied_upside=0.80, relative_span=0.05, valuation_withheld=True)

    # Assert
    assert gated == BASE_RATE


def test_dispersion_tempers_identical_upside() -> None:
    """The same expected move counts for less when methods disagree."""
    # Act
    tight = forecast_probability(implied_upside=0.30, relative_span=0.10, valuation_withheld=False)
    wide = forecast_probability(implied_upside=0.30, relative_span=1.00, valuation_withheld=False)

    # Assert
    assert tight > wide


def test_missing_inputs_fall_back_to_the_base_rate() -> None:
    """Absent evidence is treated as no evidence, never as strong evidence."""
    assert (
        forecast_probability(implied_upside=None, relative_span=0.2, valuation_withheld=False)
        == BASE_RATE
    )
    assert (
        forecast_probability(implied_upside=0.2, relative_span=None, valuation_withheld=False)
        == BASE_RATE
    )


def test_bullish_verdict_builds_an_upward_prediction() -> None:
    """A BUY with supporting evidence becomes a dated, scoreable claim."""
    # Act
    result = build_forecast(
        _inputs(), horizon=_HORIZON, recorded_at=_NOW, llm_version="qwen3", llm_cutoff=_KD
    )

    # Assert
    assert isinstance(result, Prediction)
    assert result.direction is Direction.UP
    assert result.horizon == _HORIZON
    assert result.entry_price == 200.0
    assert BASE_RATE <= result.probability <= MAX_CONFIDENCE
    assert result.llm_version == "qwen3"
    assert result.llm_cutoff == _KD


def test_hold_abstains_rather_than_predicting_weakly() -> None:
    """A HOLD is not a weak BUY; there is nothing to score."""
    # Act
    result = build_forecast(_inputs(direction=None), horizon=_HORIZON, recorded_at=_NOW)

    # Assert
    assert isinstance(result, Abstention)
    assert "directional" in result.reason


def test_bullish_verdict_with_bearish_valuation_abstains() -> None:
    """Internal contradiction is surfaced, not logged as an opinion."""
    # Act
    result = build_forecast(
        _inputs(direction=Direction.UP, implied_upside=-0.35),
        horizon=_HORIZON,
        recorded_at=_NOW,
    )

    # Assert
    assert isinstance(result, Abstention)
    assert "bullish" in result.reason
    assert "downside" in result.reason


def test_bearish_verdict_with_bullish_valuation_abstains() -> None:
    """The contradiction check works in both directions."""
    # Act
    result = build_forecast(
        _inputs(direction=Direction.DOWN, implied_upside=0.42),
        horizon=_HORIZON,
        recorded_at=_NOW,
    )

    # Assert
    assert isinstance(result, Abstention)
    assert "bearish" in result.reason


def test_withheld_valuation_still_produces_a_prediction() -> None:
    """A gated name is scored at even odds, not dropped from the record.

    Dropping it would quietly exclude exactly the cases the contract gate
    flagged, flattering the track record.
    """
    # Act — upside sign would contradict, but it was withheld, so it is unused.
    result = build_forecast(
        _inputs(valuation_withheld=True, implied_upside=-0.5),
        horizon=_HORIZON,
        recorded_at=_NOW,
    )

    # Assert
    assert isinstance(result, Prediction)
    assert result.probability == BASE_RATE


def test_horizon_must_follow_the_knowledge_date() -> None:
    """A claim that settles before it is made is rejected outright."""
    with pytest.raises(ValueError, match="horizon"):
        build_forecast(_inputs(), horizon=_KD, recorded_at=_NOW)
