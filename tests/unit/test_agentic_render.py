"""Unit tests for agentic rendering and the pure compose helpers (no network)."""

from __future__ import annotations

import json
from datetime import date, datetime

from alphalab.app.agentic_portfolio import (
    AgenticPortfolio,
    EvaluationStatus,
    OmittedCandidate,
    RunProvenance,
    TickerVerdict,
)
from alphalab.artifact.models import Verdict
from alphalab.cli.agentic_compose import ASSUMPTIONS, latest_prices
from alphalab.cli.commands.agentic_render import (
    outcome_line,
    plan_json,
    render_plan,
    render_submission,
    run_json,
)
from alphalab.domain.backtest import PriceBar
from alphalab.domain.execution import OrderIntent
from alphalab.domain.forecast import Abstention
from alphalab.domain.journal import Direction, Prediction, new_prediction_id
from alphalab.domain.types import Side, Ticker
from alphalab.report.theme import Theme

_THEME = Theme(enabled=False)
_PROV = RunProvenance(
    llm_version="qwen3-vl",
    llm_cutoff=date(2024, 10, 1),
    knowledge_date=date(2025, 1, 2),
    model_name="qwen3-vl:latest",
)


def _prediction() -> Prediction:
    return Prediction(
        prediction_id=new_prediction_id(),
        recorded_at=datetime(2025, 1, 2, 12, 0, 0),
        ticker="AAA",
        direction=Direction.UP,
        probability=0.55,
        horizon=date(2025, 1, 23),
        entry_price=100.0,
        knowledge_date=date(2025, 1, 2),
    )


def _full_portfolio() -> AgenticPortfolio:
    return AgenticPortfolio(
        targets={Ticker("AAA"): 0.10},
        verdicts=(
            TickerVerdict(
                ticker=Ticker("AAA"),
                verdict=Verdict.BUY,
                weight=0.10,
                prediction=_prediction(),
                abstention=None,
            ),
            TickerVerdict(
                ticker=Ticker("BBB"),
                verdict=Verdict.HOLD,
                weight=0.0,
                prediction=None,
                abstention=Abstention(
                    ticker="BBB", knowledge_date=date(2025, 1, 2), reason="no claim"
                ),
            ),
        ),
        omitted=(
            OmittedCandidate(
                ticker=Ticker("CCC"), status=EvaluationStatus.FAILED, reason="EDGAR down"
            ),
        ),
        evaluated_count=2,
        min_evaluated=5,
        provenance=_PROV,
        sufficient=False,
    )


def _empty_portfolio() -> AgenticPortfolio:
    return AgenticPortfolio(
        targets={},
        verdicts=(),
        omitted=(),
        evaluated_count=5,
        min_evaluated=5,
        provenance=_PROV,
        sufficient=True,
    )


def test_render_plan_shows_targets_verdicts_and_omissions() -> None:
    text = render_plan(_full_portfolio(), 12.3, _THEME)
    assert "AAA" in text and "10.00%" in text
    assert "BUY" in text and "HOLD" in text
    assert "p=0.550" in text  # probability shown for the prediction
    assert "NOT representative" in text
    assert "CCC" in text and "EDGAR down" in text


def test_render_plan_empty_shows_all_cash() -> None:
    text = render_plan(_empty_portfolio(), 1.0, _THEME)
    assert "all cash" in text
    assert "representative" in text


def test_render_submission_dry_run_notice() -> None:
    assert "dry run" in render_submission(False, [], _THEME)


def test_render_submission_no_orders() -> None:
    text = render_submission(True, [], _THEME)
    assert "no orders" in text


def test_outcome_line_and_submission_rendering() -> None:
    from alphalab.adapters.alpaca_broker import OrderOutcome, OrderStatus

    intent = OrderIntent(ticker=Ticker("AAA"), side=Side.BUY, shares=10.0, reference_price=100.0)
    outcome = OrderOutcome(intent=intent, status=OrderStatus.SUBMITTED, client_order_id="x")
    line = outcome_line(outcome)
    assert "AAA" in line and "buy" in line and "submitted" in line
    assert "AAA" in render_submission(True, [outcome], _THEME)


def test_plan_json_round_trips() -> None:
    payload = json.loads(plan_json(_full_portfolio(), 9.9))
    assert payload["sufficient"] is False
    assert payload["evaluated_count"] == 2
    assert payload["targets"] == {"AAA": 0.10}
    assert payload["omitted"][0]["status"] == "failed"
    assert payload["verdicts"][0]["verdict"] == "BUY"


def test_run_json_includes_submission_fields() -> None:
    payload = json.loads(
        run_json(_full_portfolio(), 9.9, submitted=False, outcomes=[], note="refused")
    )
    assert payload["submitted"] is False
    assert payload["note"] == "refused"
    assert payload["outcomes"] == []


def test_assumptions_are_conservative_and_frozen() -> None:
    # The LLM never supplies these; they are explicit judgement inputs.
    assert ASSUMPTIONS.cost_of_equity == 0.09
    assert ASSUMPTIONS.peer_pe == (15.0, 18.0, 20.0)


def test_latest_prices_takes_most_recent_close() -> None:
    bars = {
        Ticker("AAA"): (
            PriceBar(on=date(2025, 1, 1), close=10.0, volume=1.0),
            PriceBar(on=date(2025, 1, 2), close=11.0, volume=1.0),
        ),
        Ticker("EMPTY"): (),
    }
    prices = latest_prices(bars)  # type: ignore[arg-type]
    assert prices == {Ticker("AAA"): 11.0}
