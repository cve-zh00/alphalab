"""Tests for the agentic portfolio bridge."""

from __future__ import annotations

from datetime import date

import pytest

from alphalab.app.agentic_portfolio import (
    build_agentic_targets,
    equal_weight_targets,
)
from alphalab.artifact import EquityResearchArtifact, PriceTarget, Verdict
from alphalab.domain.types import Ticker

_AS_OF = date(2026, 10, 14)


def _artifact(ticker: str, verdict: Verdict, *, withheld: bool = False) -> EquityResearchArtifact:
    return EquityResearchArtifact(
        ticker=ticker,
        knowledge_date=_AS_OF,
        verdict=verdict,
        price_target=None if withheld else PriceTarget(value=120.0, unit="USD_per_share"),
        price_range=None,
        current_price=100.0,
        methods_used=("dcf",),
        method_values={},
        consensus_relative_span=0.1,
        narrative_slots={},
        fallback_slots={},
        contract_stamps=("[CONTRACT/x]",) if withheld else (),
        valuation_withheld=withheld,
        single_method_disclosed=False,
    )


def _tickers(*names: str) -> list[Ticker]:
    return [Ticker(n) for n in names]


def test_only_buys_receive_weight() -> None:
    """BUY earns a position; HOLD and SELL do not."""
    # Arrange
    verdicts = {"AAA": Verdict.BUY, "BBB": Verdict.HOLD, "CCC": Verdict.SELL}

    def research(ticker: Ticker) -> EquityResearchArtifact:
        return _artifact(ticker, verdicts[ticker])

    # Act
    plan = build_agentic_targets(
        as_of=_AS_OF,
        candidates=_tickers("AAA", "BBB", "CCC"),
        research=research,
        max_positions=3,
    )

    # Assert
    assert plan.approved == (Ticker("AAA"),)
    assert set(plan.targets) == {Ticker("AAA")}


def test_few_approvals_leave_cash_rather_than_concentrating() -> None:
    """Two approvals against twenty slots means 10% invested, not 100%.

    Concentrating into whichever names happened to clear the bar would turn a
    cautious result into an aggressive bet.
    """

    # Arrange
    def research(ticker: Ticker) -> EquityResearchArtifact:
        good = ticker in {Ticker("AAA"), Ticker("BBB")}
        return _artifact(ticker, Verdict.BUY if good else Verdict.HOLD)

    # Act
    plan = build_agentic_targets(
        as_of=_AS_OF,
        candidates=_tickers("AAA", "BBB", "CCC", "DDD"),
        research=research,
        max_positions=20,
    )

    # Assert
    assert len(plan.targets) == 2
    assert all(w == pytest.approx(1 / 20) for w in plan.targets.values())
    assert plan.cash_weight == pytest.approx(0.90)


def test_a_failing_ticker_does_not_abort_the_cycle() -> None:
    """One unreachable name must not discard the completed assessments."""

    # Arrange
    def research(ticker: Ticker) -> EquityResearchArtifact:
        if ticker == Ticker("BOOM"):
            raise TimeoutError("model did not respond")
        return _artifact(ticker, Verdict.BUY)

    # Act
    plan = build_agentic_targets(
        as_of=_AS_OF,
        candidates=_tickers("AAA", "BOOM", "CCC"),
        research=research,
        max_positions=5,
    )

    # Assert
    assert plan.approved == (Ticker("AAA"), Ticker("CCC"))
    boom = next(o for o in plan.outcomes if o.ticker == Ticker("BOOM"))
    assert "TimeoutError" in boom.detail
    assert plan.evaluated == 2


def test_partial_coverage_blocks_the_portfolio() -> None:
    """Below the coverage floor, no portfolio is proposed.

    A handful of survivors from a mostly failed run is an outage, not a
    decision, and acting on it would disguise one as the other.
    """

    # Arrange — four of five fail.
    def research(ticker: Ticker) -> EquityResearchArtifact:
        if ticker != Ticker("AAA"):
            raise RuntimeError("upstream unavailable")
        return _artifact(ticker, Verdict.BUY)

    # Act
    plan = build_agentic_targets(
        as_of=_AS_OF,
        candidates=_tickers("AAA", "BBB", "CCC", "DDD", "EEE"),
        research=research,
        max_positions=5,
    )

    # Assert
    assert not plan.actionable
    assert plan.targets == {}
    assert "partial coverage" in plan.note


def test_missing_data_is_skipped_not_invented() -> None:
    """A ticker the pipeline cannot assess is recorded, never assumed."""

    # Arrange
    def research(ticker: Ticker) -> EquityResearchArtifact | None:
        return None if ticker == Ticker("THIN") else _artifact(ticker, Verdict.BUY)

    # Act
    plan = build_agentic_targets(
        as_of=_AS_OF,
        candidates=_tickers("AAA", "THIN"),
        research=research,
        max_positions=5,
    )

    # Assert
    thin = next(o for o in plan.outcomes if o.ticker == Ticker("THIN"))
    assert thin.verdict is None
    assert "insufficient data" in thin.detail
    assert Ticker("THIN") not in plan.targets


def test_withheld_valuation_still_counts_as_a_verdict() -> None:
    """The contract gate suppresses the price, never the judgment."""

    # Arrange
    def research(ticker: Ticker) -> EquityResearchArtifact:
        return _artifact(ticker, Verdict.BUY, withheld=True)

    # Act
    plan = build_agentic_targets(
        as_of=_AS_OF,
        candidates=_tickers("AAA"),
        research=research,
        max_positions=5,
    )

    # Assert
    assert plan.approved == (Ticker("AAA"),)
    assert "withheld" in plan.outcomes[0].detail


def test_no_approvals_holds_cash() -> None:
    """Approving nothing is a valid outcome, reported as such."""

    # Arrange
    def research(ticker: Ticker) -> EquityResearchArtifact:
        return _artifact(ticker, Verdict.HOLD)

    # Act
    plan = build_agentic_targets(
        as_of=_AS_OF,
        candidates=_tickers("AAA", "BBB"),
        research=research,
        max_positions=5,
    )

    # Assert
    assert plan.targets == {}
    assert plan.actionable
    assert "approved no names" in plan.note
    assert plan.cash_weight == pytest.approx(1.0)


def test_equal_weight_divides_by_slots_not_by_approvals() -> None:
    """Three names in twenty slots get 5% each, not 33%."""
    # Act
    weights = equal_weight_targets(_tickers("A", "B", "C"), slots=20)

    # Assert
    assert all(w == pytest.approx(0.05) for w in weights.values())


def test_invalid_configuration_is_rejected() -> None:
    """Nonsensical sizing fails loudly."""
    with pytest.raises(ValueError, match="max_positions"):
        build_agentic_targets(as_of=_AS_OF, candidates=[], research=lambda t: None, max_positions=0)
    with pytest.raises(ValueError, match="slots"):
        equal_weight_targets([], slots=0)
