"""Tests for the verdict-stability harness.

The harness exists to answer a question before anything trades: does the
research pipeline produce judgment, or coin flips? These tests pin the four
behaviours that question depends on — perfect agreement, an alternating model,
an always-HOLD model, and a run that raises — plus the chance baseline the
measurement is judged against.

All offline: no network, no real LLM. The ``research`` seam is a Protocol, so
deterministic doubles drive the whole harness.
"""

from __future__ import annotations

from datetime import date
from itertools import cycle

import pytest

from alphalab.app.verdict_stability import (
    Assessment,
    chance_agreement,
    measure_verdict_stability,
)
from alphalab.artifact import EquityResearchArtifact, PriceTarget, Verdict

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


# -- chance baseline ------------------------------------------------------


def test_single_run_always_agrees_with_itself() -> None:
    """One run is trivially unanimous, so the baseline is 1.0."""
    assert chance_agreement(runs=1, categories=3) == pytest.approx(1.0)


def test_baseline_at_five_runs_is_five_ninths_not_one_third() -> None:
    """The naive 1/3 intuition is wrong and would flatter any measurement.

    Enumerated exactly over 3**5 draws, the expected modal share is 5/9. A
    pipeline measured at 0.55 agreement is therefore indistinguishable from
    random, even though 0.55 looks far above 0.33.
    """
    assert chance_agreement(runs=5, categories=3) == pytest.approx(5 / 9, abs=1e-12)


def test_baseline_falls_as_runs_grow() -> None:
    """More runs make accidental unanimity rarer, so the bar drops."""
    assert chance_agreement(runs=7, categories=3) < chance_agreement(runs=5, categories=3)


def test_single_category_is_certain() -> None:
    """With one possible answer, agreement is guaranteed and carries no signal."""
    assert chance_agreement(runs=4, categories=1) == pytest.approx(1.0)


def test_large_space_falls_back_to_seeded_simulation() -> None:
    """Beyond exact enumeration the estimate stays reproducible.

    Two calls with equal seeds must agree exactly; an unseeded estimate would
    make the baseline — and therefore the verdict about the pipeline — drift
    between runs.
    """
    import random

    first = chance_agreement(runs=40, categories=3, rng=random.Random(7))
    second = chance_agreement(runs=40, categories=3, rng=random.Random(7))
    assert first == second
    assert 0.0 < first < 1.0


def test_invalid_baseline_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="runs must be positive"):
        chance_agreement(runs=0)
    with pytest.raises(ValueError, match="categories must be positive"):
        chance_agreement(runs=3, categories=0)


# -- the four behaviours that matter --------------------------------------


def test_deterministic_pipeline_reports_full_agreement() -> None:
    """A pipeline that answers identically every time scores 1.0."""
    # Arrange — BUY for AAA, SELL for BBB: stable *and* discriminating.
    verdicts = {"AAA": Verdict.BUY, "BBB": Verdict.SELL}

    # Act
    report = measure_verdict_stability(
        tickers=["AAA", "BBB"],
        runs=5,
        research=lambda t: _artifact(t, verdicts[t]),
    )

    # Assert
    assert report.mean_agreement == pytest.approx(1.0)
    assert report.discrimination == pytest.approx(1.0)
    assert report.assessment is Assessment.DISCRIMINATING


def test_alternating_pipeline_is_called_arbitrary() -> None:
    """A model that flips its answer is detected, not averaged into a pass."""
    # Arrange — cycle through all three verdicts.
    source = cycle([Verdict.BUY, Verdict.HOLD, Verdict.SELL])

    # Act
    report = measure_verdict_stability(
        tickers=["AAA"],
        runs=6,
        research=lambda t: _artifact(t, next(source)),
    )

    # Assert
    assert report.mean_agreement < report.chance_baseline
    assert report.assessment is Assessment.ARBITRARY


def test_always_hold_is_stable_but_undiscriminating() -> None:
    """Perfect consistency with zero responsiveness is its own failure mode.

    This is the case raw stability hides: the pipeline never contradicts
    itself and never says anything either.
    """
    # Act
    report = measure_verdict_stability(
        tickers=["AAA", "BBB", "CCC"],
        runs=4,
        research=lambda t: _artifact(t, Verdict.HOLD),
    )

    # Assert
    assert report.mean_agreement == pytest.approx(1.0)
    assert report.discrimination == pytest.approx(0.0)
    assert report.abstention_rate == pytest.approx(1.0)
    assert report.assessment is Assessment.UNDISCRIMINATING


def test_a_failing_run_is_recorded_and_the_measurement_survives() -> None:
    """One unreachable call must not discard the runs that completed."""
    # Arrange — every third call raises.
    calls = {"n": 0}

    def research(ticker: str) -> EquityResearchArtifact:
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise TimeoutError("model did not respond")
        return _artifact(ticker, Verdict.BUY)

    # Act
    report = measure_verdict_stability(tickers=["AAA"], runs=6, research=research)

    # Assert
    assert report.failures == 2
    assert report.mean_agreement == pytest.approx(1.0)
    errors = [o.error for t in report.per_ticker for o in t.outcomes if o.error]
    assert all("TimeoutError" in e for e in errors)


def test_all_runs_failing_is_inconclusive_not_a_verdict() -> None:
    """Zero completed runs cannot support a claim about the pipeline."""

    def research(ticker: str) -> EquityResearchArtifact:
        raise ConnectionError("ollama unreachable")

    report = measure_verdict_stability(tickers=["AAA"], runs=3, research=research)

    assert report.assessment is Assessment.INCONCLUSIVE
    assert report.mean_agreement == pytest.approx(0.0)
    assert report.failures == 3


# -- reporting details ----------------------------------------------------


def test_withheld_rate_is_tracked_separately_from_the_verdict() -> None:
    """A withheld target means the probability falls to the base rate.

    The contract gate suppresses the price, never the judgment, so the verdict
    still counts — but the withholding has to be visible, because a portfolio
    of withheld targets carries no magnitude information at all.
    """
    report = measure_verdict_stability(
        tickers=["AAA"],
        runs=3,
        research=lambda t: _artifact(t, Verdict.BUY, withheld=True),
    )

    assert report.withheld_rate == pytest.approx(1.0)
    assert report.per_ticker[0].withheld_rate == pytest.approx(1.0)
    assert report.assessment is Assessment.DISCRIMINATING


def test_modal_verdict_ties_break_deterministically() -> None:
    """A tie must resolve the same way every run, or nothing is reproducible."""
    source = cycle([Verdict.BUY, Verdict.SELL])
    first = measure_verdict_stability(
        tickers=["AAA"], runs=2, research=lambda t: _artifact(t, next(source))
    )
    source2 = cycle([Verdict.SELL, Verdict.BUY])
    second = measure_verdict_stability(
        tickers=["AAA"], runs=2, research=lambda t: _artifact(t, next(source2))
    )

    assert first.per_ticker[0].modal_verdict is second.per_ticker[0].modal_verdict


def test_progress_is_emitted_per_run() -> None:
    """A twenty-minute cycle must show progress rather than appear hung."""
    seen: list[str] = []
    measure_verdict_stability(
        tickers=["AAA"],
        runs=3,
        research=lambda t: _artifact(t, Verdict.BUY),
        progress=seen.append,
    )

    assert len(seen) == 3
    assert "AAA run 1/3" in seen[0]


def test_timing_is_measured_from_the_injected_clock() -> None:
    """Durations come from an injected clock so tests stay deterministic."""
    ticks = iter([0.0, 2.5, 2.5, 4.0])
    report = measure_verdict_stability(
        tickers=["AAA"],
        runs=2,
        research=lambda t: _artifact(t, Verdict.BUY),
        monotonic=lambda: next(ticks),
    )

    assert report.per_ticker[0].mean_seconds == pytest.approx((2.5 + 1.5) / 2)


def test_invalid_measurement_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="runs must be positive"):
        measure_verdict_stability(
            tickers=["AAA"], runs=0, research=lambda t: _artifact(t, Verdict.BUY)
        )
    with pytest.raises(ValueError, match="at least one ticker"):
        measure_verdict_stability(tickers=[], runs=3, research=lambda t: _artifact(t, Verdict.BUY))
