"""Unit tests for Empirical Bayes conviction sizing in :mod:`alphalab.app.agentic_portfolio`."""

from __future__ import annotations

from datetime import date

import pytest

from alphalab.app.agentic_portfolio import (
    MAX_TILT,
    TickerOutcome,
    build_agentic_targets,
    conviction_volatility_targets,
)
from alphalab.artifact import EquityResearchArtifact, PriceTarget, Verdict
from alphalab.domain.compute.sectors import shrink_upside_empirical_bayes
from alphalab.domain.types import Ticker

_AS_OF = date(2026, 10, 14)


def _tickers(*names: str) -> list[Ticker]:
    return [Ticker(n) for n in names]


def test_empirical_bayes_dampens_wide_span_outlier_in_conviction_targets() -> None:
    """A noisy extreme upside with wide span gets lower conviction than a tight-span consensus."""
    noisy = Ticker("NOISY")
    clean = Ticker("CLEAN")
    outcomes = {
        # Raw upside/span = 1.80 / 1.20 = 1.50 without EB shrinkage,
        # but EB shrinks 1.80 with span=1.20 down to ~0.351 -> mos_ratio ~ 0.293.
        noisy: TickerOutcome(
            ticker=noisy,
            verdict=Verdict.BUY,
            upside=1.80,
            relative_span=1.20,
        ),
        # Tight span: 0.35 upside shrinks only slightly to ~0.329 -> mos_ratio = 2.0.
        clean: TickerOutcome(
            ticker=clean,
            verdict=Verdict.BUY,
            upside=0.35,
            relative_span=0.15,
        ),
    }

    weights = conviction_volatility_targets(
        [noisy, clean],
        slots=20,
        outcomes=outcomes,
        volatilities={noisy: 0.25, clean: 0.25},
        max_weight=0.10,
    )

    assert weights[clean] > weights[noisy]


def test_conviction_targets_use_exact_eb_shrunk_mos_ratio() -> None:
    """Verify conviction_volatility_targets uses shrink_upside_empirical_bayes."""
    a = Ticker("AAA")
    b = Ticker("BBB")
    outcomes = {
        a: TickerOutcome(ticker=a, verdict=Verdict.BUY, upside=0.25, relative_span=0.25),
        b: TickerOutcome(ticker=b, verdict=Verdict.BUY, upside=0.05, relative_span=0.15),
    }
    weights = conviction_volatility_targets(
        [a, b],
        slots=20,
        outcomes=outcomes,
        volatilities={a: 0.25, b: 0.25},
        max_weight=0.10,
    )

    shrunk_a = shrink_upside_empirical_bayes(0.25, relative_span=0.25)
    shrunk_b = shrink_upside_empirical_bayes(0.05, relative_span=0.15)
    conv_a = 1.0 + 0.5 * min(max(0.0, shrunk_a) / 0.25, 2.0)
    conv_b = 1.0 + 0.5 * min(max(0.0, shrunk_b) / 0.15, 2.0)
    mean_conv = 0.5 * (conv_a + conv_b)
    base_slot = 0.075  # max(1/20, 0.075)
    rel_a = min(1.0 + MAX_TILT, max(1.0 - MAX_TILT, conv_a / mean_conv))
    rel_b = min(1.0 + MAX_TILT, max(1.0 - MAX_TILT, conv_b / mean_conv))

    assert weights[a] == pytest.approx(base_slot * rel_a)
    assert weights[b] == pytest.approx(base_slot * rel_b)


def test_conviction_bounds_and_sector_caps_preserved() -> None:
    """Existing bounds ([0.01, max_weight], MAX_TILT, and sector cap 0.25) remain intact."""
    tickers = _tickers("T1", "T2", "T3", "T4", "T5")
    outcomes = {
        t: TickerOutcome(ticker=t, verdict=Verdict.BUY, upside=0.40, relative_span=0.15)
        for t in tickers
    }
    sectors = {t: "information_technology" for t in tickers[:4]} | {tickers[4]: "financials"}
    weights = conviction_volatility_targets(
        tickers,
        slots=5,
        outcomes=outcomes,
        max_weight=0.20,
        sectors=sectors,
        max_sector_weight=0.25,
    )
    assert sum(weights[t] for t in tickers[:4]) == pytest.approx(0.25)
    assert all(0.01 <= w <= 0.20 for w in weights.values())


def test_build_agentic_targets_conviction_mode_with_eb_shrinkage() -> None:
    """End-to-end build_agentic_targets in conviction mode shrinks extreme noisy upsides."""

    def research(ticker: Ticker) -> EquityResearchArtifact:
        if ticker == Ticker("STEADY"):
            return EquityResearchArtifact(
                ticker="STEADY",
                knowledge_date=_AS_OF,
                verdict=Verdict.BUY,
                price_target=PriceTarget(value=135.0, unit="USD_per_share"),
                price_range=None,
                current_price=100.0,
                methods_used=("dcf", "multiples_pe"),
                method_values={},
                consensus_relative_span=0.15,
                narrative_slots={},
                fallback_slots={},
                contract_stamps=(),
                valuation_withheld=False,
                single_method_disclosed=False,
            )
        return EquityResearchArtifact(
            ticker="OUTLIER",
            knowledge_date=_AS_OF,
            verdict=Verdict.BUY,
            price_target=PriceTarget(value=260.0, unit="USD_per_share"),
            price_range=None,
            current_price=100.0,
            methods_used=("dcf", "multiples_pe"),
            method_values={},
            consensus_relative_span=1.30,
            narrative_slots={},
            fallback_slots={},
            contract_stamps=(),
            valuation_withheld=False,
            single_method_disclosed=False,
        )

    plan = build_agentic_targets(
        as_of=_AS_OF,
        candidates=_tickers("STEADY", "OUTLIER"),
        research=research,
        max_positions=20,
        sizing_mode="conviction",
        volatilities={Ticker("STEADY"): 0.20, Ticker("OUTLIER"): 0.20},
    )
    assert plan.targets[Ticker("STEADY")] > plan.targets[Ticker("OUTLIER")]
