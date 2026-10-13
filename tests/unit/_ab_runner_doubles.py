"""Shared factories for the A/B runner tests (unit, property, acceptance).

Everything here is deterministic and LLM-free: configurable analyst/debate
pipelines built from the existing stub doubles, plus builders for
``AnalystRequest``, ``TradeCandidate`` and ``ExperimentDecision``. One builder
produces a candidate the skeptic *accepts*, another one it *vetoes*, so arm C's
divergence from arm B can be driven precisely.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from alphalab.app.ab_experiment import ExperimentDecision
from alphalab.app.debate_pipeline import DebatePipeline
from alphalab.app.inputs import AnalystRequest, ValuationBundle
from alphalab.app.pipeline import AnalystPipeline
from alphalab.artifact.models import Verdict
from alphalab.artifact.store import InMemoryArtifactStore
from alphalab.domain.analyst.contracts import RunMetadata
from alphalab.domain.analyst.tiers import ModelTier
from alphalab.domain.compute import ProvenancedValue, leaf
from alphalab.domain.skeptic import DataProvenance, TradeCandidate
from alphalab.domain.types import Side, Ticker
from tests.unit._analyst_doubles import StubExtractor, StubNarrator
from tests.unit._debate_doubles import StubBear, StubBull, StubJudge


def make_analyst_pipeline(*, verdict: Verdict = Verdict.BUY) -> AnalystPipeline:
    """A single-analyst pipeline whose narrator always returns ``verdict``."""
    return AnalystPipeline(
        extractor=StubExtractor(),
        narrator=StubNarrator(verdict=verdict),
        store=InMemoryArtifactStore(),
    )


def make_debate_pipeline(*, verdict: Verdict = Verdict.BUY) -> DebatePipeline:
    """A debate pipeline whose judge always rules ``verdict``."""
    return DebatePipeline(
        bull=StubBull(),
        bear=StubBear(),
        judge=StubJudge(verdict=verdict),
        store=InMemoryArtifactStore(),
    )


def make_request(*, knowledge_date: date, ticker: str = "SPY") -> AnalystRequest:
    """A minimal, valid pipeline request pinned to ``knowledge_date``."""
    base = leaf(value=100.0, name="dcf_input", unit="USD_per_share")
    estimate = ProvenancedValue(
        value=100.0,
        operator="dcf_value_per_share",
        operator_version="1.0.0",
        inputs=(("base", 100.0),),
        unit="USD_per_share",
        parents=(base,),
    )
    return AnalystRequest(
        ticker=ticker,
        knowledge_date=knowledge_date,
        current_price=95.0,
        documents=("A filing excerpt.",),
        valuation=ValuationBundle(estimates=(estimate,), unit="USD_per_share"),
        run_metadata=RunMetadata(
            model_name="double",
            tier=ModelTier.DEEP,
            llm_version="test-1",
            llm_cutoff=date(2000, 1, 1),
            knowledge_date=knowledge_date,
            dataset_hash="ds-1",
        ),
    )


def _accepting_returns(seed: int) -> tuple[float, ...]:
    """A long, strongly-edged, low-vol series the skeptic will accept."""
    rng = np.random.default_rng(seed)
    return tuple(float(x) for x in rng.normal(0.004, 0.004, size=300))


def make_accepting_candidate(*, seed: int = 0) -> TradeCandidate:
    """A candidate engineered to clear every skeptic criterion."""
    returns = _accepting_returns(seed)
    costs = tuple(0.00001 for _ in returns)
    return TradeCandidate(
        ticker=Ticker("SPY"),
        side=Side.BUY,
        proposed_size=10.0,
        horizon_days=5,
        backtest_returns=returns,
        cost_fractions=costs,
        n_trials=1,
        trial_sharpe_std=0.0,
        gross_edge_bps=50.0,
        adv=1_000_000.0,
        sigma_bps=50.0,
        provenance=DataProvenance(
            knowledge_date=date(2020, 1, 1),
            includes_delisted=True,
            pit_compliant=True,
            latest_observed_at=date(2019, 12, 31),
        ),
        per_slice_sharpe=(0.3, 0.3, 0.3, 0.3),
    )


def make_vetoed_candidate() -> TradeCandidate:
    """A candidate the skeptic vetoes (too few observations / no edge)."""
    return TradeCandidate(
        ticker=Ticker("SPY"),
        side=Side.BUY,
        proposed_size=10.0,
        horizon_days=5,
        backtest_returns=(0.0, 0.0, 0.0),
        cost_fractions=(0.01, 0.01, 0.01),
        n_trials=100,
        trial_sharpe_std=1.0,
        provenance=DataProvenance(
            knowledge_date=date(2020, 1, 1),
            includes_delisted=False,
            pit_compliant=True,
        ),
    )


def make_decision(
    *,
    knowledge_date: date,
    forward_return: float,
    cost_fraction: float = 0.0005,
    accept: bool = False,
    predicted_prob: float | None = None,
    seed: int = 0,
) -> ExperimentDecision:
    """Build one shared decision; ``accept`` picks the skeptic's verdict."""
    candidate = make_accepting_candidate(seed=seed) if accept else make_vetoed_candidate()
    return ExperimentDecision(
        request=make_request(knowledge_date=knowledge_date),
        forward_return=forward_return,
        cost_fraction=cost_fraction,
        candidate=candidate,
        predicted_prob=predicted_prob,
    )
