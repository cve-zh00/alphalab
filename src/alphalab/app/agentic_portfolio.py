"""Turning LLM research verdicts into portfolio weights.

What this changes
-----------------
Until now the paper account traded ``rank_ensemble`` — pure price momentum,
with no language model involved in any decision. This module makes the agentic
layer the decision maker: the portfolio holds what the research pipeline says
to buy, and nothing else.

Design choices worth defending
------------------------------
**The candidate screen must not predict returns.** Candidates are narrowed by
liquidity alone. Screening on momentum first would leave momentum making the
investment decision while the model merely decorated it, and the result would
be indistinguishable from what the account already did.

**Equal weights, never probability-weighted.** The forecast probability is
uncalibrated — zero predictions have settled — so sizing by it would amplify a
belief nobody has measured. Revisit once the journal reports a conclusive
calibration.

**Fewer approvals than slots means cash.** If the pipeline approves three
names, three are bought and the rest stays in cash. A system that is always
fully invested is not expressing an opinion, it is filling a quota.

**A thin research protocol.** The pipeline arrives as a callable, so this
module can be tested without a model, a network or a broker. Wiring EDGAR,
Ollama and the contract gate together belongs in the composition root.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from alphalab.artifact import EquityResearchArtifact, Verdict
from alphalab.domain.types import Ticker

__all__ = [
    "DEFAULT_MAX_POSITIONS",
    "MIN_EVALUATED_FRACTION",
    "AgenticPlan",
    "ResearchFunction",
    "TickerOutcome",
    "build_agentic_targets",
]

#: Names held when the pipeline approves at least that many.
DEFAULT_MAX_POSITIONS = 20

#: Fraction of candidates that must be researched successfully before a
#: portfolio is proposed at all.
#:
#: A cycle makes dozens of model calls over many minutes, so some will fail.
#: Tolerating failures is necessary; ignoring how many failed is not. If most
#: candidates never got evaluated, the surviving handful is not a considered
#: portfolio — it is whichever names happened to answer, and acting on it would
#: dress up an outage as a decision.
MIN_EVALUATED_FRACTION = 0.5


class ResearchFunction(Protocol):
    """Researches one ticker, or returns ``None`` when it cannot be assessed."""

    def __call__(self, ticker: Ticker) -> EquityResearchArtifact | None: ...


class TickerOutcome(BaseModel):
    """What the pipeline concluded about one candidate."""

    model_config = ConfigDict(frozen=True)

    ticker: Ticker
    verdict: Verdict | None = None
    detail: str = ""

    @property
    def approved(self) -> bool:
        """Whether this name earned a place in the portfolio."""
        return self.verdict is Verdict.BUY


class AgenticPlan(BaseModel):
    """Target weights decided by the research pipeline, with its reasoning trail."""

    model_config = ConfigDict(frozen=True)

    as_of: date
    targets: Mapping[Ticker, float]
    outcomes: tuple[TickerOutcome, ...]
    candidates: int = Field(ge=0)
    evaluated: int = Field(ge=0)
    actionable: bool
    note: str = ""

    @property
    def approved(self) -> tuple[Ticker, ...]:
        """Tickers the pipeline said to buy."""
        return tuple(sorted(self.targets))

    @property
    def cash_weight(self) -> float:
        """Fraction left uninvested because too few names were approved."""
        return max(0.0, 1.0 - sum(self.targets.values()))


def build_agentic_targets(
    *,
    as_of: date,
    candidates: Sequence[Ticker],
    research: ResearchFunction,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    min_evaluated_fraction: float = MIN_EVALUATED_FRACTION,
) -> AgenticPlan:
    """Research every candidate and equal-weight the approvals.

    A candidate that raises is recorded and skipped rather than aborting the
    cycle: one unreachable filing should not discard twenty completed
    assessments. The exception type and message are kept in the outcome so a
    pattern of failures is visible rather than averaged away.

    Raises:
        ValueError: if ``max_positions`` is not positive.
    """
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")

    outcomes: list[TickerOutcome] = []
    approved: list[Ticker] = []
    evaluated = 0

    for ticker in candidates:
        outcome = _research_one(ticker, research)
        outcomes.append(outcome)
        if outcome.verdict is not None:
            evaluated += 1
        if outcome.approved:
            approved.append(ticker)

    plan_note, actionable = _coverage(
        candidates=len(candidates),
        evaluated=evaluated,
        min_fraction=min_evaluated_fraction,
    )
    held = sorted(approved)[:max_positions] if actionable else []
    weight = 1.0 / max_positions if held else 0.0

    if actionable and not held:
        plan_note = "pipeline approved no names; holding cash"

    return AgenticPlan(
        as_of=as_of,
        targets={t: weight for t in held},
        outcomes=tuple(outcomes),
        candidates=len(candidates),
        evaluated=evaluated,
        actionable=actionable,
        note=plan_note,
    )


def _research_one(ticker: Ticker, research: ResearchFunction) -> TickerOutcome:
    """Assess one ticker, converting any failure into a recorded outcome."""
    try:
        artifact = research(ticker)
    except Exception as exc:
        return TickerOutcome(ticker=ticker, detail=f"research failed: {type(exc).__name__}: {exc}")
    if artifact is None:
        return TickerOutcome(ticker=ticker, detail="insufficient data to assess")
    return TickerOutcome(
        ticker=ticker,
        verdict=artifact.verdict,
        detail="valuation withheld by contract gate" if artifact.valuation_withheld else "",
    )


def _coverage(*, candidates: int, evaluated: int, min_fraction: float) -> tuple[str, bool]:
    """Decide whether enough candidates were assessed to act on the result."""
    if candidates == 0:
        return "no candidates supplied", False
    covered = evaluated / candidates
    if covered < min_fraction:
        return (
            f"only {evaluated}/{candidates} candidates were assessed "
            f"({covered:.0%} < {min_fraction:.0%} required); not proposing a "
            "portfolio built on partial coverage",
            False,
        )
    return f"assessed {evaluated}/{candidates} candidates", True


def equal_weight_targets(tickers: Sequence[Ticker], *, slots: int) -> dict[Ticker, float]:
    """Equal weights over ``slots``, leaving the remainder in cash.

    Dividing by ``slots`` rather than by the number of approvals is deliberate:
    three approvals out of twenty slots means 15% invested, not 100% crammed
    into three names. Conviction is expressed by how many names clear the bar,
    not by concentrating into whichever few did.
    """
    if slots <= 0:
        raise ValueError("slots must be positive")
    held = sorted(tickers)[:slots]
    return {t: 1.0 / slots for t in held}
