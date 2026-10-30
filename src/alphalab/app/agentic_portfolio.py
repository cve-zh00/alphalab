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
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from alphalab.artifact import EquityResearchArtifact, Verdict
from alphalab.domain.compute.sectors import (
    resolve_sector_profile,
    shrink_upside_empirical_bayes,
)
from alphalab.domain.contract.gate import conservative_c1b_upside
from alphalab.domain.types import Ticker

__all__ = [
    "DEFAULT_MAX_POSITIONS",
    "DEFAULT_MAX_SECTOR_WEIGHT",
    "MIN_EVALUATED_FRACTION",
    "AgenticPlan",
    "ResearchFunction",
    "SizingMode",
    "TickerOutcome",
    "build_agentic_targets",
    "conviction_volatility_targets",
    "equal_weight_targets",
    "resolve_sectors",
]

#: Default cap on the total weight of any one sector.
DEFAULT_MAX_SECTOR_WEIGHT = 0.25

#: Average pairwise correlation above which a name's score is penalized.
CORRELATION_PENALTY_THRESHOLD = 0.60
#: Max fractional deviation from the equal-weight base slot due to conviction/vol.
MAX_TILT = 0.25

SizingMode = Literal["equal", "conviction"]

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
    upside: float | None = None
    relative_span: float | None = None

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
    volatilities: Mapping[Ticker, float] | None = None,
    sizing_mode: SizingMode = "equal",
    max_position_weight: float = 0.10,
    correlations: Mapping[Ticker, float] | None = None,
    sectors: Mapping[Ticker, str] | None = None,
    max_sector_weight: float = DEFAULT_MAX_SECTOR_WEIGHT,
) -> AgenticPlan:
    """Research every candidate and compute portfolio weights for approvals.

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
    held = approved[:max_positions] if actionable else []
    if not held:
        targets: dict[Ticker, float] = {}
    elif sizing_mode == "conviction":
        outcome_by_ticker = {o.ticker: o for o in outcomes}
        targets = conviction_volatility_targets(
            held,
            slots=max_positions,
            outcomes=outcome_by_ticker,
            volatilities=volatilities,
            max_weight=max_position_weight,
            correlations=correlations,
            sectors=resolve_sectors(held, sectors),
            max_sector_weight=max_sector_weight,
        )
    else:
        weight = 1.0 / max_positions
        targets = {t: weight for t in held}

    if actionable and not held:
        plan_note = "pipeline approved no names; holding cash"

    return AgenticPlan(
        as_of=as_of,
        targets=targets,
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
    upside: float | None = None
    if (
        not artifact.valuation_withheld
        and artifact.price_target is not None
        and artifact.current_price > 0.0
    ):
        upside = artifact.price_target.value / artifact.current_price - 1.0
    detail = "valuation withheld by contract gate" if artifact.valuation_withheld else ""
    if upside is None:
        conservative = conservative_c1b_upside(artifact)
        if conservative is not None:
            upside = conservative
            detail = "conservative C1b target"
    return TickerOutcome(
        ticker=ticker,
        verdict=artifact.verdict,
        detail=detail,
        upside=upside,
        relative_span=artifact.consensus_relative_span,
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


def conviction_volatility_targets(
    tickers: Sequence[Ticker],
    *,
    slots: int,
    outcomes: Mapping[Ticker, TickerOutcome] | None = None,
    volatilities: Mapping[Ticker, float] | None = None,
    max_weight: float = 0.10,
    correlations: Mapping[Ticker, float] | None = None,
    sectors: Mapping[Ticker, str] | None = None,
    max_sector_weight: float = DEFAULT_MAX_SECTOR_WEIGHT,
) -> dict[Ticker, float]:
    """Weight approved names by ``conviction / sigma_60d``, bounded by ``max_weight``.

    * **Conviction** increases with dispersion-adjusted Margin of Safety
      ``min(max(upside, 0) / max(span, 0.15), 2.0)`` in ``[1.0, 2.0]`` (or
      ``0.75`` when valuation was withheld).
    * **Inverse volatility** scales down high-volatility names using 60-day
      annualized volatility clamped to ``[0.10, 0.80]``.
    * **Equal-weight base with a bounded tilt**: each approved name receives a
      base budget of ``min(max_weight, max(1 / slots, 0.075))`` scaled by its
      relative conviction/volatility score, with the tilt limited to
      ``+/- MAX_TILT`` (25%), clipped to ``[0.01, max_weight]`` and gross
      exposure capped at ``1.0``.
    """
    if slots <= 0:
        raise ValueError("slots must be positive")
    if max_weight <= 0.0 or max_weight > 1.0:
        raise ValueError("max_weight must be in (0, 1]")
    if max_sector_weight <= 0.0 or max_sector_weight > 1.0:
        raise ValueError("max_sector_weight must be in (0, 1]")
    held = list(dict.fromkeys(tickers))[:slots]
    if not held:
        return {}

    raw_scores: dict[Ticker, float] = {}
    for ticker in held:
        outcome = outcomes.get(ticker) if outcomes is not None else None
        if outcome is None or outcome.upside is None:
            conviction = 0.75 if (outcome and "withheld" in outcome.detail) else 1.0
        else:
            span = max(outcome.relative_span or 0.15, 0.15)
            shrunk_upside = shrink_upside_empirical_bayes(
                outcome.upside,
                relative_span=outcome.relative_span,
            )
            mos_ratio = min(max(0.0, shrunk_upside) / span, 2.0)
            conviction = 1.0 + 0.5 * mos_ratio
        raw_vol = volatilities.get(ticker, 0.25) if volatilities is not None else 0.25
        sigma = max(0.10, min(0.80, raw_vol if raw_vol > 0.0 else 0.25))
        score = conviction / sigma
        avg_corr = correlations.get(ticker) if correlations is not None else None
        if avg_corr is not None and avg_corr > CORRELATION_PENALTY_THRESHOLD:
            score *= max(0.5, 1.0 - 0.5 * (avg_corr - CORRELATION_PENALTY_THRESHOLD))
        raw_scores[ticker] = score

    mean_score = sum(raw_scores.values()) / len(held)
    base_slot = min(max_weight, max(1.0 / slots, 0.075))
    weights: dict[Ticker, float] = {}
    for ticker in held:
        rel = raw_scores[ticker] / mean_score if mean_score > 0.0 else 1.0
        rel = min(1.0 + MAX_TILT, max(1.0 - MAX_TILT, rel))
        w = min(max_weight, max(0.01, base_slot * rel))
        weights[ticker] = w

    total = sum(weights.values())
    if total > 1.0:
        scale = 1.0 / total
        weights = {t: w * scale for t, w in weights.items()}
    if sectors is not None:
        weights = _apply_sector_cap(weights, sectors, max_sector_weight)
    return weights


def resolve_sectors(
    tickers: Sequence[Ticker], sectors: Mapping[Ticker, str] | None
) -> dict[Ticker, str]:
    """Use supplied sectors, falling back to the static sector profile."""
    out: dict[Ticker, str] = {}
    for t in tickers:
        known = sectors.get(t) if sectors is not None else None
        if known:
            out[t] = known
            continue
        sector = str(resolve_sector_profile(t).sector.value)
        # Unclassified names are not one real sector; never cap them together.
        out[t] = "" if sector == "market_default" else sector
    return out


def _apply_sector_cap(
    weights: dict[Ticker, float], sectors: Mapping[Ticker, str], cap: float
) -> dict[Ticker, float]:
    """Scale each over-cap sector down to exactly ``cap`` (others untouched)."""
    totals: dict[str, float] = {}
    for t, w in weights.items():
        sector = sectors.get(t)
        if sector:
            totals[sector] = totals.get(sector, 0.0) + w
    out = dict(weights)
    for t, w in weights.items():
        sector = sectors.get(t)
        if sector and totals[sector] > cap:
            out[t] = w * cap / totals[sector]
    return out
