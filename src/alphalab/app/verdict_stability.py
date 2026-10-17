"""Measuring whether the research pipeline produces judgment or coin flips.

Why this exists before anything trades
--------------------------------------
The system is about to let a local model decide a real portfolio in a paper
account. There is a measured reason to suspect its verdicts may be close to
arbitrary:

* ``build_request`` passes ``documents=("(no narrative documents supplied)",)``
  by default, so the model receives **no qualitative text at all**.
* ``_valuation_summary`` hands the debaters exactly four numbers:
  ``consensus_mid``, ``consensus_low``, ``consensus_high``, ``current_price``.
* With generic assumptions, AAPL produced DCF 88.0 / DDM 6.3 / P/E 49.0 against
  a price of 204 -- a 14x spread -- and the contract gate withheld the target.

If verdicts are coin flips on that input, building forty of them yields a
portfolio that is noise with extra steps. This module measures it cheaply, in
minutes, so the answer arrives before the plumbing is built rather than after.

Two dimensions, because one is not enough
-----------------------------------------
**Stability** asks whether the same input yields the same verdict. High
stability alone proves nothing: a pipeline that answers HOLD to everything is
perfectly stable and perfectly useless.

**Discrimination** asks whether the verdict varies across different names. A
system needs both -- repeatable *and* responsive to its input -- so both are
reported side by side and :attr:`StabilityReport.assessment` refuses to call a
pipeline sound on stability alone.

The chance baseline is computed, not guessed
--------------------------------------------
With three possible verdicts, agreement-with-the-modal-verdict under random
choice is **not** 1/3. Sampling concentration alone pushes it much higher:
enumerated exactly, it is 5/9 (0.5556) at five runs and 383/729 (0.5254) at
seven. Comparing a measured 0.6 against a naive 0.33 would declare success
where none exists, so :func:`chance_agreement` derives the real baseline by
exact enumeration.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from enum import StrEnum
from itertools import product
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from alphalab.artifact import EquityResearchArtifact, Verdict

__all__ = [
    "Assessment",
    "ResearchRunner",
    "RunOutcome",
    "StabilityReport",
    "TickerStability",
    "chance_agreement",
    "measure_verdict_stability",
]

#: Enumerate exactly below this many combinations; simulate above it.
_EXACT_LIMIT = 200_000

#: Samples used when the exact enumeration would be too large.
_SIMULATION_SAMPLES = 100_000

#: How far above the chance baseline mean agreement must sit to count as
#: stable. A pipeline barely above chance is not demonstrating judgment, and a
#: margin keeps a near-tie from being read as a pass.
_STABILITY_MARGIN = 0.15

#: Minimum share of ticker pairs that must disagree for the pipeline to count
#: as discriminating. Set low deliberately: the bar here is only "the verdict
#: responds to its input at all", not "the verdicts are correct".
_MIN_DISCRIMINATION = 0.1


class Assessment(StrEnum):
    """The honest verdict about the pipeline itself."""

    ARBITRARY = "arbitrary"
    """Agreement is near chance: the same input yields different answers."""

    UNDISCRIMINATING = "consistent but undiscriminating"
    """Stable, yet the answer barely changes across different companies."""

    DISCRIMINATING = "stable and discriminating"
    """Repeatable and responsive to its input. The weakest claim worth making."""

    INCONCLUSIVE = "inconclusive"
    """Too few completed runs to judge. Reported, never guessed around."""


class ResearchRunner(Protocol):
    """Runs the research pipeline once for a ticker."""

    def __call__(self, ticker: str) -> EquityResearchArtifact: ...


class RunOutcome(BaseModel):
    """One pipeline run: its verdict, or the failure that replaced it."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    verdict: Verdict | None = None
    withheld: bool = False
    seconds: float = Field(ge=0.0, default=0.0)
    error: str = ""

    @property
    def completed(self) -> bool:
        """Whether this run produced a verdict."""
        return self.verdict is not None


class TickerStability(BaseModel):
    """Repeated runs of one ticker, and what they agreed on."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    outcomes: tuple[RunOutcome, ...]

    @property
    def verdicts(self) -> tuple[Verdict, ...]:
        """Verdicts from the runs that completed."""
        return tuple(o.verdict for o in self.outcomes if o.verdict is not None)

    @property
    def modal_verdict(self) -> Verdict | None:
        """Most frequent verdict, ties broken by name for determinism."""
        verdicts = self.verdicts
        if not verdicts:
            return None
        counts = Counter(verdicts)
        best = max(counts.values())
        return min((v for v, c in counts.items() if c == best), key=lambda v: v.value)

    @property
    def agreement(self) -> float:
        """Share of completed runs matching the modal verdict."""
        verdicts = self.verdicts
        if not verdicts:
            return 0.0
        return Counter(verdicts).most_common(1)[0][1] / len(verdicts)

    @property
    def withheld_rate(self) -> float:
        """Share of completed runs where the gate suppressed the target."""
        completed = [o for o in self.outcomes if o.completed]
        if not completed:
            return 0.0
        return sum(1 for o in completed if o.withheld) / len(completed)

    @property
    def failures(self) -> int:
        """Runs that raised instead of answering."""
        return sum(1 for o in self.outcomes if not o.completed)

    @property
    def mean_seconds(self) -> float:
        """Average wall-clock seconds per completed run."""
        completed = [o for o in self.outcomes if o.completed]
        if not completed:
            return 0.0
        return sum(o.seconds for o in completed) / len(completed)


class StabilityReport(BaseModel):
    """The measurement, with its baseline attached so it can be judged."""

    model_config = ConfigDict(frozen=True)

    per_ticker: tuple[TickerStability, ...]
    runs: int = Field(gt=0)
    chance_baseline: float = Field(ge=0.0, le=1.0)

    @property
    def mean_agreement(self) -> float:
        """Average intra-ticker agreement across tickers that answered."""
        scored = [t for t in self.per_ticker if t.verdicts]
        if not scored:
            return 0.0
        return sum(t.agreement for t in scored) / len(scored)

    @property
    def modal_verdicts(self) -> tuple[Verdict, ...]:
        """Each ticker's modal verdict, for the discrimination check."""
        return tuple(t.modal_verdict for t in self.per_ticker if t.modal_verdict is not None)

    @property
    def discrimination(self) -> float:
        """Share of ticker pairs whose modal verdicts differ.

        Zero means the pipeline answered every company identically -- the
        failure mode that raw stability hides.
        """
        verdicts = self.modal_verdicts
        if len(verdicts) < 2:
            return 0.0
        pairs = [(a, b) for i, a in enumerate(verdicts) for b in verdicts[i + 1 :]]
        return sum(1 for a, b in pairs if a is not b) / len(pairs)

    @property
    def withheld_rate(self) -> float:
        """Share of completed runs, overall, where the gate withheld the target."""
        completed = [o for t in self.per_ticker for o in t.outcomes if o.completed]
        if not completed:
            return 0.0
        return sum(1 for o in completed if o.withheld) / len(completed)

    @property
    def abstention_rate(self) -> float:
        """Share of completed runs that answered HOLD.

        HOLD is a first-class abstention in this system, not a weak BUY, so it
        is counted separately rather than folded into the verdict mix.
        """
        completed = [o for t in self.per_ticker for o in t.outcomes if o.completed]
        if not completed:
            return 0.0
        return sum(1 for o in completed if o.verdict is Verdict.HOLD) / len(completed)

    @property
    def failures(self) -> int:
        """Total runs that raised."""
        return sum(t.failures for t in self.per_ticker)

    @property
    def assessment(self) -> Assessment:
        """Classify the pipeline, refusing to pass on stability alone."""
        if not self.modal_verdicts:
            return Assessment.INCONCLUSIVE
        if self.mean_agreement < self.chance_baseline + _STABILITY_MARGIN:
            return Assessment.ARBITRARY
        if len(self.per_ticker) >= 2 and self.discrimination < _MIN_DISCRIMINATION:
            return Assessment.UNDISCRIMINATING
        return Assessment.DISCRIMINATING


def chance_agreement(
    *,
    runs: int,
    categories: int = 3,
    rng: random.Random | None = None,
) -> float:
    """Expected agreement-with-the-modal-verdict under uniform random choice.

    This is the number a measured agreement must beat. It is far above
    ``1 / categories`` because concentration arises by chance: across five
    random choices from three options, the most common one already appears
    5/9 of the time (0.5556). Treating 1/3 as the baseline would mistake
    sampling noise for judgment.

    Computed by exact enumeration while the space is small enough, and by
    seeded simulation beyond that so the result stays reproducible.

    Raises:
        ValueError: if ``runs`` or ``categories`` is not positive.
    """
    if runs <= 0:
        raise ValueError("runs must be positive")
    if categories <= 0:
        raise ValueError("categories must be positive")
    if categories == 1:
        return 1.0
    if categories**runs <= _EXACT_LIMIT:
        totals = [
            max(Counter(draw).values()) / runs for draw in product(range(categories), repeat=runs)
        ]
        return sum(totals) / len(totals)
    source = rng or random.Random(0)
    sampled = [
        max(Counter(source.randrange(categories) for _ in range(runs)).values()) / runs
        for _ in range(_SIMULATION_SAMPLES)
    ]
    return sum(sampled) / len(sampled)


def measure_verdict_stability(
    *,
    tickers: Sequence[str],
    runs: int,
    research: ResearchRunner,
    monotonic: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] | None = None,
) -> StabilityReport:
    """Run each ticker ``runs`` times and report stability and discrimination.

    A run that raises is recorded with its exception type and message and the
    measurement continues: a single unreachable call must not discard the runs
    that completed, and a pattern of failures is itself a finding.

    Raises:
        ValueError: if ``runs`` is not positive or ``tickers`` is empty.
    """
    if runs <= 0:
        raise ValueError("runs must be positive")
    if not tickers:
        raise ValueError("at least one ticker is required")

    per_ticker = tuple(
        TickerStability(
            ticker=ticker,
            outcomes=tuple(
                _one_run(ticker, research, monotonic, progress, index=i + 1, total=runs)
                for i in range(runs)
            ),
        )
        for ticker in tickers
    )
    return StabilityReport(
        per_ticker=per_ticker,
        runs=runs,
        chance_baseline=chance_agreement(runs=runs, categories=len(Verdict)),
    )


def _one_run(
    ticker: str,
    research: ResearchRunner,
    monotonic: Callable[[], float],
    progress: Callable[[str], None] | None,
    *,
    index: int,
    total: int,
) -> RunOutcome:
    """Execute one run, turning any failure into a recorded outcome."""
    started = monotonic()
    try:
        artifact = research(ticker)
    except Exception as exc:  # one failure must not stop the measurement
        outcome = RunOutcome(
            ticker=ticker,
            seconds=monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    else:
        outcome = RunOutcome(
            ticker=ticker,
            verdict=artifact.verdict,
            withheld=artifact.valuation_withheld,
            seconds=monotonic() - started,
        )
    if progress is not None:
        label = outcome.verdict.value if outcome.verdict else f"FAILED ({outcome.error})"
        progress(f"  {ticker} run {index}/{total}: {label} [{outcome.seconds:.1f}s]")
    return outcome
