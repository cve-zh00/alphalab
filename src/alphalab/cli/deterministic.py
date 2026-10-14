"""Deterministic, offline stand-ins for the LLM providers (``--no-llm``).

The real analysis calls a 27B model that takes ~60s per request and needs
Ollama running. For a report the user can read *today* — with no wait and no
server — these template providers implement the exact same domain ports
(:class:`ExtractionProvider` / :class:`NarrationProvider` and the debate
ports) and return fixed, clearly-labelled prose.

They invent **no numbers**: the extraction schema has no numeric slot and the
narration carries only text plus a directional verdict. The verdict is derived
from the already-computed valuation summary (consensus vs. current price) so it
is deterministic and consistent with the code-calculated figures — the CLI is
orchestrating, not doing analysis.

This module lives in ``cli/`` (outside ``domain/``); it holds no business
logic beyond selecting a verdict label from numbers the compute layer produced.
"""

from __future__ import annotations

from alphalab.artifact.models import Verdict
from alphalab.domain.analyst.contracts import (
    ExtractionRequest,
    NarrationRequest,
    QualitativeFinding,
    ThesisNarrative,
)
from alphalab.domain.analyst.direction import Direction
from alphalab.domain.debate.contracts import BearCase, BullCase, JudgeRuling
from alphalab.domain.debate.ports import BearRequest, BullRequest, JudgeRequest

_TEMPLATE_SOURCE = "deterministic-template (--no-llm)"

#: Buy/sell band around parity. A consensus more than this fraction above the
#: current price is a BUY, more than this fraction below is a SELL, else HOLD.
#: This is a *presentation* threshold for the offline template only.
_VERDICT_BAND = 0.10


def _summary_lookup(summary: tuple[tuple[str, float], ...]) -> dict[str, float]:
    return dict(summary)


def verdict_from_summary(summary: tuple[tuple[str, float], ...]) -> Verdict:
    """Pick a deterministic verdict from the code-computed numbers.

    Uses ``consensus_mid`` vs ``current_price``; this reads *only* numbers the
    compute layer already produced and never invents a magnitude.
    """
    values = _summary_lookup(summary)
    mid = values.get("consensus_mid")
    current = values.get("current_price")
    if mid is None or current is None or current <= 0.0:
        return Verdict.HOLD
    ratio = (mid - current) / current
    if ratio > _VERDICT_BAND:
        return Verdict.BUY
    if ratio < -_VERDICT_BAND:
        return Verdict.SELL
    return Verdict.HOLD


class TemplateExtractionProvider:
    """Offline extractor returning fixed, clearly-labelled qualitative facts."""

    def extract(self, request: ExtractionRequest) -> tuple[QualitativeFinding, ...]:
        """Return template findings; invents no numbers."""
        return (
            QualitativeFinding(
                category="data_provenance",
                direction=Direction.NEUTRAL,
                evidence_quote=(
                    f"Offline template run for {request.ticker}; "
                    "no live qualitative extraction performed."
                ),
                source=_TEMPLATE_SOURCE,
                confidence=1.0,
            ),
        )


class TemplateNarrationProvider:
    """Offline narrator producing template prose and a numeric-derived verdict."""

    def narrate(self, request: NarrationRequest) -> ThesisNarrative:
        """Return a template narrative; the verdict follows the computed numbers."""
        verdict = verdict_from_summary(request.valuation_summary)
        return ThesisNarrative(
            thesis=(
                f"Template thesis for {request.ticker}: the directional call is derived "
                "from the code-calculated consensus relative to the current price. "
                "Run without --no-llm for an LLM-authored narrative."
            ),
            risks=(
                "Template run: qualitative risks are not analysed offline. "
                "The headline number is still fully code-calculated and traceable."
            ),
            catalysts=("Template run: catalysts are not analysed offline."),
            verdict=verdict,
        )


class TemplateBullProvider:
    """Offline bull debater producing template prose (--no-llm)."""

    def argue(self, request: BullRequest) -> BullCase:
        """Return a template bullish case; invents no numbers."""
        return BullCase(
            argument=(
                f"Template bull case for {request.ticker} (offline). "
                "Run without --no-llm for an LLM-authored argument."
            ),
            catalysts="Template run: catalysts not analysed offline.",
            evidence=(),
            confidence=0.5,
        )


class TemplateBearProvider:
    """Offline bear debater producing template prose (--no-llm)."""

    def argue(self, request: BearRequest) -> BearCase:
        """Return a template bearish case; invents no numbers."""
        return BearCase(
            argument=(
                f"Template bear case for {request.ticker} (offline). "
                "Run without --no-llm for an LLM-authored argument."
            ),
            risks="Template run: risks not analysed offline.",
            evidence=(),
            confidence=0.5,
        )


class TemplateJudgeProvider:
    """Offline arbiter; verdict follows the computed numbers (--no-llm)."""

    def rule(self, request: JudgeRequest) -> JudgeRuling:
        """Return a template ruling whose verdict follows the computed numbers."""
        verdict = verdict_from_summary(request.valuation_summary)
        return JudgeRuling(
            verdict=verdict,
            rationale=(
                f"Template debate ruling for {request.ticker}: verdict derived from the "
                "code-calculated consensus. Run without --no-llm for a real debate."
            ),
            bull_points_weighed=("template bull point",),
            bear_points_weighed=("template bear point",),
        )
