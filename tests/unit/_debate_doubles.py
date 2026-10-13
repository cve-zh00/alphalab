"""Deterministic test doubles implementing the debate domain ports.

These let the debate tests run with zero network and zero LLM latency while
exercising the full conductor, router, assembly, and gating logic. The doubles
also record the exact request they received, so context-isolation can be
asserted (the bull must never see the bear's private case), and one judge
double can be told to *attempt* to smuggle a fabricated price into its prose so
the "numbers come from code, never the model" rule can be proven end to end.
"""

from __future__ import annotations

from alphalab.artifact.models import Verdict
from alphalab.domain.debate.contracts import BearCase, BullCase, JudgeRuling
from alphalab.domain.debate.ports import BearRequest, BullRequest, JudgeRequest


class StubBull:
    """Returns a fixed bullish case and records every request it saw."""

    def __init__(self, *, confidence: float = 0.7) -> None:
        self._confidence = confidence
        self.requests: list[BullRequest] = []

    def argue(self, request: BullRequest) -> BullCase:
        self.requests.append(request)
        n = len(self.requests)
        return BullCase(
            argument=f"Bull argument #{n}: durable franchise, expanding margins.",
            catalysts="New product cycle and buyback.",
            evidence=("Revenue guidance raised.",),
            confidence=self._confidence,
        )


class StubBear:
    """Returns a fixed bearish case and records every request it saw."""

    def __init__(self, *, confidence: float = 0.6) -> None:
        self._confidence = confidence
        self.requests: list[BearRequest] = []

    def argue(self, request: BearRequest) -> BearCase:
        self.requests.append(request)
        n = len(self.requests)
        return BearCase(
            argument=f"Bear argument #{n}: regulatory overhang, margin pressure.",
            risks="Antitrust probe and slowing demand.",
            evidence=("New antitrust inquiry opened.",),
            confidence=self._confidence,
        )


class StubJudge:
    """Returns a fixed ruling; can be told to inject a fake number in prose.

    When ``injected_price`` is set, the double writes that fabricated figure
    into the rationale prose — simulating an arbiter that tries to smuggle a
    valuation number. The pipeline/gate must ensure that number never becomes
    the artifact's headline target.
    """

    def __init__(
        self,
        *,
        verdict: Verdict = Verdict.BUY,
        injected_price: float | None = None,
    ) -> None:
        self._verdict = verdict
        self._injected_price = injected_price
        self.requests: list[JudgeRequest] = []

    def rule(self, request: JudgeRequest) -> JudgeRuling:
        self.requests.append(request)
        rationale = "On balance the bull case on margins outweighs the bear's risks."
        if self._injected_price is not None:
            rationale = (
                f"We independently peg fair value at ${self._injected_price:.2f} "
                "per share, above the current price."
            )
        return JudgeRuling(
            verdict=self._verdict,
            rationale=rationale,
            bull_points_weighed=("durable franchise", "expanding margins"),
            bear_points_weighed=("regulatory overhang",),
        )
