"""Immutable contracts for the Bull / Bear / Judge debate.

These are the structured shapes the LLM debaters and the arbiter must produce,
plus the auditable transcript that records the whole exchange. They mirror the
discipline of ``domain/analyst/contracts.py``:

* Debater outputs (:class:`BullCase`, :class:`BearCase`) carry **only text**
  plus a single bounded ``confidence`` in ``[0, 1]`` — there is deliberately
  **no** numeric valuation field, so a model cannot smuggle a price target
  through its structured output. The headline number is injected downstream
  from ``domain/compute``; it is *structurally* impossible to emit here.
* :class:`JudgeRuling` carries the directional :class:`Verdict`, prose
  reasoning, and *which arguments from each side weighed* — again no free
  numeric field.
* :class:`DebateTranscript` is the audit artefact: the ordered sequence of
  turns plus the ruling, from which "who said what, in what order" is fully
  reconstructable.

Everything is frozen and ``extra="forbid"`` so neither mutation nor an
unexpected numeric key can appear after construction.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from alphalab.artifact.models import Verdict
from alphalab.domain.debate.roles import DebateRole


class DebateTurn(BaseModel):
    """A single utterance in the debate, recorded for the audit transcript.

    Attributes:
        role: Which role spoke (BULL/BEAR/JUDGE).
        turn_number: Zero-based position in the debate sequence; strictly
            increasing across the transcript so order is reconstructable.
        argument: The prose argument the role made.
        evidence: Verbatim evidence snippets the role cited, in order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: DebateRole
    turn_number: int = Field(ge=0)
    argument: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()


class BullCase(BaseModel):
    """The bull debater's structured output — text only, no valuation number.

    Like :class:`~alphalab.domain.analyst.contracts.ThesisNarrative`, this type
    has no ``price_target`` / ``fair_value`` / numeric valuation slot of any
    kind. The only number is ``confidence``, hard-bounded to ``[0, 1]``, so the
    model has nowhere to put a magnitude.

    Attributes:
        argument: The bullish thesis, in prose.
        catalysts: Catalysts that could realise the upside, in prose.
        evidence: Verbatim supporting quotes the bull relied on.
        confidence: Bull's confidence in its own case, bounded to ``[0, 1]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    argument: str = Field(min_length=1)
    catalysts: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)


class BearCase(BaseModel):
    """The bear debater's structured output — text only, no valuation number.

    The mirror image of :class:`BullCase`: prose plus a bounded confidence, and
    no numeric valuation field whatsoever.

    Attributes:
        argument: The bearish thesis, in prose.
        risks: Risks that could sink the thesis, in prose.
        evidence: Verbatim supporting quotes the bear relied on.
        confidence: Bear's confidence in its own case, bounded to ``[0, 1]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    argument: str = Field(min_length=1)
    risks: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)


class JudgeRuling(BaseModel):
    """The arbiter's resolution of the debate — directional, no free numbers.

    The judge issues a directional :class:`Verdict` and explains it, naming
    which bull and which bear points carried weight. It has no numeric
    valuation field: the headline target is injected by the pipeline from
    ``domain/compute``, never by the arbiter.

    Attributes:
        verdict: The directional BUY/HOLD/SELL call.
        rationale: Prose explanation of the decision.
        bull_points_weighed: Bull arguments the judge found persuasive.
        bear_points_weighed: Bear arguments the judge found persuasive.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: Verdict
    rationale: str = Field(min_length=1)
    bull_points_weighed: tuple[str, ...] = ()
    bear_points_weighed: tuple[str, ...] = ()


class DebateTranscript(BaseModel):
    """The full, auditable record of one debate: ordered turns plus the ruling.

    This is the artefact that makes the debate reproducible and reviewable: the
    ``turns`` are in the exact order they were emitted and the ``ruling`` is the
    arbiter's resolution. Nothing here is mutable after construction.

    Attributes:
        ticker: The instrument the debate concerned.
        turns: The ordered sequence of debater turns (excludes the ruling).
        ruling: The arbiter's final resolution.
        max_rounds: The hard per-side round budget the debate ran under.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    turns: tuple[DebateTurn, ...]
    ruling: JudgeRuling
    max_rounds: int = Field(ge=1)
