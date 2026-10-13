"""Immutable contracts the LLM must produce, and the run metadata.

The models here are the structured output *schemas* for the two LLM steps.
They are frozen and carefully shaped so a valuation number cannot slip out of
the model:

* :class:`QualitativeFinding` carries text and a single bounded
  ``confidence`` in ``[0, 1]`` — there is no free-form numeric field for a
  price, ratio, or multiple.
* :class:`ThesisNarrative` carries only text slots and a directional verdict.
  It has **no price-target field at all**; the headline number is injected by
  the pipeline from ``domain/compute``. The LLM therefore cannot emit it
  because the shape it fills has nowhere to put it.
* :class:`RunMetadata` makes the model identity, its knowledge cutoff, the PIT
  knowledge date and the dataset hash *mandatory* — construction fails without
  them (dossier §6: a silently-drifting frontier model destroys reproducibility).
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from alphalab.artifact.models import Verdict
from alphalab.domain.analyst.direction import Direction
from alphalab.domain.analyst.tiers import ModelTier


class QualitativeFinding(BaseModel):
    """A single qualitative fact extracted from source text by the LLM.

    Deliberately carries *no* free numeric valuation field. The only number is
    ``confidence``, hard-bounded to ``[0, 1]`` by the schema, so the model
    cannot smuggle a price or multiple through a numeric slot.

    Attributes:
        category: The kind of fact (e.g. ``"regulatory_risk"``,
            ``"competitive_shift"``, ``"management_guidance"``).
        direction: Whether the fact helps, hurts, or is neutral to the thesis.
        evidence_quote: A verbatim quote from the source backing the finding.
        source: Provenance of the quote (e.g. ``"10-K FY2023"``, ``"Reuters"``).
        confidence: Model confidence in the extraction, bounded to ``[0, 1]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: str = Field(min_length=1)
    direction: Direction
    evidence_quote: str = Field(min_length=1)
    source: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class ThesisNarrative(BaseModel):
    """The synthesised narrative the deep-think model produces.

    Only text slots and a directional verdict. There is intentionally **no**
    ``price_target`` / ``value`` / ``fair_value`` field: the headline number is
    computed by ``domain/compute`` and injected by the pipeline, never emitted
    by the model. Any number the model writes into the prose is scrubbed by the
    contract gate at persistence; this type guarantees it is never a
    *first-class* output.

    Attributes:
        thesis: The core investment thesis, in prose.
        risks: The principal risks to the thesis, in prose.
        catalysts: The catalysts that could realise the thesis, in prose.
        verdict: The directional BUY/HOLD/SELL call.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    thesis: str = Field(min_length=1)
    risks: str = Field(min_length=1)
    catalysts: str = Field(min_length=1)
    verdict: Verdict


class RunMetadata(BaseModel):
    """Mandatory, first-class provenance of one analyst run.

    Per dossier §6 the model identity *and* its knowledge cutoff are
    load-bearing metadata, equal in standing to the dataset hash: a frontier
    model that updates silently invalidates the temporal boundary of the
    experiment. Every field here is required — there is no default — so a run
    cannot be constructed that forgets to pin them.

    Attributes:
        model_name: Concrete model identifier (e.g. ``"qwen3.8:27b-mlx"``).
        tier: Which tier produced the headline synthesis.
        llm_version: Pinned version string of the model.
        llm_cutoff: The model's knowledge cutoff date.
        knowledge_date: The point-in-time knowledge date of the analysis.
        dataset_hash: Canonical hash of the dataset the numbers came from.
    """

    # ``model_name`` would collide with Pydantic v2's protected ``model_``
    # namespace; disable the guard here deliberately since these are plain
    # metadata strings, not Pydantic model internals.
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    model_name: str = Field(min_length=1)
    tier: ModelTier
    llm_version: str = Field(min_length=1)
    llm_cutoff: date
    knowledge_date: date
    dataset_hash: str = Field(min_length=1)


class ExtractionRequest(BaseModel):
    """Typed input for the extraction step (quick-think).

    Attributes:
        ticker: The instrument under analysis.
        documents: Source texts (news, filings) to extract facts from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    documents: tuple[str, ...] = Field(min_length=1)


class NarrationRequest(BaseModel):
    """Typed input for the narration step (deep-think).

    The numeric valuation summary is passed as **already-computed** context so
    the model narrates *around* numbers it did not produce. It is a plain
    read-only mapping of labels to pre-computed values; the model cannot change
    them, only describe them.

    Attributes:
        ticker: The instrument under analysis.
        findings: Qualitative facts gathered by the extraction step.
        valuation_summary: Pre-computed numeric context (name -> value), e.g.
            ``{"consensus_mid": 42.0, "current_price": 38.0}``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    findings: tuple[QualitativeFinding, ...]
    valuation_summary: tuple[tuple[str, float], ...] = ()
