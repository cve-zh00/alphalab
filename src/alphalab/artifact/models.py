"""Immutable data model for an equity-research artifact (US equities).

The artifact is the *whole assembled research product* — the headline price
target, the directional verdict, the per-method valuations, and the narrative
slots a language model will later fill. It is the single object the output
contract gate (``alphalab.domain.contract``) inspects at the persistence
boundary. Per-fragment checks are blind; only the fully assembled artifact lets
the gate verify that the conclusion agrees with the headline.

The model is frozen (immutable): the gate never mutates it, it returns a new,
degraded copy. Mutation in place would make the "withhold the price, never the
judgment" contract impossible to reason about.

A deliberate design choice: ``price_target`` is typed against the structural
:class:`HasValueAndUnit` ``Protocol`` rather than the concrete
``ProvenancedValue``. The gate only needs ``.value`` and ``.unit``; typing
against the shape lets tests pass trivial doubles while ``ProvenancedValue``
satisfies it structurally (duck typing checked by the type checker).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class Verdict(StrEnum):
    """The directional judgment. This survives contract enforcement always."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@runtime_checkable
class HasValueAndUnit(Protocol):
    """Structural contract for anything carrying a numeric value and a unit.

    The gate reasons only about a price target's magnitude and its unit, so it
    depends on this minimal shape rather than on ``ProvenancedValue``. Any
    object exposing ``value: float`` and ``unit: str`` qualifies.
    """

    @property
    def value(self) -> float: ...

    @property
    def unit(self) -> str: ...


class PriceTarget(BaseModel):
    """A concrete, frozen value+unit pair satisfying :class:`HasValueAndUnit`.

    Provided so the artifact (and tests) have a lightweight concrete type;
    ``ProvenancedValue`` also satisfies the protocol structurally.
    """

    model_config = ConfigDict(frozen=True)

    value: float
    unit: str = Field(min_length=1)


class PriceRange(BaseModel):
    """A frozen low/mid/high valuation range retained when a point is degraded."""

    model_config = ConfigDict(frozen=True)

    low: float
    mid: float
    high: float


class EquityResearchArtifact(BaseModel):
    """The whole assembled research artifact inspected at the save boundary.

    Attributes:
        ticker: US-equity symbol.
        knowledge_date: Point-in-time knowledge date of the analysis.
        verdict: Directional call (BUY/HOLD/SELL); preserved through the gate.
        price_target: The headline point target, or ``None`` once withheld.
        price_range: The low/mid/high range; retained when a point is degraded.
        current_price: The reference entry/current price per share.
        methods_used: Names of the valuation methods that fed the target.
        method_values: Per-method implied values (name -> value per share).
        consensus_relative_span: The ``relative_span`` emitted by
            ``consensus(...)`` across methods; the gate thresholds on it
            without recomputing anything.
        narrative_slots: Where the LLM text will live; a number may appear
            embedded in prose here and the gate must scrub it from the text.
        fallback_slots: Secondary slots from which a withheld value could
            "resurrect"; scanned by the no-resurrection clause.
        contract_stamps: Where the gate stamps ``[CONTRACT/*]`` violations.
        valuation_withheld: Set ``True`` by the gate when a hard clause fires.
        single_method_disclosed: Whether the artifact declares it lacks a
            cross-check (consumed by the soft disclosure clause).
    """

    # ``arbitrary_types_allowed`` lets the ``price_target`` field accept any
    # object satisfying the runtime-checkable ``HasValueAndUnit`` protocol;
    # Pydantic validates membership with an ``isinstance`` check.
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    ticker: str = Field(min_length=1)
    knowledge_date: date | None = None
    verdict: Verdict
    price_target: HasValueAndUnit | None = None
    price_range: PriceRange | None = None
    current_price: float
    methods_used: tuple[str, ...] = ()
    method_values: Mapping[str, float] = Field(default_factory=dict)
    consensus_relative_span: float | None = None
    narrative_slots: Mapping[str, str] = Field(default_factory=dict)
    fallback_slots: Mapping[str, str] = Field(default_factory=dict)
    contract_stamps: tuple[str, ...] = ()
    valuation_withheld: bool = False
    single_method_disclosed: bool = False
