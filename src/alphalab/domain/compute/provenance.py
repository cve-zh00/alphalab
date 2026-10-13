"""The ``ProvenancedValue`` type and the ``provenanced`` operator helper.

This module encodes the thesis of the whole compute layer: *numbers are
computed by code, judgement is supplied elsewhere; every number must be
traceable to the call that produced it.* A :class:`ProvenancedValue` is a
single immutable node in a derivation **DAG**. Leaves are raw inputs; every
derived value carries references to the exact ``ProvenancedValue`` parents it
was computed from, together with the operator, its version, the inputs, the
unit, and the point-in-time knowledge date.

Design of the composable trace
-------------------------------
Each node stores its direct ``parents`` as an immutable tuple of
``ProvenancedValue``. Because every parent is itself a fully-formed node, the
structure is a directed acyclic graph that can be walked to answer "where did
this number come from?":

* :meth:`ProvenancedValue.leaves` returns the raw input nodes (no parents).
* :meth:`ProvenancedValue.tree` returns a readable nested dict of the whole
  derivation.

A tuple-of-nodes (rather than, say, storing flattened strings) was chosen
because it keeps the *full* structured ancestry addressable and immutable, lets
shared sub-expressions be referenced by multiple parents without duplication,
and makes the leaf/tree queries pure traversals. Cycles are impossible to build
(a parent must exist before its child), and traversal is still guarded by
:data:`MAX_PROVENANCE_DEPTH` and an identity-based visited set for safety.

Everything here is pure: no I/O, no mutation, no global state.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from alphalab.domain.compute.thresholds import MAX_PROVENANCE_DEPTH

# An input pair is a (name, value) tuple; the full input set is a tuple of
# those pairs. Tuples are hashable and immutable, so a node's recorded inputs
# can never be mutated after construction.
InputPair = tuple[str, float]
InputPairs = tuple[InputPair, ...]


class ProvenanceError(Exception):
    """Base class for provenance construction/traversal errors."""


class ProvenanceDepthError(ProvenanceError):
    """Raised when a derivation tree exceeds :data:`MAX_PROVENANCE_DEPTH`."""


class ProvenancedValue(BaseModel):
    """An immutable number annotated with its full derivation lineage.

    The only way the rest of the system should obtain a number is as one of
    these. ``value`` is the number; everything else explains it.
    """

    model_config = ConfigDict(frozen=True)

    value: float
    operator: str = Field(min_length=1)
    operator_version: str = Field(min_length=1)
    inputs: InputPairs
    unit: str = Field(min_length=1)
    parents: tuple[ProvenancedValue, ...] = ()
    dataset_hash: str | None = None
    knowledge_date: date | None = None

    def is_leaf(self) -> bool:
        """True when this node has no parents (a raw input)."""
        return len(self.parents) == 0

    def leaves(self) -> tuple[ProvenancedValue, ...]:
        """Return every leaf (parent-less) ancestor, deduplicated by identity.

        Walking to the leaves answers "which raw inputs ultimately produced
        this number?". Order is deterministic: a stable depth-first,
        left-to-right traversal with first-seen-wins deduplication.
        """
        ordered: list[ProvenancedValue] = []
        seen: set[int] = set()
        self._collect_leaves(ordered, seen, depth=0)
        return tuple(ordered)

    def _collect_leaves(
        self,
        ordered: list[ProvenancedValue],
        seen: set[int],
        *,
        depth: int,
    ) -> None:
        if depth > MAX_PROVENANCE_DEPTH:
            raise ProvenanceDepthError(f"derivation exceeded max depth {MAX_PROVENANCE_DEPTH}")
        if self.is_leaf():
            key = id(self)
            if key not in seen:
                seen.add(key)
                ordered.append(self)
            return
        for parent in self.parents:
            parent._collect_leaves(ordered, seen, depth=depth + 1)

    def tree(self) -> dict[str, Any]:
        """Return a readable, nested view of the full derivation.

        The returned dict is a fresh, plain structure safe to serialise or
        print; it never exposes the live model objects for mutation.
        """
        return self._tree(depth=0)

    def _tree(self, *, depth: int) -> dict[str, Any]:
        if depth > MAX_PROVENANCE_DEPTH:
            raise ProvenanceDepthError(f"derivation exceeded max depth {MAX_PROVENANCE_DEPTH}")
        return {
            "operator": self.operator,
            "operator_version": self.operator_version,
            "value": self.value,
            "unit": self.unit,
            "inputs": [list(pair) for pair in self.inputs],
            "parents": [p._tree(depth=depth + 1) for p in self.parents],
        }


def leaf(
    *,
    value: float,
    name: str,
    unit: str,
    dataset_hash: str | None = None,
    knowledge_date: date | None = None,
) -> ProvenancedValue:
    """Construct a raw-input leaf node.

    Leaves are the only nodes allowed to originate a number without parents;
    they record the name of the input as their sole input pair.
    """
    return ProvenancedValue(
        value=value,
        operator="leaf",
        operator_version="1",
        inputs=((name, value),),
        unit=unit,
        parents=(),
        dataset_hash=dataset_hash,
        knowledge_date=knowledge_date,
    )


def _merge_knowledge_date(parents: Iterable[ProvenancedValue]) -> date | None:
    """Most-conservative PIT date across parents (the latest known date).

    A derived value cannot be "known" earlier than the newest input it needed,
    so the earliest-safe knowledge date of a composition is the maximum of its
    parents' dates. ``None`` parents are ignored; all-``None`` yields ``None``.
    """
    dates = [p.knowledge_date for p in parents if p.knowledge_date is not None]
    return max(dates) if dates else None


def provenanced(
    *,
    operator: str,
    version: str,
    unit: str,
) -> Callable[[Callable[..., float]], Callable[..., ProvenancedValue]]:
    """Decorator that turns a pure numeric function into a provenanced operator.

    The wrapped function receives the *same* keyword arguments the caller
    passes and must return a plain ``float``. The decorator takes care of the
    tedious, error-prone bookkeeping — collecting parent nodes, recording their
    scalar values as inputs, and propagating the conservative knowledge date —
    so writing a correctly-traced operator is the path of least resistance and
    forgetting the provenance is simply not possible.

    Rules enforced:

    * Every keyword argument that is a :class:`ProvenancedValue` becomes a
      parent and contributes a ``(name, value)`` input pair.
    * Plain numeric keyword arguments are also recorded as input pairs so the
      node is self-describing, but do not become parents.
    * The decorated callable must be invoked with keyword arguments only; this
      keeps the recorded input names meaningful.
    """

    def decorator(fn: Callable[..., float]) -> Callable[..., ProvenancedValue]:
        @functools.wraps(fn)
        def wrapper(**kwargs: Any) -> ProvenancedValue:
            parents: list[ProvenancedValue] = []
            inputs: list[InputPair] = []
            scalar_kwargs: dict[str, float] = {}
            for key, arg in kwargs.items():
                if isinstance(arg, ProvenancedValue):
                    parents.append(arg)
                    inputs.append((key, arg.value))
                    scalar_kwargs[key] = arg.value
                else:
                    inputs.append((key, float(arg)))
                    scalar_kwargs[key] = float(arg)
            result = fn(**scalar_kwargs)
            return ProvenancedValue(
                value=float(result),
                operator=operator,
                operator_version=version,
                inputs=tuple(inputs),
                unit=unit,
                parents=tuple(parents),
                knowledge_date=_merge_knowledge_date(parents),
            )

        return wrapper

    return decorator


def combine(
    *,
    value: float,
    operator: str,
    version: str,
    unit: str,
    parents: tuple[ProvenancedValue, ...],
    extra_inputs: Mapping[str, float] | None = None,
) -> ProvenancedValue:
    """Build a derived node explicitly from a value and its parents.

    Use this when an operator computes its number through several intermediate
    steps and wants to attach a specific set of parents and extra scalar inputs
    rather than go through :func:`provenanced`. Parents still drive the inputs
    and the conservative knowledge date.
    """
    inputs: list[InputPair] = [(p.operator, p.value) for p in parents]
    if extra_inputs is not None:
        inputs.extend((k, float(v)) for k, v in extra_inputs.items())
    return ProvenancedValue(
        value=float(value),
        operator=operator,
        operator_version=version,
        inputs=tuple(inputs),
        unit=unit,
        parents=parents,
        knowledge_date=_merge_knowledge_date(parents),
    )
