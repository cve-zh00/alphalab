"""Consensus aggregation across independent valuation methods.

Given several :class:`~alphalab.domain.compute.provenance.ProvenancedValue`
estimates produced by different methods (DCF, DDM, multiples, ...), compute a
central point and the surrounding range. The result exposes the **relative
span** so the downstream contract gate (Task 7) can decide — *without
recomputing anything* — whether the methods agree closely enough to report a
single point, or diverge so much that it must degrade to a range.

All constituent estimates become parents of the consensus node, so the full
derivation of every method flows into the aggregate and remains traceable to
the leaves.

Pure: numpy only, no I/O, no state.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from alphalab.domain.compute.errors import InvalidValuationInputError
from alphalab.domain.compute.provenance import ProvenancedValue, combine
from alphalab.domain.compute.thresholds import CONSENSUS_WIDE_SPAN_THRESHOLD

_CONSENSUS_VERSION = "1.1.0"

INTRINSIC_VALUATION_OPERATORS: frozenset[str] = frozenset(
    {
        "dcf_value_per_share",
        "ddm_value_per_share",
        "residual_income_value_per_share",
        "graham_value_per_share",
        "epv_value_per_share",
        "dcf",
        "ddm",
        "residual_income",
        "graham",
        "epv",
    }
)

RELATIVE_MULTIPLES_OPERATORS: frozenset[str] = frozenset(
    {
        "pe_implied_value",
        "ev_ebitda_implied_value",
        "pb_implied_value",
        "ffo_implied_value",
        "multiples_pe",
        "multiples_ev_ebitda",
        "multiples_pb",
        "multiples_ffo",
        "pe",
        "ev_ebitda",
        "pb",
        "ffo",
    }
)


def family_balanced_midpoint(pairs: Sequence[tuple[str, float]]) -> float:
    """Compute a family-balanced consensus midpoint across valuation methods.

    When both intrinsic valuation methods (DCF, DDM, Residual Income, ...) and
    relative multiples methods (P/E, EV/EBITDA, P/B, P/FFO) are present with no
    unclassified operators, each family first collapses to its own median and
    the consensus midpoint is the 50/50 average (2-element median) of the two
    family medians. This prevents 2-3 relative multiples from outvoting 1-2
    intrinsic methods in flat median aggregation.
    """
    if len(pairs) == 0:
        raise InvalidValuationInputError("estimates must be non-empty")
    intrinsic = [val for op, val in pairs if op in INTRINSIC_VALUATION_OPERATORS]
    relative = [val for op, val in pairs if op in RELATIVE_MULTIPLES_OPERATORS]
    other = [
        val
        for op, val in pairs
        if op not in INTRINSIC_VALUATION_OPERATORS and op not in RELATIVE_MULTIPLES_OPERATORS
    ]
    if intrinsic and relative and not other:
        intrinsic_mid = float(np.median(np.asarray(intrinsic, dtype=float)))
        relative_mid = float(np.median(np.asarray(relative, dtype=float)))
        return 0.5 * (intrinsic_mid + relative_mid)
    return float(np.median(np.asarray([val for _, val in pairs], dtype=float)))


class ConsensusRange(BaseModel):
    """Immutable summary of a cross-method valuation consensus.

    Attributes:
        point: The central (median) estimate, carrying the full provenance of
            all constituent methods as parents.
        low: Minimum across methods.
        mid: Median across methods (equals ``point.value``).
        high: Maximum across methods.
        relative_span: ``(high - low) / |mid|`` — the dispersion measure the
            gate thresholds on. Zero when all methods agree.
        is_wide: ``True`` when ``relative_span`` exceeds the calibrated
            :data:`CONSENSUS_WIDE_SPAN_THRESHOLD`; the gate should then prefer a
            range over a point.
    """

    model_config = ConfigDict(frozen=True)

    point: ProvenancedValue
    low: float
    mid: float
    high: float
    relative_span: float
    is_wide: bool


def consensus(
    *,
    estimates: Sequence[ProvenancedValue],
    unit: str,
) -> ConsensusRange:
    """Aggregate several method estimates into a central point and a range.

    Args:
        estimates: Non-empty sequence of provenanced estimates, expected to all
            share ``unit``. A unit mismatch is rejected — mixing units here is
            exactly the calibre error the audit layer guards against.
        unit: The common unit the estimates must share.

    Returns:
        A frozen :class:`ConsensusRange` whose ``point`` has every estimate as a
        parent.

    Raises:
        InvalidValuationInputError: on an empty set or a unit mismatch.
    """
    if len(estimates) == 0:
        raise InvalidValuationInputError("estimates must be non-empty")
    for est in estimates:
        if est.unit != unit:
            raise InvalidValuationInputError(
                f"estimate from '{est.operator}' has unit '{est.unit}', expected '{unit}'"
            )

    values = np.asarray([e.value for e in estimates], dtype=float)
    low = float(np.min(values))
    high = float(np.max(values))
    mid = family_balanced_midpoint([(e.operator, e.value) for e in estimates])

    denom = abs(mid)
    relative_span = (high - low) / denom if denom > 0.0 else 0.0
    is_wide = relative_span > CONSENSUS_WIDE_SPAN_THRESHOLD

    point = combine(
        value=mid,
        operator="consensus",
        version=_CONSENSUS_VERSION,
        unit=unit,
        parents=tuple(estimates),
        extra_inputs={
            "n_methods": float(len(estimates)),
            "low": low,
            "high": high,
            "relative_span": relative_span,
        },
    )

    return ConsensusRange(
        point=point,
        low=low,
        mid=mid,
        high=high,
        relative_span=relative_span,
        is_wide=is_wide,
    )
