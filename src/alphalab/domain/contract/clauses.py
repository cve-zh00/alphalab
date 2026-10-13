"""Pure contract clauses for the output gate.

Each clause is a small, independent, pure function
``check(artifact) -> AuditFinding | None``. It returns ``None`` when the
artifact is fine and an :class:`AuditFinding` (``passed is False``) when it is
not. HARD findings cause the gate to withhold the price; SOFT findings are
annotated but never withhold.

The clauses never recompute the valuation — they verify an already-computed
result against calibrated bands imported from :mod:`.thresholds`. A verifier
that recomputed could reproduce the very bug it is meant to catch.

No I/O, no mutation, no global state.
"""

from __future__ import annotations

import math
import re

from alphalab.artifact.models import EquityResearchArtifact
from alphalab.domain.compute import AuditFinding, Severity
from alphalab.domain.compute.operators.audit.findings import fail
from alphalab.domain.contract._text import (
    has_double_decimal,
    has_glued_digits,
    text_contains_amount,
)
from alphalab.domain.contract.thresholds import (
    METHOD_CORROBORATION_SPAN_K,
    MULTI_METHOD_RATIO_K,
    PRICE_TEXT_DECIMALS,
    SINGLE_METHOD_RATIO_K,
)


def _is_multi_method(artifact: EquityResearchArtifact) -> bool:
    """True when more than one valuation method fed the target."""
    return len(artifact.methods_used) > 1


def check_c1_upside_band(artifact: EquityResearchArtifact) -> AuditFinding | None:
    """C1 (HARD): target/current_price must fall in the calibrated band.

    Single method: ``[1/2x, 2x]``. Multiple corroborating methods: ``[1/4x,
    4x]``. A non-finite or non-positive price/target is structurally impossible
    and fails hard.
    """
    target = artifact.price_target
    if target is None:
        return None
    value = target.value
    current = artifact.current_price
    if not math.isfinite(value) or not math.isfinite(current) or current <= 0.0 or value <= 0.0:
        return fail(
            "contract_c1_upside_band",
            Severity.HARD,
            f"target {value!r} or current price {current!r} is non-finite or non-positive",
            values=(("target", _finite(value)), ("current_price", _finite(current))),
        )
    k = MULTI_METHOD_RATIO_K if _is_multi_method(artifact) else SINGLE_METHOD_RATIO_K
    ratio = value / current
    if ratio < (1.0 / k) or ratio > k:
        return fail(
            "contract_c1_upside_band",
            Severity.HARD,
            f"target/current ratio {ratio:.3f} outside calibrated band [1/{k:g}x, {k:g}x]",
            values=(("ratio", ratio), ("band_k", k)),
        )
    return None


def check_c1b_corroboration(artifact: EquityResearchArtifact) -> AuditFinding | None:
    """C1b (HARD): methods must corroborate; else degrade point to range.

    Uses the ``consensus(...)`` ``relative_span`` already attached to the
    artifact — never recomputes it. Only meaningful with several methods; a
    single-method artifact has nothing to corroborate.
    """
    if not _is_multi_method(artifact):
        return None
    span = artifact.consensus_relative_span
    if span is None:
        return None
    if not math.isfinite(span) or span > METHOD_CORROBORATION_SPAN_K:
        return fail(
            "contract_c1b_corroboration",
            Severity.HARD,
            f"method dispersion span {span!r} exceeds tolerated {METHOD_CORROBORATION_SPAN_K:g}",
            values=(("relative_span", _finite(span)), ("span_k", METHOD_CORROBORATION_SPAN_K)),
        )
    return None


def check_c2_numeric_concatenation(artifact: EquityResearchArtifact) -> AuditFinding | None:
    """C2 (HARD): detect malformed amounts in the textual slots.

    A double-decimal amount (``1.234.56``) or an implausibly long glued digit
    run betrays a string that was concatenated wrong. Scans the narrative and
    fallback slots.
    """
    for slot_name, text in _iter_text_slots(artifact):
        if has_double_decimal(text) or has_glued_digits(text):
            return fail(
                "contract_c2_numeric_concatenation",
                Severity.HARD,
                f"slot '{slot_name}' contains a malformed concatenated amount",
                values=(),
            )
    return None


def check_c3_conclusion_headline(artifact: EquityResearchArtifact) -> AuditFinding | None:
    """C3 (HARD): the conclusion amount must equal the headline target.

    The artifact is seen whole, so the gate can confirm the number cited in a
    ``conclusion`` narrative slot matches the headline ``price_target``. An
    internal inconsistency withholds the price. Only runs when a finite target
    and a conclusion slot both exist.
    """
    target = artifact.price_target
    if target is None or not math.isfinite(target.value):
        return None
    conclusion = artifact.narrative_slots.get("conclusion")
    if conclusion is None:
        return None
    # The headline amount must appear verbatim in the conclusion. If any amount
    # is cited there but it is not the headline, that is a desync.
    if text_contains_amount(conclusion, target.value, decimals=PRICE_TEXT_DECIMALS):
        return None
    if not _mentions_any_amount(conclusion):
        return None
    return fail(
        "contract_c3_conclusion_headline",
        Severity.HARD,
        "conclusion cites an amount that disagrees with the headline target",
        values=(("headline_target", target.value),),
    )


def check_c4_unit_caliber(artifact: EquityResearchArtifact) -> AuditFinding | None:
    """C4 (HARD): the target's unit must be a per-share price caliber.

    A target whose ``unit`` is a dimensionless 'ratio' (or anything that is not
    a per-share price) cannot be used as a price per share. This is a backstop
    for a unit/scale blunder that slipped past the compute layer.
    """
    target = artifact.price_target
    if target is None:
        return None
    unit = target.unit
    if not _is_price_unit(unit):
        return fail(
            "contract_c4_unit_caliber",
            Severity.HARD,
            f"target unit '{unit}' is not a per-share price caliber",
            values=(),
        )
    return None


def check_c6_single_method_disclosure(artifact: EquityResearchArtifact) -> AuditFinding | None:
    """C6 (SOFT): a single-method artifact must disclose the lack of cross-check.

    Soft: it never withholds the number, but it demands the disclosure flag and
    stamps the omission so a reader knows the point rests on one method.
    """
    if _is_multi_method(artifact):
        return None
    if artifact.single_method_disclosed:
        return None
    return fail(
        "contract_c6_single_method_disclosure",
        Severity.SOFT,
        "single-method target without a disclosed lack of cross-check",
        values=(),
    )


def check_c7_no_resurrection(
    artifact: EquityResearchArtifact, withheld_value: float
) -> AuditFinding | None:
    """C7 (HARD, internal): a withheld value must not survive anywhere.

    Run *after* a withhold, this scans every slot — narrative text, fallback
    slots, the headline point and the range — to confirm the retired number has
    not resurrected. The classic hole it closes: the main slot is nulled but
    the value re-emerges from a fallback.
    """
    if not math.isfinite(withheld_value):
        return None
    for slot_name, text in _iter_text_slots(artifact):
        if text_contains_amount(text, withheld_value, decimals=PRICE_TEXT_DECIMALS):
            return fail(
                "contract_c7_no_resurrection",
                Severity.HARD,
                f"withheld value {withheld_value} resurrected in slot '{slot_name}'",
                values=(("withheld_value", withheld_value),),
            )
    target = artifact.price_target
    if (
        target is not None
        and math.isfinite(target.value)
        and round(target.value, PRICE_TEXT_DECIMALS) == round(withheld_value, PRICE_TEXT_DECIMALS)
    ):
        return fail(
            "contract_c7_no_resurrection",
            Severity.HARD,
            f"withheld value {withheld_value} resurrected in price_target",
            values=(("withheld_value", withheld_value),),
        )
    return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _iter_text_slots(artifact: EquityResearchArtifact) -> list[tuple[str, str]]:
    """All (slot_name, text) pairs across narrative and fallback slots."""
    slots: list[tuple[str, str]] = []
    slots.extend((f"narrative:{k}", v) for k, v in artifact.narrative_slots.items())
    slots.extend((f"fallback:{k}", v) for k, v in artifact.fallback_slots.items())
    return slots


def _mentions_any_amount(text: str) -> bool:
    """True when the text contains any numeric amount at all."""
    return re.search(r"\d", text) is not None


def _is_price_unit(unit: str) -> bool:
    """True when ``unit`` denotes a per-share price (not a bare ratio)."""
    normalized = unit.strip().lower()
    if normalized in {"ratio", "x", "multiple", "percent", "%", "bps", ""}:
        return False
    return "per_share" in normalized or "price" in normalized or normalized.startswith("usd")


def _finite(x: float) -> float:
    """Coerce a non-finite number to 0.0 for safe inclusion in findings."""
    return x if math.isfinite(x) else 0.0
