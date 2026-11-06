"""The output contract gate: value-withhold, never refuse-to-rate.

The gate runs at the single persistence boundary, sees the whole assembled
artifact, runs every clause, and returns a *new* artifact (it never mutates the
input). On a hard violation it withholds the price — nulls the number in every
slot, stamps the violation visibly, sets ``valuation_withheld`` — while the
directional verdict and the narrative prose survive untouched.

Guiding rule: *corrupt or out-of-band data withholds the price, never the
judgment.* The gate therefore **never raises**. If a clause explodes on a
degenerate input, the gate degrades with grace and stamps the failure; an
exception here would blank the whole run and discard a valid analysis.

Pure: no I/O, no mutation of the input, no global state.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from alphalab.artifact.models import EquityResearchArtifact, PriceRange
from alphalab.domain.compute import AuditFinding, Severity
from alphalab.domain.compute.operators.consensus import family_balanced_midpoint
from alphalab.domain.contract import clauses
from alphalab.domain.contract._text import scrub_amount
from alphalab.domain.contract.thresholds import PRICE_TEXT_DECIMALS

# Clauses that, on failure, degrade the whole point (scrub everywhere).
_HARD_CLAUSES: tuple[Callable[[EquityResearchArtifact], AuditFinding | None], ...] = (
    clauses.check_c0_unit_plausibility,
    clauses.check_c1_upside_band,
    clauses.check_c2_numeric_concatenation,
    clauses.check_c3_conclusion_headline,
    clauses.check_c4_unit_caliber,
)

# Soft clauses: annotate and stamp, but never withhold.
_SOFT_CLAUSES: tuple[Callable[[EquityResearchArtifact], AuditFinding | None], ...] = (
    clauses.check_c6_single_method_disclosure,
)


class GateResult:
    """Immutable outcome of running the gate: the artifact plus the findings."""

    __slots__ = ("artifact", "findings")

    def __init__(
        self, artifact: EquityResearchArtifact, findings: tuple[AuditFinding, ...]
    ) -> None:
        self.artifact = artifact
        self.findings = findings


def _safe_check(
    clause: Callable[[EquityResearchArtifact], AuditFinding | None],
    artifact: EquityResearchArtifact,
) -> AuditFinding | None:
    """Run a clause, converting any explosion into a stamped hard finding.

    A clause must never be able to abort the gate: an unexpected input that
    raises becomes a HARD finding so the price is withheld and the failure is
    visible, rather than propagating and blanking the run.
    """
    try:
        return clause(artifact)
    except Exception as exc:
        return AuditFinding(
            check=f"{getattr(clause, '__name__', 'clause')}_raised",
            passed=False,
            severity=Severity.HARD,
            message=f"clause raised {type(exc).__name__}: {exc}",
            values=(),
        )


def _stamp(check: str) -> str:
    """Render a visible contract stamp for a violation."""
    return f"[CONTRACT/{check}]"


def _scrub_text_slots(slots: dict[str, str] | None, amount: float) -> dict[str, str]:
    """Return a copy of the slot mapping with ``amount`` redacted from each."""
    if slots is None:
        return {}
    return {k: scrub_amount(v, amount, decimals=PRICE_TEXT_DECIMALS) for k, v in slots.items()}


def enforce_contract(artifact: EquityResearchArtifact) -> GateResult:
    """Run all clauses and return a (possibly degraded) new artifact.

    Never raises. On a hard violation the price is withheld in every slot, the
    violation is stamped, ``valuation_withheld`` is set, and the no-resurrection
    clause verifies the number is truly gone. The verdict and narrative prose
    always survive.
    """
    findings: list[AuditFinding] = []
    hard: list[AuditFinding] = []

    for clause in _HARD_CLAUSES:
        finding = _safe_check(clause, artifact)
        if finding is not None:
            findings.append(finding)
            if finding.is_blocking():
                hard.append(finding)

    corroboration = _safe_check(clauses.check_c1b_corroboration, artifact)
    if corroboration is not None:
        findings.append(corroboration)

    for clause in _SOFT_CLAUSES:
        finding = _safe_check(clause, artifact)
        if finding is not None:
            findings.append(finding)

    # Soft findings never withhold, but they are always stamped — in every path
    # — so the result does not depend on whether a hard clause also fired. This
    # is what makes a second pass produce the same stamps as the first.
    soft_stamps = tuple(
        _stamp(f.check) for f in findings if not f.passed and f.severity is Severity.SOFT
    )

    result_artifact = artifact

    # C1b degrades to a range (drop the point, keep the range) when it is the
    # *only* complaint; a harder clause always wins and scrubs everywhere.
    if not hard and corroboration is not None and corroboration.is_blocking():
        result_artifact = _degrade_to_range(result_artifact, corroboration)
    elif hard:
        result_artifact = _withhold(result_artifact, hard, corroboration)
        c7 = _safe_check(
            lambda a: clauses.check_c7_no_resurrection(a, _withheld_value(artifact)),
            result_artifact,
        )
        if c7 is not None:
            findings.append(c7)

    result_artifact = _apply_stamps(result_artifact, soft_stamps)
    return GateResult(result_artifact, tuple(findings))


def _apply_stamps(
    artifact: EquityResearchArtifact, extra: tuple[str, ...]
) -> EquityResearchArtifact:
    """Add stamps, deduplicated and order-preserving, for idempotence."""
    if not extra:
        return artifact
    seen = set(artifact.contract_stamps)
    additions = tuple(s for s in extra if s not in seen)
    if not additions:
        return artifact
    return artifact.model_copy(update={"contract_stamps": artifact.contract_stamps + additions})


def _withheld_value(artifact: EquityResearchArtifact) -> float:
    """The headline target value to scrub, or NaN when there is none."""
    target = artifact.price_target
    if target is None:
        return math.nan
    return target.value


def _withhold(
    artifact: EquityResearchArtifact,
    hard: list[AuditFinding],
    corroboration: AuditFinding | None,
) -> EquityResearchArtifact:
    """Produce a new artifact with the price withheld in every slot."""
    amount = _withheld_value(artifact)
    stamps = tuple(_stamp(f.check) for f in hard)
    if corroboration is not None and corroboration.is_blocking():
        stamps = (*stamps, _stamp(corroboration.check))
    seen = set(artifact.contract_stamps)
    new_stamps = tuple(s for s in stamps if s not in seen)

    update: dict[str, object] = {
        "price_target": None,
        "price_range": None,
        "valuation_withheld": True,
        "contract_stamps": artifact.contract_stamps + new_stamps,
    }
    if math.isfinite(amount):
        update["narrative_slots"] = _scrub_text_slots(dict(artifact.narrative_slots), amount)
        update["fallback_slots"] = _scrub_text_slots(dict(artifact.fallback_slots), amount)
    return artifact.model_copy(update=update)


def _degrade_to_range(
    artifact: EquityResearchArtifact, corroboration: AuditFinding
) -> EquityResearchArtifact:
    """Drop the point target but keep the range when methods diverge."""
    stamp = _stamp(corroboration.check)
    existing = artifact.contract_stamps
    new_stamps = () if stamp in existing else (stamp,)
    update: dict[str, object] = {
        "price_target": None,
        "valuation_withheld": True,
        "contract_stamps": existing + new_stamps,
    }
    # If no range exists we have nothing to fall back to; the point is simply
    # withheld. The range (when present) is preserved deliberately.
    if artifact.price_range is None:
        update["price_range"] = _range_from_methods(artifact)
    return artifact.model_copy(update=update)


def _range_from_methods(artifact: EquityResearchArtifact) -> PriceRange | None:
    """Build a low/mid/high range from per-method values, if any exist."""
    finite_pairs = [(k, v) for k, v in artifact.method_values.items() if math.isfinite(v)]
    if not finite_pairs:
        return None
    values = [v for _, v in finite_pairs]
    low = min(values)
    high = max(values)
    mid = family_balanced_midpoint(finite_pairs)
    return PriceRange(low=low, mid=mid, high=high)


C1B_STAMP = "[CONTRACT/contract_c1b_corroboration]"
C1B_TARGET_DETAIL = "conservative C1b target"


def conservative_c1b_target(artifact: EquityResearchArtifact) -> float | None:
    """Conservative point value derived from a C1b-degraded range, else ``None``.

    Applies only when the gate degraded the point because of method dispersion
    (C1b stamp present, point absent) and every method agrees directionally
    against the current price: bullish (``low > price``) gives
    ``0.5 * (low + mid)``; bearish (``high < price``) gives ``0.5 * (mid + high)``.
    A range straddling the price (methods contradict) yields ``None``. Hard
    withholds drop the range, so they never produce a value here.
    """
    rng = artifact.price_range
    price = artifact.current_price
    if (
        rng is None
        or artifact.price_target is not None
        or C1B_STAMP not in artifact.contract_stamps
        or not (math.isfinite(price) and price > 0.0)
        or not all(math.isfinite(v) for v in (rng.low, rng.mid, rng.high))
    ):
        return None
    if rng.low > price:
        return 0.5 * (rng.low + rng.mid)
    if rng.high < price:
        return 0.5 * (rng.mid + rng.high)
    return None


def conservative_c1b_upside(artifact: EquityResearchArtifact) -> float | None:
    """Implied fractional upside of :func:`conservative_c1b_target`, if any."""
    value = conservative_c1b_target(artifact)
    if value is None:
        return None
    return value / artifact.current_price - 1.0
