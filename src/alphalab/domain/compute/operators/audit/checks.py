"""Number-vs-number coherence audit checks.

Each check inspects already-computed numbers (and their units) for internal
inconsistencies and returns a structured
:class:`~alphalab.domain.compute.operators.audit.findings.AuditFinding` — never
a bare bool. The checks are the coherence subset of FinRobot's audit suite;
the narrative-vs-number checks arrive once a narrative layer exists.

Every check is written so it marks the *bad* case and leaves the *good* case
alone: a detector that always alarms is worthless.

Pure: numpy only, no I/O, no state.
"""

from __future__ import annotations

from alphalab.domain.compute.operators.audit.findings import (
    AuditFinding,
    Severity,
    fail,
    ok,
)
from alphalab.domain.compute.thresholds import (
    EV_BRIDGE_REL_TOLERANCE,
    MAX_PLAUSIBLE_EV_EBITDA,
    MAX_PLAUSIBLE_MARGIN,
    MAX_PLAUSIBLE_PE,
    MAX_PLAUSIBLE_PRICE_PER_SHARE,
    MIN_PLAUSIBLE_MARGIN,
)


def currency_caliber(
    *,
    left_unit: str,
    right_unit: str,
    left_value: float,
    right_value: float,
) -> AuditFinding:
    """Reject comparing/combining two magnitudes in incompatible units.

    Applying a multiple derived in one currency to a price in another, or
    comparing a ``USD`` figure to a ``bps`` figure, is a silent calibre error.
    The check is a HARD failure because the resulting number is meaningless.
    """
    values = (("left", left_value), ("right", right_value))
    if left_unit != right_unit:
        return fail(
            "currency_caliber",
            Severity.HARD,
            f"incompatible units: '{left_unit}' vs '{right_unit}'",
            values,
        )
    return ok(
        "currency_caliber",
        f"units match ('{left_unit}')",
        values,
    )


def ev_bridge_check(
    *,
    equity_value: float,
    net_debt: float,
    enterprise_value: float,
) -> AuditFinding:
    """Verify ``equity_value + net_debt`` reconciles with ``enterprise_value``.

    A mismatch beyond :data:`EV_BRIDGE_REL_TOLERANCE` means the bridge was
    applied inconsistently (a classic error) and is a HARD failure.
    """
    implied = equity_value + net_debt
    scale = max(abs(enterprise_value), abs(implied), 1.0)
    rel_error = abs(implied - enterprise_value) / scale
    values = (
        ("equity_value", equity_value),
        ("net_debt", net_debt),
        ("enterprise_value", enterprise_value),
        ("implied_ev", implied),
    )
    if rel_error > EV_BRIDGE_REL_TOLERANCE:
        return fail(
            "ev_bridge_check",
            Severity.HARD,
            f"EV bridge mismatch: equity+net_debt={implied:.4f} vs "
            f"EV={enterprise_value:.4f} (rel error {rel_error:.4%})",
            values,
        )
    return ok(
        "ev_bridge_check",
        f"EV bridge reconciles within tolerance (rel error {rel_error:.4%})",
        values,
    )


def ttm_period_check(
    *,
    left_period: str,
    right_period: str,
) -> AuditFinding:
    """Flag mixing incompatible accounting periods.

    Combining a trailing-twelve-months figure with a single fiscal-year or
    fiscal-quarter figure double- or under-counts. Periods must match exactly;
    a mismatch is a HARD failure.
    """
    values: tuple[tuple[str, float], ...] = ()
    if left_period != right_period:
        return fail(
            "ttm_period_check",
            Severity.HARD,
            f"period mismatch: '{left_period}' combined with '{right_period}'",
            values,
        )
    return ok(
        "ttm_period_check",
        f"periods match ('{left_period}')",
        values,
    )


def sector_sign_check(
    *,
    ebitda: float,
    applied_multiple: float,
    margin: float,
) -> AuditFinding:
    """Flag structurally impossible signs / out-of-range margins.

    A positive multiple applied to negative EBITDA yields a negative implied
    value presented as positive — a sign blunder (HARD). A margin outside the
    plausible band is also structurally impossible (HARD).
    """
    values = (
        ("ebitda", ebitda),
        ("applied_multiple", applied_multiple),
        ("margin", margin),
    )
    if ebitda < 0.0 and applied_multiple > 0.0:
        return fail(
            "sector_sign_check",
            Severity.HARD,
            f"positive multiple {applied_multiple:.2f} applied to negative EBITDA {ebitda:.2f}",
            values,
        )
    if not MIN_PLAUSIBLE_MARGIN <= margin <= MAX_PLAUSIBLE_MARGIN:
        return fail(
            "sector_sign_check",
            Severity.HARD,
            f"margin {margin:.4f} outside plausible band "
            f"[{MIN_PLAUSIBLE_MARGIN}, {MAX_PLAUSIBLE_MARGIN}]",
            values,
        )
    return ok(
        "sector_sign_check",
        "signs and margin within structural bounds",
        values,
    )


def magnitude_sanity(
    *,
    price_per_share: float,
    pe: float,
    ev_ebitda: float,
) -> AuditFinding:
    """Flag values outside plausible orders of magnitude for US equities.

    A non-positive share price, a P/E beyond :data:`MAX_PLAUSIBLE_PE`, or an
    EV/EBITDA beyond :data:`MAX_PLAUSIBLE_EV_EBITDA` almost always indicates a
    near-zero denominator or a scale error. Reported HARD.
    """
    values = (
        ("price_per_share", price_per_share),
        ("pe", pe),
        ("ev_ebitda", ev_ebitda),
    )
    if price_per_share <= 0.0 or price_per_share > MAX_PLAUSIBLE_PRICE_PER_SHARE:
        return fail(
            "magnitude_sanity",
            Severity.HARD,
            f"price per share {price_per_share:.4f} outside (0, {MAX_PLAUSIBLE_PRICE_PER_SHARE}]",
            values,
        )
    if abs(pe) > MAX_PLAUSIBLE_PE:
        return fail(
            "magnitude_sanity",
            Severity.HARD,
            f"P/E {pe:.2f} beyond plausible max {MAX_PLAUSIBLE_PE}",
            values,
        )
    if abs(ev_ebitda) > MAX_PLAUSIBLE_EV_EBITDA:
        return fail(
            "magnitude_sanity",
            Severity.HARD,
            f"EV/EBITDA {ev_ebitda:.2f} beyond plausible max {MAX_PLAUSIBLE_EV_EBITDA}",
            values,
        )
    return ok(
        "magnitude_sanity",
        "price, P/E and EV/EBITDA within plausible magnitudes",
        values,
    )
