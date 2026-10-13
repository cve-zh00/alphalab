"""The deterministic rejection criteria — one pure function per criterion.

Each criterion inspects a :class:`~alphalab.domain.skeptic.candidate.TradeCandidate`
and returns a structured :class:`~alphalab.domain.compute.AuditFinding`
(reused, never a bare bool). ``passed is True`` means the criterion is satisfied
(no objection); ``passed is False`` means the skeptic objects, with the exact
numbers stamped into ``message`` and ``values`` so the rejection is auditable.

Every function is **pure and total**: no I/O, no mutation, and it must not
raise on any input — degenerate candidates (NaN, inf, empty series, negative
size) resolve to a HARD finding, never an exception. The aggregator relies on
this totality to guarantee it never explodes.

Criteria and their thresholds are documented in
:mod:`alphalab.domain.skeptic.thresholds` with the source of each number.
"""

from __future__ import annotations

import math

import numpy as np

from alphalab.domain.compute import AuditFinding, Severity
from alphalab.domain.costs import capacity_size, sharpe_gross_net
from alphalab.domain.skeptic import thresholds
from alphalab.domain.skeptic.candidate import TradeCandidate
from alphalab.domain.stats import (
    certify_strategy,
    effective_sample_size,
)


def _finding(
    check: str,
    *,
    passed: bool,
    severity: Severity,
    message: str,
    values: tuple[tuple[str, float], ...] = (),
) -> AuditFinding:
    """Build a finding, guaranteeing a non-empty message (model invariant)."""
    return AuditFinding(
        check=check,
        passed=passed,
        severity=severity,
        message=message or check,
        values=values,
    )


def _returns_array(candidate: TradeCandidate) -> np.ndarray:
    """Return the backtest returns as a float array (may be empty/degenerate)."""
    return np.asarray(candidate.backtest_returns, dtype=np.float64)


def check_sample_sufficiency(candidate: TradeCandidate) -> AuditFinding:
    """S1 — reject when the effective sample ``T_eff`` is below the floor."""
    arr = _returns_array(candidate)
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return _finding(
            "skeptic.sample_sufficiency",
            passed=False,
            severity=Severity.HARD,
            message=(
                f"effective sample unmeasurable: {arr.size} finite-checked observations "
                f"(need >= {thresholds.MIN_EFFECTIVE_SAMPLE:.0f})"
            ),
            values=(("n_obs", float(arr.size)), ("min_effective", thresholds.MIN_EFFECTIVE_SAMPLE)),
        )
    t_eff = effective_sample_size(arr)
    passed = t_eff >= thresholds.MIN_EFFECTIVE_SAMPLE
    return _finding(
        "skeptic.sample_sufficiency",
        passed=passed,
        severity=Severity.HARD,
        message=(
            f"T_eff {t_eff:.2f} {'>=' if passed else '<'} "
            f"minimum {thresholds.MIN_EFFECTIVE_SAMPLE:.0f}"
        ),
        values=(("t_eff", t_eff), ("min_effective", thresholds.MIN_EFFECTIVE_SAMPLE)),
    )


def check_cost_survival(candidate: TradeCandidate) -> AuditFinding:
    """S2 — reject when net Sharpe collapses below the floor or flips sign."""
    gross = _returns_array(candidate)
    drag = np.asarray(candidate.cost_fractions, dtype=np.float64)
    if (
        gross.size < 2
        or gross.shape != drag.shape
        or not np.all(np.isfinite(gross))
        or not np.all(np.isfinite(drag))
    ):
        return _finding(
            "skeptic.cost_survival",
            passed=False,
            severity=Severity.HARD,
            message=(
                "cost survival unmeasurable: returns/costs degenerate or misaligned "
                f"({gross.size} returns, {drag.size} cost points)"
            ),
            values=(("n_returns", float(gross.size)), ("n_costs", float(drag.size))),
        )
    result = sharpe_gross_net(tuple(gross), tuple(drag))
    sign_flipped = result.gross > 0.0 and result.net <= 0.0
    passed = (result.net >= thresholds.MIN_NET_SHARPE) and not sign_flipped
    reason = (
        f"net Sharpe {result.net:.4f} < floor {thresholds.MIN_NET_SHARPE:.4f}"
        if result.net < thresholds.MIN_NET_SHARPE
        else f"edge sign flipped under costs (gross {result.gross:.4f} -> net {result.net:.4f})"
        if sign_flipped
        else f"net Sharpe {result.net:.4f} survives costs (gross {result.gross:.4f})"
    )
    return _finding(
        "skeptic.cost_survival",
        passed=passed,
        severity=Severity.HARD,
        message=reason,
        values=(
            ("gross_sharpe", result.gross),
            ("net_sharpe", result.net),
            ("min_net_sharpe", thresholds.MIN_NET_SHARPE),
        ),
    )


def check_capacity(candidate: TradeCandidate) -> AuditFinding:
    """S3 — reject when the proposed size exceeds the capacity utilisation cap."""
    size = candidate.proposed_size
    if not math.isfinite(size) or size < 0.0:
        return _finding(
            "skeptic.capacity",
            passed=False,
            severity=Severity.HARD,
            message=f"proposed size is degenerate: {size!r}",
            values=(("proposed_size", _safe_float(size)),),
        )
    if candidate.adv <= 0.0 or candidate.sigma_bps <= 0.0 or candidate.gross_edge_bps <= 0.0:
        return _finding(
            "skeptic.capacity",
            passed=False,
            severity=Severity.HARD,
            message=(
                "capacity unmeasurable: non-positive adv/sigma/edge "
                f"(adv={candidate.adv}, sigma_bps={candidate.sigma_bps}, "
                f"gross_edge_bps={candidate.gross_edge_bps})"
            ),
            values=(
                ("adv", _safe_float(candidate.adv)),
                ("sigma_bps", _safe_float(candidate.sigma_bps)),
                ("gross_edge_bps", _safe_float(candidate.gross_edge_bps)),
            ),
        )
    ceiling = capacity_size(
        candidate.gross_edge_bps,
        candidate.adv,
        candidate.sigma_bps,
        eta=candidate.cost_spec.impact_eta or 1.0,
        alpha=candidate.cost_spec.impact_alpha,
    )
    allowed = thresholds.MAX_CAPACITY_UTILISATION * ceiling
    passed = ceiling > 0.0 and size <= allowed
    return _finding(
        "skeptic.capacity",
        passed=passed,
        severity=Severity.HARD,
        message=(
            f"size {size:.2f} {'<=' if passed else '>'} "
            f"{thresholds.MAX_CAPACITY_UTILISATION:.0%} of capacity ceiling {ceiling:.2f} "
            f"(= {allowed:.2f})"
        ),
        values=(
            ("proposed_size", size),
            ("capacity_ceiling", ceiling),
            ("allowed_size", allowed),
        ),
    )


def check_survivorship(candidate: TradeCandidate) -> AuditFinding:
    """S4 — reject when the universe excluded delisted names (upward bias)."""
    passed = candidate.provenance.includes_delisted
    return _finding(
        "skeptic.survivorship",
        passed=passed,
        severity=Severity.HARD,
        message=(
            "universe includes delisted names (no survivorship bias)"
            if passed
            else "universe excludes delisted names: results biased upward by survivorship"
        ),
        values=(("includes_delisted", float(passed)),),
    )


def check_leakage(candidate: TradeCandidate) -> AuditFinding:
    """S5 — hard reject on declared non-PIT resolution or future-dated lineage."""
    prov = candidate.provenance
    future_dated = (
        prov.latest_observed_at is not None and prov.latest_observed_at > prov.knowledge_date
    )
    passed = prov.pit_compliant and not future_dated
    if future_dated:
        message = (
            f"lineage observed_at {prov.latest_observed_at} is after "
            f"knowledge_date {prov.knowledge_date}: look-ahead leakage"
        )
    elif not prov.pit_compliant:
        message = "a feature was resolved outside the point-in-time path: declared leakage"
    else:
        message = "no leakage: PIT-compliant and lineage within knowledge date"
    return _finding(
        "skeptic.leakage",
        passed=passed,
        severity=Severity.HARD,
        message=message,
        values=(
            ("pit_compliant", float(prov.pit_compliant)),
            ("future_dated", float(future_dated)),
        ),
    )


def check_statistical_significance(
    candidate: TradeCandidate, *, rng: np.random.Generator
) -> AuditFinding:
    """S6 — reject on failed DSR / PBO gate / bootstrap CI / HLZ t-stat."""
    arr = _returns_array(candidate)
    if arr.size < 3 or not np.all(np.isfinite(arr)):
        return _finding(
            "skeptic.significance",
            passed=False,
            severity=Severity.HARD,
            message=(
                f"significance unmeasurable: {arr.size} finite-checked observations (need >= 3)"
            ),
            values=(("n_obs", float(arr.size)),),
        )
    reasons: list[str] = []
    try:
        config_matrix = arr.reshape(-1, 1)
        cert = certify_strategy(
            arr,
            config_matrix,
            n_trials=max(candidate.n_trials, 2),
            trial_sharpe_std=max(candidate.trial_sharpe_std, 0.0),
            rng=rng,
            n_pbo_null_sims=16,
            n_pbo_bootstrap=64,
            n_resamples=256,
        )
        if not cert.dsr > thresholds.DSR_THRESHOLD:
            reasons.append(f"DSR {cert.dsr:.4f} <= {thresholds.DSR_THRESHOLD:.2f}")
        if not cert.pbo_gate.passed:
            reasons.append(f"PBO gate red (pbo={cert.pbo_gate.pbo:.4f})")
        if not cert.bootstrap_ci.excludes_zero:
            reasons.append(
                f"bootstrap CI [{cert.bootstrap_ci.lower:.4f}, "
                f"{cert.bootstrap_ci.upper:.4f}] includes zero"
            )
        t_eff = cert.t_eff
        observed = cert.observed_sharpe
    except Exception as exc:  # totality: a stats failure becomes a hard finding
        return _finding(
            "skeptic.significance",
            passed=False,
            severity=Severity.HARD,
            message=f"significance evaluation failed: {type(exc).__name__}: {exc}",
            values=(("n_obs", float(arr.size)),),
        )
    t_stat = observed * math.sqrt(t_eff) if t_eff > 0.0 else 0.0
    if not t_stat > thresholds.HARVEY_LIU_ZHU_T_STAT:
        reasons.append(f"HLZ t-stat {t_stat:.3f} <= {thresholds.HARVEY_LIU_ZHU_T_STAT:.1f}")
    passed = not reasons
    message = "; ".join(reasons) if reasons else "DSR, PBO, bootstrap CI and HLZ t-stat all clear"
    return _finding(
        "skeptic.significance",
        passed=passed,
        severity=Severity.HARD,
        message=message,
        values=(("t_stat", t_stat), ("hlz_threshold", thresholds.HARVEY_LIU_ZHU_T_STAT)),
    )


def check_cross_robustness(candidate: TradeCandidate) -> AuditFinding:
    """S7 — reject when the edge survives in too few assets / sub-periods."""
    slices = np.asarray(candidate.per_slice_sharpe, dtype=np.float64)
    finite = slices[np.isfinite(slices)]
    if finite.size < thresholds.MIN_SLICES_FOR_ROBUSTNESS:
        return _finding(
            "skeptic.cross_robustness",
            passed=False,
            severity=Severity.SOFT,
            message=(
                f"only {finite.size} usable slice(s); need "
                f">= {thresholds.MIN_SLICES_FOR_ROBUSTNESS} to corroborate robustness"
            ),
            values=(("n_slices", float(finite.size)),),
        )
    positive_fraction = float((finite > 0.0).mean())
    passed = positive_fraction >= thresholds.MIN_ROBUST_SLICE_FRACTION
    return _finding(
        "skeptic.cross_robustness",
        passed=passed,
        severity=Severity.SOFT,
        message=(
            f"edge positive in {positive_fraction:.0%} of {finite.size} slices "
            f"({'>=' if passed else '<'} required {thresholds.MIN_ROBUST_SLICE_FRACTION:.0%})"
        ),
        values=(
            ("positive_fraction", positive_fraction),
            ("min_fraction", thresholds.MIN_ROBUST_SLICE_FRACTION),
        ),
    )


def check_contradictory_data(candidate: TradeCandidate) -> AuditFinding:
    """S8 — reject when a HARD compute-audit finding is attached to the candidate."""
    blocking = tuple(f for f in candidate.audit_findings if f.is_blocking())
    passed = not blocking
    if passed:
        message = "no blocking audit findings attached to the candidate"
    else:
        names = ", ".join(f.check for f in blocking)
        message = f"{len(blocking)} blocking audit finding(s): {names}"
    return _finding(
        "skeptic.contradictory_data",
        passed=passed,
        severity=Severity.HARD,
        message=message,
        values=(("n_blocking", float(len(blocking))),),
    )


def _safe_float(value: float) -> float:
    """Coerce a possibly non-finite float into something a finding can carry."""
    return value if math.isfinite(value) else 0.0


# Deterministic criteria evaluated without randomness, in a fixed report order.
DETERMINISTIC_CRITERIA = (
    check_sample_sufficiency,
    check_cost_survival,
    check_capacity,
    check_survivorship,
    check_leakage,
    check_cross_robustness,
    check_contradictory_data,
)
