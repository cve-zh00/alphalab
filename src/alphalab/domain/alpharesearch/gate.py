"""Pure 3-role discernment engine for the AlphaResearch weekly committee.

Every candidate :class:`ResearchProposal` produced by Role 1 (``ScoutProposer``)
must pass both:

* **Role 2 — SkepticReviewer** (:func:`run_skeptic_review`): evaluates 6
  non-short-circuiting methodological and architectural criteria (graveyard
  anti-snooping deduplication, Point-in-Time discipline, LLM numeric boundary,
  hexagonal domain purity, NOFX risk guard preservation, and Occam's Razor /
  turnover drag).
* **Role 3 — EmpiricalGatekeeper** (:func:`run_empirical_gatekeeper`):
  increments the honest trial counter ``n_trials`` and certifies the proposal
  against the current production baseline using Deflated Sharpe Ratio (DSR),
  hardened PBO/CSCV, Politis-Romano stationary block-bootstrap Sharpe
  difference, and maximum-drawdown non-degradation.

Pure domain module: numpy/pydantic only, no I/O, no filesystem, no network.
"""

from __future__ import annotations

import numpy as np

from alphalab.domain.alpharesearch.contracts import (
    CommitteeDecision,
    EmpiricalScorecard,
    ProposalDecisionStatus,
    ResearchProposal,
    SkepticCritique,
)
from alphalab.domain.compute import AuditFinding, Severity
from alphalab.domain.experiment.difference import paired_sharpe_difference
from alphalab.domain.stats._moments import sharpe_ratio
from alphalab.domain.stats.certification import DSR_THRESHOLD, certify_strategy

__all__ = [
    "MAX_COMPLEXITY_LOC_DELTA",
    "MAX_DRAWDOWN_DEGRADATION_TOLERANCE",
    "MAX_TURNOVER_INCREASE_PCT",
    "judge_proposal",
    "run_empirical_gatekeeper",
    "run_skeptic_review",
]

MAX_COMPLEXITY_LOC_DELTA: int = 250
MAX_TURNOVER_INCREASE_PCT: float = 25.0
MAX_DRAWDOWN_DEGRADATION_TOLERANCE: float = 0.015

_LOOKAHEAD_RED_FLAGS: tuple[str, ...] = (
    "future_close",
    "next_day_open_as_signal",
    "ignore filed_at",
    "bypass pit",
    "full_sample_zscore",
    "in_sample_only",
)


def _max_drawdown(returns: tuple[float, ...]) -> float:
    if not returns:
        return 0.0
    arr = np.asarray(returns, dtype=np.float64)
    equity = np.cumprod(1.0 + arr)
    peaks = np.maximum.accumulate(equity)
    dd = equity / peaks - 1.0
    return float(dd.min())


def run_skeptic_review(
    proposal: ResearchProposal,
    *,
    rejected_proposal_ids: frozenset[str] = frozenset(),
    max_loc_delta: int = MAX_COMPLEXITY_LOC_DELTA,
    max_turnover_increase_pct: float = MAX_TURNOVER_INCREASE_PCT,
) -> SkepticCritique:
    """Run all 6 Role-2 skeptic checks without short-circuiting."""
    findings: list[AuditFinding] = []
    safeguards: list[str] = []

    # 1. Graveyard anti-snooping deduplication
    is_repeat = proposal.proposal_id in rejected_proposal_ids
    findings.append(
        AuditFinding(
            check="alpharesearch.skeptic.graveyard_dedup",
            passed=not is_repeat,
            severity=Severity.HARD,
            message=(
                f"proposal {proposal.proposal_id} was already rejected in a prior committee cycle"
                if is_repeat
                else f"proposal {proposal.proposal_id} is novel"
            ),
            values=(),
        )
    )

    # 2. Point-in-Time (PIT) anti-leakage check
    text_blob = f"{proposal.mathematical_formulation} {proposal.falsifiable_hypothesis}".lower()
    flagged_terms = [term for term in _LOOKAHEAD_RED_FLAGS if term in text_blob]
    pit_ok = proposal.pit_safe and not flagged_terms
    findings.append(
        AuditFinding(
            check="alpharesearch.skeptic.pit_discipline",
            passed=pit_ok,
            severity=Severity.HARD,
            message=(
                "proposal respects Point-in-Time (filed_at <= knowledge_date) discipline"
                if pit_ok
                else (
                    f"PIT look-ahead risk detected "
                    f"(pit_safe={proposal.pit_safe}, flags={flagged_terms})"
                )
            ),
            values=(),
        )
    )
    safeguards.append("Verify test_pit_no_leakage passes on all modified operators.")

    # 3. LLM numeric boundary invariant
    llm_ok = proposal.llm_numeric_boundary_safe
    findings.append(
        AuditFinding(
            check="alpharesearch.skeptic.llm_numeric_boundary",
            passed=llm_ok,
            severity=Severity.HARD,
            message=(
                "LLM numeric boundary preserved (code computes numbers; LLM only argues)"
                if llm_ok
                else (
                    "proposal violates LLM numeric boundary by letting LLM emit "
                    "free numbers/weights"
                )
            ),
            values=(),
        )
    )

    # 4. Hexagonal domain purity invariant
    domain_ok = proposal.domain_pure
    findings.append(
        AuditFinding(
            check="alpharesearch.skeptic.domain_purity",
            passed=domain_ok,
            severity=Severity.HARD,
            message=(
                "Hexagonal domain purity preserved (zero I/O in src/alphalab/domain/)"
                if domain_ok
                else "proposal introduces I/O or adapter coupling inside src/alphalab/domain/"
            ),
            values=(),
        )
    )

    # 5. NOFX execution & risk guard preservation
    guards_ok = proposal.preserves_nofx_guards
    findings.append(
        AuditFinding(
            check="alpharesearch.skeptic.nofx_risk_preservation",
            passed=guards_ok,
            severity=Severity.HARD,
            message=(
                "NOFX runtime risk guards (Lot-1 stop, O-U drawdown exit, Dykstra caps) preserved"
                if guards_ok
                else (
                    "proposal weakens NOFX runtime stop-loss, drawdown, "
                    "or sector concentration guards"
                )
            ),
            values=(),
        )
    )
    safeguards.append("Keep Lot-1 stop-loss anchoring and Dykstra post-trade projection active.")

    # 6. Occam's Razor complexity & turnover drag check
    turnover_delta = proposal.ablation_evidence.turnover_delta_pct
    loc_ok = proposal.estimated_loc_delta <= max_loc_delta
    efficiency_ok = not (
        proposal.estimated_loc_delta > 150 and proposal.expected_sharpe_gain < 0.05
    )
    turnover_ok = turnover_delta <= max_turnover_increase_pct
    occam_ok = loc_ok and efficiency_ok and turnover_ok
    findings.append(
        AuditFinding(
            check="alpharesearch.skeptic.occam_and_turnover",
            passed=occam_ok,
            severity=Severity.HARD,
            message=(
                f"Occam & turnover check passed (loc_delta={proposal.estimated_loc_delta}, "
                f"turnover_delta={turnover_delta:+.1f}%)"
                if occam_ok
                else (
                    f"Excessive complexity or turnover drag: "
                    f"loc_delta={proposal.estimated_loc_delta} (max {max_loc_delta}), "
                    f"expected_sharpe_gain={proposal.expected_sharpe_gain:.3f}, "
                    f"turnover_delta={turnover_delta:+.1f}% "
                    f"(max +{max_turnover_increase_pct:.1f}%)"
                )
            ),
            values=(
                ("loc_delta", float(proposal.estimated_loc_delta)),
                ("expected_sharpe_gain", float(proposal.expected_sharpe_gain)),
                ("turnover_delta", float(turnover_delta)),
            ),
        )
    )

    veto_reasons = tuple(f"{f.check}: {f.message}" for f in findings if not f.passed)
    n_failed = len(veto_reasons)
    base_risk = min(1.0, n_failed / len(findings))
    complexity_bump = min(0.25, max(0, proposal.estimated_loc_delta) / 1000.0)
    risk_score = round(min(1.0, base_risk + (0.0 if n_failed else complexity_bump)), 4)

    return SkepticCritique(
        accepted=(n_failed == 0),
        risk_score=risk_score,
        findings=tuple(findings),
        veto_reasons=veto_reasons,
        required_safeguards=tuple(safeguards),
    )


def run_empirical_gatekeeper(
    proposal: ResearchProposal,
    *,
    prior_n_trials: int,
    rng: np.random.Generator,
    trial_id: str | None = None,
    trial_sharpe_std: float | None = None,
    dsr_threshold: float = DSR_THRESHOLD,
    n_resamples: int = 1_000,
    n_pbo_bootstrap: int = 200,
    n_pbo_null_sims: int = 40,
) -> EmpiricalScorecard:
    """Run Role-3 statistical & invariant certification for ``proposal``."""
    n_trials_total = max(1, prior_n_trials + 1)
    ablation = proposal.ablation_evidence
    base_ret = ablation.baseline_returns
    chal_ret = ablation.challenger_returns

    if len(base_ret) >= 20 and len(chal_ret) == len(base_ret):
        base_arr = np.asarray(base_ret, dtype=np.float64)
        chal_arr = np.asarray(chal_ret, dtype=np.float64)

        if ablation.configuration_matrix_rows:
            config_matrix = np.asarray(ablation.configuration_matrix_rows, dtype=np.float64)
        else:
            # Construct a realistic multi-configuration matrix around baseline & challenger
            noise_rng, cert_rng, diff_rng = rng.spawn(3)
            synth_cols = [
                base_arr,
                chal_arr,
                0.5 * (base_arr + chal_arr),
                0.75 * chal_arr + 0.25 * base_arr,
            ]
            for _ in range(4):
                jitter = noise_rng.normal(
                    0.0, max(1e-4, float(np.std(base_arr)) * 0.15), len(base_arr)
                )
                synth_cols.append(chal_arr + jitter)
            config_matrix = np.column_stack(synth_cols)
        if ablation.configuration_matrix_rows:
            cert_rng, diff_rng = rng.spawn(2)

        if trial_sharpe_std is None:
            col_srs = [
                float(sharpe_ratio(config_matrix[:, j]))
                for j in range(config_matrix.shape[1])
            ]
            empirical_std = float(np.std(col_srs, ddof=1)) if len(col_srs) > 1 else 0.03
            effective_sr_std = max(0.02, empirical_std)
        else:
            effective_sr_std = max(1e-3, trial_sharpe_std)

        n_obs = len(chal_arr)
        n_strips = 8 if n_obs >= 32 else 4
        cert = certify_strategy(
            chal_arr,
            config_matrix,
            n_trials=max(2, n_trials_total),
            trial_sharpe_std=effective_sr_std,
            rng=cert_rng,
            n_resamples=n_resamples,
            n_strips=n_strips,
            n_pbo_bootstrap=n_pbo_bootstrap,
            n_pbo_null_sims=n_pbo_null_sims,
        )
        diff = paired_sharpe_difference(
            chal_ret,
            base_ret,
            name_a="challenger",
            name_b="baseline",
            rng=diff_rng,
            alpha=0.05,
            n_comparisons=1,
            n_resamples=n_resamples,
        )
        sr_base = float(sharpe_ratio(base_arr))
        sr_chal = float(sharpe_ratio(chal_arr))
        mdd_base = _max_drawdown(base_ret)
        mdd_chal = _max_drawdown(chal_ret)
        dd_delta = mdd_chal - mdd_base

        reasons: list[str] = []
        if cert.dsr <= dsr_threshold:
            reasons.append(
                f"DSR {cert.dsr:.4f} <= threshold {dsr_threshold:.2f} (n_trials={n_trials_total})"
            )
        if not cert.bootstrap_ci.excludes_zero:
            reasons.append(
                f"Challenger Sharpe CI [{cert.bootstrap_ci.lower:.4f}, "
                f"{cert.bootstrap_ci.upper:.4f}] includes zero"
            )
        if diff.delta <= 0.0 or not diff.excludes_zero:
            reasons.append(
                f"Paired Sharpe gain {diff.delta:+.4f} CI [{diff.ci_low:+.4f}, "
                f"{diff.ci_high:+.4f}] does not beat baseline"
            )
        if dd_delta < -MAX_DRAWDOWN_DEGRADATION_TOLERANCE:
            tol_pct = MAX_DRAWDOWN_DEGRADATION_TOLERANCE * 100
            reasons.append(
                f"Max drawdown worsened by {dd_delta * 100:.2f}% beyond tolerance (-{tol_pct:.2f}%)"
            )

        return EmpiricalScorecard(
            accepted=(len(reasons) == 0),
            evaluation_mode="backtest_certified",
            n_trials_total=n_trials_total,
            trial_id=trial_id,
            observed_sharpe_baseline=sr_base,
            observed_sharpe_challenger=sr_chal,
            sharpe_delta=float(diff.delta),
            sharpe_diff_ci_lower=float(diff.ci_low),
            sharpe_diff_ci_upper=float(diff.ci_high),
            dsr=float(cert.dsr),
            pbo=float(cert.pbo),
            max_drawdown_baseline=mdd_base,
            max_drawdown_challenger=mdd_chal,
            drawdown_delta=dd_delta,
            rejection_reasons=tuple(reasons),
        )

    if ablation.invariants_verified:
        return EmpiricalScorecard(
            accepted=True,
            evaluation_mode="structural_invariant",
            n_trials_total=n_trials_total,
            trial_id=trial_id,
            rejection_reasons=(),
        )

    return EmpiricalScorecard(
        accepted=False,
        evaluation_mode="insufficient_evidence",
        n_trials_total=n_trials_total,
        trial_id=trial_id,
        rejection_reasons=(
            "No out-of-sample ablation return series (>=20 obs) or deterministic "
            "invariant proof attached",
        ),
    )


def judge_proposal(
    proposal: ResearchProposal,
    *,
    prior_n_trials: int,
    rng: np.random.Generator,
    rejected_proposal_ids: frozenset[str] = frozenset(),
    trial_id: str | None = None,
    trial_sharpe_std: float | None = None,
) -> CommitteeDecision:
    """Run all 3 committee roles on ``proposal`` and emit an immutable decision."""
    skeptic = run_skeptic_review(
        proposal,
        rejected_proposal_ids=rejected_proposal_ids,
    )
    scorecard = run_empirical_gatekeeper(
        proposal,
        prior_n_trials=prior_n_trials,
        rng=rng,
        trial_id=trial_id,
        trial_sharpe_std=trial_sharpe_std,
    )

    if not skeptic.accepted:
        status = ProposalDecisionStatus.REJECTED_BY_SKEPTIC
        reason = "; ".join(skeptic.veto_reasons)
    elif not scorecard.accepted:
        if scorecard.evaluation_mode == "insufficient_evidence":
            status = ProposalDecisionStatus.HOLD_FOR_EVIDENCE
        else:
            status = ProposalDecisionStatus.REJECTED_BY_EMPIRICAL_GATE
        reason = "; ".join(scorecard.rejection_reasons)
    else:
        status = ProposalDecisionStatus.PROMOTED_TO_PLAN
        if scorecard.evaluation_mode == "backtest_certified":
            reason = (
                f"Certified by all 3 roles: DSR={scorecard.dsr:.3f}, "
                f"Sharpe delta={scorecard.sharpe_delta:+.3f} "
                f"[{scorecard.sharpe_diff_ci_lower:+.3f}, {scorecard.sharpe_diff_ci_upper:+.3f}], "
                f"MDD delta={scorecard.drawdown_delta:+.2%} (n_trials={scorecard.n_trials_total})"
            )
        else:
            reason = (
                f"Certified via deterministic structural invariants "
                f"({len(proposal.ablation_evidence.invariants_verified)} invariants verified, "
                f"skeptic risk={skeptic.risk_score:.2f})"
            )

    return CommitteeDecision(
        proposal=proposal,
        skeptic_critique=skeptic,
        empirical_scorecard=scorecard,
        status=status,
        summary_reason=reason,
    )
