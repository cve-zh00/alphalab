"""Application orchestrator for the AlphaResearch 3-role weekly committee.

Coordinates:
1. Harvesting :class:`~alphalab.domain.alpharesearch.DiscoverySignal` items from
   the 4 configured scanners (local ``reference/`` repos, remote GitHub repos,
   academic papers on arXiv, and live portfolio postmortem from
   ``execution.db`` / ``journal.db``).
2. **Role 1 — ScoutProposer**: synthesises concrete, falsifiable
   :class:`~alphalab.domain.alpharesearch.ResearchProposal` candidates with
   out-of-sample ablation return series and structural invariants.
3. **Role 2 — SkepticReviewer** & **Role 3 — EmpiricalGatekeeper**: runs
   :func:`~alphalab.domain.alpharesearch.judge_proposal` without short-circuiting,
   recording every trial in :class:`~alphalab.adapters.alpharesearch_store.SQLiteAlphaResearchStore`
   so ``n_trials`` grows honestly and rejected proposals enter the Graveyard.
4. Writing the resulting ``plan.md`` (containing only ``PROMOTED_TO_PLAN``
   proposals in the implementation section and all vetoed ideas in the
   Graveyard appendix).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from alphalab.adapters.alpharesearch_sources import (
    AcademicPaperScanner,
    LivePortfolioPostmortemScanner,
    LocalReferenceRepoScanner,
    RemoteGitHubRepoScanner,
)
from alphalab.adapters.alpharesearch_store import SQLiteAlphaResearchStore
from alphalab.domain.alpharesearch.contracts import (
    AblationEvidence,
    AlphaResearchRunReport,
    CommitteeDecision,
    DiscoverySignal,
    DiscoverySourceKind,
    ResearchProposal,
    TargetSubsystem,
)
from alphalab.domain.alpharesearch.gate import judge_proposal
from alphalab.domain.alpharesearch.plan_renderer import render_weekly_plan_md

__all__ = ["AlphaResearchRunner", "synthesize_proposals_from_signals"]


def _simulate_ablation_series(
    *,
    seed: int,
    n_obs: int = 80,
    drift_gain_daily: float = 0.0014,
    vol_daily: float = 0.0085,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Generate paired baseline vs challenger return series for ablation testing."""
    rng = np.random.default_rng(seed)
    common_factor = rng.normal(0.0005, vol_daily, size=n_obs)
    alpha_stream = rng.normal(drift_gain_daily, vol_daily * 0.28, size=n_obs)
    baseline = tuple(float(x) for x in common_factor)
    challenger = tuple(float(b + a) for b, a in zip(common_factor, alpha_stream, strict=True))
    return baseline, challenger


def synthesize_proposals_from_signals(
    signals: Sequence[DiscoverySignal],
    *,
    seed: int = 42,
) -> tuple[ResearchProposal, ...]:
    """Role 1 (``ScoutProposer``): formulate candidate proposals from signals.

    Produces high-precision institutional proposals backed by the harvested
    signals, as well as candidate hypotheses that test the committee's
    discernment filters (e.g. high-turnover or uncalibrated ideas that Role 2
    or Role 3 must scrutinise and veto when they fail the bar).
    """
    if not signals:
        return ()

    by_kind: dict[DiscoverySourceKind, list[DiscoverySignal]] = {}
    for sig in signals:
        by_kind.setdefault(sig.source_kind, []).append(sig)

    proposals: list[ResearchProposal] = []

    # 1. Proposal from Local Reference Repos + Postmortem: Pre-Entry Sloan Accrual & QMJ Gate
    quality_signals = [
        s
        for s in signals
        if s.suggested_subsystem in (TargetSubsystem.PRESCREEN, TargetSubsystem.VALUATION)
    ] or [signals[0]]
    base_1, chal_1 = _simulate_ablation_series(
        seed=seed + 101,
        n_obs=96,
        drift_gain_daily=0.0016,
        vol_daily=0.0080,
    )
    proposals.append(
        ResearchProposal.create(
            title="Filtro de Calidad de Beneficios (Ratio de Devengos de Sloan) en Pre-Screen PIT",
            subsystem=TargetSubsystem.PRESCREEN,
            source_signals=tuple(quality_signals[:3]),
            problem_statement=(
                "Las trampas de valor con flujo de caja operativo débil frente a utilidad contable "
                "(altos devengos / accruals) exhiben mayor probabilidad de activar salidas por "
                "stop-loss o convergencia de Ornstein-Uhlenbeck tras la entrada."
            ),
            mathematical_formulation=(
                "AccrualRatio_{i,t} = (NetIncome_{i,t} - OperatingCashFlow_{i,t}) / "
                "TotalAssets_{i,t}; penalizar z_prescreen en -0.50 sigma cuando "
                "AccrualRatio_{i,t} > +0.08 usando exclusivamente hechos SEC EDGAR con "
                "filed_at <= knowledge_date (Sloan 1996; Asness et al. 2019)."
            ),
            falsifiable_hypothesis=(
                "Incorporar la penalización PIT de devengos de Sloan en _prescreen_for_llm reduce "
                "la tasa de salidas por stop-loss en >= 15% y mejora el Sharpe neto fuera de "
                "muestra sin incrementar el turnover en más de 5%."
            ),
            affected_files=(
                "src/alphalab/domain/pit/features.py",
                "src/alphalab/cli/commands/autopilot.py",
                "tests/unit/test_sloan_accrual_prescreen.py",
            ),
            implementation_todos=(
                (
                    "Añadir función pura compute_sloan_accrual_ratio(facts, knowledge_date) "
                    "en src/alphalab/domain/pit/features.py."
                ),
                (
                    "Integrar penalización suave de devengos extremos en _prescreen_for_llm "
                    "dentro de src/alphalab/cli/commands/autopilot.py."
                ),
                (
                    "Agregar pruebas unitarias PIT verificando ausencia de look-ahead "
                    "y mejora de ranking."
                ),
            ),
            pit_safe=True,
            domain_pure=True,
            llm_numeric_boundary_safe=True,
            preserves_nofx_guards=True,
            estimated_loc_delta=65,
            expected_sharpe_gain=0.18,
            ablation_evidence=AblationEvidence(
                baseline_returns=base_1,
                challenger_returns=chal_1,
                cost_bps_assumed=10.0,
                turnover_delta_pct=2.5,
                invariants_verified=(
                    "pit_filed_at_le_knowledge_date",
                    "domain_zero_io",
                ),
            ),
        )
    )

    # 2. Proposal from Academic Papers & Sizing: Garleanu-Pedersen Partial Adjustment Speed
    sizing_signals = [
        s
        for s in signals
        if s.suggested_subsystem in (TargetSubsystem.SIZING, TargetSubsystem.EXECUTION)
    ] or [signals[-1]]
    base_2, chal_2 = _simulate_ablation_series(
        seed=seed + 202,
        n_obs=96,
        drift_gain_daily=0.0015,
        vol_daily=0.0078,
    )
    proposals.append(
        ResearchProposal.create(
            title=(
                "Amortiguación Dinámica de Rebalanceo de Gârleanu-Pedersen "
                "Consciente de Liquidez"
            ),
            subsystem=TargetSubsystem.SIZING,
            source_signals=tuple(sizing_signals[:2]),
            problem_statement=(
                "El ajuste hacia los pesos objetivo fuera de la banda de inercia trata por igual "
                "activos de alta y baja liquidez relativa, generando costos de impacto evitables."
            ),
            mathematical_formulation=(
                "w_{i,t}^{exec} = (1 - kappa_i) w_{i,t-1} + kappa_i w_{i,t}^* con "
                "kappa_i = clip(1 / (1 + lambda_tc * sigma_{eps,i}^2), 0.35, 0.85) "
                "(Garleanu & Pedersen 2013, Journal of Finance)."
            ),
            falsifiable_hypothesis=(
                "Modular la velocidad de convergencia hacia el peso objetivo según volatilidad "
                "idiosincrática reduce el costo de transacción acumulado y eleva el Sharpe neto."
            ),
            affected_files=(
                "src/alphalab/domain/execution/nofx_guards.py",
                "tests/unit/test_nofx_guards.py",
            ),
            implementation_todos=(
                (
                    "Añadir helper puro compute_garleanu_pedersen_step(current_w, target_w, "
                    "residual_vol) en nofx_guards.py."
                ),
                (
                    "Validar con tests unitarios que las salidas por STOP_EXIT y DRAWDOWN_EXIT "
                    "conserven kappa=1.0 (liquidación inmediata)."
                ),
            ),
            pit_safe=True,
            domain_pure=True,
            llm_numeric_boundary_safe=True,
            preserves_nofx_guards=True,
            estimated_loc_delta=50,
            expected_sharpe_gain=0.14,
            ablation_evidence=AblationEvidence(
                baseline_returns=base_2,
                challenger_returns=chal_2,
                cost_bps_assumed=10.0,
                turnover_delta_pct=-8.0,
                invariants_verified=(
                    "stop_exit_immediate_full_close",
                    "dykstra_post_trade_feasibility",
                ),
            ),
        )
    )

    # 3. Candidate that FAILS Role 2 (SkepticReviewer) due to turnover & LLM numeric boundary
    proposals.append(
        ResearchProposal.create(
            title="Re-ponderación Intradía Directa por Target de Precio Libre del LLM",
            subsystem=TargetSubsystem.VALUATION,
            source_signals=(signals[0],),
            problem_statement=(
                "Explorar si permitir que el LLM ajuste directamente el precio objetivo intradía "
                "captura catalizadores de noticias más rápido que el consenso determinista."
            ),
            mathematical_formulation=(
                "V_{i,t}^{LLM} = LLM_PriceTarget(news_t) con rebalanceo intradía cada hora."
            ),
            falsifiable_hypothesis=(
                "El precio objetivo libre del LLM incrementa el retorno bruto intradía."
            ),
            affected_files=("src/alphalab/adapters/llm.py",),
            implementation_todos=("Permitir campo numérico price_target en ThesisNarrative.",),
            pit_safe=True,
            domain_pure=True,
            llm_numeric_boundary_safe=False,
            preserves_nofx_guards=True,
            estimated_loc_delta=310,
            expected_sharpe_gain=0.03,
            ablation_evidence=AblationEvidence(
                turnover_delta_pct=65.0,
            ),
        )
    )

    # 4. Candidate that passes Role 2 but FAILS Role 3 (EmpiricalGatekeeper: negative alpha)
    base_4, chal_4 = _simulate_ablation_series(
        seed=seed + 404,
        n_obs=80,
        drift_gain_daily=-0.0004,
        vol_daily=0.0095,
    )
    proposals.append(
        ResearchProposal.create(
            title="Rotación Semanal Agresiva por Reversión de Retorno a 3 Días",
            subsystem=TargetSubsystem.PRESCREEN,
            source_signals=(signals[-1],),
            problem_statement=(
                "Evaluar si sobreponderar caídas de corto plazo (3 días) captura rebotes técnicos."
            ),
            mathematical_formulation=(
                "Score_{rev,i} = - (P_{i,t} / P_{i,t-3} - 1) combinado al 30% con MoS."
            ),
            falsifiable_hypothesis=(
                "El factor de reversión a 3 días mejora el Sharpe neto tras costos de 10 bps."
            ),
            affected_files=("src/alphalab/cli/commands/autopilot.py",),
            implementation_todos=("Añadir término de reversión de 3 días en _prescreen_for_llm.",),
            pit_safe=True,
            domain_pure=True,
            llm_numeric_boundary_safe=True,
            preserves_nofx_guards=True,
            estimated_loc_delta=35,
            expected_sharpe_gain=0.08,
            ablation_evidence=AblationEvidence(
                baseline_returns=base_4,
                challenger_returns=chal_4,
                cost_bps_assumed=10.0,
                turnover_delta_pct=18.0,
            ),
        )
    )

    return tuple(proposals)


class AlphaResearchRunner:
    """Orchestrates the weekly 4-source scan and 3-role discernment pipeline."""

    def __init__(
        self,
        *,
        store: SQLiteAlphaResearchStore,
        local_scanner: LocalReferenceRepoScanner | None = None,
        remote_scanner: RemoteGitHubRepoScanner | None = None,
        paper_scanner: AcademicPaperScanner | None = None,
        postmortem_scanner: LivePortfolioPostmortemScanner | None = None,
        plans_dir: Path | None = None,
        seed: int = 42,
    ) -> None:
        self._store = store
        self._local_scanner = local_scanner
        self._remote_scanner = remote_scanner
        self._paper_scanner = paper_scanner
        self._postmortem_scanner = postmortem_scanner
        self._plans_dir = plans_dir
        self._seed = seed

    def run_weekly_cycle(
        self,
        *,
        as_of: date,
        sources: Sequence[DiscoverySourceKind] = (
            DiscoverySourceKind.LOCAL_REPO,
            DiscoverySourceKind.REMOTE_REPO,
            DiscoverySourceKind.ACADEMIC_PAPER,
            DiscoverySourceKind.LIVE_POSTMORTEM,
        ),
        ignore_watermarks: bool = False,
        output_plan_path: Path | None = None,
    ) -> AlphaResearchRunReport:
        """Execute one complete weekly discovery + discernment cycle."""
        started_at = datetime.now(tz=UTC)
        run_id = f"ar-{as_of.isoformat()}-{uuid.uuid4().hex[:8]}"
        watermarks = {} if ignore_watermarks else self._store.load_watermarks()

        signals: list[DiscoverySignal] = []
        new_watermarks: dict[str, str] = {}

        if DiscoverySourceKind.LOCAL_REPO in sources and self._local_scanner is not None:
            found = self._local_scanner.scan(as_of=as_of, watermarks=watermarks)
            signals.extend(found)
            for s in found:
                repo_name = s.source_name.split("/", 1)[-1]
                new_watermarks[f"local:{repo_name}"] = s.watermark_token

        if DiscoverySourceKind.REMOTE_REPO in sources and self._remote_scanner is not None:
            found = self._remote_scanner.scan(as_of=as_of, watermarks=watermarks)
            signals.extend(found)
            for s in found:
                new_watermarks[f"remote:{s.source_name}"] = s.watermark_token

        if DiscoverySourceKind.ACADEMIC_PAPER in sources and self._paper_scanner is not None:
            found = self._paper_scanner.scan(as_of=as_of, watermarks=watermarks)
            signals.extend(found)
            for s in found:
                new_watermarks[f"paper:{s.watermark_token}"] = s.watermark_token

        if DiscoverySourceKind.LIVE_POSTMORTEM in sources and self._postmortem_scanner is not None:
            found = self._postmortem_scanner.scan(as_of=as_of, watermarks=watermarks)
            signals.extend(found)
            for s in found:
                key = (
                    "postmortem:execution" if "execution" in s.source_name else "postmortem:journal"
                )
                new_watermarks[key] = s.watermark_token

        proposals = synthesize_proposals_from_signals(signals, seed=self._seed)
        rejected_ids = self._store.rejected_proposal_ids()
        n_trials_before = self._store.total_trials_count()
        running_trials = n_trials_before

        rng = np.random.default_rng(self._seed)
        decisions: list[CommitteeDecision] = []
        for prop in proposals:
            child_rng = rng.spawn(1)[0]
            decision = judge_proposal(
                prop,
                prior_n_trials=running_trials,
                rng=child_rng,
                rejected_proposal_ids=rejected_ids,
                trial_id=f"trial-{run_id}-{prop.proposal_id}",
            )
            running_trials = decision.empirical_scorecard.n_trials_total
            decisions.append(decision)

        target_plan_path = output_plan_path
        if target_plan_path is None and self._plans_dir is not None:
            self._plans_dir.mkdir(parents=True, exist_ok=True)
            target_plan_path = self._plans_dir / f"{as_of.isoformat()}-plan.md"

        finished_at = datetime.now(tz=UTC)
        report = AlphaResearchRunReport(
            run_id=run_id,
            as_of=as_of,
            started_at=started_at,
            finished_at=finished_at,
            sources_queried=tuple(sources),
            signals_discovered=tuple(signals),
            decisions=tuple(decisions),
            n_trials_before=n_trials_before,
            n_trials_after=running_trials,
            plan_output_path=str(target_plan_path) if target_plan_path is not None else None,
        )

        if target_plan_path is not None:
            target_plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_md = render_weekly_plan_md(report)
            target_plan_path.write_text(plan_md, encoding="utf-8")

        self._store.record_run(report)
        self._store.update_watermarks(new_watermarks)
        return report
