"""Pure Markdown generator for the AlphaResearch weekly ``plan.md`` artifact.

Only proposals with status :attr:`ProposalDecisionStatus.PROMOTED_TO_PLAN` are
promoted into the actionable implementation sections of ``plan.md``. Proposals
vetoed by Role 2 (``SkepticReviewer``) or Role 3 (``EmpiricalGatekeeper``) are
recorded in the **Graveyard / Rejected Proposals Appendix** at the bottom of the
document so the user has complete visibility into what was filtered out and why.

Pure domain module: string formatting only, no I/O.
"""

from __future__ import annotations

from alphalab.domain.alpharesearch.contracts import (
    AlphaResearchRunReport,
    CommitteeDecision,
)

__all__ = ["render_weekly_plan_md"]


def _format_scorecard_line(decision: CommitteeDecision) -> str:
    sc = decision.empirical_scorecard
    if sc.evaluation_mode == "backtest_certified" and sc.dsr is not None:
        dd_val = sc.drawdown_delta or 0.0
        return (
            f"Mode=`backtest_certified` | DSR=`{sc.dsr:.3f}` | PBO=`{sc.pbo:.3f}` | "
            f"ΔSharpe=`{sc.sharpe_delta:+.3f}` CI `[{(sc.sharpe_diff_ci_lower or 0.0):+.3f}, "
            f"{(sc.sharpe_diff_ci_upper or 0.0):+.3f}]` | ΔMDD=`{dd_val:+.2%}` | "
            f"Trial `N={sc.n_trials_total}`"
        )
    if sc.evaluation_mode == "structural_invariant":
        invs = ", ".join(decision.proposal.ablation_evidence.invariants_verified)
        return (
            f"Mode=`structural_invariant` | Verified Invariants: `{invs}` | "
            f"Trial `N={sc.n_trials_total}`"
        )
    return f"Mode=`{sc.evaluation_mode}` | Trial `N={sc.n_trials_total}`"


def render_weekly_plan_md(report: AlphaResearchRunReport) -> str:
    """Render an actionable ``plan.md`` from a weekly :class:`AlphaResearchRunReport`."""
    promoted = report.promoted_decisions
    rejected = report.rejected_decisions
    sources_str = ", ".join(s.value for s in report.sources_queried)

    lines: list[str] = [
        f"# Plan: AlphaResearch Weekly Committee Synthesis ({report.as_of.isoformat()})",
        "",
        "## Problema y Enfoque Propuesto",
        (
            f"Síntesis semanal generada por el **Comité de Discernimiento AlphaResearch** "
            f"(`run_id={report.run_id}`). "
            f"Se escanearon **{len(report.signals_discovered)} señales** desde `{sources_str}`, "
            f"se formularon **{len(report.decisions)} propuestas formales** y se promovieron "
            f"únicamente **{len(promoted)} mejoras certificadas** tras superar los 3 roles "
            f"adversarios (`ScoutProposer` → `SkepticReviewer` → `EmpiricalGatekeeper`, "
            f"contador acumulado de ensayos `n_trials: {report.n_trials_before} → "
            f"{report.n_trials_after}`)."
        ),
        "",
        "### Resumen de Discernimiento del Comité",
        f"- **Fecha de corte (`as_of`):** `{report.as_of.isoformat()}`",
        f"- **Señales crudas ingeridas:** `{len(report.signals_discovered)}`",
        f"- **Propuestas evaluadas:** `{len(report.decisions)}`",
        f"- **Promovidas a este Plan (`PROMOTED_TO_PLAN`):** `{len(promoted)}`",
        f"- **Vetadas / En espera (`Graveyard`):** `{len(rejected)}`",
        "",
        "---",
        "",
        "## Plan de Implementación (Todos Promovidos)",
        "",
    ]

    if not promoted:
        lines.extend(
            [
                "> **Ninguna propuesta superó las compuertas del comité esta semana.**",
                "> El sistema de producción permanece sin cambios (principio *Do No Harm*).",
                "",
            ]
        )
    else:
        for idx, decision in enumerate(promoted, start=1):
            prop = decision.proposal
            todo_slug = f"ar-{report.as_of.strftime('%m%d')}-{idx}-{prop.subsystem.value}"
            lines.extend(
                [
                    (
                        f"### {idx}. `{todo_slug}` — {prop.title} "
                        f"(`{prop.subsystem.value}`, ID `{prop.proposal_id}`)"
                    ),
                    f"- **Problema detectado:** {prop.problem_statement}",
                    f"- **Formulación matemática:** {prop.mathematical_formulation}",
                    f"- **Hipótesis falsable:** {prop.falsifiable_hypothesis}",
                    "- **Fuentes de origen:**",
                ]
            )
            for sig in prop.source_signals:
                lines.append(
                    f"  - `[{sig.source_kind.value}]` **{sig.source_name}** — "
                    f"{sig.title} (`{sig.reference_uri}`)"
                )
            lines.append(
                f"- **Archivos objetivo:** {', '.join(f'`{f}`' for f in prop.affected_files)}"
            )
            lines.append(f"- **Certificación del Comité:** {_format_scorecard_line(decision)}")
            lines.append("- **Pasos de implementación:**")
            for step in prop.implementation_todos:
                lines.append(f"  - {step}")
            verif_str = ", ".join(f"`{c}`" for c in prop.verification_commands)
            lines.append(f"- **Verificación requerida:** {verif_str}")
            lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Apéndice de Discernimiento: Propuestas Vetadas (Graveyard)",
            "",
        ]
    )

    if not rejected:
        lines.append("_Todas las propuestas formuladas en este ciclo superaron las compuertas._")
    else:
        lines.extend(
            [
                "| ID | Subsistema | Propuesta | Estado | Razón de Veto / Retención |",
                "|---|---|---|---|---|",
            ]
        )
        for decision in rejected:
            prop = decision.proposal
            reason_clean = decision.summary_reason.replace("|", "/")
            lines.append(
                f"| `{prop.proposal_id}` | `{prop.subsystem.value}` | {prop.title} | "
                f"`{decision.status.value}` | {reason_clean} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Notas y Consideraciones",
            (
                "- **Invariante Anti-Snooping:** Todas las propuestas evaluadas "
                "(aprobadas y rechazadas) quedaron registradas en `data/alpharesearch.db` "
                "y en el `TrialLedger` (`n_trials`)."
            ),
            (
                "- **Ejecución Controlada:** Este plan no modifica el código fuente "
                "automáticamente; úsalo en Plan Mode o revísalo antes de aplicar los cambios."
            ),
            "",
        ]
    )
    return "\n".join(lines)
