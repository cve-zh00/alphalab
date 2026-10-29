"""Human-readable and JSON rendering for ``alphalab autopilot`` commands.

Pure formatting: takes already-computed :class:`PreflightReport` and
:class:`AutopilotCycleResult` structures and returns strings or JSON-ready
dicts. No I/O or network calls happen here.
"""

from __future__ import annotations

from alphalab.adapters.kill_switch import KillSwitchState
from alphalab.app.autopilot_preflight import PreflightReport
from alphalab.app.autopilot_runner import AGENTIC_STRATEGY_NAME, AutopilotCycleResult
from alphalab.app.paper_models import Reconciliation, ShadowBenchmark
from alphalab.domain.execution import effective_stop_loss_pct
from alphalab.report.theme import Theme

__all__ = [
    "autopilot_cycle_json",
    "preflight_json",
    "render_autopilot_run",
    "render_autopilot_status",
    "render_preflight",
]


def render_preflight(report: PreflightReport, theme: Theme) -> str:
    """Render NOFX launch preflight checks."""
    verdict = (
        theme.paint("READY", "bold", "green")
        if report.ready
        else theme.paint("BLOCKED", "bold", "red")
    )
    lines = [f"{theme.bold('AUTOPILOT LAUNCH PREFLIGHT')} — {verdict}"]
    for check in report.checks:
        if check.passed:
            badge = theme.paint("PASS", "bold", "green")
        elif check.blocking:
            badge = theme.paint("FAIL", "bold", "red")
        else:
            badge = theme.paint("WARN", "bold", "yellow")
        lines.append(f"  [{badge}] {check.name}: {check.detail}")
    return "\n".join(lines)


def render_autopilot_status(
    result: AutopilotCycleResult, kill_switch: KillSwitchState, theme: Theme
) -> str:
    """Render the autonomous agentic status view."""
    cycle = result.cycle
    lines = [theme.bold("AUTOPILOT AGENT STATUS  (The model proposes. The runtime disposes.)")]
    lines.append(f"  strategy:      {AGENTIC_STRATEGY_NAME} (agentic + NOFX runtime guards)")
    lines.append(f"  equity:        ${cycle.account.equity:,.2f}")
    lines.append(f"  buying power:  ${cycle.account.buying_power:,.2f}")
    lines.append("  market:        " + _market_label(cycle.market_open, theme))
    lines.append("  kill switch:   " + _kill_label(kill_switch, theme))
    lines.append("  runtime mode:  " + _health_label(result, theme))
    lines.extend(_position_risk_lines(result, theme))
    lines.extend(_research_lines(result, theme))
    lines.extend(_guard_lines(result, theme))
    lines.append(_shadow_line(cycle.shadow))
    lines.append(theme.bold("  RECONCILIATION (actual vs guarded target)"))
    lines.extend(_reconciliation_lines(cycle.reconciliation, theme))
    return "\n".join(lines)


def render_autopilot_run(result: AutopilotCycleResult, theme: Theme) -> str:
    """Render one autonomous agentic cycle (dry-run or submitted)."""
    cycle = result.cycle
    header = "AUTOPILOT CYCLE — DRY RUN" if not cycle.submitted else "AUTOPILOT CYCLE — SUBMITTED"
    lines = [theme.bold(header)]
    lines.append(f"  cycle:         {cycle.cycle_id}")
    lines.append(f"  as of:         {cycle.as_of.isoformat()}")
    lines.append("  market:        " + _market_label(cycle.market_open, theme))
    lines.append("  runtime mode:  " + _health_label(result, theme))
    lines.append(
        f"  journal:       settled={result.settled_predictions}, "
        f"registered={result.predictions_registered}, "
        f"abstained={result.abstentions_registered}"
    )
    if cycle.skipped_reason:
        lines.append("  " + theme.paint(cycle.skipped_reason, "yellow"))
    if cycle.halted_reason:
        lines.append("  " + theme.paint(cycle.halted_reason, "red"))
    lines.extend(_position_risk_lines(result, theme))
    lines.extend(_research_lines(result, theme))
    lines.extend(_guard_lines(result, theme))
    lines.append(theme.bold("  PLANNED ORDERS"))
    lines.extend(_plan_lines(result, theme))
    if cycle.plan.rejected:
        lines.append(theme.bold("  REJECTED BY RISK LIMITS"))
        for rejected in cycle.plan.rejected:
            lines.append(f"    {rejected.ticker}: {rejected.reason}")
    if cycle.outcomes:
        lines.append(theme.bold("  SUBMISSION OUTCOMES"))
        for outcome in cycle.outcomes:
            lines.append(
                f"    {outcome.intent.ticker} {outcome.status.value}"
                + (f" ({outcome.detail})" if outcome.detail else "")
            )
    lines.append(_shadow_line(cycle.shadow))
    return "\n".join(lines)


def preflight_json(report: PreflightReport) -> dict[str, object]:
    """Serialize :class:`PreflightReport` to a JSON-ready dictionary."""
    return {
        "ready": report.ready,
        "blocking_failures": [c.name for c in report.blocking_failures],
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "blocking": c.blocking,
                "detail": c.detail,
            }
            for c in report.checks
        ],
    }


def autopilot_cycle_json(
    result: AutopilotCycleResult, kill_switch: KillSwitchState | None = None
) -> dict[str, object]:
    """Serialize :class:`AutopilotCycleResult` to a JSON-ready dictionary."""
    cycle = result.cycle
    payload: dict[str, object] = {
        "strategy": AGENTIC_STRATEGY_NAME,
        "cycle_id": cycle.cycle_id,
        "as_of": cycle.as_of.isoformat(),
        "submitted": cycle.submitted,
        "market_open": cycle.market_open,
        "halted_reason": cycle.halted_reason,
        "skipped_reason": cycle.skipped_reason,
        "account": {
            "equity": cycle.account.equity,
            "buying_power": cycle.account.buying_power,
            "trading_blocked": cycle.account.trading_blocked,
            "account_blocked": cycle.account.account_blocked,
        },
        "health": {
            "safe_mode": result.health.safe_mode,
            "safe_mode_reason": result.health.safe_mode_reason,
            "evaluated_candidates": result.health.evaluated_candidates,
            "total_candidates": result.health.total_candidates,
            "kill_switch_engaged": result.health.kill_switch_engaged,
            "kill_switch_reason": result.health.kill_switch_reason,
        },
        "journal": {
            "settled_predictions": result.settled_predictions,
            "predictions_registered": result.predictions_registered,
            "abstentions_registered": result.abstentions_registered,
        },
        "research_outcomes": [
            {
                "ticker": str(o.ticker),
                "verdict": o.verdict.value if o.verdict is not None else None,
                "detail": o.detail,
            }
            for o in result.agentic_plan.outcomes
        ],
        "exposure": (
            {
                "beta_ols": result.exposure.beta_ols,
                "beta_welch": result.exposure.beta_welch,
                "active_beta_vs_spy": result.exposure.active_beta_vs_spy,
                "ex_ante_tracking_error": result.exposure.ex_ante_tracking_error,
                "sector_weights": dict(result.exposure.sector_weights),
                "active_sector_gaps_vs_spy": dict(result.exposure.active_sector_gaps_vs_spy),
                "warnings": list(result.exposure.warnings),
            }
            if result.exposure is not None
            else None
        ),
        "position_risks": [
            {
                "ticker": str(pos.ticker),
                "shares": pos.shares,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "peak_price": pos.peak_price,
                "price_pnl_pct": pos.price_pnl_pct,
                "peak_pnl_pct": pos.peak_pnl_pct,
                "profit_giveback_pct": pos.profit_giveback_pct,
                "stop_loss_pct": effective_stop_loss_pct(pos),
            }
            for pos in result.position_risks.values()
        ],
        "guard_actions": [
            {
                "ticker": str(a.ticker),
                "kind": a.kind.value,
                "reason": a.reason,
            }
            for a in result.guarded.actions
        ],
        "targets": {str(k): v for k, v in result.guarded.targets.items()},
        "planned_orders": [
            {
                "ticker": str(i.ticker),
                "side": i.side.value,
                "shares": i.shares,
                "reference_price": i.reference_price,
                "notional": i.notional,
                "client_order_id": i.client_order_id(
                    strategy=AGENTIC_STRATEGY_NAME, as_of=cycle.as_of
                ),
            }
            for i in cycle.plan.intents
        ],
        "outcomes": [
            {
                "ticker": str(o.intent.ticker),
                "side": o.intent.side.value,
                "shares": o.intent.shares,
                "status": o.status.value,
                "client_order_id": o.client_order_id,
                "broker_order_id": o.broker_order_id,
                "detail": o.detail,
            }
            for o in cycle.outcomes
        ],
    }
    if kill_switch is not None:
        payload["kill_switch"] = {
            "engaged": kill_switch.engaged,
            "reason": kill_switch.reason(),
        }
    return payload


def _market_label(is_open: bool, theme: Theme) -> str:
    return theme.paint("OPEN", "green") if is_open else theme.paint("CLOSED", "yellow")


def _kill_label(state: KillSwitchState, theme: Theme) -> str:
    if state.engaged:
        return theme.paint(state.reason(), "red")
    return theme.paint("clear", "green")


def _health_label(result: AutopilotCycleResult, theme: Theme) -> str:
    h = result.health
    cov = f"({h.evaluated_candidates}/{h.total_candidates} candidates evaluated)"
    if h.safe_mode:
        reason = h.safe_mode_reason or "low evaluation coverage"
        return theme.paint(f"SAFE MODE — {reason} {cov}", "red")
    return theme.paint(f"ACTIVE {cov}", "green")


def _position_risk_lines(result: AutopilotCycleResult, theme: Theme) -> list[str]:
    lines = [theme.bold("  OPEN POSITION RISK MONITOR (NOFX drawdown & stop-loss)")]
    if not result.position_risks:
        lines.append("    (no open positions)")
        return lines
    for pos in result.position_risks.values():
        pnl_col = "green" if pos.price_pnl_pct >= 0 else "red"
        pnl_txt = theme.paint(f"{pos.price_pnl_pct:+.2f}%", pnl_col)
        stop_floor = effective_stop_loss_pct(pos)
        lines.append(
            f"    {pos.ticker:<8} qty={pos.shares:g} "
            f"entry=${pos.entry_price:,.2f} now=${pos.current_price:,.2f} "
            f"peak=${pos.peak_price:,.2f} pnl={pnl_txt} "
            f"(peak {pos.peak_pnl_pct:+.2f}%, giveback {pos.profit_giveback_pct:.1f}%, "
            f"stop {stop_floor:+.2f}%)"
        )
    return lines


def _research_lines(result: AutopilotCycleResult, theme: Theme) -> list[str]:
    lines = [theme.bold("  AI RESEARCH PROPOSALS")]
    if not result.agentic_plan.outcomes:
        lines.append("    (no candidates evaluated)")
        return lines
    for outcome in result.agentic_plan.outcomes:
        if outcome.verdict is not None:
            colour = {
                "BUY": "green",
                "HOLD": "yellow",
                "SELL": "red",
            }.get(outcome.verdict.value, "reset")
            badge = theme.paint(outcome.verdict.value, "bold", colour)
            lines.append(f"    {outcome.ticker:<8} -> {badge}")
        else:
            badge = theme.paint("SKIPPED", "yellow")
            lines.append(f"    {outcome.ticker:<8} -> {badge} ({outcome.detail})")
    return lines


def _guard_lines(result: AutopilotCycleResult, theme: Theme) -> list[str]:
    lines = [theme.bold("  NOFX RUNTIME DISPOSITION")]
    if not result.guarded.actions:
        lines.append("    (all proposals passed without guard overrides)")
        return lines
    for action in result.guarded.actions:
        colour = "yellow" if "retained" in action.kind.value else "red"
        badge = theme.paint(action.kind.value.upper(), colour)
        lines.append(f"    [{badge}] {action.ticker}: {action.reason}")
    return lines


def _shadow_line(shadow: ShadowBenchmark | None) -> str:
    if shadow is None:
        return "  shadow SPY benchmark: unavailable (no SPY price series)"
    return (
        f"  shadow SPY buy&hold: {shadow.total_return:+.2%} since inception "
        f"(entry ${shadow.entry_price:,.2f}, now ${shadow.latest_price:,.2f}, "
        "no orders sent)"
    )


def _reconciliation_lines(reconciliation: Reconciliation, theme: Theme) -> list[str]:
    if not reconciliation.rows:
        return ["    (no positions and no targets)"]
    out: list[str] = []
    for row in reconciliation.rows:
        colour = "green" if abs(row.deviation) <= 0.005 else "yellow"
        out.append(
            "    "
            + f"{row.ticker:<8} target={row.target_weight:6.2%} "
            + f"actual={row.actual_weight:6.2%} "
            + theme.paint(f"dev={row.deviation:+6.2%}", colour)
        )
    status = "aligned" if reconciliation.is_aligned else "DRIFT DETECTED"
    colour = "green" if reconciliation.is_aligned else "yellow"
    out.append("    " + theme.paint(status, colour))
    return out


def _plan_lines(result: AutopilotCycleResult, theme: Theme) -> list[str]:
    cycle = result.cycle
    if cycle.plan.is_empty:
        return ["    (no orders — portfolio already matches guarded target)"]
    out: list[str] = []
    for intent in cycle.plan.intents:
        coid = intent.client_order_id(strategy=AGENTIC_STRATEGY_NAME, as_of=cycle.as_of)
        out.append(
            "    "
            + f"{intent.side.value.upper():<4} {intent.shares:g} {intent.ticker:<8} "
            + f"@~${intent.reference_price:,.2f} "
            + f"(notional ${intent.notional:,.2f}) id={coid}"
        )
    return out
