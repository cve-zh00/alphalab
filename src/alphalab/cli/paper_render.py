"""Human-readable rendering for the ``paper`` commands.

Pure formatting: every function takes already-computed values and a
:class:`~alphalab.report.theme.Theme` and returns a string. No I/O, no network,
no business logic, so the renderers are trivially testable and the command
modules stay about orchestration.

The copy is deliberately plain about what this is: a paper forward test with no
demonstrated edge. The shadow SPY line sits next to the strategy's equity so a
reader never sees a return without its reference.
"""

from __future__ import annotations

from alphalab.adapters.execution_log import ExecutionEvent
from alphalab.adapters.kill_switch import KillSwitchState
from alphalab.app.paper_models import Reconciliation, ShadowBenchmark
from alphalab.app.paper_runner import STRATEGY_NAME, CycleResult
from alphalab.report.theme import Theme

__all__ = ["render_log", "render_run", "render_status"]


def render_status(result: CycleResult, kill_switch: KillSwitchState, theme: Theme) -> str:
    """Render the read-only account snapshot."""
    lines = [theme.bold("PAPER ACCOUNT STATUS")]
    lines.append(f"  strategy: {STRATEGY_NAME} (paper; no demonstrated edge)")
    lines.append(f"  equity:        ${result.account.equity:,.2f}")
    lines.append(f"  buying power:  ${result.account.buying_power:,.2f}")
    lines.append("  market: " + _market_label(result.market_open, theme))
    lines.append("  kill switch: " + _kill_label(kill_switch, theme))
    lines.extend(_account_flag_lines(result, theme))
    lines.append(_shadow_line(result.shadow, theme))
    lines.append(theme.bold("  RECONCILIATION (actual vs target)"))
    lines.extend(_reconciliation_lines(result.reconciliation, theme))
    return "\n".join(lines)


def render_run(result: CycleResult, theme: Theme) -> str:
    """Render the outcome of one cycle (dry-run or submitted)."""
    header = "PAPER RUN — DRY RUN" if not result.submitted else "PAPER RUN — SUBMITTED"
    lines = [theme.bold(header)]
    lines.append(f"  cycle: {result.cycle_id}")
    lines.append(f"  as of: {result.as_of.isoformat()}")
    lines.append("  market: " + _market_label(result.market_open, theme))
    if result.skipped_reason:
        lines.append("  " + theme.paint(result.skipped_reason, "yellow"))
    if result.halted_reason:
        lines.append("  " + theme.paint(result.halted_reason, "red"))
    lines.append(theme.bold("  PLANNED ORDERS"))
    lines.extend(_plan_lines(result, theme))
    if result.plan.rejected:
        lines.append(theme.bold("  REJECTED BY RISK LIMITS"))
        for rejected in result.plan.rejected:
            lines.append(f"    {rejected.ticker}: {rejected.reason}")
    if result.outcomes:
        lines.append(theme.bold("  SUBMISSION OUTCOMES"))
        for outcome in result.outcomes:
            lines.append(
                f"    {outcome.intent.ticker} {outcome.status.value}"
                + (f" ({outcome.detail})" if outcome.detail else "")
            )
    lines.append(_shadow_line(result.shadow, theme))
    return "\n".join(lines)


def render_log(events: list[ExecutionEvent], theme: Theme) -> str:
    """Render the append-only execution history."""
    lines = [theme.bold("EXECUTION LOG")]
    if not events:
        lines.append("  No events recorded yet.")
        return "\n".join(lines)
    lines.append(f"  {len(events)} event(s), append-only.")
    for event in events:
        stamp = event.recorded_at.isoformat(timespec="seconds")
        detail = f" — {event.detail}" if event.detail else ""
        shares = f" {event.shares:g}" if event.shares is not None else ""
        side = f" {event.side}" if event.side else ""
        lines.append(
            f"    {stamp} [{_kind_label(event.kind.value, theme)}] "
            f"{event.ticker}{side}{shares}{detail}"
        )
    return "\n".join(lines)


# -- shared fragments -----------------------------------------------------


def _market_label(is_open: bool, theme: Theme) -> str:
    return theme.paint("OPEN", "green") if is_open else theme.paint("CLOSED", "yellow")


def _kill_label(state: KillSwitchState, theme: Theme) -> str:
    if state.engaged:
        return theme.paint(state.reason(), "red")
    return theme.paint("clear", "green")


def _account_flag_lines(result: CycleResult, theme: Theme) -> list[str]:
    out: list[str] = []
    if result.account.trading_blocked:
        out.append("  " + theme.paint("WARNING: trading_blocked is set", "red"))
    if result.account.account_blocked:
        out.append("  " + theme.paint("WARNING: account_blocked is set", "red"))
    return out


def _shadow_line(shadow: ShadowBenchmark | None, theme: Theme) -> str:
    if shadow is None:
        return "  shadow SPY benchmark: unavailable (no SPY price series)"
    return (
        f"  shadow SPY buy&hold: ${shadow.current_value:,.2f} "
        f"({shadow.total_return:+.2%} since inception, no orders sent)"
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


def _plan_lines(result: CycleResult, theme: Theme) -> list[str]:
    if result.plan.is_empty:
        return ["    (no orders — portfolio already matches target)"]
    out: list[str] = []
    for intent in result.plan.intents:
        coid = intent.client_order_id(strategy=STRATEGY_NAME, as_of=result.as_of)
        out.append(
            "    "
            + f"{intent.side.value.upper():<4} {intent.shares:g} {intent.ticker:<8} "
            + f"@~${intent.reference_price:,.2f} "
            + f"(notional ${intent.notional:,.2f}) id={coid}"
        )
    return out


def _kind_label(kind: str, theme: Theme) -> str:
    colours = {
        "intent": "blue",
        "submitted": "green",
        "duplicate": "cyan",
        "halted": "red",
        "failed": "red",
        "rejected": "yellow",
    }
    return theme.paint(kind.upper(), colours.get(kind, "reset"))
