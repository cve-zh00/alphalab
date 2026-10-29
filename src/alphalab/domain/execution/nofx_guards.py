"""NOFX-inspired deterministic runtime risk guards for autonomous agentic trading.

"The model proposes. The runtime disposes."
-------------------------------------------
An LLM research pipeline proposes which equities deserve capital, but the
runtime owns survival. Adapted from ``NoFxAiOS/nofx`` (``auto_trader_risk.go``,
``auto_trader_throttle.go``, ``auto_trader_signal_policy.go``, and
``runtime_health.go``), this module sits between the agentic target builder
(:mod:`alphalab.app.agentic_portfolio`) and the pre-trade order planner
(:mod:`alphalab.domain.execution.planner`):

1. **Safe Mode on Model Degradation.** When candidate evaluation coverage falls
   below the required threshold, the runtime enters Safe Mode: **all new BUY
   entries are blocked**, while existing positions remain protected by
   deterministic stop-loss, peak-drawdown, and explicit SELL exits rather than
   being blindly liquidated during a temporary model outage.
2. **Hard Stop-Loss Auto-Close.** Any open position whose price P&L drops at or
   below ``stop_loss_pct`` (default ``-8.0%``) is closed unconditionally,
   regardless of what the model says.
3. **Peak-Profit Drawdown Auto-Close.** Mirroring NOFX's
   ``shouldDrawdownClose`` (``drawdownClosePriceGainPct = 5.0%``,
   ``drawdownCloseGivebackPct = 40.0%``), once a position's peak price gain has
   reached ``+5.0%`` from entry, giving back ``>= 40%`` of that peak profit
   triggers an immediate protective exit.
4. **Signal-Managed Position Policy.** An explicit ``Verdict.SELL`` on an open
   position exits immediately. An open position evaluated as ``Verdict.HOLD``
   retains its slot weight (when ``hold_keeps_existing_position=True``) so
   ``HOLD`` genuinely holds an existing position without opening new ones.
5. **Trade Throttling (Re-entry Cooldown).** A flat symbol that was closed
   within ``reentry_cooldown_days`` is blocked from re-opening so the agent
   cannot churn in and out of the same name across consecutive cycles.

Pure by construction: no I/O, no network, no clock lookup.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from alphalab.artifact import Verdict
from alphalab.domain.types import Ticker

__all__ = [
    "DEFAULT_DRAWDOWN_ARM_GAIN_PCT",
    "DEFAULT_DRAWDOWN_GIVEBACK_PCT",
    "DEFAULT_REENTRY_COOLDOWN_DAYS",
    "DEFAULT_STOP_LOSS_MAX_PCT",
    "DEFAULT_STOP_LOSS_MIN_PCT",
    "DEFAULT_STOP_LOSS_PCT",
    "DEFAULT_STOP_LOSS_VOL_MULT",
    "AutopilotGuardConfig",
    "GuardAction",
    "GuardActionKind",
    "GuardedTargets",
    "PositionRiskSnapshot",
    "apply_nofx_guards",
    "effective_stop_loss_pct",
    "project_feasible_weights_dykstra",
    "should_drawdown_close",
]

DEFAULT_STOP_LOSS_PCT = -8.0
DEFAULT_STOP_LOSS_VOL_MULT = 3.5
DEFAULT_STOP_LOSS_MIN_PCT = -6.0
DEFAULT_STOP_LOSS_MAX_PCT = -12.0
DEFAULT_DRAWDOWN_ARM_GAIN_PCT = 5.0
DEFAULT_DRAWDOWN_GIVEBACK_PCT = 40.0
DEFAULT_REENTRY_COOLDOWN_DAYS = 3


class AutopilotGuardConfig(BaseModel):
    """Deterministic runtime protection parameters enforced over AI proposals."""

    model_config = ConfigDict(frozen=True)

    stop_loss_pct: float = Field(default=DEFAULT_STOP_LOSS_PCT, lt=0.0)
    stop_loss_vol_mult: float = Field(default=DEFAULT_STOP_LOSS_VOL_MULT, gt=0.0)
    stop_loss_min_pct: float = Field(default=DEFAULT_STOP_LOSS_MIN_PCT, lt=0.0)
    stop_loss_max_pct: float = Field(default=DEFAULT_STOP_LOSS_MAX_PCT, lt=0.0)
    volatility_scaled_stop_loss: bool = True
    drawdown_arm_gain_pct: float = Field(default=DEFAULT_DRAWDOWN_ARM_GAIN_PCT, gt=0.0)
    drawdown_giveback_pct: float = Field(default=DEFAULT_DRAWDOWN_GIVEBACK_PCT, gt=0.0, le=100.0)
    reentry_cooldown_days: int = Field(default=DEFAULT_REENTRY_COOLDOWN_DAYS, ge=0)
    exit_on_sell_verdict: bool = True
    sell_confirmations_required: int = Field(default=1, ge=1)
    hold_keeps_existing_position: bool = True
    retain_on_research_failure: bool = True


class PositionRiskSnapshot(BaseModel):
    """Mark-to-market and peak-profit state for one open long position."""

    model_config = ConfigDict(frozen=True)

    ticker: Ticker
    shares: float = Field(gt=0.0)
    entry_price: float = Field(gt=0.0)
    current_price: float = Field(gt=0.0)
    peak_price: float = Field(gt=0.0)
    annualized_volatility: float | None = Field(default=None, ge=0.0)
    fair_value_upside_pct: float | None = Field(default=None)

    @property
    def effective_peak_price(self) -> float:
        """Peak price is never below entry or current price."""
        return max(self.entry_price, self.current_price, self.peak_price)

    @property
    def price_pnl_pct(self) -> float:
        """Signed percentage move from entry price to current price."""
        return ((self.current_price - self.entry_price) / self.entry_price) * 100.0

    @property
    def peak_pnl_pct(self) -> float:
        """Highest percentage gain achieved since entry."""
        return ((self.effective_peak_price - self.entry_price) / self.entry_price) * 100.0

    @property
    def profit_giveback_pct(self) -> float:
        """Fraction of peak profit given back (0..100+ when below peak)."""
        peak = self.peak_pnl_pct
        current = self.price_pnl_pct
        if peak <= 0.0 or current >= peak:
            return 0.0
        return ((peak - current) / peak) * 100.0


class GuardActionKind(StrEnum):
    """Deterministic intervention applied by the NOFX runtime layer."""

    STOP_LOSS_EXIT = "stop_loss_exit"
    DRAWDOWN_EXIT = "drawdown_exit"
    SELL_VERDICT_EXIT = "sell_verdict_exit"
    SELL_PENDING_CONFIRMATION = "sell_pending_confirmation"
    HOLD_RETAINED = "hold_retained"
    RESEARCH_FAILED_RETAINED = "research_failed_retained"
    COOLDOWN_BLOCKED = "cooldown_blocked"
    SAFE_MODE_BLOCKED = "safe_mode_blocked"
    SECTOR_CAP_BLOCKED = "sector_cap_blocked"


class GuardAction(BaseModel):
    """Audit record of one runtime intervention on a symbol."""

    model_config = ConfigDict(frozen=True)

    ticker: Ticker
    kind: GuardActionKind
    reason: str = Field(min_length=1)


class GuardedTargets(BaseModel):
    """Post-guard target weights and the audit trail of every intervention."""

    model_config = ConfigDict(frozen=True)

    targets: Mapping[Ticker, float]
    actions: tuple[GuardAction, ...]
    safe_mode: bool = False
    safe_mode_reason: str | None = None


def effective_stop_loss_pct(
    snapshot: PositionRiskSnapshot,
    config: AutopilotGuardConfig | None = None,
) -> float:
    """Return the effective stop-loss floor (in %) for ``snapshot``.

    When ``volatility_scaled_stop_loss`` is enabled and the position carries a
    positive 60-day annualized volatility $\\sigma_{\\text{ann}}$, the stop floor
    scales to $-k_{\\sigma} \\cdot (\\sigma_{\\text{ann}} / \\sqrt{252}) \\cdot 100$
    and is clamped between ``stop_loss_max_pct`` (default ``-12.0%``) and
    ``stop_loss_min_pct`` (default ``-6.0%``). Falls back to ``stop_loss_pct``
    when volatility is unavailable.
    """
    rules = config or AutopilotGuardConfig()
    vol = snapshot.annualized_volatility
    if not rules.volatility_scaled_stop_loss or vol is None or vol <= 0.0:
        return rules.stop_loss_pct
    daily_vol_pct = (vol / math.sqrt(252.0)) * 100.0
    raw_stop = -rules.stop_loss_vol_mult * daily_vol_pct
    low_bound = min(rules.stop_loss_min_pct, rules.stop_loss_max_pct)
    high_bound = max(rules.stop_loss_min_pct, rules.stop_loss_max_pct)
    return max(low_bound, min(high_bound, raw_stop))


def _effective_drawdown_thresholds(
    snapshot: PositionRiskSnapshot,
    config: AutopilotGuardConfig | None = None,
) -> tuple[float, float]:
    """Return ``(arm_pct, giveback_pct)`` after Ornstein-Uhlenbeck convergence scaling."""
    rules = config or AutopilotGuardConfig()
    arm_pct = rules.drawdown_arm_gain_pct
    giveback_pct = rules.drawdown_giveback_pct
    upside = snapshot.fair_value_upside_pct
    if upside is not None:
        if upside <= 2.0:
            arm_pct = min(arm_pct, 3.0)
            giveback_pct = min(giveback_pct, 30.0)
        elif upside >= 25.0:
            arm_pct = max(arm_pct, min(10.0, 0.25 * upside))
    return arm_pct, giveback_pct


def should_drawdown_close(
    snapshot: PositionRiskSnapshot,
    config: AutopilotGuardConfig | None = None,
) -> bool:
    """Whether peak-profit giveback protection should close ``snapshot``.

    Mirrors NOFX's ``shouldDrawdownClose``: the monitor arms only once the
    position's peak price gain reaches ``drawdown_arm_gain_pct`` (+5%), and
    triggers when at least ``drawdown_giveback_pct`` (40%) of that peak gain has
    been given back. When ``fair_value_upside_pct`` is available, thresholds
    scale by Ornstein-Uhlenbeck convergence distance (Leung & Li 2015).
    """
    arm_pct, giveback_pct = _effective_drawdown_thresholds(snapshot, config)
    return (
        snapshot.peak_pnl_pct >= arm_pct
        and snapshot.profit_giveback_pct >= giveback_pct
    )


def apply_nofx_guards(
    *,
    as_of: date,
    proposed_targets: Mapping[Ticker, float],
    verdicts: Mapping[Ticker, Verdict | None],
    positions: Mapping[Ticker, PositionRiskSnapshot],
    recent_sell_dates: Mapping[Ticker, date] | None = None,
    prior_sell_counts: Mapping[Ticker, int] | None = None,
    current_weights: Mapping[Ticker, float] | None = None,
    actionable: bool = True,
    safe_mode_reason: str | None = None,
    max_positions: int = 20,
    config: AutopilotGuardConfig | None = None,
    sectors: Mapping[Ticker, str] | None = None,
    max_sector_weight: float = 0.25,
) -> GuardedTargets:
    """Apply NOFX runtime risk guards to model-proposed target weights.

    Raises:
        ValueError: if ``max_positions`` is not positive.
    """
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")

    rules = config or AutopilotGuardConfig()
    sells = recent_sell_dates or {}
    sell_counts = prior_sell_counts or {}
    slot_weight = 1.0 / max_positions
    actions: list[GuardAction] = []
    surviving: dict[Ticker, float] = {}

    _evaluate_open_positions(
        positions=positions,
        proposed_targets=proposed_targets,
        verdicts=verdicts,
        prior_sell_counts=sell_counts,
        current_weights=current_weights,
        actionable=actionable,
        slot_weight=slot_weight,
        rules=rules,
        surviving=surviving,
        actions=actions,
    )
    _evaluate_new_entries(
        as_of=as_of,
        proposed_targets=proposed_targets,
        positions=positions,
        recent_sell_dates=sells,
        actionable=actionable,
        safe_mode_reason=safe_mode_reason,
        slot_weight=slot_weight,
        rules=rules,
        surviving=surviving,
        actions=actions,
    )
    capped = _enforce_slot_ceiling(surviving, positions=positions, max_positions=max_positions)
    if sectors is not None:
        _enforce_sector_cap(
            capped,
            positions=positions,
            current_weights=current_weights,
            sectors=sectors,
            cap=max_sector_weight,
            actions=actions,
        )
    capped = _enforce_cash_headroom(capped, positions=positions)
    reason = safe_mode_reason or "model evaluation coverage below minimum threshold"
    return GuardedTargets(
        targets=capped,
        actions=tuple(actions),
        safe_mode=not actionable,
        safe_mode_reason=None if actionable else reason,
    )


def _evaluate_open_positions(
    *,
    positions: Mapping[Ticker, PositionRiskSnapshot],
    proposed_targets: Mapping[Ticker, float],
    verdicts: Mapping[Ticker, Verdict | None],
    prior_sell_counts: Mapping[Ticker, int],
    current_weights: Mapping[Ticker, float] | None,
    actionable: bool,
    slot_weight: float,
    rules: AutopilotGuardConfig,
    surviving: dict[Ticker, float],
    actions: list[GuardAction],
) -> None:
    """Apply protective exits or retention policy to every currently held name."""
    for ticker in sorted(positions):
        pos = positions[ticker]
        verdict = verdicts.get(ticker)
        prior_sells = prior_sell_counts.get(ticker, 0)
        retained_weight = (
            current_weights.get(ticker, slot_weight)
            if current_weights is not None
            else slot_weight
        )
        exit_action = _protective_exit_for(pos, verdict, prior_sells, rules)
        if exit_action is not None:
            actions.append(exit_action)
            continue
        if ticker in proposed_targets and proposed_targets[ticker] > 0.0:
            proposed_w = proposed_targets[ticker]
            if current_weights is not None and abs(proposed_w - retained_weight) < 0.02:
                surviving[ticker] = retained_weight
            else:
                surviving[ticker] = proposed_w
            continue
        if not actionable:
            surviving[ticker] = retained_weight
            continue
        if rules.exit_on_sell_verdict and verdict is Verdict.SELL:
            streak = prior_sells + 1
            surviving[ticker] = retained_weight
            actions.append(
                GuardAction(
                    ticker=ticker,
                    kind=GuardActionKind.SELL_PENDING_CONFIRMATION,
                    reason=(
                        f"AI verdict is SELL ({streak}/{rules.sell_confirmations_required} "
                        f"required to exit; {pos.price_pnl_pct:+.2f}% P&L); "
                        "retaining slot pending confirmation"
                    ),
                )
            )
            continue
        if rules.hold_keeps_existing_position and verdict is Verdict.HOLD:
            surviving[ticker] = retained_weight
            actions.append(
                GuardAction(
                    ticker=ticker,
                    kind=GuardActionKind.HOLD_RETAINED,
                    reason=(
                        f"AI verdict is HOLD on open position ({pos.price_pnl_pct:+.2f}% P&L); "
                        "retaining existing slot"
                    ),
                )
            )
            continue
        if rules.retain_on_research_failure and verdict is None:
            surviving[ticker] = retained_weight
            actions.append(
                GuardAction(
                    ticker=ticker,
                    kind=GuardActionKind.RESEARCH_FAILED_RETAINED,
                    reason=(
                        f"research unavailable on open position ({pos.price_pnl_pct:+.2f}% P&L); "
                        "retaining existing slot"
                    ),
                )
            )


def _protective_exit_for(
    pos: PositionRiskSnapshot,
    verdict: Verdict | None,
    prior_sells: int,
    rules: AutopilotGuardConfig,
) -> GuardAction | None:
    """Return a forced-exit action if a risk or signal rule fires, else None."""
    stop_floor = effective_stop_loss_pct(pos, rules)
    if pos.price_pnl_pct <= stop_floor:
        return GuardAction(
            ticker=pos.ticker,
            kind=GuardActionKind.STOP_LOSS_EXIT,
            reason=(
                f"stop-loss triggered: price P&L {pos.price_pnl_pct:+.2f}% "
                f"<= {stop_floor:+.2f}% floor"
            ),
        )
    if should_drawdown_close(pos, rules):
        _, giveback_pct = _effective_drawdown_thresholds(pos, rules)
        return GuardAction(
            ticker=pos.ticker,
            kind=GuardActionKind.DRAWDOWN_EXIT,
            reason=(
                f"peak-profit drawdown triggered: peak {pos.peak_pnl_pct:+.2f}%, "
                f"current {pos.price_pnl_pct:+.2f}%, giveback {pos.profit_giveback_pct:.1f}% "
                f">= {giveback_pct:.1f}%"
            ),
        )
    if rules.exit_on_sell_verdict and verdict is Verdict.SELL:
        streak = prior_sells + 1
        if streak >= rules.sell_confirmations_required:
            return GuardAction(
                ticker=pos.ticker,
                kind=GuardActionKind.SELL_VERDICT_EXIT,
                reason=f"AI verdict is SELL ({pos.price_pnl_pct:+.2f}% P&L); closing position",
            )
    return None


def _evaluate_new_entries(
    *,
    as_of: date,
    proposed_targets: Mapping[Ticker, float],
    positions: Mapping[Ticker, PositionRiskSnapshot],
    recent_sell_dates: Mapping[Ticker, date],
    actionable: bool,
    safe_mode_reason: str | None,
    slot_weight: float,
    rules: AutopilotGuardConfig,
    surviving: dict[Ticker, float],
    actions: list[GuardAction],
) -> None:
    """Filter proposed new entries through Safe Mode and re-entry cooldown."""
    for ticker in sorted(proposed_targets):
        weight = proposed_targets[ticker]
        if weight <= 0.0 or ticker in positions:
            continue
        if not actionable:
            actions.append(
                GuardAction(
                    ticker=ticker,
                    kind=GuardActionKind.SAFE_MODE_BLOCKED,
                    reason=(
                        "safe mode active: new BUY entry blocked "
                        f"({safe_mode_reason or 'insufficient evaluation coverage'})"
                    ),
                )
            )
            continue
        cooldown_action = _cooldown_block_for(
            ticker=ticker,
            as_of=as_of,
            last_sell=recent_sell_dates.get(ticker),
            cooldown_days=rules.reentry_cooldown_days,
        )
        if cooldown_action is not None:
            actions.append(cooldown_action)
            continue
        surviving[ticker] = weight if weight > 0.0 else slot_weight


def _cooldown_block_for(
    *,
    ticker: Ticker,
    as_of: date,
    last_sell: date | None,
    cooldown_days: int,
) -> GuardAction | None:
    """Return a cooldown block if ``ticker`` was sold inside the cooldown window."""
    if cooldown_days <= 0 or last_sell is None:
        return None
    elapsed_days = (as_of - last_sell).days
    if 0 <= elapsed_days < cooldown_days:
        remaining = cooldown_days - elapsed_days
        return GuardAction(
            ticker=ticker,
            kind=GuardActionKind.COOLDOWN_BLOCKED,
            reason=(
                f"trade throttle: {ticker} was closed {elapsed_days}d ago "
                f"({last_sell.isoformat()}); wait {remaining}d more before re-entry"
            ),
        )
    return None


def _enforce_sector_cap(
    surviving: dict[Ticker, float],
    *,
    positions: Mapping[Ticker, PositionRiskSnapshot],
    current_weights: Mapping[Ticker, float] | None,
    sectors: Mapping[Ticker, str],
    cap: float,
    actions: list[GuardAction],
) -> None:
    """Scale down / block *new entries* that push a sector above ``cap``.

    Held positions are never reduced or sold because of the cap; their weight
    simply consumes sector headroom first. New entries are admitted in ticker
    order until headroom runs out (the last one is scaled to fit).
    """
    used: dict[str, float] = {}
    for t in positions:
        sector = sectors.get(t)
        if not sector:
            continue
        if current_weights is not None and t in current_weights:
            used[sector] = used.get(sector, 0.0) + current_weights[t]
        elif t in surviving:
            used[sector] = used.get(sector, 0.0) + surviving[t]
    for t in sorted(surviving):
        sector = sectors.get(t)
        if not sector or t not in positions:
            continue
        if current_weights is None or t not in current_weights:
            continue
        held_w = current_weights[t]
        delta = surviving[t] - held_w
        if delta <= 1e-12:
            continue
        room = max(0.0, cap - used.get(sector, 0.0))
        if delta <= room + 1e-12:
            used[sector] = used.get(sector, 0.0) + delta
        else:
            new_w = held_w + room
            surviving[t] = new_w
            used[sector] = used.get(sector, 0.0) + room
            actions.append(
                GuardAction(
                    ticker=t,
                    kind=GuardActionKind.SECTOR_CAP_BLOCKED,
                    reason=(
                        f"sector cap: {sector} limited to {cap:.0%}; "
                        f"top-up capped at {new_w:.2%}"
                    ),
                )
            )
    new_entries = sorted(
        (t for t in surviving if t not in positions),
        key=lambda t: (-surviving[t], str(t)),
    )
    for t in new_entries:
        sector = sectors.get(t)
        if not sector:
            continue
        weight = surviving[t]
        room = cap - used.get(sector, 0.0)
        if weight <= room + 1e-12:
            used[sector] = used.get(sector, 0.0) + weight
            continue
        if room <= 1e-9:
            del surviving[t]
            actions.append(
                GuardAction(
                    ticker=t,
                    kind=GuardActionKind.SECTOR_CAP_BLOCKED,
                    reason=f"sector cap: {sector} already at {cap:.0%}; new entry blocked",
                )
            )
        else:
            surviving[t] = room
            used[sector] = cap
            actions.append(
                GuardAction(
                    ticker=t,
                    kind=GuardActionKind.SECTOR_CAP_BLOCKED,
                    reason=(
                        f"sector cap: {sector} limited to {cap:.0%}; "
                        f"entry scaled {weight:.2%} -> {room:.2%}"
                    ),
                )
            )


def _enforce_slot_ceiling(
    surviving: Mapping[Ticker, float],
    *,
    positions: Mapping[Ticker, PositionRiskSnapshot],
    max_positions: int,
) -> dict[Ticker, float]:
    """Keep at most ``max_positions`` names, prioritizing existing holdings and conviction."""
    if len(surviving) <= max_positions:
        out = dict(surviving)
    else:
        held_first = sorted(
            surviving,
            key=lambda t: (t not in positions, -surviving[t], str(t)),
        )
        kept = held_first[:max_positions]
        out = {t: surviving[t] for t in sorted(kept)}
    total = sum(out.values())
    if not positions and total > 1.0 + 1e-9:
        out = {t: w / total for t, w in out.items()}
    return out


def _project_simplex_ball(values: list[float], cap: float) -> list[float]:
    """Euclidean projection of ``values`` onto ``{x >= 0 : sum(x) <= cap}``."""
    non_neg = [max(0.0, v) for v in values]
    if cap <= 0.0 or not non_neg:
        return [0.0 for _ in values]
    if sum(non_neg) <= cap + 1e-12:
        return non_neg
    u = sorted(values, reverse=True)
    cumsum = 0.0
    rho = 0
    theta = 0.0
    for j, val in enumerate(u, start=1):
        cumsum += val
        candidate = (cumsum - cap) / j
        if val - candidate > 0.0:
            rho = j
            theta = candidate
    if rho == 0:
        return [0.0 for _ in values]
    return [max(0.0, v - theta) for v in values]


def project_feasible_weights_dykstra(
    weights: Mapping[Ticker, float],
    *,
    max_position_weight: float = 0.10,
    sectors: Mapping[Ticker, str] | None = None,
    max_sector_weight: float = 0.25,
    max_gross_exposure: float = 1.0,
    locked_tickers: frozenset[Ticker] = frozenset(),
    iterations: int = 20,
) -> dict[Ticker, float]:
    """Project target weights onto box, sector, and gross constraints via Dykstra (1983)."""
    if max_position_weight <= 0.0:
        raise ValueError("max_position_weight must be positive")
    if max_sector_weight <= 0.0:
        raise ValueError("max_sector_weight must be positive")
    if max_gross_exposure <= 0.0:
        raise ValueError("max_gross_exposure must be positive")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not weights:
        return {}

    tickers = sorted(weights)
    n = len(tickers)
    locked_mask = [t in locked_tickers for t in tickers]
    x0 = [float(weights[t]) for t in tickers]
    x = list(x0)

    p_box = [0.0] * n
    p_sec = [0.0] * n
    p_gross = [0.0] * n

    sector_groups: dict[str, list[int]] = {}
    if sectors is not None:
        for idx, t in enumerate(tickers):
            sec = sectors.get(t)
            if sec:
                sector_groups.setdefault(sec, []).append(idx)

    unlocked_indices = [i for i in range(n) if not locked_mask[i]]

    def _proj_box(vec: list[float]) -> list[float]:
        out = [0.0] * n
        for i, val in enumerate(vec):
            if locked_mask[i]:
                out[i] = max(0.0, x0[i])
            else:
                out[i] = min(max(0.0, val), max_position_weight)
        return out

    def _proj_sector(vec: list[float]) -> list[float]:
        if not sector_groups:
            return list(vec)
        out = list(vec)
        for indices in sector_groups.values():
            locked_sum = sum(max(0.0, out[i]) for i in indices if locked_mask[i])
            unlocked_sec = [i for i in indices if not locked_mask[i]]
            if not unlocked_sec:
                continue
            rem_cap = max(0.0, max_sector_weight - locked_sum)
            proj_vals = _project_simplex_ball([out[i] for i in unlocked_sec], rem_cap)
            for i, pv in zip(unlocked_sec, proj_vals, strict=True):
                out[i] = pv
        return out

    def _proj_gross(vec: list[float]) -> list[float]:
        if not unlocked_indices:
            return list(vec)
        out = list(vec)
        locked_sum = sum(max(0.0, out[i]) for i in range(n) if locked_mask[i])
        rem_cap = max(0.0, max_gross_exposure - locked_sum)
        proj_vals = _project_simplex_ball([out[i] for i in unlocked_indices], rem_cap)
        for i, pv in zip(unlocked_indices, proj_vals, strict=True):
            out[i] = pv
        return out

    for _ in range(iterations):
        y_box = [x[i] + p_box[i] for i in range(n)]
        x = _proj_box(y_box)
        p_box = [y_box[i] - x[i] for i in range(n)]

        y_sec = [x[i] + p_sec[i] for i in range(n)]
        x = _proj_sector(y_sec)
        p_sec = [y_sec[i] - x[i] for i in range(n)]

        y_gross = [x[i] + p_gross[i] for i in range(n)]
        x = _proj_gross(y_gross)
        p_gross = [y_gross[i] - x[i] for i in range(n)]

    # Final feasibility pass ensures exact satisfaction of all constraints.
    x = _proj_gross(_proj_sector(_proj_box(x)))
    return {t: w for t, w in zip(tickers, x, strict=True) if w > 1e-6}


def _enforce_cash_headroom(
    surviving: Mapping[Ticker, float],
    *,
    positions: Mapping[Ticker, PositionRiskSnapshot],
) -> dict[Ticker, float]:
    """Bound new entries to available cash headroom so existing positions are never trimmed."""
    if not positions or sum(surviving.values()) <= 1.0 + 1e-9:
        return dict(surviving)
    out: dict[Ticker, float] = {t: w for t, w in surviving.items() if t in positions}
    room = max(0.0, 1.0 - sum(out.values()))
    new_entries = sorted(
        (t for t in surviving if t not in positions),
        key=lambda t: (-surviving[t], str(t)),
    )
    for t in new_entries:
        w = surviving[t]
        if w <= room + 1e-12:
            out[t] = w
            room = max(0.0, room - w)
        elif room >= 0.01:
            out[t] = room
            room = 0.0
    return out
