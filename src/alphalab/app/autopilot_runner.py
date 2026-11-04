"""NOFX-inspired autonomous agentic trading runtime over Alpaca.

"The model proposes. The runtime disposes."
-------------------------------------------
This module unifies ``alphalab``'s research and forecasting stack with NOFX's
autonomous execution discipline:

1. **Settle due forecasts** so calibration advances on every cycle.
2. **Screen candidates by liquidity only** (plus any currently open broker
   positions so held names are always re-assessed).
3. **Run the research pipeline** (:class:`AnalystPipeline` or
   :class:`DebatePipeline`) per candidate and record a forward-dated
   :class:`Prediction` or :class:`Abstention` — *every decision leaves a paper
   trail*.
4. **Build equal-weight agentic targets** (:func:`build_agentic_targets`),
   preserving cash when fewer names are approved than ``max_positions``.
5. **Enforce deterministic NOFX runtime guards** (:func:`apply_nofx_guards`):
   Safe Mode on low evaluation coverage, hard stop-loss auto-close, peak-profit
   drawdown giveback auto-close, SELL verdict exits, HOLD slot retention, and
   re-entry cooldown throttling.
6. **Plan and submit orders** through :func:`plan_trades` and
   :class:`AlpacaBroker` (sells-first, deterministic ``client_order_id``,
   :class:`KillSwitch` checked before every POST), logging every intent, guard
   block, and broker outcome to :class:`SqliteExecutionLog`.
"""

from __future__ import annotations

import contextlib
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from alphalab.adapters.alpaca_broker import (
    AccountSnapshot,
    AlpacaBroker,
    BrokerOrder,
    BrokerPosition,
    MarketClock,
    OrderOutcome,
    OrderStatus,
)
from alphalab.adapters.execution_log import (
    ExecutionEvent,
    ExecutionEventKind,
    SqliteExecutionLog,
)
from alphalab.adapters.kill_switch import KillSwitch
from alphalab.app.agentic_portfolio import (
    DEFAULT_MAX_POSITIONS,
    MIN_EVALUATED_FRACTION,
    AgenticPlan,
    ResearchFunction,
    SizingMode,
    build_agentic_targets,
    resolve_sectors,
)
from alphalab.app.agentic_universe import (
    DEFAULT_CANDIDATE_CAP,
    LIQUIDITY_MIN_DOLLAR_VOLUME,
    LIQUIDITY_MIN_PRICE,
    LIQUIDITY_WINDOW,
)
from alphalab.app.paper_runner import (
    CycleResult,
    PriceProvider,
    _blocked_reason,
    _holdings_from_positions,
    _latest_prices,
    _reconcile,
    _sells_first,
    _shadow_benchmark,
)
from alphalab.artifact import EquityResearchArtifact, Verdict
from alphalab.domain.backtest import PriceBar, SignalContext, liquidity_screened
from alphalab.domain.compute.sectors import (
    SPY_SECTOR_WEIGHTS,
    estimate_beta_from_prices,
    estimate_welch_beta,
)
from alphalab.domain.execution import (
    AutopilotGuardConfig,
    GuardAction,
    GuardActionKind,
    GuardedTargets,
    PositionRiskSnapshot,
    RiskLimits,
    TradePlan,
    apply_nofx_guards,
    plan_trades,
)
from alphalab.domain.forecast import Abstention
from alphalab.domain.journal import Prediction
from alphalab.domain.types import Side, Ticker

__all__ = [
    "AGENTIC_STRATEGY_NAME",
    "AutopilotCycleResult",
    "AutopilotRunner",
    "CandidateSelector",
    "ExposureReport",
    "ForecastRecorder",
    "IntradayHighsProvider",
    "RuntimeHealth",
    "SettlementRunner",
    "select_liquid_candidates",
]

AGENTIC_STRATEGY_NAME = "nofx_agentic"
_SPY = Ticker("SPY")
_NYSE_TZ = ZoneInfo("America/New_York")

ForecastRecorder = Callable[[EquityResearchArtifact, date], Prediction | Abstention | None]
SettlementRunner = Callable[[date], int]
IntradayHighsProvider = Callable[[Sequence[Ticker]], Mapping[Ticker, float]]
CandidateSelector = Callable[
    [date, Mapping[Ticker, Sequence[PriceBar]], Sequence[Ticker]],
    Sequence[Ticker],
]


@dataclass(frozen=True)
class RuntimeHealth:
    """Observable health state of the autonomous trader (mirrors NOFX GetStatus)."""

    safe_mode: bool
    safe_mode_reason: str | None
    trading_blocked: bool
    account_blocked: bool
    kill_switch_engaged: bool
    kill_switch_reason: str | None
    evaluated_candidates: int
    total_candidates: int


@dataclass(frozen=True)
class ExposureReport:
    """Active benchmark exposure and beta diagnostics vs SPY per cycle."""

    beta_ols: float
    beta_welch: float
    active_beta_vs_spy: float
    ex_ante_tracking_error: float
    sector_weights: Mapping[str, float]
    active_sector_gaps_vs_spy: Mapping[str, float]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AutopilotCycleResult:
    """Complete output of one autonomous agentic cycle."""

    cycle: CycleResult
    agentic_plan: AgenticPlan
    guarded: GuardedTargets
    position_risks: Mapping[Ticker, PositionRiskSnapshot]
    health: RuntimeHealth
    predictions_registered: int = 0
    abstentions_registered: int = 0
    settled_predictions: int = 0
    exposure: ExposureReport | None = None


def select_liquid_candidates(
    as_of: date,
    bars: Mapping[Ticker, Sequence[PriceBar]],
    held_tickers: Sequence[Ticker] = (),
    *,
    cap: int = DEFAULT_CANDIDATE_CAP,
) -> tuple[Ticker, ...]:
    """Screen ``bars`` by liquidity only, rank by median dollar volume, and include holdings."""
    tradable = {t for t in bars if t != _SPY}
    context = SignalContext(as_of=as_of, prices=bars)

    def nominate(context: SignalContext) -> Mapping[Ticker, float]:
        del context
        return {t: 0.0 for t in tradable}

    screen = liquidity_screened(
        nominate,
        min_price=LIQUIDITY_MIN_PRICE,
        min_dollar_volume=LIQUIDITY_MIN_DOLLAR_VOLUME,
        window=LIQUIDITY_WINDOW,
    )
    scores = screen(context)
    liquid = [t for t, s in scores.items() if s == 0.0]
    liquid.sort(key=lambda t: (-_median_dollar_volume(context, t), str(t)))
    selected = set(liquid[:cap]) | {t for t in held_tickers if t != _SPY}
    return tuple(sorted(selected))


def _median_dollar_volume(context: SignalContext, ticker: Ticker) -> float:
    window = context.history(ticker)[-LIQUIDITY_WINDOW:]
    dvols = [b.close * (b.volume or 0.0) for b in window if b.volume is not None]
    return float(statistics.median(dvols)) if dvols else 0.0


class AutopilotRunner:
    """Autonomous agentic trader: LLM proposes, NOFX runtime guards dispose."""

    def __init__(
        self,
        *,
        broker: AlpacaBroker,
        prices: PriceProvider,
        execution_log: SqliteExecutionLog,
        research: ResearchFunction,
        kill_switch: KillSwitch | None = None,
        candidate_selector: CandidateSelector | None = None,
        forecast_recorder: ForecastRecorder | None = None,
        settlement_runner: SettlementRunner | None = None,
        intraday_highs: IntradayHighsProvider | None = None,
        limits: RiskLimits | None = None,
        guard_config: AutopilotGuardConfig | None = None,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        candidate_cap: int = DEFAULT_CANDIDATE_CAP,
        min_evaluated_fraction: float = MIN_EVALUATED_FRACTION,
        sizing_mode: SizingMode = "conviction",
        require_market_open: bool = False,
        strategy_name: str = AGENTIC_STRATEGY_NAME,
        now: datetime | None = None,
    ) -> None:
        self._broker = broker
        self._prices = prices
        self._log = execution_log
        self._research = research
        self._kill_switch = kill_switch or KillSwitch()
        self._candidate_selector = candidate_selector
        self._forecast_recorder = forecast_recorder
        self._settlement_runner = settlement_runner
        self._intraday_highs = intraday_highs
        self._limits = limits or RiskLimits(whole_shares_only=False)
        self._guard_config = guard_config or AutopilotGuardConfig()
        self._max_positions = max_positions
        self._candidate_cap = candidate_cap
        self._min_evaluated_fraction = min_evaluated_fraction
        self._sizing_mode: SizingMode = sizing_mode
        self._require_market_open = require_market_open
        self._strategy_name = strategy_name
        self._now = now
        self._sell_streaks: dict[Ticker, int] = _prior_sell_streaks(execution_log)
        self._peak_prices: dict[tuple[Ticker, date], float] = {}

    def run_cycle(self, *, as_of: date | None = None, submit: bool = False) -> AutopilotCycleResult:
        """Execute one full autonomous agentic trading cycle."""
        stamp = self._now or datetime.now(UTC)
        cycle_id = f"{self._strategy_name}-{stamp.isoformat()}"
        account = self._broker.account()
        raw_positions = self._broker.positions()
        clock = self._broker.clock()
        open_orders = self._broker.open_orders() if hasattr(self._broker, "open_orders") else ()
        self._sync_broker_fills(cycle_id=cycle_id)
        decision_date = _resolve_session_date(as_of, clock, stamp)

        settled_count = self._settlement_runner(decision_date) if self._settlement_runner else 0
        if submit and self._require_market_open and not clock.is_open:
            clock, account, raw_positions, open_orders = self._await_or_refresh_open(
                clock, account, raw_positions, open_orders
            )
            decision_date = _resolve_session_date(as_of, clock, stamp)

        raw_bars = self._prices.recent_bars()
        latest_prices = {
            **_latest_prices(raw_bars),
            **{t: pos.price for t, pos in raw_positions.items() if pos.price > 0.0},
        }
        bars = _overlay_live_bars(raw_bars, latest_prices, decision_date)
        positions = _reconcile_positions_with_open_orders(raw_positions, open_orders, latest_prices)
        volatilities = _volatilities_60d(bars, decision_date)
        sizing_volatilities = _residual_volatilities_60d(bars, decision_date)

        candidates = self._choose_candidates(decision_date, bars, tuple(positions))
        agentic_plan, preds, absts = self._evaluate_candidates(
            decision_date,
            candidates,
            volatilities=sizing_volatilities,
            correlations=_avg_correlations_60d(bars, decision_date, candidates),
        )

        live_highs: Mapping[Ticker, float] = {}
        if self._intraday_highs is not None and positions:
            with contextlib.suppress(Exception):
                live_highs = self._intraday_highs(tuple(positions))
        fair_upsides = {
            o.ticker: o.upside * 100.0 for o in agentic_plan.outcomes if o.upside is not None
        }
        pos_risks = _build_position_risks(
            positions,
            bars,
            self._log,
            decision_date,
            intraday_highs=live_highs,
            peak_memory=self._peak_prices,
            volatilities=volatilities,
            fair_value_upsides=fair_upsides,
        )
        verdict_map: dict[Ticker, Verdict | None] = {
            o.ticker: o.verdict for o in agentic_plan.outcomes
        }
        equity = max(account.equity, 1e-9)
        current_weights = {t: (pos.shares * pos.price) / equity for t, pos in positions.items()}
        resolved_sectors = resolve_sectors(sorted({*agentic_plan.targets, *positions}), None)
        spy_bench_sectors = {s.value: w for s, w in SPY_SECTOR_WEIGHTS.items()}
        guarded = apply_nofx_guards(
            as_of=decision_date,
            proposed_targets=agentic_plan.targets,
            verdicts=verdict_map,
            positions=pos_risks,
            recent_sell_dates=_recent_sell_dates(self._log),
            prior_sell_counts=self._sell_streaks,
            current_weights=current_weights,
            actionable=agentic_plan.actionable,
            safe_mode_reason=None if agentic_plan.actionable else agentic_plan.note,
            max_positions=self._max_positions,
            config=self._guard_config,
            sectors=resolved_sectors,
            benchmark_sector_weights=spy_bench_sectors,
        )
        if not self._require_market_open or clock.is_open:
            _update_sell_streaks(self._sell_streaks, positions, verdict_map, guarded.actions)

        plan = plan_trades(
            as_of=decision_date,
            equity=equity,
            holdings=_holdings_from_positions(positions),
            targets=guarded.targets,
            prices=latest_prices,
            limits=self._limits,
        )
        self._record_plan_and_guards(cycle_id, stamp, plan, verdict_map, guarded.actions)
        outcomes, halted, skipped = self._submit_plan(
            cycle_id=cycle_id,
            stamp=stamp,
            plan=plan,
            account=account,
            market_open=clock.is_open,
            open_orders=open_orders,
            submit=submit,
        )
        exposure = _build_exposure_report(
            targets=guarded.targets,
            bars=bars,
            as_of=decision_date,
            sectors=resolved_sectors,
        )
        cycle = CycleResult(
            cycle_id=cycle_id,
            as_of=decision_date,
            account=account,
            market_open=clock.is_open,
            submitted=submit and not halted and not skipped,
            plan=plan,
            outcomes=tuple(outcomes),
            reconciliation=_reconcile(guarded.targets, positions, account.equity),
            shadow=_shadow_benchmark(bars, account.equity, self._inception(decision_date)),
            halted_reason=halted,
            skipped_reason=skipped,
            targets=guarded.targets,
        )
        return AutopilotCycleResult(
            cycle=cycle,
            agentic_plan=agentic_plan,
            guarded=guarded,
            position_risks=pos_risks,
            health=self._build_health(account, agentic_plan, guarded),
            predictions_registered=preds,
            abstentions_registered=absts,
            settled_predictions=settled_count,
            exposure=exposure,
        )

    def run_loop(
        self,
        *,
        as_of: date | None = None,
        submit: bool = False,
        interval_seconds: float = 3600.0,
        max_cycles: int | None = None,
        stop_requested: Callable[[], bool] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        on_cycle: Callable[[AutopilotCycleResult], None] | None = None,
    ) -> list[AutopilotCycleResult]:
        """Run autonomous cycles continuously until halted, stopped, or ``max_cycles``."""
        results: list[AutopilotCycleResult] = []
        while True:
            if stop_requested is not None and stop_requested():
                break
            if self._kill_switch.engaged():
                break
            try:
                result = self.run_cycle(as_of=as_of, submit=submit)
            except Exception:
                if max_cycles is not None:
                    raise
                sleep_fn(min(60.0, max(5.0, interval_seconds)))
                continue
            results.append(result)
            if on_cycle is not None:
                on_cycle(result)
            if result.cycle.halted_reason is not None:
                break
            if max_cycles is not None and len(results) >= max_cycles:
                break
            if stop_requested is not None and stop_requested():
                break
            sleep_fn(self._next_sleep_seconds(interval_seconds, result.cycle.market_open))
        return results

    def _await_or_refresh_open(
        self,
        clock: MarketClock,
        account: AccountSnapshot,
        raw_positions: dict[Ticker, BrokerPosition],
        open_orders: tuple[BrokerOrder, ...],
    ) -> tuple[
        MarketClock,
        AccountSnapshot,
        dict[Ticker, BrokerPosition],
        tuple[BrokerOrder, ...],
    ]:
        with contextlib.suppress(Exception):
            clock = self._broker.clock()
            if not clock.is_open and clock.next_open is not None and clock.timestamp is not None:
                rem = (clock.next_open - clock.timestamp).total_seconds()
                if 0.0 < rem <= 300.0:
                    time.sleep(rem + 15.0)
                    clock = self._broker.clock()
            if clock.is_open:
                account = self._broker.account()
                raw_positions = self._broker.positions()
                open_orders = (
                    self._broker.open_orders() if hasattr(self._broker, "open_orders") else ()
                )
        return clock, account, raw_positions, open_orders

    def _sync_broker_fills(self, *, cycle_id: str) -> int:
        """Idempotently record newly filled broker orders in the execution log."""
        filled_fn = getattr(self._broker, "filled_orders", None)
        if not callable(filled_fn):
            return 0
        try:
            fills = filled_fn()
        except Exception:
            return 0
        append_fill = getattr(self._log, "append_fill_if_new", None)
        inserted = 0
        for fill in fills:
            event = ExecutionEvent(
                cycle_id=cycle_id,
                recorded_at=fill.filled_at,
                kind=ExecutionEventKind.FILLED,
                ticker=str(fill.ticker),
                side=fill.side.value,
                shares=fill.filled_qty,
                reference_price=fill.filled_avg_price,
                client_order_id=fill.client_order_id,
                broker_order_id=fill.broker_order_id,
                detail="broker fill synced",
            )
            with contextlib.suppress(Exception):
                if callable(append_fill):
                    if append_fill(event):
                        inserted += 1
                else:
                    self._log.append(event)
                    inserted += 1
        return inserted

    def _next_sleep_seconds(self, interval_seconds: float, was_market_open: bool) -> float:
        base_wait = max(0.0, interval_seconds)
        if base_wait <= 0.0:
            return base_wait
        with contextlib.suppress(Exception):
            clock = self._broker.clock()
            if clock.timestamp is None:
                return base_wait
            if not clock.is_open and clock.next_open is not None:
                until_open = (clock.next_open - clock.timestamp).total_seconds()
                pre_open_wait = max(5.0, until_open - 300.0)
                if 0.0 < until_open <= base_wait + 300.0:
                    return min(base_wait, pre_open_wait)
            if clock.is_open and clock.next_close is not None:
                until_close = (clock.next_close - clock.timestamp).total_seconds()
                pre_close_wait = until_close - 600.0
                if 60.0 < pre_close_wait < base_wait:
                    return pre_close_wait
        return base_wait

    def _choose_candidates(
        self,
        as_of: date,
        bars: Mapping[Ticker, Sequence[PriceBar]],
        held_tickers: Sequence[Ticker],
    ) -> Sequence[Ticker]:
        if self._candidate_selector is not None:
            return self._candidate_selector(as_of, bars, held_tickers)
        return select_liquid_candidates(as_of, bars, held_tickers, cap=self._candidate_cap)

    def _evaluate_candidates(
        self,
        as_of: date,
        candidates: Sequence[Ticker],
        *,
        volatilities: Mapping[Ticker, float] | None = None,
        correlations: Mapping[Ticker, float] | None = None,
    ) -> tuple[AgenticPlan, int, int]:
        predictions = 0
        abstentions = 0

        def _wrapped_research(ticker: Ticker) -> EquityResearchArtifact | None:
            nonlocal predictions, abstentions
            artifact = self._research(ticker)
            if artifact is not None and self._forecast_recorder is not None:
                recorded = self._forecast_recorder(artifact, as_of)
                if isinstance(recorded, Prediction):
                    predictions += 1
                elif isinstance(recorded, Abstention):
                    abstentions += 1
            return artifact

        plan = build_agentic_targets(
            as_of=as_of,
            candidates=candidates,
            research=_wrapped_research,
            max_positions=self._max_positions,
            min_evaluated_fraction=self._min_evaluated_fraction,
            volatilities=volatilities,
            sizing_mode=self._sizing_mode,
            max_position_weight=self._limits.max_position_weight,
            correlations=correlations,
            sectors=resolve_sectors(candidates, None),
        )
        return plan, predictions, abstentions

    def _submit_plan(
        self,
        *,
        cycle_id: str,
        stamp: datetime,
        plan: TradePlan,
        account: AccountSnapshot,
        market_open: bool,
        open_orders: Sequence[BrokerOrder] = (),
        submit: bool,
    ) -> tuple[list[OrderOutcome], str | None, str | None]:
        if not submit:
            return [], None, "dry-run: no orders submitted (pass --submit to send)"
        if self._kill_switch.engaged():
            return [], self._kill_switch.state().reason(), None
        if account.is_blocked:
            return [], None, _blocked_reason(account)
        if self._require_market_open and not market_open:
            return [], None, "market is closed; market orders would dangle or be rejected"
        working_by_side: dict[tuple[Ticker, Side], BrokerOrder] = {
            (o.ticker, o.side): o for o in open_orders
        }
        outcomes: list[OrderOutcome] = []
        halted_reason: str | None = None
        for intent in _sells_first(plan.intents):
            existing = working_by_side.get((intent.ticker, intent.side))
            if existing is not None:
                outcome = OrderOutcome(
                    intent=intent,
                    status=OrderStatus.DUPLICATE,
                    client_order_id=intent.client_order_id(
                        strategy=self._strategy_name, as_of=plan.as_of
                    ),
                    broker_order_id=existing.broker_order_id,
                    detail="open order already working for ticker/side",
                )
            else:
                outcome = self._broker.submit(
                    intent, strategy=self._strategy_name, as_of=plan.as_of
                )
            outcomes.append(outcome)
            self._log.append(
                ExecutionEvent(
                    cycle_id=cycle_id,
                    recorded_at=stamp,
                    kind=ExecutionEventKind(outcome.status.value),
                    ticker=str(outcome.intent.ticker),
                    side=outcome.intent.side.value,
                    shares=outcome.intent.shares,
                    reference_price=outcome.intent.reference_price,
                    client_order_id=outcome.client_order_id,
                    broker_order_id=outcome.broker_order_id,
                    detail=outcome.detail,
                )
            )
            if outcome.status is OrderStatus.HALTED and halted_reason is None:
                halted_reason = outcome.detail
        return outcomes, halted_reason, None

    def _record_plan_and_guards(
        self,
        cycle_id: str,
        stamp: datetime,
        plan: TradePlan,
        verdicts: Mapping[Ticker, Verdict | None],
        actions: Sequence[GuardAction],
    ) -> None:
        action_by_ticker = {a.ticker: a for a in actions}
        for intent in plan.intents:
            detail = _intent_detail(intent.ticker, verdicts.get(intent.ticker), action_by_ticker)
            self._log.append(
                ExecutionEvent(
                    cycle_id=cycle_id,
                    recorded_at=stamp,
                    kind=ExecutionEventKind.INTENT,
                    ticker=str(intent.ticker),
                    side=intent.side.value,
                    shares=intent.shares,
                    reference_price=intent.reference_price,
                    client_order_id=intent.client_order_id(
                        strategy=self._strategy_name, as_of=plan.as_of
                    ),
                    broker_order_id=None,
                    detail=detail,
                )
            )
        for rejected in plan.rejected:
            self._log.append(
                ExecutionEvent(
                    cycle_id=cycle_id,
                    recorded_at=stamp,
                    kind=ExecutionEventKind.REJECTED,
                    ticker=str(rejected.ticker),
                    side=None,
                    shares=None,
                    reference_price=None,
                    client_order_id=None,
                    broker_order_id=None,
                    detail=rejected.reason,
                )
            )
        for action in actions:
            if action.kind in (
                GuardActionKind.COOLDOWN_BLOCKED,
                GuardActionKind.SAFE_MODE_BLOCKED,
                GuardActionKind.SELL_PENDING_CONFIRMATION,
            ):
                self._log.append(
                    ExecutionEvent(
                        cycle_id=cycle_id,
                        recorded_at=stamp,
                        kind=ExecutionEventKind.REJECTED,
                        ticker=str(action.ticker),
                        side=None,
                        shares=None,
                        reference_price=None,
                        client_order_id=None,
                        broker_order_id=None,
                        detail=f"{action.kind.value}: {action.reason}",
                    )
                )

    def _build_health(
        self,
        account: AccountSnapshot,
        agentic_plan: AgenticPlan,
        guarded: GuardedTargets,
    ) -> RuntimeHealth:
        ks = self._kill_switch.state()
        return RuntimeHealth(
            safe_mode=guarded.safe_mode,
            safe_mode_reason=guarded.safe_mode_reason,
            trading_blocked=account.trading_blocked,
            account_blocked=account.account_blocked,
            kill_switch_engaged=ks.engaged,
            kill_switch_reason=ks.reason() if ks.engaged else None,
            evaluated_candidates=agentic_plan.evaluated,
            total_candidates=agentic_plan.candidates,
        )

    def _inception(self, decision_date: date) -> date:
        earliest = decision_date
        for event in self._log:
            seen = event.recorded_at.date()
            if seen < earliest:
                earliest = seen
        return earliest


def _intent_detail(
    ticker: Ticker,
    verdict: Verdict | None,
    actions: Mapping[Ticker, GuardAction],
) -> str:
    if ticker in actions:
        action = actions[ticker]
        return f"{action.kind.value}: {action.reason}"
    verdict_label = verdict.value if verdict is not None else "NONE"
    return f"verdict={verdict_label} (nofx_agentic)"


def _overlay_live_bars(
    bars: Mapping[Ticker, Sequence[PriceBar]],
    latest_prices: Mapping[Ticker, float],
    as_of: date,
) -> dict[Ticker, Sequence[PriceBar]]:
    """Overlay live post-open prices onto ``bars`` so research and cache see ``P_open``."""
    out: dict[Ticker, Sequence[PriceBar]] = dict(bars)
    for ticker, live_px in latest_prices.items():
        if not math.isfinite(live_px) or live_px <= 0.0:
            continue
        series = list(bars.get(ticker, ()))
        if not series:
            out[ticker] = (PriceBar(on=as_of, close=live_px, raw_close=live_px),)
            continue
        last = series[-1]
        if last.on < as_of:
            series.append(
                PriceBar(
                    on=as_of,
                    close=live_px,
                    raw_close=live_px,
                    volume=last.volume,
                )
            )
            out[ticker] = tuple(series)
        elif last.on == as_of and abs(last.close - live_px) > 1e-6:
            series[-1] = PriceBar(
                on=as_of,
                close=live_px,
                raw_close=live_px,
                volume=last.volume,
                high=max(last.high or live_px, live_px),
                low=min(last.low or live_px, live_px),
            )
            out[ticker] = tuple(series)
    return out


def _volatilities_60d(
    bars: Mapping[Ticker, Sequence[PriceBar]], as_of: date
) -> dict[Ticker, float]:
    """Compute forward annualized volatility via Variance-Targeted GARCH(1,1).

    Uses up to 252 trailing closes for QML estimation with a 126-day variance
    target and 10-day forward horizon, falling back to trailing sample stdev
    when fewer than 60 daily returns are available.
    """
    from alphalab.domain.compute.volatility import forecast_garch11_volatility

    out: dict[Ticker, float] = {}
    for ticker, series in bars.items():
        window = [b.close for b in series if b.on <= as_of and b.close > 0.0][-253:]
        if len(window) < 10:
            continue
        rets = [window[i] / window[i - 1] - 1.0 for i in range(1, len(window))]
        if len(rets) >= 2:
            out[ticker] = forecast_garch11_volatility(
                rets, horizon_days=10, min_garch_obs=60, as_of=as_of
            ).annualized_volatility
    return out


def _residual_volatilities_60d(
    bars: Mapping[Ticker, Sequence[PriceBar]], as_of: date
) -> dict[Ticker, float]:
    """Compute annualized residual (idiosyncratic) forward volatility vs SPY.

    Following Baker, Bradley & Wurgler (2011, FAJ), sizing by total volatility
    $\\sqrt{\\beta_i^2 \\sigma_m^2 + \\sigma_{\\varepsilon,i}^2}$ mechanically
    penalizes market beta and crowds the portfolio into low-beta bond proxies.
    Residual volatility $\\sigma_{\\varepsilon,i}$ makes conviction/volatility
    sizing invariant to market beta. Falls back to total volatility when SPY
    bars are unavailable.
    """
    from alphalab.domain.compute.volatility import forecast_garch11_volatility

    total_vols = _volatilities_60d(bars, as_of)
    spy_series = [b for b in bars.get(_SPY, ()) if b.on <= as_of and b.close > 0.0][-127:]
    if len(spy_series) < 16:
        return total_vols
    spy_rets = {
        spy_series[i].on: spy_series[i].close / spy_series[i - 1].close - 1.0
        for i in range(1, len(spy_series))
    }
    out = dict(total_vols)
    for ticker, series in bars.items():
        if ticker == _SPY:
            continue
        window = [b for b in series if b.on <= as_of and b.close > 0.0][-127:]
        if len(window) < 16:
            continue
        asset_rets = {
            window[i].on: window[i].close / window[i - 1].close - 1.0 for i in range(1, len(window))
        }
        common = sorted(asset_rets.keys() & spy_rets.keys())
        if len(common) < 15:
            continue
        ri = [asset_rets[d] for d in common]
        rm = [spy_rets[d] for d in common]
        mean_i = statistics.mean(ri)
        mean_m = statistics.mean(rm)
        var_m = sum((x - mean_m) ** 2 for x in rm)
        if var_m <= 1e-12:
            continue
        cov_im = sum((x - mean_i) * (y - mean_m) for x, y in zip(ri, rm, strict=True))
        beta_hat = cov_im / var_m
        alpha_hat = mean_i - beta_hat * mean_m
        resids = [x - alpha_hat - beta_hat * y for x, y in zip(ri, rm, strict=True)]
        if len(resids) >= 2:
            resid_vol = forecast_garch11_volatility(
                resids, horizon_days=10, min_garch_obs=60, as_of=as_of
            ).annualized_volatility
            out[ticker] = max(0.05, resid_vol)
    return out


def _build_exposure_report(
    *,
    targets: Mapping[Ticker, float],
    bars: Mapping[Ticker, Sequence[PriceBar]],
    as_of: date,
    sectors: Mapping[Ticker, str],
) -> ExposureReport:
    """Compute portfolio beta (OLS + Welch 2022), SPY sector gaps, and tracking error."""
    spy_closes = [b.close for b in bars.get(_SPY, ()) if b.on <= as_of and b.close > 0.0]
    total_w = sum(w for w in targets.values() if w > 0.0)
    beta_ols = 0.0
    beta_welch = 0.0
    resid_var_sum = 0.0
    resid_vols = _residual_volatilities_60d(bars, as_of)
    for ticker, weight in targets.items():
        if weight <= 0.0:
            continue
        closes = [b.close for b in bars.get(ticker, ()) if b.on <= as_of and b.close > 0.0]
        b_ols = estimate_beta_from_prices(closes, spy_closes if spy_closes else None)
        b_welch = estimate_welch_beta(closes, spy_closes if spy_closes else None)
        beta_ols += weight * b_ols
        beta_welch += weight * b_welch
        sig_e = resid_vols.get(ticker, 0.20)
        resid_var_sum += (weight * sig_e) ** 2

    spy_vol = _volatilities_60d(bars, as_of).get(_SPY, 0.16)
    active_beta = beta_welch - 1.0
    ex_ante_te = math.sqrt((active_beta * spy_vol) ** 2 + resid_var_sum)

    sec_weights: dict[str, float] = {}
    for ticker, weight in targets.items():
        if weight <= 0.0:
            continue
        sec = sectors.get(ticker, "market_default")
        sec_weights[sec] = sec_weights.get(sec, 0.0) + weight

    spy_sec = {s.value: w for s, w in SPY_SECTOR_WEIGHTS.items()}
    all_secs = sorted(set(sec_weights) | set(spy_sec))
    gaps = {
        sec: sec_weights.get(sec, 0.0) - spy_sec.get(sec, 0.0)
        for sec in all_secs
        if sec != "market_default" or sec in sec_weights
    }
    warnings: list[str] = []
    if total_w > 0.0 and (beta_welch < 0.80 or beta_welch > 1.10):
        warnings.append(
            f"portfolio beta_welch={beta_welch:.2f} outside [0.80, 1.10] "
            f"(active_beta={active_beta:+.2f})"
        )
    for sec, gap in sorted(gaps.items()):
        if abs(gap) > 0.15:
            warnings.append(f"sector active gap on {sec} is {gap:+.1%} (> 15 pp vs SPY)")

    return ExposureReport(
        beta_ols=beta_ols,
        beta_welch=beta_welch,
        active_beta_vs_spy=active_beta,
        ex_ante_tracking_error=ex_ante_te,
        sector_weights=sec_weights,
        active_sector_gaps_vs_spy=gaps,
        warnings=tuple(warnings),
    )


def _avg_correlations_60d(
    bars: Mapping[Ticker, Sequence[PriceBar]],
    as_of: date,
    tickers: Sequence[Ticker],
) -> dict[Ticker, float]:
    """Average pairwise 60d return correlation of each ticker to the others.

    Returns are aligned by date; pairs with fewer than 20 shared observations
    (or a flat series) are ignored, and tickers with no usable pair are omitted.
    """
    rets: dict[Ticker, dict[date, float]] = {}
    for t in dict.fromkeys(tickers):
        window = [b for b in bars.get(t, ()) if b.on <= as_of and b.close > 0.0][-61:]
        if len(window) >= 21:
            rets[t] = {
                window[i].on: window[i].close / window[i - 1].close - 1.0
                for i in range(1, len(window))
            }
    sums: dict[Ticker, list[float]] = {t: [] for t in rets}
    names = sorted(rets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = sorted(rets[a].keys() & rets[b].keys())
            if len(common) < 20:
                continue
            xs = [rets[a][d] for d in common]
            ys = [rets[b][d] for d in common]
            try:
                corr = statistics.correlation(xs, ys)
            except statistics.StatisticsError:
                continue
            sums[a].append(corr)
            sums[b].append(corr)
    return {t: sum(v) / len(v) for t, v in sums.items() if v}


def _recent_sell_dates(log: SqliteExecutionLog) -> dict[Ticker, date]:
    sells: dict[Ticker, date] = {}
    for event in log:
        if (
            event.kind in (ExecutionEventKind.FILLED, ExecutionEventKind.SUBMITTED)
            and event.side == "sell"
        ):
            ticker = Ticker(event.ticker)
            seen = event.recorded_at.date()
            if ticker not in sells or seen > sells[ticker]:
                sells[ticker] = seen
    return sells


def _build_position_risks(
    positions: Mapping[Ticker, BrokerPosition],
    bars: Mapping[Ticker, Sequence[PriceBar]],
    log: SqliteExecutionLog,
    as_of: date,
    *,
    intraday_highs: Mapping[Ticker, float] | None = None,
    peak_memory: dict[tuple[Ticker, date], float] | None = None,
    volatilities: Mapping[Ticker, float] | None = None,
    fair_value_upsides: Mapping[Ticker, float] | None = None,
) -> dict[Ticker, PositionRiskSnapshot]:
    held_shares = {t: p.shares for t, p in positions.items() if p.shares > 0.0}
    entries = _latest_entries(log, held_shares=held_shares)
    late_fills = _late_entry_fills(log, entries)
    out: dict[Ticker, PositionRiskSnapshot] = {}
    highs = intraday_highs or {}
    vols = volatilities or {}
    upsides = fair_value_upsides or {}
    for ticker, pos in positions.items():
        if pos.shares <= 0.0 or pos.price <= 0.0:
            continue
        entry_date, logged_entry_price = entries.get(ticker, (as_of, pos.price))
        entry_price = (
            max(logged_entry_price, pos.avg_entry_price)
            if pos.avg_entry_price is not None and pos.avg_entry_price > 0.0
            else logged_entry_price
        )
        include_entry_day_high = (as_of > entry_date) and (ticker not in late_fills)
        series = bars.get(ticker, ())
        since_entry = [
            max(b.close, b.high or b.close)
            if (b.on > entry_date or include_entry_day_high)
            else max(entry_price, b.close)
            for b in series
            if entry_date <= b.on <= as_of
        ]
        live_high = highs.get(ticker, 0.0) if as_of > entry_date else 0.0
        remembered = peak_memory.get((ticker, entry_date), 0.0) if peak_memory is not None else 0.0
        peak = max([entry_price, pos.price, live_high, remembered, *since_entry])
        if peak_memory is not None:
            peak_memory[(ticker, entry_date)] = peak
        out[ticker] = PositionRiskSnapshot(
            ticker=ticker,
            shares=pos.shares,
            entry_price=entry_price,
            current_price=pos.price,
            peak_price=peak,
            annualized_volatility=vols.get(ticker),
            fair_value_upside_pct=upsides.get(ticker),
        )
    return out


def _latest_entries(
    log: SqliteExecutionLog,
    *,
    held_shares: Mapping[Ticker, float] | None = None,
) -> dict[Ticker, tuple[date, float]]:
    """Entry (date, price) of the first still-open buy per ticker.

    Prefers ``FILLED`` events for any ticker that has recorded fills, falling
    back to ``SUBMITTED`` events when no ``FILLED`` rows exist for that ticker.
    When ``held_shares`` is provided from the live broker, walks backwards over
    buy events after the last flat point until cumulative bought shares cover
    ``held_shares[ticker]``, preventing stale/canceled historical submissions
    from backdating a fresh entry.
    """
    events = list(log)
    tickers_with_fills = {Ticker(e.ticker) for e in events if e.kind is ExecutionEventKind.FILLED}
    entries: dict[Ticker, tuple[date, float]] = {}
    net: dict[Ticker, float] = {}
    open_buys: dict[Ticker, list[tuple[date, float, float]]] = {}
    for event in events:
        ticker = Ticker(event.ticker)
        target_kind = (
            ExecutionEventKind.FILLED
            if ticker in tickers_with_fills
            else ExecutionEventKind.SUBMITTED
        )
        if event.kind is not target_kind:
            continue
        shares = event.shares or 0.0
        if event.side == "buy" and event.reference_price and event.reference_price > 0.0:
            if net.get(ticker, 0.0) <= 1e-6:
                entries[ticker] = (event.recorded_at.date(), event.reference_price)
                net[ticker] = 0.0
                open_buys[ticker] = []
            net[ticker] += shares
            open_buys.setdefault(ticker, []).append(
                (event.recorded_at.date(), event.reference_price, shares)
            )
        elif event.side == "sell":
            net[ticker] = net.get(ticker, 0.0) - shares
            if net[ticker] <= 1e-6:
                entries.pop(ticker, None)
                open_buys.pop(ticker, None)
                net[ticker] = 0.0
    if held_shares:
        for ticker, target_qty in held_shares.items():
            lots = open_buys.get(ticker)
            if not lots or target_qty <= 0.0:
                continue
            acc = 0.0
            for lot_date, lot_price, lot_shares in reversed(lots):
                entries[ticker] = (lot_date, lot_price)
                acc += lot_shares
                if acc >= target_qty * 0.99:
                    break
    return entries


def _late_entry_fills(
    log: SqliteExecutionLog,
    entries: Mapping[Ticker, tuple[date, float]],
) -> set[Ticker]:
    """Tickers whose Lot-1 FILLED event occurred after the morning open window (10:00 ET)."""
    late: set[Ticker] = set()
    for event in log:
        if event.kind is not ExecutionEventKind.FILLED or event.side != "buy":
            continue
        ticker = Ticker(event.ticker)
        entry = entries.get(ticker)
        if entry is None or event.recorded_at.date() != entry[0]:
            continue
        ny_time = event.recorded_at.astimezone(_NYSE_TZ)
        if (ny_time.hour, ny_time.minute) >= (10, 0):
            late.add(ticker)
    return late


def _resolve_session_date(
    as_of: date | None,
    clock: MarketClock,
    stamp: datetime,
) -> date:
    """Resolve the active or next NYSE session date in Eastern Time."""
    if as_of is not None:
        return as_of
    if not clock.is_open and clock.next_open is not None:
        return clock.next_open.astimezone(_NYSE_TZ).date()
    if clock.timestamp is not None:
        return clock.timestamp.astimezone(_NYSE_TZ).date()
    if stamp.tzinfo is not None and stamp.hour == 0 and stamp.minute == 0:
        return stamp.date()
    return stamp.astimezone(_NYSE_TZ).date()


def _reconcile_positions_with_open_orders(
    positions: Mapping[Ticker, BrokerPosition],
    open_orders: Sequence[BrokerOrder],
    latest_prices: Mapping[Ticker, float],
) -> dict[Ticker, BrokerPosition]:
    """Merge live positions with working orders (Positions + OpenOrders)."""
    effective: dict[Ticker, BrokerPosition] = dict(positions)
    for order in open_orders:
        if order.shares <= 0.0:
            continue
        current = effective.get(order.ticker)
        if order.side is Side.BUY:
            if current is not None:
                effective[order.ticker] = BrokerPosition(
                    ticker=order.ticker,
                    shares=current.shares + order.shares,
                    price=current.price,
                    avg_entry_price=current.avg_entry_price,
                )
            else:
                ref_price = latest_prices.get(order.ticker, 0.0)
                if ref_price > 0.0:
                    effective[order.ticker] = BrokerPosition(
                        ticker=order.ticker,
                        shares=order.shares,
                        price=ref_price,
                        avg_entry_price=ref_price,
                    )
        elif order.side is Side.SELL and current is not None:
            remaining = current.shares - order.shares
            if remaining > 1e-6:
                effective[order.ticker] = BrokerPosition(
                    ticker=order.ticker,
                    shares=remaining,
                    price=current.price,
                    avg_entry_price=current.avg_entry_price,
                )
            else:
                effective.pop(order.ticker, None)
    return effective


def _prior_sell_streaks(log: SqliteExecutionLog) -> dict[Ticker, int]:
    """Reconstruct consecutive pending-SELL streaks from the execution log."""
    streaks: dict[Ticker, int] = {}
    for event in log:
        ticker = Ticker(event.ticker)
        detail = event.detail or ""
        if event.kind is ExecutionEventKind.REJECTED and detail.startswith(
            GuardActionKind.SELL_PENDING_CONFIRMATION.value
        ):
            streaks[ticker] = streaks.get(ticker, 0) + 1
        elif event.kind is ExecutionEventKind.SUBMITTED:
            streaks.pop(ticker, None)
    return streaks


def _update_sell_streaks(
    streaks: dict[Ticker, int],
    positions: Mapping[Ticker, BrokerPosition],
    verdicts: Mapping[Ticker, Verdict | None],
    actions: Sequence[GuardAction],
) -> None:
    """Update in-memory consecutive SELL verdict counts after guard evaluation."""
    action_kinds = {a.ticker: a.kind for a in actions}
    for ticker in list(streaks):
        if ticker not in positions:
            streaks.pop(ticker, None)
    for ticker in positions:
        kind = action_kinds.get(ticker)
        verdict = verdicts.get(ticker)
        if kind is GuardActionKind.SELL_PENDING_CONFIRMATION:
            streaks[ticker] = streaks.get(ticker, 0) + 1
        elif kind is GuardActionKind.SELL_VERDICT_EXIT or verdict in (Verdict.BUY, Verdict.HOLD):
            streaks.pop(ticker, None)
