"""The paper-trading cycle: decide, plan, (optionally) submit, record, reconcile.

Honesty about what this runs
----------------------------
No strategy here has a demonstrated edge. The momentum ensemble's out-of-sample
information ratio was -0.10 and the turbulence overlay did not beat random
abstention. Running this against a paper account is not an attempt to make
money; it is (a) a forward test of the execution plumbing with real fills and
real timing, and (b) an honest, hindsight-proof forward record. Nothing in this
module should be read as expecting a profit.

The cycle is idempotent
------------------------
1. Read account and positions from the broker.
2. Build target weights from the *already-measured* signal
   (``liquidity_screened(rank_ensemble_signal(lookbacks=(60, 120, 250)))``,
   top 5, equal weight) — not a new signal invented here.
3. Hand holdings, targets and prices to :func:`plan_trades`, the single
   sanctioned path for deciding orders. The runner never computes deltas itself.
4. Submit the plan's intents (only when ``submit=True``; the default is a
   dry run that sends nothing). Each order carries a deterministic
   ``client_order_id``, so re-running the same cycle is a no-op at the broker.
5. Record every intent, outcome and rejection in the append-only execution log.

It also **reconciles** realised positions against targets and records a
**shadow SPY buy-and-hold** benchmark, so a forward return is never reported
without a reference. Blocked accounts and a closed market short-circuit
submission with an actionable report rather than a dangling order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Protocol

from alphalab.adapters.alpaca_broker import (
    AccountSnapshot,
    AlpacaBroker,
    BrokerPosition,
    OrderOutcome,
    OrderStatus,
)
from alphalab.adapters.execution_log import (
    ExecutionEvent,
    ExecutionEventKind,
    SqliteExecutionLog,
)
from alphalab.app.paper_models import Reconciliation, ReconciliationRow, ShadowBenchmark
from alphalab.domain.backtest import (
    PriceBar,
    SignalContext,
    liquidity_screened,
    rank_ensemble_signal,
)
from alphalab.domain.execution import Holding, RiskLimits, TradePlan, plan_trades
from alphalab.domain.types import Ticker

__all__ = [
    "DEFAULT_LOOKBACKS",
    "DEFAULT_TOP_N",
    "STRATEGY_NAME",
    "CycleResult",
    "PaperRunner",
    "PriceProvider",
    "build_targets",
]

#: The ensemble lookbacks already measured in backtest. Not re-tuned here.
DEFAULT_LOOKBACKS = (60, 120, 250)

#: Number of names held, equal-weighted. Matches the measured variant.
DEFAULT_TOP_N = 5

#: Identifier baked into every ``client_order_id``, so re-runs are idempotent.
STRATEGY_NAME = "rank_ensemble"

#: The benchmark the shadow buy-and-hold tracks.
_SPY = Ticker("SPY")


class PriceProvider(Protocol):
    """Supplies recent daily bars per ticker.

    Behind this protocol the CLI wires the live Alpaca bars client or a local
    CSV source; unit tests pass a dict-backed double. Keeping it a protocol is
    what lets the runner's logic be tested without a network.
    """

    def recent_bars(self) -> Mapping[Ticker, Sequence[PriceBar]]: ...


@dataclass(frozen=True)
class CycleResult:
    """Everything one cycle produced, for rendering and assertions."""

    cycle_id: str
    as_of: date
    account: AccountSnapshot
    market_open: bool
    submitted: bool
    plan: TradePlan
    outcomes: tuple[OrderOutcome, ...]
    reconciliation: Reconciliation
    shadow: ShadowBenchmark | None
    halted_reason: str | None = None
    skipped_reason: str | None = None
    targets: Mapping[Ticker, float] = field(default_factory=dict)

    @property
    def orders_reached_broker(self) -> int:
        """How many intents now exist at the broker (submitted or duplicate)."""
        return sum(1 for o in self.outcomes if o.reached_broker)


def build_targets(
    *,
    as_of: date,
    prices: Mapping[Ticker, Sequence[PriceBar]],
    top_n: int = DEFAULT_TOP_N,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
) -> dict[Ticker, float]:
    """Compute equal-weight target weights from the measured ensemble signal.

    Selection mirrors the backtest engine exactly: score every name with
    ``liquidity_screened(rank_ensemble_signal(...))`` on a past-only context,
    drop NaNs, take the ``top_n`` highest scores with ticker as the
    deterministic tie-break, and equal-weight the survivors. This is the same
    specification already measured out-of-sample; the runner does not invent a
    new one.
    """
    signal = liquidity_screened(rank_ensemble_signal(lookbacks=tuple(lookbacks)))
    context = SignalContext(as_of=as_of, prices=prices)
    scores = signal(context)
    eligible = [
        (score, ticker)
        for ticker, score in scores.items()
        if context.latest_close(ticker) is not None and not _is_nan(score)
    ]
    eligible.sort(key=lambda pair: (-pair[0], pair[1]))
    selected = [ticker for _, ticker in eligible[:top_n]]
    if not selected:
        return {}
    weight = 1.0 / len(selected)
    return {ticker: weight for ticker in selected}


class PaperRunner:
    """Runs one paper-trading cycle against an injected broker and log."""

    def __init__(
        self,
        *,
        broker: AlpacaBroker,
        prices: PriceProvider,
        execution_log: SqliteExecutionLog,
        limits: RiskLimits | None = None,
        top_n: int = DEFAULT_TOP_N,
        lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
        now: datetime | None = None,
    ) -> None:
        self._broker = broker
        self._prices = prices
        self._log = execution_log
        self._limits = limits or RiskLimits()
        self._top_n = top_n
        self._lookbacks = tuple(lookbacks)
        self._now = now

    def _inception(self, decision_date: date) -> date:
        """Earliest date this account acted, from the append-only log.

        The log is the authoritative record of the account's life, so its
        oldest event dates inception. The current cycle's own events may
        already be written by the time this is called, which is why the
        decision date is folded in with ``min``: inception must never drift
        later than it already is, or the benchmark would silently re-anchor on
        every run and stop being a fixed yardstick.

        Deriving inception from the price window instead would make the
        benchmark move whenever the signal's lookback changed.
        """
        earliest = decision_date
        for event in self._log:
            seen = event.recorded_at.date()
            if seen < earliest:
                earliest = seen
        return earliest

    def run_cycle(self, *, as_of: date | None = None, submit: bool = False) -> CycleResult:
        """Execute one cycle. ``submit=False`` (the default) sends nothing."""
        stamp = self._timestamp()
        cycle_id = f"{STRATEGY_NAME}-{stamp.isoformat()}"
        decision_date = as_of or stamp.date()

        account = self._broker.account()
        positions = self._broker.positions()
        clock = self._broker.clock()
        bars = self._prices.recent_bars()

        targets = build_targets(
            as_of=decision_date, prices=bars, top_n=self._top_n, lookbacks=self._lookbacks
        )
        prices_now = _latest_prices(bars)
        holdings = _holdings_from_positions(positions)
        plan = plan_trades(
            as_of=decision_date,
            equity=max(account.equity, 1e-9),
            holdings=holdings,
            targets=targets,
            prices=prices_now,
            limits=self._limits,
        )
        self._record_plan(cycle_id, stamp, plan)

        outcomes, halted, skipped = self._maybe_submit(
            cycle_id=cycle_id,
            stamp=stamp,
            plan=plan,
            account=account,
            market_open=clock.is_open,
            submit=submit,
        )
        reconciliation = _reconcile(targets, positions, account.equity)
        shadow = _shadow_benchmark(bars, account.equity, self._inception(decision_date))
        return CycleResult(
            cycle_id=cycle_id,
            as_of=decision_date,
            account=account,
            market_open=clock.is_open,
            submitted=submit and not halted and not skipped,
            plan=plan,
            outcomes=tuple(outcomes),
            reconciliation=reconciliation,
            shadow=shadow,
            halted_reason=halted,
            skipped_reason=skipped,
            targets=targets,
        )

    # -- submission -------------------------------------------------------

    def _maybe_submit(
        self,
        *,
        cycle_id: str,
        stamp: datetime,
        plan: TradePlan,
        account: AccountSnapshot,
        market_open: bool,
        submit: bool,
    ) -> tuple[list[OrderOutcome], str | None, str | None]:
        """Submit the plan's intents unless a gate forbids it.

        Returns the outcomes plus, when emission was blocked, a halt reason or
        a skip reason. Gates are checked before any POST so a blocked account or
        closed market produces zero order attempts, not rejected orders.
        """
        if not submit:
            return [], None, "dry-run: no orders submitted (pass --submit to send)"
        if account.is_blocked:
            return [], None, _blocked_reason(account)
        if not market_open:
            return [], None, "market is closed; market orders would dangle or be rejected"
        outcomes: list[OrderOutcome] = []
        halted_reason: str | None = None
        for intent in plan.intents:
            outcome = self._broker.submit(intent, strategy=STRATEGY_NAME, as_of=plan.as_of)
            outcomes.append(outcome)
            self._record_outcome(cycle_id, stamp, outcome)
            if outcome.status is OrderStatus.HALTED and halted_reason is None:
                halted_reason = outcome.detail
        return outcomes, halted_reason, None

    # -- logging ----------------------------------------------------------

    def _record_plan(self, cycle_id: str, stamp: datetime, plan: TradePlan) -> None:
        """Record every intent and every rejection before any submission."""
        for intent in plan.intents:
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
                        strategy=STRATEGY_NAME, as_of=plan.as_of
                    ),
                    broker_order_id=None,
                    detail=None,
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

    def _record_outcome(self, cycle_id: str, stamp: datetime, outcome: OrderOutcome) -> None:
        """Record the result of submitting one intent."""
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

    def _timestamp(self) -> datetime:
        return self._now or datetime.now(UTC)


# -- pure helpers ---------------------------------------------------------


def _holdings_from_positions(
    positions: Mapping[Ticker, BrokerPosition],
) -> dict[Ticker, Holding]:
    """Translate broker positions into planner holdings, dropping empties."""
    out: dict[Ticker, Holding] = {}
    for ticker, pos in positions.items():
        if pos.shares == 0.0 or pos.price <= 0.0:
            continue
        out[ticker] = Holding(ticker=ticker, shares=pos.shares, price=pos.price)
    return out


def _latest_prices(prices: Mapping[Ticker, Sequence[PriceBar]]) -> dict[Ticker, float]:
    """Most recent close per ticker, for sizing orders."""
    out: dict[Ticker, float] = {}
    for ticker, bars in prices.items():
        if bars:
            out[ticker] = bars[-1].close
    return out


def _reconcile(
    targets: Mapping[Ticker, float],
    positions: Mapping[Ticker, BrokerPosition],
    equity: float,
) -> Reconciliation:
    """Compare realised position weights against targets, name by name."""
    actual = _actual_weights(positions, equity)
    names = sorted(set(targets) | set(actual))
    rows = tuple(
        ReconciliationRow(
            ticker=ticker,
            target_weight=targets.get(ticker, 0.0),
            actual_weight=actual.get(ticker, 0.0),
        )
        for ticker in names
    )
    return Reconciliation(rows=rows)


def _actual_weights(
    positions: Mapping[Ticker, BrokerPosition], equity: float
) -> dict[Ticker, float]:
    if equity <= 0.0:
        return {}
    return {
        ticker: (pos.shares * pos.price) / equity
        for ticker, pos in positions.items()
        if pos.price > 0.0
    }


def _shadow_benchmark(
    prices: Mapping[Ticker, Sequence[PriceBar]],
    equity: float,
    inception: date,
) -> ShadowBenchmark | None:
    """Build the SPY buy-and-hold yardstick, anchored at account inception.

    The entry price must be SPY's close on the first day this account traded,
    **not** the first bar of the fetched window. The window spans the signal's
    full lookback (250 trading days), so using ``bars[0]`` measured SPY over
    years the paper account did not exist — it reported a +719% benchmark
    against a portfolio days old, which would make any honest comparison
    impossible.

    Returns ``None`` only when SPY has no bars at all.
    """
    bars = prices.get(_SPY)
    if not bars:
        return None
    entry = next((bar for bar in bars if bar.on >= inception), None)
    if entry is None:
        # Inception is newer than the freshest bar, which happens on the first
        # cycle because the data feed lags the calendar by a day. Entering at
        # the most recent available close is what an actual purchase would do
        # with the same information, so the yardstick starts alongside the
        # strategy instead of staying blank forever — the lag is permanent, so
        # waiting for a bar dated today would never resolve.
        entry = bars[-1]
    return ShadowBenchmark(
        symbol=_SPY,
        initial_capital=equity,
        entry_price=entry.close,
        latest_price=bars[-1].close,
    )


def _blocked_reason(account: AccountSnapshot) -> str:
    flags = []
    if account.trading_blocked:
        flags.append("trading_blocked")
    if account.account_blocked:
        flags.append("account_blocked")
    return f"account flags forbid trading: {', '.join(flags)}"


def _is_nan(value: float) -> bool:
    return value != value
