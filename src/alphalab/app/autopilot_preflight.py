"""NOFX-inspired launch preflight checks for autonomous agentic trading.

Before an autonomous trading loop is allowed to evaluate candidates or route
orders to Alpaca, preflight verifies that every runtime dependency is healthy:

1. **Kill switch** — neither the ``HALT`` file nor ``ALPHALAB_PAPER_HALT`` is
   engaged.
2. **Broker account** — Alpaca credentials authenticate, ``equity > 0``, and
   neither ``trading_blocked`` nor ``account_blocked`` is set.
3. **Market clock** — Alpaca clock endpoint responds (and, when
   ``require_market_open=True``, the market is currently open).
4. **Price feed** — the bar provider returns non-empty history including the
   ``SPY`` benchmark.
5. **Fundamentals source** — when ``source="live"``, ``ALPHALAB_EDGAR_USER_AGENT``
   is configured for SEC Fair Access compliance.
6. **AI model provider** — when ``use_llm=True``, the configured LLM endpoint
   responds; when ``use_llm=False``, deterministic template mode is acknowledged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from alphalab.adapters.alpaca_broker import AlpacaBroker
from alphalab.adapters.kill_switch import KillSwitch
from alphalab.app.paper_runner import PriceProvider
from alphalab.domain.types import Ticker

__all__ = [
    "PreflightCheck",
    "PreflightReport",
    "run_preflight",
]

_SPY = Ticker("SPY")


@dataclass(frozen=True)
class PreflightCheck:
    """Result of one launch preflight check."""

    name: str
    passed: bool
    detail: str
    blocking: bool = True


@dataclass(frozen=True)
class PreflightReport:
    """Aggregated launch preflight readiness report."""

    checks: tuple[PreflightCheck, ...]

    @property
    def ready(self) -> bool:
        """Whether every blocking check passed."""
        return all(check.passed or not check.blocking for check in self.checks)

    @property
    def blocking_failures(self) -> tuple[PreflightCheck, ...]:
        """Blocking checks that failed."""
        return tuple(check for check in self.checks if not check.passed and check.blocking)


def run_preflight(
    *,
    broker: AlpacaBroker,
    kill_switch: KillSwitch,
    prices: PriceProvider,
    source: str = "live",
    use_llm: bool = True,
    edgar_user_agent: str | None = None,
    llm_probe: Callable[[], tuple[bool, str]] | None = None,
    require_market_open: bool = False,
) -> PreflightReport:
    """Run all launch preflight checks without mutating broker or store state."""
    checks = (
        _check_kill_switch(kill_switch),
        _check_broker_account(broker),
        _check_market_clock(broker, require_open=require_market_open),
        _check_price_feed(prices),
        _check_fundamentals_source(source=source, edgar_user_agent=edgar_user_agent),
        _check_llm_provider(use_llm=use_llm, llm_probe=llm_probe),
    )
    return PreflightReport(checks=checks)


def _check_kill_switch(kill_switch: KillSwitch) -> PreflightCheck:
    state = kill_switch.state()
    if state.engaged:
        return PreflightCheck(
            name="kill_switch",
            passed=False,
            detail=state.reason(),
            blocking=True,
        )
    return PreflightCheck(
        name="kill_switch",
        passed=True,
        detail="clear (neither HALT file nor ALPHALAB_PAPER_HALT is set)",
        blocking=True,
    )


def _check_broker_account(broker: AlpacaBroker) -> PreflightCheck:
    try:
        account = broker.account()
    except Exception as exc:
        return PreflightCheck(
            name="broker_account",
            passed=False,
            detail=f"Alpaca account unreachable: {type(exc).__name__}: {exc}",
            blocking=True,
        )
    if account.trading_blocked or account.account_blocked:
        flags = []
        if account.trading_blocked:
            flags.append("trading_blocked=True")
        if account.account_blocked:
            flags.append("account_blocked=True")
        return PreflightCheck(
            name="broker_account",
            passed=False,
            detail=f"account blocked ({', '.join(flags)})",
            blocking=True,
        )
    if account.equity <= 0.0:
        return PreflightCheck(
            name="broker_account",
            passed=False,
            detail=f"non-positive account equity (${account.equity:,.2f})",
            blocking=True,
        )
    return PreflightCheck(
        name="broker_account",
        passed=True,
        detail=f"ready (equity=${account.equity:,.2f}, buying_power=${account.buying_power:,.2f})",
        blocking=True,
    )


def _check_market_clock(broker: AlpacaBroker, *, require_open: bool) -> PreflightCheck:
    try:
        clock = broker.clock()
    except Exception as exc:
        return PreflightCheck(
            name="market_clock",
            passed=False,
            detail=f"Alpaca clock unreachable: {type(exc).__name__}: {exc}",
            blocking=True,
        )
    if clock.is_open:
        return PreflightCheck(
            name="market_clock",
            passed=True,
            detail="market is OPEN",
            blocking=require_open,
        )
    return PreflightCheck(
        name="market_clock",
        passed=not require_open,
        detail="market is CLOSED (dry-run works; live submission requires market open)",
        blocking=require_open,
    )


def _check_price_feed(prices: PriceProvider) -> PreflightCheck:
    try:
        panel = prices.recent_bars()
    except Exception as exc:
        return PreflightCheck(
            name="price_feed",
            passed=False,
            detail=f"price provider failed: {type(exc).__name__}: {exc}",
            blocking=True,
        )
    if not panel:
        return PreflightCheck(
            name="price_feed",
            passed=False,
            detail="price provider returned 0 symbols",
            blocking=True,
        )
    if _SPY not in panel or not panel[_SPY]:
        return PreflightCheck(
            name="price_feed",
            passed=False,
            detail=f"loaded {len(panel)} symbol(s), but benchmark SPY is missing",
            blocking=True,
        )
    return PreflightCheck(
        name="price_feed",
        passed=True,
        detail=f"{len(panel)} symbol(s) loaded with SPY benchmark",
        blocking=True,
    )


def _check_fundamentals_source(*, source: str, edgar_user_agent: str | None) -> PreflightCheck:
    if source != "live":
        return PreflightCheck(
            name="fundamentals_source",
            passed=True,
            detail=f"source={source!r} (local fixtures; SEC EDGAR not required)",
            blocking=False,
        )
    if edgar_user_agent and edgar_user_agent.strip():
        return PreflightCheck(
            name="fundamentals_source",
            passed=True,
            detail="source='live' (ALPHALAB_EDGAR_USER_AGENT configured)",
            blocking=True,
        )
    return PreflightCheck(
        name="fundamentals_source",
        passed=False,
        detail="source='live' requires ALPHALAB_EDGAR_USER_AGENT for SEC Fair Access",
        blocking=True,
    )


def _check_llm_provider(
    *,
    use_llm: bool,
    llm_probe: Callable[[], tuple[bool, str]] | None,
) -> PreflightCheck:
    if not use_llm:
        return PreflightCheck(
            name="ai_model",
            passed=True,
            detail="deterministic template narrator (--no-llm)",
            blocking=False,
        )
    if llm_probe is None:
        return PreflightCheck(
            name="ai_model",
            passed=True,
            detail="LLM enabled (no probe supplied)",
            blocking=True,
        )
    ok, detail = llm_probe()
    return PreflightCheck(
        name="ai_model",
        passed=ok,
        detail=detail,
        blocking=True,
    )
