"""The ``autopilot`` commands: ``preflight``, ``status``, ``run``, and ``loop``.

This is the CLI composition root for the NOFX-inspired autonomous agentic
trading system:

* **preflight** — runs :func:`run_preflight` to verify KillSwitch, Alpaca
  account, market clock, price feed, EDGAR ``User-Agent``, and Ollama LLM
  readiness before trading.
* **status** — runs a read-only agentic assessment cycle (``submit=False``) and
  displays open position risk (PnL%, peak PnL%, drawdown giveback%), AI
  verdicts, NOFX runtime guard actions, and target reconciliation.
* **run** — runs one full autonomous cycle (dry-run by default; pass
  ``--submit`` to route live market orders to the Alpaca paper account).
* **loop** — runs continuous autonomous cycles on a configurable interval until
  stopped or halted by the kill switch.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TextIO

from alphalab.adapters.abstention_store import SqliteAbstentionStore
from alphalab.adapters.alpaca_bars import AlpacaBarsClient
from alphalab.adapters.alpaca_broker import AlpacaBroker
from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.execution_log import SqliteExecutionLog
from alphalab.adapters.journal_sqlite import SqliteJournal
from alphalab.adapters.kill_switch import KillSwitch
from alphalab.adapters.llm_verdict_cache import (
    LlmVerdictCache,
    compute_request_fingerprint,
)
from alphalab.adapters.prices import CsvPriceSeriesSource, PricePoint, PriceSeriesSource
from alphalab.app.agentic_portfolio import DEFAULT_MAX_POSITIONS, ResearchFunction
from alphalab.app.autopilot_preflight import run_preflight
from alphalab.app.autopilot_runner import (
    AutopilotCycleResult,
    AutopilotRunner,
    CandidateSelector,
    ForecastRecorder,
    SettlementRunner,
    _recent_sell_dates,
    select_liquid_candidates,
)
from alphalab.app.forecast_runner import register_forecast, settle_due_predictions
from alphalab.app.paper_runner import PriceProvider
from alphalab.artifact import EquityResearchArtifact, Verdict
from alphalab.cli.autopilot_render import (
    autopilot_cycle_json,
    preflight_json,
    render_autopilot_run,
    render_autopilot_status,
    render_preflight,
)
from alphalab.cli.commands.doctor import CheckStatus, _check_ollama
from alphalab.cli.commands.paper import _price_provider
from alphalab.cli.commands.predict import DEFAULT_HORIZON_DAYS
from alphalab.cli.commands.research import _run_pipeline
from alphalab.cli.fixtures import build_fixture_request
from alphalab.cli.ingest import load_live_builder, load_news_sentiment_fn
from alphalab.cli.paper_transport import HttpxWriteTransport
from alphalab.domain.backtest import PriceBar, ProportionalCostModel
from alphalab.domain.compute.sectors import shrink_upside_empirical_bayes
from alphalab.domain.contract.gate import conservative_c1b_upside
from alphalab.domain.costs import CostSpec
from alphalab.domain.execution import AutopilotGuardConfig
from alphalab.domain.forecast import Abstention
from alphalab.domain.journal import Prediction
from alphalab.domain.types import Ticker
from alphalab.report.theme import theme_for

__all__ = [
    "DEFAULT_AUTOPILOT_CANDIDATE_CAP",
    "SP500_ALIASES",
    "AutopilotCycleOptions",
    "AutopilotPreflightOptions",
    "BrokerFactory",
    "PanelPriceSeriesSource",
    "PricesFactory",
    "expand_ticker_aliases",
    "load_sp500_tickers",
    "run_autopilot_loop",
    "run_autopilot_preflight",
    "run_autopilot_run",
    "run_autopilot_status",
]

#: Interactive CLI default cap so a status/run cycle completes quickly while
#: still screening the liquid universe. Configurable via ``--candidate-cap``.
DEFAULT_AUTOPILOT_CANDIDATE_CAP = 8
_DEFAULT_HALF_SPREAD_BPS = 5.0
SP500_ALIASES: frozenset[str] = frozenset({"SP500", "SPY500", "S&P500"})
_SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)
_DEFAULT_SP500_CACHE = Path("data/sp500_constituents.json")

BrokerFactory = Callable[[KillSwitch], AlpacaBroker]
PricesFactory = Callable[[Path], PriceProvider]


def load_sp500_tickers(cache_path: Path = _DEFAULT_SP500_CACHE) -> tuple[str, ...]:
    """Load S&P 500 constituent symbols from local cache or canonical dataset."""
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, list) and cached:
            return tuple(str(t).strip().upper() for t in cached if str(t).strip())
    import csv
    import io

    import httpx

    resp = httpx.get(_SP500_CSV_URL, timeout=20.0)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    symbols = tuple(
        row["Symbol"].strip().upper() for row in reader if row.get("Symbol", "").strip()
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(list(symbols), indent=2), encoding="utf-8")
    return symbols


def expand_ticker_aliases(
    raw_tickers: Sequence[str],
    *,
    sp500_loader: Callable[[], Sequence[str]] = load_sp500_tickers,
) -> tuple[str, ...]:
    """Expand ``sp500`` / ``spy500`` tokens into the full S&P 500 symbol list."""
    expanded: list[str] = []
    for raw in raw_tickers:
        token = raw.strip().upper()
        if not token:
            continue
        if token in SP500_ALIASES:
            expanded.extend(t.strip().upper() for t in sp500_loader() if t.strip())
        else:
            expanded.append(token)
    return tuple(dict.fromkeys(expanded))


def _default_broker(kill_switch: KillSwitch) -> AlpacaBroker:
    settings = AlpacaSettings()  # type: ignore[call-arg]
    return AlpacaBroker(settings, transport=HttpxWriteTransport(), kill_switch=kill_switch)


def _default_prices(
    prices_dir: Path,
    *,
    tickers: tuple[str, ...] | None = None,
    held: Sequence[Ticker] = (),
) -> PriceProvider:
    settings = AlpacaSettings()  # type: ignore[call-arg]
    if tickers:
        resolved = expand_ticker_aliases(tickers)
        explicit = [Ticker(t) for t in resolved]
        combined = list(dict.fromkeys([*explicit, *held, Ticker("SPY")]))
        return _price_provider(settings, prices_dir, tickers=combined)
    return _price_provider(settings, prices_dir)


class PanelPriceSeriesSource:
    """Bridges an in-memory ``PriceBar`` panel to :class:`PriceSeriesSource` with CSV fallback."""

    def __init__(
        self,
        panel: Mapping[Ticker, Sequence[PriceBar]],
        fallback: PriceSeriesSource | None = None,
        current_as_of: date | None = None,
    ) -> None:
        self._panel = panel
        self._fallback = fallback
        self.current_as_of = current_as_of

    def update_panel(self, panel: Mapping[Ticker, Sequence[PriceBar]]) -> None:
        """Refresh the underlying bar panel on a new cycle."""
        self._panel = panel

    def series(self, ticker: Ticker) -> list[PricePoint]:
        bars = self._panel.get(ticker)
        if bars:
            return [PricePoint(on=b.on, close=b.close, raw_close=b.raw_close) for b in bars]
        if self._fallback is not None:
            return self._fallback.series(ticker)
        return []


@dataclass(frozen=True)
class AutopilotPreflightOptions:
    """Resolved options for ``alphalab autopilot preflight``."""

    prices_dir: Path
    halt_path: Path | None
    source: str
    use_llm: bool
    require_market_open: bool
    as_json: bool
    no_color: bool


@dataclass(frozen=True)
class AutopilotCycleOptions:
    """Resolved options for ``autopilot status``, ``run``, and ``loop``."""

    execution_db: Path
    journal_db: Path
    abstentions_db: Path
    prices_dir: Path
    halt_path: Path | None
    as_of: date | None
    source: str
    use_llm: bool
    debate: bool
    submit: bool
    candidate_cap: int = DEFAULT_AUTOPILOT_CANDIDATE_CAP
    max_positions: int = DEFAULT_MAX_POSITIONS
    stop_loss_pct: float = -8.0
    drawdown_trigger_pct: float = 5.0
    drawdown_giveback_pct: float = 40.0
    cooldown_days: int = 3
    sell_confirmations_required: int = 2
    require_market_open: bool = True
    tickers: tuple[str, ...] | None = None
    assume_verdict: Verdict | None = None
    interval_seconds: float = 3600.0
    max_cycles: int | None = None
    as_json: bool = False
    no_color: bool = False


def run_autopilot_preflight(
    opts: AutopilotPreflightOptions,
    *,
    out: TextIO,
    err: TextIO,
    broker_factory: BrokerFactory | None = None,
    prices_factory: PricesFactory | None = None,
    llm_probe: Callable[[], tuple[bool, str]] | None = None,
) -> int:
    """Execute launch preflight checks and report readiness."""
    kill_switch = KillSwitch(halt_path=opts.halt_path)
    try:
        broker = (broker_factory or _default_broker)(kill_switch)
        prices = (prices_factory or _default_prices)(opts.prices_dir)
    except Exception as exc:
        print(f"error: {exc}", file=err)
        return 1

    report = run_preflight(
        broker=broker,
        kill_switch=kill_switch,
        prices=prices,
        source=opts.source,
        use_llm=opts.use_llm,
        edgar_user_agent=os.environ.get("ALPHALAB_EDGAR_USER_AGENT"),
        llm_probe=llm_probe or _default_llm_probe,
        require_market_open=opts.require_market_open,
    )
    if opts.as_json:
        print(json.dumps(preflight_json(report), indent=2, sort_keys=True), file=out)
    else:
        theme = theme_for(out, force_disable=opts.no_color)
        print(render_preflight(report, theme), file=out)
    return 0 if report.ready else 1


def run_autopilot_status(
    opts: AutopilotCycleOptions,
    *,
    out: TextIO,
    err: TextIO,
    broker_factory: BrokerFactory | None = None,
    prices_factory: PricesFactory | None = None,
    research_fn: ResearchFunction | None = None,
) -> int:
    """Run a read-only agentic assessment cycle and render status."""
    kill_switch = KillSwitch(halt_path=opts.halt_path)
    result = _execute_single_cycle(
        opts,
        submit=False,
        kill_switch=kill_switch,
        err=err,
        broker_factory=broker_factory or _default_broker,
        prices_factory=prices_factory,
        research_fn=research_fn,
    )
    if result is None:
        return 1
    if opts.as_json:
        payload = autopilot_cycle_json(result, kill_switch.state())
        print(json.dumps(payload, indent=2, sort_keys=True), file=out)
        return 0
    theme = theme_for(out, force_disable=opts.no_color)
    print(render_autopilot_status(result, kill_switch.state(), theme), file=out)
    return 0


def run_autopilot_run(
    opts: AutopilotCycleOptions,
    *,
    out: TextIO,
    err: TextIO,
    broker_factory: BrokerFactory | None = None,
    prices_factory: PricesFactory | None = None,
    research_fn: ResearchFunction | None = None,
) -> int:
    """Run one autonomous agentic cycle (dry-run unless ``opts.submit`` is True)."""
    kill_switch = KillSwitch(halt_path=opts.halt_path)
    result = _execute_single_cycle(
        opts,
        submit=opts.submit,
        kill_switch=kill_switch,
        err=err,
        broker_factory=broker_factory or _default_broker,
        prices_factory=prices_factory,
        research_fn=research_fn,
    )
    if result is None:
        return 1
    if opts.as_json:
        print(json.dumps(autopilot_cycle_json(result), indent=2, sort_keys=True), file=out)
    else:
        theme = theme_for(out, force_disable=opts.no_color)
        print(render_autopilot_run(result, theme), file=out)
    if result.cycle.halted_reason is not None:
        return 1
    return 0


def run_autopilot_loop(
    opts: AutopilotCycleOptions,
    *,
    out: TextIO,
    err: TextIO,
    broker_factory: BrokerFactory | None = None,
    prices_factory: PricesFactory | None = None,
    research_fn: ResearchFunction | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> int:
    """Run continuous autonomous cycles until stopped, halted, or ``max_cycles``."""
    kill_switch = KillSwitch(halt_path=opts.halt_path)
    try:
        broker = (broker_factory or _default_broker)(kill_switch)
        prices = (
            prices_factory(opts.prices_dir)
            if prices_factory is not None
            else _default_prices(
                opts.prices_dir,
                tickers=opts.tickers,
                held=tuple(broker.positions()),
            )
        )
        panel = prices.recent_bars()
    except Exception as exc:
        print(f"error: {exc}", file=err)
        return 1

    price_source = PanelPriceSeriesSource(
        panel,
        fallback=CsvPriceSeriesSource(opts.prices_dir),
        current_as_of=opts.as_of,
    )
    cached_prices = _CachedPriceProvider(
        panel,
        delegate=prices,
        on_refresh=price_source.update_panel,
    )
    theme = theme_for(out, force_disable=opts.no_color)

    def _on_cycle(res: AutopilotCycleResult) -> None:
        if opts.as_json:
            print(json.dumps(autopilot_cycle_json(res), sort_keys=True), file=out, flush=True)
        else:
            print(render_autopilot_run(res, theme), file=out, flush=True)

    def _default_loop_sleep(seconds: float) -> None:
        import time

        target = time.time() + max(0.0, seconds)
        if not opts.as_json:
            wake_local = datetime.fromtimestamp(target).strftime("%H:%M:%S")
            print(
                f"  [loop] sleeping {int(seconds)}s — next cycle at {wake_local}...",
                file=err,
                flush=True,
            )
        while True:
            remaining = target - time.time()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))
        if not opts.as_json:
            now_local = datetime.now().strftime("%H:%M:%S")
            print(f"  [loop] starting new cycle at {now_local}...", file=err, flush=True)

    opts.execution_db.parent.mkdir(parents=True, exist_ok=True)
    opts.journal_db.parent.mkdir(parents=True, exist_ok=True)
    opts.abstentions_db.parent.mkdir(parents=True, exist_ok=True)
    exec_log = SqliteExecutionLog(opts.execution_db)
    journal = SqliteJournal(opts.journal_db)
    abstentions = SqliteAbstentionStore(opts.abstentions_db)
    try:
        runner = _build_runner(
            opts,
            broker=broker,
            prices=cached_prices,
            exec_log=exec_log,
            journal=journal,
            abstentions=abstentions,
            kill_switch=kill_switch,
            price_source=price_source,
            research_fn=research_fn,
            err=err,
        )
        try:
            kwargs: dict[str, object] = {
                "as_of": opts.as_of,
                "submit": opts.submit,
                "interval_seconds": opts.interval_seconds,
                "max_cycles": opts.max_cycles,
                "on_cycle": _on_cycle,
                "sleep_fn": sleep_fn if sleep_fn is not None else _default_loop_sleep,
            }
            results = runner.run_loop(**kwargs)  # type: ignore[arg-type]
        except KeyboardInterrupt:
            return 0
    finally:
        exec_log.close()
        journal.close()
        abstentions.close()

    if any(r.cycle.halted_reason is not None for r in results):
        return 1
    return 0


def _execute_single_cycle(
    opts: AutopilotCycleOptions,
    *,
    submit: bool,
    kill_switch: KillSwitch,
    err: TextIO,
    broker_factory: BrokerFactory,
    prices_factory: PricesFactory | None,
    research_fn: ResearchFunction | None,
) -> AutopilotCycleResult | None:
    try:
        broker = broker_factory(kill_switch)
        prices = (
            prices_factory(opts.prices_dir)
            if prices_factory is not None
            else _default_prices(
                opts.prices_dir,
                tickers=opts.tickers,
                held=tuple(broker.positions()),
            )
        )
        panel = prices.recent_bars()
    except Exception as exc:
        print(f"error: {exc}", file=err)
        return None

    price_source = PanelPriceSeriesSource(
        panel,
        fallback=CsvPriceSeriesSource(opts.prices_dir),
        current_as_of=opts.as_of,
    )
    cached_prices = _CachedPriceProvider(
        panel,
        delegate=prices,
        on_refresh=price_source.update_panel,
    )
    opts.execution_db.parent.mkdir(parents=True, exist_ok=True)
    opts.journal_db.parent.mkdir(parents=True, exist_ok=True)
    opts.abstentions_db.parent.mkdir(parents=True, exist_ok=True)

    exec_log = SqliteExecutionLog(opts.execution_db)
    journal = SqliteJournal(opts.journal_db)
    abstentions = SqliteAbstentionStore(opts.abstentions_db)
    try:
        runner = _build_runner(
            opts,
            broker=broker,
            prices=cached_prices,
            exec_log=exec_log,
            journal=journal,
            abstentions=abstentions,
            kill_switch=kill_switch,
            price_source=price_source,
            research_fn=research_fn,
            err=err,
        )
        try:
            return runner.run_cycle(as_of=opts.as_of, submit=submit)
        except Exception as exc:
            print(f"error: autopilot cycle failed: {type(exc).__name__}: {exc}", file=err)
            return None
    finally:
        exec_log.close()
        journal.close()
        abstentions.close()


def _build_runner(
    opts: AutopilotCycleOptions,
    *,
    broker: AlpacaBroker,
    prices: PriceProvider,
    exec_log: SqliteExecutionLog,
    journal: SqliteJournal,
    abstentions: SqliteAbstentionStore,
    kill_switch: KillSwitch,
    price_source: PriceSeriesSource,
    research_fn: ResearchFunction | None,
    err: TextIO | None = None,
) -> AutopilotRunner:
    guard_cfg = AutopilotGuardConfig(
        stop_loss_pct=opts.stop_loss_pct,
        drawdown_arm_gain_pct=opts.drawdown_trigger_pct,
        drawdown_giveback_pct=opts.drawdown_giveback_pct,
        reentry_cooldown_days=opts.cooldown_days,
        sell_confirmations_required=opts.sell_confirmations_required,
    )
    if opts.tickers:
        selector: CandidateSelector | None = _explicit_selector(
            opts.tickers,
            cap=opts.candidate_cap,
            use_llm=opts.use_llm,
            source=opts.source,
            as_of=opts.as_of,
            price_source=price_source,
            exec_log=exec_log,
            cooldown_days=opts.cooldown_days,
        )
    else:

        def _default_selector(
            eval_date: date,
            bars: Mapping[Ticker, Sequence[PriceBar]],
            held_tickers: Sequence[Ticker],
        ) -> Sequence[Ticker]:
            if isinstance(price_source, PanelPriceSeriesSource):
                price_source.current_as_of = opts.as_of or eval_date
                price_source.update_panel(bars)
            return select_liquid_candidates(
                eval_date, bars, held_tickers, cap=opts.candidate_cap
            )

        selector = _default_selector
    research = research_fn or _make_research_fn(opts, price_source=price_source, err=err)
    recorder = _make_forecast_recorder(opts, journal=journal, abstentions=abstentions)
    settler = _make_settlement_runner(journal=journal, price_source=price_source)
    intraday_highs = _default_intraday_highs() if opts.source == "live" else None
    return AutopilotRunner(
        broker=broker,
        prices=prices,
        execution_log=exec_log,
        research=research,
        kill_switch=kill_switch,
        candidate_selector=selector,
        forecast_recorder=recorder,
        settlement_runner=settler,
        intraday_highs=intraday_highs,
        guard_config=guard_cfg,
        max_positions=opts.max_positions,
        candidate_cap=opts.candidate_cap,
        require_market_open=opts.require_market_open,
        strategy_name="nofx_llm_agent" if opts.use_llm else "nofx_agentic",
    )


def _default_intraday_highs() -> Callable[[Sequence[Ticker]], Mapping[Ticker, float]] | None:
    with contextlib.suppress(Exception):
        client = AlpacaBarsClient(AlpacaSettings(), transport=HttpxWriteTransport())  # type: ignore[call-arg]
        return lambda syms: client.intraday_highs(list(syms))
    return None


def _explicit_selector(
    tickers: tuple[str, ...],
    *,
    cap: int = 20,
    use_llm: bool = False,
    source: str = "fixture",
    as_of: date | None = None,
    price_source: PriceSeriesSource | None = None,
    exec_log: SqliteExecutionLog | None = None,
    cooldown_days: int = 0,
) -> CandidateSelector:
    explicit = [Ticker(t) for t in expand_ticker_aliases(tickers)]
    seen_by_date: dict[date, set[Ticker]] = {}

    def _select(
        eval_date: date,
        bars: Mapping[Ticker, Sequence[PriceBar]],
        held_tickers: Sequence[Ticker],
    ) -> Sequence[Ticker]:
        effective_date = as_of or eval_date
        if isinstance(price_source, PanelPriceSeriesSource):
            price_source.current_as_of = effective_date
            price_source.update_panel(bars)
        combined = dict.fromkeys([*explicit, *held_tickers])
        combined.pop(Ticker("SPY"), None)
        names = list(combined)
        if len(names) > 1:
            import statistics

            def _med_dv(t: Ticker) -> float:
                hist = bars.get(t, ())[-60:]
                dvols = [b.close * (b.volume or 0.0) for b in hist if b.volume is not None]
                return float(statistics.median(dvols)) if dvols else 0.0

            names.sort(key=_med_dv, reverse=True)
        if not use_llm or len(names) <= cap:
            return tuple(names)
        cooldown_blocked: set[Ticker] = set()
        if exec_log is not None and cooldown_days > 0:
            cooldown_blocked = {
                t
                for t, sell_d in _recent_sell_dates(exec_log).items()
                if 0 <= (effective_date - sell_d).days < cooldown_days
            }
        seen_today = seen_by_date.setdefault(effective_date, set())
        held_set = set(held_tickers)
        deprioritized = seen_today - held_set
        chosen = _prescreen_for_llm(
            names,
            held_tickers=held_tickers,
            cap=cap,
            source=source,
            as_of=effective_date,
            price_source=price_source,
            excluded_tickers=cooldown_blocked,
            deprioritized_tickers=deprioritized,
        )
        new_candidates = [t for t in chosen if t not in held_set]
        if any(t in deprioritized for t in new_candidates):
            # Wrapped around the rotation pool: reset seen_today to current batch
            seen_today.clear()
        seen_today.update(new_candidates)
        return chosen

    return _select


def _prescreen_for_llm(
    names: Sequence[Ticker],
    *,
    held_tickers: Sequence[Ticker],
    cap: int,
    source: str,
    as_of: date,
    price_source: PriceSeriesSource | None,
    sentiment_fn: Callable[[Ticker], float | None] | None = None,
    excluded_tickers: Set[Ticker] = frozenset(),
    deprioritized_tickers: Set[Ticker] = frozenset(),
    rotation_pool_size: int = 30,
) -> tuple[Ticker, ...]:
    """Two-stage S&P 500 funnel: fast PIT + ContractGate screen before live LLM.

    Existing held positions are always evaluated first and do *not* consume
    ``cap`` (which bounds new candidate discovery).
    """
    held_set = {t for t in held_tickers if t != Ticker("SPY")}
    selected: list[Ticker] = list(dict.fromkeys(t for t in held_tickers if t in held_set))
    live_builder = (
        load_live_builder(price_source=price_source, include_news=False)
        if source == "live" and price_source is not None
        else None
    )
    if live_builder is None:
        new_added = 0
        for t in names:
            if t in selected or t in excluded_tickers:
                continue
            if new_added >= cap:
                break
            selected.append(t)
            new_added += 1
        return tuple(selected)

    scored_buys: list[tuple[float, int, Ticker]] = []
    for idx, t in enumerate(names):
        if t in selected or t in excluded_tickers:
            continue
        try:
            req = live_builder(str(t), knowledge_date=as_of)
            art = _run_pipeline(req, use_llm=False, debate=False)
        except Exception:
            continue
        if art.verdict is Verdict.BUY:
            score = _dispersion_adjusted_mos(art)
            if score > 0.0:
                scored_buys.append((score, -idx, t))
    scored_buys.sort(reverse=True)
    pool_limit = max(rotation_pool_size, 3 * max(1, cap))
    if sentiment_fn is None:
        news_fn = None
        with contextlib.suppress(Exception):
            news_fn = load_news_sentiment_fn()
        if news_fn is not None:
            sentiment_fn = lambda t: news_fn(t, as_of)  # noqa: E731
    if sentiment_fn is not None:
        scored_buys = _apply_sentiment_penalty(scored_buys, sentiment_fn, limit=pool_limit)
    pool = [t for _score, _neg_idx, t in scored_buys[:pool_limit]]
    fresh = [t for t in pool if t not in deprioritized_tickers]
    target_cap = max(0, cap)
    batch = fresh[:target_cap]
    if len(batch) < target_cap:
        rank_out_cap = max(target_cap, min(len(pool), 2 * max(1, target_cap)))
        for rank_idx, t in enumerate(pool):
            if len(batch) >= target_cap:
                break
            if t not in batch and rank_idx < rank_out_cap:
                batch.append(t)
    selected.extend(batch)
    return tuple(selected)


def _apply_sentiment_penalty(
    scored: list[tuple[float, int, Ticker]],
    sentiment_fn: Callable[[Ticker], float | None],
    *,
    limit: int,
) -> list[tuple[float, int, Ticker]]:
    """Down-weight top candidates with very negative news (knife-catching guard)."""
    out: list[tuple[float, int, Ticker]] = []
    for pos, (score, neg_idx, t) in enumerate(scored):
        if pos < limit:
            net: float | None = None
            with contextlib.suppress(Exception):
                net = sentiment_fn(t)
            if net is not None and net <= -0.50:
                score = score * 0.25 if score > 0 else score
        out.append((score, neg_idx, t))
    out.sort(reverse=True)
    return out


def _dispersion_adjusted_mos(art: EquityResearchArtifact) -> float:
    """Compute Empirical-Bayes dispersion-adjusted Margin of Safety.

    Applies Smith & Winkler (2006) Empirical Bayes shrinkage toward the cross-
    sectional prior before dividing by dispersion ``max(span, floor)``. Supports
    both exact point targets (``not art.valuation_withheld``) and unanimous
    multi-method C1b ranges via ``conservative_c1b_upside(art)``.
    """
    if art.current_price <= 0.0:
        return -1e9
    if not art.valuation_withheld and art.price_target is not None:
        upside = art.price_target.value / art.current_price - 1.0
    else:
        c1b_upside = conservative_c1b_upside(art)
        if c1b_upside is None or c1b_upside < 0.10 or c1b_upside > 2.0:
            return -1e9
        upside = c1b_upside
    if upside <= 0.0:
        return -1e9
    raw_span = art.consensus_relative_span if art.consensus_relative_span is not None else 0.15
    span_floor = 0.15 if len(art.methods_used) >= 2 else 1.0
    if upside > 1.0:
        upside = 1.0 + max(
            0.0,
            shrink_upside_empirical_bayes(
                upside - 1.0,
                relative_span=raw_span,
                methods_count=len(art.methods_used),
            ),
        )
    return upside / max(raw_span, span_floor)


def _make_research_fn(
    opts: AutopilotCycleOptions,
    *,
    price_source: PriceSeriesSource,
    err: TextIO | None = None,
) -> ResearchFunction:
    live_builder = load_live_builder(price_source=price_source) if opts.source == "live" else None
    verdict_cache: LlmVerdictCache | None = None
    llm_version = ""
    if opts.use_llm and opts.source == "live":
        from alphalab.adapters.llm_config import LLMSettings

        settings = LLMSettings()
        llm_version = (
            f"{settings.quick_version}+{settings.deep_version}"
            if opts.debate
            else settings.deep_version
        )
        cache_path = Path(opts.execution_db).parent / "llm_verdict_cache.json"
        verdict_cache = LlmVerdictCache(cache_path)

    def _research(ticker: Ticker) -> EquityResearchArtifact | None:
        effective_as_of = (
            opts.as_of
            or getattr(price_source, "current_as_of", None)
            or datetime.now(UTC).date()
        )
        if opts.use_llm and not opts.as_json and err is not None:
            print(f"  [llm] evaluating {ticker}...", end="", file=err, flush=True)
        try:
            if live_builder is not None:
                request = live_builder(str(ticker), knowledge_date=effective_as_of)
            else:
                request = build_fixture_request(
                    str(ticker), knowledge_date=effective_as_of
                ).request
            fp: str | None = None
            cached: EquityResearchArtifact | None = None
            if verdict_cache is not None:
                fp = compute_request_fingerprint(
                    request, llm_version=llm_version, debate=opts.debate
                )
                candidate_cached = verdict_cache.get(fp)
                if candidate_cached is not None and candidate_cached.verdict is not Verdict.SELL:
                    cached = candidate_cached.model_copy(
                        update={"current_price": request.current_price}
                    )
            if cached is not None:
                artifact = cached
            else:
                artifact = _run_pipeline(request, use_llm=opts.use_llm, debate=opts.debate)
                if (
                    verdict_cache is not None
                    and fp is not None
                    and artifact.verdict is not Verdict.SELL
                ):
                    verdict_cache.put(fp, artifact)
            if opts.assume_verdict is not None and not opts.use_llm:
                artifact = artifact.model_copy(update={"verdict": opts.assume_verdict})
        except Exception:
            if opts.use_llm and not opts.as_json and err is not None:
                print(" FAILED", file=err, flush=True)
            raise
        if opts.use_llm and not opts.as_json and err is not None:
            suffix = " (cached)" if cached is not None else ""
            print(f" -> {artifact.verdict.value}{suffix}", file=err, flush=True)
        return artifact

    return _research


def _make_forecast_recorder(
    opts: AutopilotCycleOptions,
    *,
    journal: SqliteJournal,
    abstentions: SqliteAbstentionStore,
) -> ForecastRecorder:
    llm_version: str | None = "deterministic-template"
    llm_cutoff: date | None = None
    if opts.use_llm:
        from alphalab.adapters.llm_config import LLMSettings

        settings = LLMSettings()
        llm_version = (
            f"{settings.quick_version}+{settings.deep_version}"
            if opts.debate
            else settings.deep_version
        )
        llm_cutoff = (
            max(settings.quick_cutoff, settings.deep_cutoff)
            if opts.debate
            else settings.deep_cutoff
        )

    def _record(artifact: EquityResearchArtifact, as_of: date) -> Prediction | Abstention | None:
        dated = (
            artifact
            if artifact.knowledge_date is not None
            else artifact.model_copy(update={"knowledge_date": as_of})
        )
        assert dated.knowledge_date is not None
        if dated.verdict in (Verdict.BUY, Verdict.SELL):
            existing = journal.find_prediction_for_date(dated.ticker, dated.knowledge_date)
            if existing is not None:
                return None
        horizon = dated.knowledge_date + timedelta(days=DEFAULT_HORIZON_DAYS)
        try:
            return register_forecast(
                dated,
                horizon=horizon,
                recorded_at=datetime.now(UTC),
                journal=journal,
                abstentions=abstentions,
                llm_version=llm_version,
                llm_cutoff=llm_cutoff,
            )
        except Exception:
            return None

    return _record


def _make_settlement_runner(
    *, journal: SqliteJournal, price_source: PriceSeriesSource
) -> SettlementRunner:
    def _settle(as_of: date) -> int:
        run = settle_due_predictions(
            journal=journal,
            prices=price_source,
            as_of=as_of,
            settled_at=datetime.now(UTC),
            cost_model=ProportionalCostModel(CostSpec(half_spread_bps=_DEFAULT_HALF_SPREAD_BPS)),
        )
        return len(run.settled)

    return _settle


def _default_llm_probe() -> tuple[bool, str]:
    check = _check_ollama()
    return (check.status is CheckStatus.OK, check.detail)


class _CachedPriceProvider:
    """Returns a pre-fetched bar panel on the first cycle and delegates thereafter."""

    def __init__(
        self,
        initial: Mapping[Ticker, Sequence[PriceBar]],
        *,
        delegate: PriceProvider,
        on_refresh: Callable[[Mapping[Ticker, Sequence[PriceBar]]], None] | None = None,
    ) -> None:
        self._initial: Mapping[Ticker, Sequence[PriceBar]] | None = initial
        self._delegate = delegate
        self._on_refresh = on_refresh

    def recent_bars(self) -> Mapping[Ticker, Sequence[PriceBar]]:
        if self._initial is not None:
            panel = self._initial
            self._initial = None
            return panel
        panel = self._delegate.recent_bars()
        if self._on_refresh is not None:
            self._on_refresh(panel)
        return panel
