#!/usr/bin/env python3
"""Live Alpaca portfolio performance vs. ETFs and non-agentic baselines."""

from __future__ import annotations

import os
import site
import sys
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
site.addsitedir(str(_ROOT / ".venv/lib/python3.12/site-packages"))
sys.path.insert(0, str(_ROOT / "src"))

from alphalab.adapters.alpaca_bars import AlpacaBarsClient, DataFeed  # noqa: E402
from alphalab.adapters.alpaca_broker import AlpacaBroker  # noqa: E402
from alphalab.adapters.alpaca_config import AlpacaSettings  # noqa: E402
from alphalab.adapters.kill_switch import KillSwitch  # noqa: E402
from alphalab.cli.paper_transport import HttpxWriteTransport  # noqa: E402
from alphalab.domain.types import Ticker  # noqa: E402

AGENTIC_BASE_EQUITY = 98_515.96
NOMINAL_INITIAL_EQUITY = 100_000.00
INCEPTION_DATE = date(2026, 10, 15)

BENCHMARK_ETFS = (
    ("SPY", "S&P 500 (Cap-Weight)"),
    ("QQQ", "Nasdaq-100"),
    ("DIA", "Dow Jones (DJIA)"),
    ("IWM", "Russell 2000 (Small Cap)"),
    ("RSP", "S&P 500 Equal-Weight"),
    ("VTV", "Vanguard Value"),
    ("VUG", "Vanguard Growth"),
)


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    _load_dotenv()
    settings = AlpacaSettings()
    transport = HttpxWriteTransport()  # type: ignore[call-arg]
    broker = AlpacaBroker(
        settings,
        transport=transport,
        kill_switch=KillSwitch(),
    )
    bars_client = AlpacaBarsClient(
        settings,
        transport=transport,
        feed=DataFeed.SIP,
        with_raw_close=True,
    )

    acct = broker.account()
    raw_acct = transport.get(
        f"{settings.trading_base_url()}/v2/account",
        headers=settings.auth_headers(),
    ).json
    positions = broker.positions()
    equity = float(acct.equity)
    last_equity = float(
        raw_acct.get("last_equity", equity) if isinstance(raw_acct, dict) else equity
    )
    cash = float(raw_acct.get("cash", 0.0) if isinstance(raw_acct, dict) else 0.0)

    day_pnl = equity - last_equity
    day_pct = (equity / last_equity - 1.0) * 100.0 if last_equity > 0 else 0.0
    cum_pnl = equity - AGENTIC_BASE_EQUITY
    cum_pct = (equity / AGENTIC_BASE_EQUITY - 1.0) * 100.0

    print("=== ALPHALAB LIVE AGENTIC PORTFOLIO (nofx_llm_agent) ===")
    print(f"Agentic Inception Base (2026-10-15): ${AGENTIC_BASE_EQUITY:,.2f}")
    print(f"Previous Close Equity              : ${last_equity:,.2f}")
    print(f"Current Live Equity                : ${equity:,.2f} (Cash: ${cash:,.2f})")
    print(f"Today P&L                          : ${day_pnl:+,.2f} ({day_pct:+.2f}%)")
    print(f"Cumulative vs Agentic Base         : ${cum_pnl:+,.2f} ({cum_pct:+.2f}%)")
    print(f"Open Positions ({len(positions)})                :")
    for p in sorted(positions.values(), key=lambda x: str(x.ticker)):
        mv = p.shares * p.price
        pnl_pct = (
            (p.price / p.avg_entry_price - 1.0) * 100.0 if p.avg_entry_price else 0.0
        )
        print(
            f"  {p.ticker!s:<6} qty={p.shares:<9.4f} avg=${(p.avg_entry_price or 0.0):<8.2f} "
            f"now=${p.price:<8.2f} mv=${mv:<9.2f} pnl={pnl_pct:+.2f}%"
        )

    etf_tickers = [Ticker(sym) for sym, _ in BENCHMARK_ETFS]
    end_d = date.today() - timedelta(days=1)
    daily = bars_client.daily_bars(etf_tickers, start=INCEPTION_DATE - timedelta(days=5), end=end_d)
    snaps_url = (
        f"{settings.data_base_url()}/v2/stocks/snapshots"
        f"?symbols={','.join(str(t) for t in etf_tickers)}&feed=iex"
    )
    snap_resp = transport.get(snaps_url, headers=settings.auth_headers()).json

    print("\n=== MULTI-BENCHMARK COMPARISON (Base $98,515.96 from 2026-10-15) ===")
    header = (
        f"{'Benchmark':<28} | {'Today %':>8} | {'Cumul %':>8} | "
        f"{'Cumul $ PnL':>12} | {'Sim Equity':>12} | {'Agentic Alpha':>13}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'nofx_llm_agent (Live AI)':<28} | {day_pct:>+7.2f}% | {cum_pct:>+7.2f}% | "
        f"${cum_pnl:>+10,.2f} | ${equity:>10,.2f} | {'—':>13}"
    )
    for sym, label in BENCHMARK_ETFS:
        series = daily.get(Ticker(sym), ())
        inc_bars = [b for b in series if b.on <= INCEPTION_DATE]
        p0 = (
            (inc_bars[-1].raw_close or inc_bars[-1].close)
            if inc_bars
            else (series[0].raw_close or series[0].close)
        )
        prev_p = series[-1].raw_close or series[-1].close
        snap = snap_resp.get(sym, {}) if isinstance(snap_resp, dict) else {}
        live_p = (
            (snap.get("latestTrade") or {}).get("p")
            or (snap.get("dailyBar") or {}).get("c")
            or prev_p
        )
        live_p = float(live_p)
        b_day = (live_p / prev_p - 1.0) * 100.0
        b_cum = (live_p / p0 - 1.0) * 100.0
        sim_eq = AGENTIC_BASE_EQUITY * (1.0 + b_cum / 100.0)
        sim_pnl = sim_eq - AGENTIC_BASE_EQUITY
        alpha_bps = (cum_pct - b_cum) * 100.0
        print(
            f"{f'{sym} ({label[:18]})':<28} | {b_day:>+7.2f}% | {b_cum:>+7.2f}% | "
            f"${sim_pnl:>+10,.2f} | ${sim_eq:>10,.2f} | {alpha_bps:>+9.0f} bps"
        )


if __name__ == "__main__":
    main()
