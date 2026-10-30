---
name: live-performance-benchmark
description: Audits and compares Alphalab's live Alpaca Paper/Live portfolio performance from the true agentic starting capital ($98,515.96 on Oct 15) against major ETFs/funds (SPY, QQQ, DIA, IWM, RSP, VTV, VUG) and internal non-agentic baselines (deterministic nofx_agentic Top-20 valuation basket and rank_ensemble momentum basket), plus ExposureReport and wealth e-process diagnostics. Trigger whenever the user asks "como nos fue hoy?", "como vamos?", asks for a portfolio summary, or requests a comparison against SPY, Nasdaq, funds, or non-agentic algorithms.
---

# Live Performance & Multi-Benchmark Comparison Skill

Use this skill whenever the user asks how the portfolio performed today or cumulatively (`"como nos fue hoy"`, `"como vamos"`, `"comparalo con los fondos, spy, nasdaq y nuestro algoritmo sin agentes"`).

---

## Critical Accounting Baseline Rules

1. **True Agentic Starting Capital (`$98,515.96`, NOT `$100,000.00`):**
   - The Alpaca Paper account started at `$100,000.00` on Oct 14, 2026, when the **non-agentic momentum algorithm (`rank_ensemble`)** ran first and lost **`-$1,484.04` (`-1.48%`)**, leaving **`$98,515.96`** upon liquidation on Oct 15.
   - The **autonomous AI agent (`nofx_llm_agent`)** started operating on **Oct 15, 2026** with a true starting capital of **`$98,515.96`**.
   - Always compute cumulative agentic return and dollar PnL relative to **`$98,515.96`** (while noting the `$100,000.00` account nominal base for transparency).

2. **Required Comparison Benchmarks (All Scaled to the Same `$98,515.96` Oct 15 Base):**
   - **Market & Style ETFs:**
     - `SPY` (S&P 500 Cap-Weight)
     - `QQQ` (Nasdaq-100)
     - `DIA` (Dow Jones Industrial Average)
     - `IWM` (Russell 2000 Small Cap)
     - `RSP` (S&P 500 Equal-Weight)
     - `VTV` (Vanguard Value)
     - `VUG` (Vanguard Growth)
   - **Internal Non-Agentic Algorithms (Without LLM Agents):**
     1. **`nofx_agentic` (`use_llm=False`):** Our deterministic Top-20 fundamental valuation basket (same EDGAR DCF/DDM/RI/Graham models, equal-weight, without LLM qualitative filtering or runtime trailing stops).
     2. **`rank_ensemble` (Pure Price Momentum 12-1/6-1):** Report both its **actual realized live loss (`-$1,484.04` / `-1.48%` on Oct 14)** and its hypothetical Buy & Hold trajectory since Oct 15.

---

## Automated Benchmark Script

Run the bundled script to fetch live Alpaca account equity, open positions, ETF returns (`SPY`, `QQQ`, `DIA`, `IWM`, `RSP`, `VTV`, `VUG`), and non-agentic baseline returns in one command:

```bash
.venv/bin/python scripts/benchmark_live_portfolio.py
```
