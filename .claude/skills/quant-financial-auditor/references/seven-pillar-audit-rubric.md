# Seven-Pillar Quantitative & Agentic Audit Rubric

Use this rubric to systematically inspect every layer of a quantitative and agentic trading repository. For each pillar, verify the code against the checklist, diagnose concrete failure modes, and formulate mathematically rigorous upgrades.

## Table of Contents
1. [Pillar 1: Point-in-Time (PIT) Data Integrity & Anti-Leakage](#pillar-1-point-in-time-pit-data-integrity--anti-leakage)
2. [Pillar 2: Valuation Engine & Financial Statement Mathematics](#pillar-2-valuation-engine--financial-statement-mathematics)
3. [Pillar 3: Agentic Topology, Adversarial Debate & Contract Gate](#pillar-3-agentic-topology-adversarial-debate--contract-gate)
4. [Pillar 4: Epistemic Calibration, Forecasting Journal & Reflexion](#pillar-4-epistemic-calibration-forecasting-journal--reflexion)
5. [Pillar 5: Statistical Certification & Regime-Aware Backtesting](#pillar-5-statistical-certification--regime-aware-backtesting)
6. [Pillar 6: Portfolio Construction, Sizing & Cost/Capacity Modeling](#pillar-6-portfolio-construction-sizing--costcapacity-modeling)
7. [Pillar 7: Autonomous Broker Execution & NOFX Runtime Guards](#pillar-7-autonomous-broker-execution--nofx-runtime-guards)

---

## Pillar 1: Point-in-Time (PIT) Data Integrity & Anti-Leakage

### What to Inspect
- SEC EDGAR ingestion (`adapters/edgar.py`, `adapters/edgar_live.py`, `cli/ingest.py`)
- PIT storage and temporal filtering (`domain/pit/`, `adapters/pit_store.py`)
- Universe selection (`data/sp500_constituents.json`, `adapters/universe.py`)

### High-Value Audit Checks & Common Flaws
1. **Filing Date vs. Period End Date:** Confirm that every fundamental fact filters on `filed <= knowledge_date` (or SEC `acceptance_datetime`), never `period_end <= knowledge_date` (since 10-K/10-Q filings arrive 30–60 days after quarter end).
2. **Historical S&P 500 Survivorship Bias:** Using *today's* `sp500_constituents.json` for historical backtests introduces survivorship bias (including winners added later like `UBER`, `CVNA`, `PLTR` and excluding delisted/bankrupt constituents like `SIVB`, `FRC`). Flag if historical backtests use a static constituent list without effective-date intervals.
3. **Restatement Leakage:** Verify that amended filings (`10-K/A`, `10-Q/A`) filed at $t_2 > \text{knowledge\_date}$ do not overwrite the original value known at $t_1 \le \text{knowledge\_date}$.

---

## Pillar 2: Valuation Engine & Financial Statement Mathematics

### What to Inspect
- `domain/compute/operators/dcf.py`, `ddm.py`, `multiples.py`, `consensus.py`
- `app/ingest.py` and `cli/ingest.py` (how XBRL facts map into `DCF`, `DDM`, `P/E` inputs)

### High-Value Audit Checks & Common Flaws
1. **The Non-Dividend / Low-Payout DDM Distortion (Critical):**
   - *Symptom:* Tech/growth compounders (`CRM`, `UBER`, `ADBE`, `NOW`) pay tiny or zero dividends, or return capital via buybacks. If `ddm.py` produces a near-zero implied price (e.g., `$15` for a `$240` stock) while `DCF` is `$180` and `P/E` is `$310`, then `consensus.relative_span = (high - low) / mid` explodes above `1.0`.
   - *Consequence:* `ContractGate` (`consensus_relative_span > 1.0`) withholds the valuation on high-quality compounders solely because `DDM` is an inappropriate model for low-payout firms!
   - *Upgrade:*
     - Replace pure cash dividends in `DDM` with **Total Shareholder Yield** ($\text{Dividends} + \text{Net Share Repurchases}$, Boudoukh et al., 2007).
     - Exclude `DDM` from the valuation bundle when dividend payout ratio $< 15\%$ and replace it with **EV/EBITDA** or **EV/FCF** (`multiples.py`).
2. **Financial Sector & REIT Mis-Specification:**
   - Standard `DCF` (`OCF - CapEx`) fails for banks/insurers (`JPM`, `BAC`, `PGR`, `AON`, `BRK.B`) because working capital and debt are raw operating inputs.
   - *Upgrade:* Add a **Residual Income / Excess Return Model** ($V_0 = B_0 + \sum \frac{(\text{ROE}_t - r_e) B_{t-1}}{(1+r_e)^t}$) or **P/B vs. ROE** operator for GICS Financials.
3. **Static WACC & Terminal Growth Assumptions:**
   - Check whether discount rates (`discount_rate`) and terminal growth (`terminal_growth`) are hardcoded across all tickers instead of adapting to company leverage ($\text{Net Debt} / \text{Enterprise Value}$), beta, and the risk-free rate.
4. **Structured Accounting Signals (Kim, Muhn, Nikolaev 2024, `arXiv:2407.17866`):**
   - Enrich the `documents` / quantitative payload passed to the LLM with **Piotroski F-Score (9 binary accounting health signals)**, **Sloan Accrual Ratio** ($\frac{\text{NI} - \text{OCF}}{\text{Assets}}$), and **Novy-Marx Gross Profitability** ($\frac{\text{GP}}{\text{Assets}}$).

---

## Pillar 3: Agentic Topology, Adversarial Debate & Contract Gate

### What to Inspect
- `app/pipeline.py`, `app/debate_pipeline.py`, `domain/debate/conductor.py`, `domain/debate/router.py`
- `domain/contract/gate.py`, `domain/contract/clauses.py`
- `adapters/llm.py`, `adapters/debate_llm.py`

### High-Value Audit Checks & Common Flaws
1. **Single-Analyst Default vs. Selective Debate Routing:**
   - Running 3 LLM calls (`Bull` + `Bear` + `Judge`) on every candidate triples latency, while running single-analyst (`--no-debate`) on borderline/high-dispersion names risks overconfident false positives.
   - *Upgrade:* Activate **Selective Debate Routing** (`domain/debate/router.py`): run the fast single Analyst first, and automatically escalate to the full `Bull/Bear/Judge` Debate when `0.45 <=consensus_relative_span <= 1.0` or when the single analyst recommends `BUY` with high model disagreement.
2. **Adversarial Reverse-DCF Prompting for the Bear Agent (`arXiv:2405.16334`):**
   - Generic Bear prompts produce generic macro boilerplate ("macroeconomic headwinds, competition").
   - *Upgrade:* Supply the Bear agent with the deterministic **Implied Reverse-DCF Growth Rate** ($g_{\text{implied}}$ that equates `DCF` to `current_price`) and require the Bear to argue why historical FCF growth cannot sustain $g_{\text{implied}}$.
3. **LLM XML / Tool-Call Parsing Resilience:**
   - Local quantized/MLX models (e.g., `qwen3.8:27b-mlx`) occasionally emit malformed XML tool calls (`status_code: 500, XML syntax error: element <parameter> closed by </function>`).
   - *Upgrade:* Add automatic retry with JSON-schema mode (`format="json"`) fallback in `OllamaLLMAdapter` so a transient tool-call formatting hiccup does not skip a valid S&P 500 candidate (as happened with `UAL` on Cycle #1).

---

## Pillar 4: Epistemic Calibration, Forecasting Journal & Reflexion

### What to Inspect
- `domain/forecast/probability.py`, `domain/forecast/builder.py`
- `domain/journal/calibration.py`, `domain/journal/reflect.py`, `domain/journal/settle.py`

### High-Value Audit Checks & Common Flaws
1. **Open-Loop vs. Closed-Loop Learning:**
   - Check if `domain/journal/reflect.py` and `domain/journal/calibration.py` only *write* lessons and Brier scores to `journal.db`, or if those settled lessons and empirical base rates are **fed back into `_make_research_fn`**!
   - *Upgrade (Closed-Loop PIT Reflexion):* Query `journal.db` for settled predictions with `settled_at <= as_of` and inject:
     1. The model's historical empirical hit rate (`base_rate`) into `probability.py` (replacing a hardcoded `0.50` prior once $N \ge 30$).
     2. Past post-mortem `lessons` for that ticker/sector into the Analyst/Judge context.

---

## Pillar 5: Statistical Certification & Regime-Aware Backtesting

### What to Inspect
- `domain/stats/deflated_sharpe.py`, `pbo.py`, `pbo_gate.py`, `certification.py`, `bootstrap.py`
- `domain/backtest/engine.py`, `turbulence.py`, `domain/skeptic/`

### High-Value Audit Checks & Common Flaws
1. **Regime-Conditional Evaluation (*StockBench*, `arXiv:2510.02209`):**
   - Check if backtest/paper reports break down alpha across **Bull vs. Bear vs. High-Turbulence** sub-periods (`domain/backtest/turbulence.py`).
   - Connect the Mahalanobis `turbulence` detector (`domain/backtest/turbulence.py`) directly to `AutopilotRunner` so extreme market turbulence scales down gross exposure automatically.

---

## Pillar 6: Portfolio Construction, Sizing & Cost/Capacity Modeling

### What to Inspect
- `app/agentic_portfolio.py`, `domain/execution/planner.py`, `domain/costs/`

### High-Value Audit Checks & Common Flaws
1. **Structural Cash Drag When Approvals $<$ Max Slots:**
   - In `agentic_portfolio.py`, `weight = 1.0 / max_positions` means that if `max_positions = 20` (`5%` per slot) and `--candidate-cap 10` is used (yielding 6 `BUY`s), the portfolio invests only `30%` and leaves `70%` in idle cash (`0%` return).
   - *Upgrade:*
     - Dynamic equal-weight or **Inverse-Volatility / ATR Risk-Parity Sizing** (`Moreira & Muir 2017`) where each approved name receives $\min\left(w_{\max}, \frac{1}{\max(M, N_{\text{min\_diversification}})}\right)$ (e.g., with $w_{\max} = 10\%$ and $M = 6$, each gets $10\% \rightarrow 60\%$ invested, or optionally sweeping residual cash into `SPY` / short-term T-bill ETF `SGOV`/`BIL` when market regime is non-turbulent).
2. **Whole-Share Truncation Dropping High-Priced S&P 500 Stocks:**
   - In `domain/execution/planner.py`, `math.floor(notional / price)` turns a `$4,900` slot allocation on `NVR` (`~$7,500`/share) or `BKNG` (`~$5,000`/share) into `0` shares, silently discarding an LLM-approved `BUY`!
   - *Upgrade:* Support **fractional shares** (`round(notional / price, 4)`) in `plan_trades` when supported by Alpaca Paper/Live (`fractionable=True` in `alpaca_assets.py`).
3. **Pre-Screen Ordering Bias in Large Universes (`spy500`):**
   - When screening 503 S&P 500 symbols down to `--candidate-cap K` for the LLM:
     - Screening * purely by liquidity first* and stopping at the first $K$ undervalued names biases toward mega-cap tech/telecom and ignores high-upside mid-cap S&P 500 names (`rank 50–250` by liquidity).
     - *Upgrade:* Require a minimum liquidity floor (e.g., `median_dollar_volume >= $25M/day`, which all 503 S&P 500 stocks pass!), and then rank the `ContractGate`-passing S&P 500 candidates by **Risk-Adjusted Valuation Margin of Safety** ($\frac{\text{consensus\_mid} - \text{price}}{\text{price} \cdot (1 + \text{relative\_span})}$) so the LLM evaluates the **top $K$ most undervalued S&P 500 stocks**, not just the $K$ largest!

---

## Pillar 7: Autonomous Broker Execution & NOFX Runtime Guards

### What to Inspect
- `domain/execution/nofx_guards.py`, `app/autopilot_runner.py`, `adapters/alpaca_broker.py`

### High-Value Audit Checks & Common Flaws
1. **Pre-Open Pending Order Blindness Across Date Rollovers (Critical Operational Hazard):**
   - *Symptom:* If Cycle A runs at `22:00` on Day $T$ (market closed) and submits `DAY` orders (`nofx_llm_agent-T-UBER-buy`, `status=accepted`), and Cycle B runs at `08:30` on Day $T+1$ (before the `09:30 ET` opening bell), `broker.positions()` is still empty (`{}`) because the `accepted` orders haven't filled yet! Because the date changed to $T+1$, Cycle B generates new `client_order_id`s (`nofx_llm_agent-(T+1)-UBER-buy`) and submits **duplicate buy orders** that double the position at the `09:30 ET` opening bell!
   - *Upgrade:* Reconcile **open/accepted broker orders** (`GET /v2/orders?status=open`) into effective holdings (`holdings + pending_open_buys - pending_open_sells`) inside `AutopilotRunner` before calling `plan_trades`!
2. **Fixed `-8%` Stop-Loss vs. Volatility-Scaled (ATR) NOFX Guards:**
   - Upgrade `AutopilotGuardConfig` to support ATR-multiples (`3.0 * ATR_14`) alongside fixed percentages so high-beta and low-beta stocks receive volatility-consistent risk bounds.
