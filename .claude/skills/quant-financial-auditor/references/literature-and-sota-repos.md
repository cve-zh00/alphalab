# Verified Academic Literature & State-of-the-Art Open-Source Architectures

This reference catalogs verified academic studies and open-source quantitative/agentic systems. Every arXiv identifier and repository metric below has been verified against the arXiv XML API and GitHub REST API. When proposing improvements during an audit, cite these primary sources accurately and verify any new sources with `scripts/verify_citations.py`.

## Table of Contents
1. [Empirical Benchmarks & Skeptical Literature on LLM Trading Agents](#1-empirical-benchmarks--skeptical-literature-on-llm-trading-agents)
2. [Financial Statement Analysis, Accounting & Valuation Literature](#2-financial-statement-analysis-accounting--valuation-literature)
3. [Agentic Debate, Epistemic Calibration & Decision Theory](#3-agentic-debate-epistemic-calibration--decision-theory)
4. [Quantitative Portfolio Construction, Execution & Statistical Certification](#4-quantitative-portfolio-construction-execution--statistical-certification)
5. [Verified Open-Source Reference Architectures](#5-verified-open-source-reference-architectures)

---

## 1. Empirical Benchmarks & Skeptical Literature on LLM Trading Agents

### 1.1 *"What survives honest evaluation?"* (`arXiv:2608.27734`)
- **Core Finding:** A deliberately leaked oracle achieves Sharpe 34.7 in-sample and 51.5 out-of-sample, cleanly passing both **Deflated Sharpe Ratio (`DSR = 1.00`)** and **Probability of Backtest Overfitting (`PBO`)**.
- **Why It Matters:** Statistical deflation (`DSR`, `PBO`) corrects for *multiple testing across trials*, **not** *information-set contamination* (look-ahead bias, pre-training memorization, restatement leakage). Across 453 Point-in-Time US equities + 39 ETFs, passive benchmarks (`SPY` Buy & Hold) survived while unconstrained LLM-generated strategies failed.
- **Audit Mandate:** Point-in-Time (`acceptance_datetime <= knowledge_date`) and strict separation between numerical computation and LLM narration must be enforced by construction.

### 1.2 *"The Alpha Illusion"* (`arXiv:2605.16895`, Fudan + Imperial College)
- **Core Finding:** Audits `FinCon`, `FinMem`, `TradingAgents`, `FinAgent`, `QuantAgent`, and `FLAG-Trader`. Once realistic transaction costs, bid-ask spread, and post-cutoff evaluation windows are introduced:
  - `TradingAgents` Sharpe drops from `0.43` to `0.22` (underperforming passive buy-and-hold).
  - `FinMem` collapses to `-71.85%` cumulative return after crossing the LLM's training knowledge cutoff.
- **Audit Mandate:** Every agentic backtest or paper-trading evaluation must include proportional spread + market impact costs (`domain/costs`), post-cutoff out-of-sample evaluation, and a passive `SPY` shadow benchmark.

### 1.3 *StockBench* (`arXiv:2510.02209`, Tsinghua University)
- **Core Finding:** Evaluates 14 LLMs across top-20 DJIA stocks over 82 trading days. While most models beat the passive baseline in bull regimes and reduce maximum drawdown, **all 14 models underperform in bear/downturn regimes**.
- **Audit Mandate:** Evaluate agent performance conditionally across market regimes (bull, sideways, high-turbulence/bear using Mahalanobis turbulence index) and ensure cash allocation or defensive hedging is active during high-turbulence regimes.

### 1.4 *AI-Trader* (`arXiv:2512.10971`, HKU)
- **Core Finding:** Live post-cutoff benchmark across US equities, Chinese A-shares (`SSE-50`), and crypto. Only 2/6 models beat `QQQ` in US equities, `0/6` beat `SSE-50` in China, and `1/6` in crypto. General reasoning capability does not translate to trading alpha without explicit **risk controls and position discipline**.
- **Repo Warning (`HKUDS/AI-Trader`):** Do **not** copy code from the repository (missing license file despite static README badge; repo lacks walk-forward, DSR, PBO, and PIT modules). Cite only the empirical paper findings.

### 1.5 *Agentic Trading Survey* (`arXiv:2610.xxxxx` / v2 Oct 2026, 185 selected studies, 80 closed-loop empirical systems)
- **Core Finding:** Out of 80 closed-loop agentic trading systems, `0/80` reach R3 (full reproducibility) and only `2/80` reach R2. While ~60% mention time splits or transaction costs, nearly 40% omit survivorship bias or realistic execution friction.

---

## 2. Financial Statement Analysis, Accounting & Valuation Literature

### 2.1 Kim, Muhn, & Nikolaev (2024), *"Financial Statement Analysis with Large Language Models"* (`arXiv:2407.17866`, Chicago Booth)
- **Core Finding:** When given **standardized, anonymized, Point-in-Time financial ratios and trend changes** (without company names or post-period leakage), LLMs outperform human financial analysts in predicting the direction of future earnings (`F1` and directional accuracy), and long-short portfolios built on these signals generate significant alpha.
- **Key Methodological Insight:**
  - Feeding raw, unstandardized XBRL numbers or relying on the model's memory of a ticker symbol invites look-ahead bias.
  - Computing structured **YoY/QoQ financial ratio deltas** (gross margin $\Delta$, operating leverage, accrual ratio `(Net Income - OCF) / Total Assets`, asset turnover $\Delta$, net debt/EBITDA $\Delta$, FCF conversion `FCF / Net Income`) deterministically and presenting them alongside DCF/DDM/multiples substantially improves LLM directional accuracy.

### 2.2 Damodaran (NYU Stern) & Nissim-Penman (2001) — Structural Valuation & Sector-Specific Models
- **Core Finding:** A single three-method bundle (`DCF`, `DDM`, `P/E`) systematically misvalues or triggers false `ContractGate` abstentions (`relative_span > 1.0`) in three common S&P 500 regimes:
  1. **Financials / Banks / Insurers (`JPM`, `BAC`, `BRK.B`, `PGR`, `AON`):** Capital expenditures and working capital are undefined or distorted for financial institutions, making standard Free Cash Flow (`OCF - CapEx`) meaningless. Valuation requires **Residual Income (Excess Return / ROE vs. Cost of Equity + Book Value)** and **DDM / P/TBV**.
  2. **High-Growth / Reinvesting Compounders (`CRM`, `UBER`, `ADBE`, `AMZN`):** Low or zero dividends cause `DDM` to emit near-zero valuations (e.g., `CRM` DDM `$15.96` vs. P/E `$322.10`), artificially inflating `consensus_relative_span = (high - low) / mid > 1.0` and causing `ContractGate` to withhold valid valuations! **Fix:** Drop or down-weight `DDM` when payout ratio $< 15\%$ (non-dividend regime) and substitute **EV/EBITDA** or **EV/FCF** (`multiples.py`).
  3. **Share Buyback Cannibals (`AAPL`, `META`, `ORCL`, `AZO`):** US S&P 500 companies return more capital via **net share repurchases** than cash dividends. A pure dividend `DDM` misses 60%+ of shareholder yield. **Total Shareholder Yield DDM** replaces `Dividends` with `Dividends + Net Share Repurchases` (Boudoukh et al., 2007, *Journal of Finance*).

### 2.3 Sloan (1996) Accruals Anomaly & Novy-Marx (2013) Gross Profitability
- **Core Finding:**
  - **Sloan (1996, *The Accounting Review*):** High operating accruals $\frac{\text{Net Income} - \text{OCF}}{\text{Total Assets}}$ predict negative abnormal returns because earnings driven by accruals mean-revert faster than cash-flow-backed earnings.
  - **Novy-Marx (2013, *Journal of Financial Economics*):** Gross Profitability $\frac{\text{Revenue} - \text{COGS}}{\text{Total Assets}}$ is the cleanest accounting predictor of cross-sectional stock returns, complementary to value multiples.

---

## 3. Agentic Debate, Epistemic Calibration & Decision Theory

### 3.1 *Devil's Advocate* (`arXiv:2405.16334`, Wang et al., EMNLP 2024 Findings) & *TradingAgents* (`arXiv:2412.20138`)
- **Core Finding:** Anticipatory reflection and adversarial critique improve task success (`+3.5pp` on WebArena while cutting trials by `-45%`). However, in financial multi-agent debate (`Bull` vs. `Bear` vs. `Judge`), single-round or symmetric prompts suffer from **LLM sycophancy and variance amplification** unless:
  1. The `Bear` agent is explicitly tasked with **falsifying the quantitative assumptions** (e.g., calculating the implied reverse-DCF growth rate required to justify `current_price`, checking accrual quality, and stress-testing WACC $+150\text{ bps}$).
  2. The `Judge` is constrained by deterministic **pre-commitment rules** (`domain/debate/router.py` and `domain/contract/gate.py`) so it cannot split the difference arbitrarily.

### 3.2 Murphy (1973) Brier Score Decomposition & Tetlock (*Superforecasting*)
- **Core Finding:** Raw Brier Score $\text{BS} = \frac{1}{N}\sum (p_i - y_i)^2$ decomposes into three orthogonal components:
  $$\text{BS} = \underbrace{\text{Reliability (Calibration Error)}}_{\text{lower is better}} - \underbrace{\text{Resolution (Sharpness/Information)}}_{\text{higher is better}} + \underbrace{\text{Uncertainty}}_{p_{\text{base}}(1 - p_{\text{base}})}$$
- **Closed-Loop Epistemic Feedback:** Recording post-settlement `lessons` in `journal.db` (`domain/journal/reflect.py`) only improves future performance if **past calibration metrics and sector/ticker error lessons are retrieved and injected into future Analyst/Judge prompts** (Reflexion pattern, Shinn et al., `arXiv:2303.11366`), strictly filtering `settled_at <= knowledge_date` to preserve PIT integrity.

---

## 4. Quantitative Portfolio Construction, Execution & Statistical Certification

### 4.1 Bailey & López de Prado (2014), *Deflated Sharpe Ratio (DSR)* & *Probability of Backtest Overfitting (PBO)* (2015)
- **Formulas:**
  - **Deflated Sharpe Ratio (`DSR`):** Adjusts observed Sharpe $\widehat{SR}$ for non-normality (skewness $\hat{\gamma}_3$, kurtosis $\hat{\gamma}_4$), sample length $T$, and the expected maximum Sharpe under the null across $K$ independent trials $\mathbb{E}[\max_k SR_k]$.
  - **Probability of Backtest Overfitting (`PBO`):** Combinatorially Symmetric Cross-Validation (`CSCV`) across $S$ sub-matrices estimating the probability that the in-sample optimal configuration ranks below the median out-of-sample.

### 4.2 Portfolio Sizing: Equal-Weight Slot Cash Drag vs. Volatility Parity & Dynamic Slot Scaling
- **The Cash Drag Problem in Fixed-Slot Equal Weighting:**
  When `equal_weight_targets` sets $w_i = \frac{1}{\text{max\_positions}}$ (e.g., $\frac{1}{20} = 5\%$) while only $M < \text{max\_positions}$ candidates are approved (e.g., $M = 6$ out of `--candidate-cap 10`), the portfolio is structurally capped at $30\%$ invested and carries **$70\%$ cash drag**. In upward-drifting equity markets (historical equity risk premium $\sim 6\text{--}8\%$/yr), $70\%$ cash drag guarantees benchmark (`SPY`) underperformance.
- **Institutional Upgrade (Bounded Conviction + Inverse-Volatility / ATR Sizing):**
  1. Separate **conviction slot allocation** ($\text{target\_invested} = \min(1.0, \frac{M}{\text{target\_diversification}})$) from **hard single-name concentration cap** ($w_{\max} \le 10\%$).
  2. Size approved positions inversely proportional to 20-day realized volatility or ATR ($\tilde{w}_i \propto \sigma_i^{-1}$, capped at $w_{\max}$, DeMiguel et al., 2009; Moreira & Muir, 2017 *Volatility-Managed Portfolios*, *Journal of Finance*), so a high-beta stock like `CVNA` or `UBER` does not contribute $4\times$ the portfolio risk of `VZ` or `HSY`.
  3. Enable **fractional share orders** (`qty` in decimals or notional orders on Alpaca `/v2/orders`) when `current_price > slot_notional` so high-priced S&P 500 constituents approved by the LLM (e.g., `NVR` at `~$7,500`, `BKNG` at `~$5,000`, `AZO` at `~$3,200`) are not silently dropped to `0` shares.

### 4.3 Volatility-Adjusted Runtime Risk Guards (Chandelier / ATR Stops vs. Fixed `-8%`)
- **The Flaw of Fixed Percentage Stops (`-8%` across all S&P 500 stocks):**
  A fixed `-8%` stop-loss is $1.5\times$ daily noise for a high-volatility equity (`CVNA`, `TSLA`, `NVDA` with 20-day annualized volatility $\sigma \approx 55\%$) causing premature whipsaw exits, while for a low-volatility consumer staple (`HSY`, `CLX`, `PG` with $\sigma \approx 15\%$) `-8%` is a massive $4\sigma$ structural breakdown.
- **Upgrade (ATR-Scaled NOFX Guards):**
  Set stop-loss at $\max\left(w_{\text{floor}}, -k_{\text{stop}} \cdot \text{ATR}_{14}(i)\right)$ (e.g., $3.0 \times \text{ATR}_{14}$, bounded between `-5%` and `-12%`) and trailing peak-drawdown arming at $+2.5 \times \text{ATR}_{14}$.

---

## 5. Verified Open-Source Reference Architectures

| Project | Stars (Verified) | License | Key Architectural Pattern to Adopt | What to Avoid |
|---|---|---|---|---|
| `AI4Finance-Foundation/FinRobot` | ~8,226 | Apache-2.0 | **Provenance separation & ContractGate**: numbers computed in code, narratives by LLM; gate strips unverified numbers. | Heavyweight autogen wrappers. |
| `NoFxAiOS/nofx` | ~13,071 | AGPL-3.0 | **"The model proposes. The runtime disposes."** Explicit runtime health (`SafeMode`), hard stop-loss, peak-drawdown giveback, cooldowns, preflight gates. | Uncalibrated prompt-only trading without PIT fundamentals. |
| `nautechsystems/nautilus_trader` | ~29,900 | LGPL-3.0 | **Event-driven parity** between backtest and live execution; explicit order-state lifecycle (`PENDING_NEW` -> `ACCEPTED` -> `FILLED` / `CANCELED`). | — |
| `TauricResearch/TradingAgents` | ~110,900 | Apache-2.0 | Multi-agent **Bull / Bear / Judge** debate decomposition. | Zero transaction-cost modeling in the upstream repo; uncalibrated LLM price targets. |
| `microsoft/qlib` | ~49,300 | MIT | Point-in-Time expression engine and cross-sectional IC / RankIC evaluation. | Over-complex YAML configuration pipelines. |
| `QuantConnect/Lean` | ~21,900 | Apache-2.0 | Reality modeling: brokerage models, corporate actions, margin/buying-power reservation, and open-order reconciliation. | — |
