# Alphalab

**Alphalab** is an autonomous, institutional-grade quantitative equity research and agentic trading system for the S&P 500, combining **strict point-in-time (PIT) fundamental valuation**, **local LLM multi-agent reasoning (`pydantic-ai` + Ollama)**, **FinBERT news sentiment**, and **NOFX-style runtime portfolio risk guards** connected to **Alpaca Paper/Live Trading**.

> **Core Principle — *Code Calculates, the Model Argues and Judges*:**  
> Language models never invent financial figures or fair-value targets. Deterministic operators compute intrinsic value ranges, accounting audits, cost-of-equity ($k_e$), and provenance trees from SEC EDGAR XBRL filings; the LLM agents extract qualitative risk/catalyst findings, debate Bull vs. Bear theses, and emit a contract-gated verdict (`BUY` / `HOLD` / `SELL`).

---

## Key Capabilities & Quantitative Foundations

### 1. Point-in-Time (PIT) Data & Fundamental Valuation Engine
- **Zero Look-Ahead Bias (`domain/pit`, `adapters/pit_store.py`, `adapters/edgar.py`):** Every SEC EDGAR 10-K, 10-Q, and 8-K fact is indexed by its SEC acceptance date (`observed_at <= knowledge_date`) in a partitioned Parquet + DuckDB store.
- **Multi-Model Intrinsic Valuation (`domain/compute`):**
  - **2-Stage FCFF / FCFE Discounted Cash Flow (DCF)** with TTM flow aggregation and net-buyback yield adjustments.
  - **Dividend Discount Model (DDM)** and **Residual Income (Edwards-Bell-Ohlson)** for Financials, Utilities, and REITs.
  - **Benjamin Graham Intrinsic Value** and **Earnings Power Value (EPV)** multiples.
  - **Piotroski 9-Point F-Score** and automated accounting red-flag audits (accrual bloat, dilution, leverage spikes).
- **Scale & Contract Plausibility Gates (`domain/contract`):** Enforces hard pre-verdict gates (`C0` unit plausibility $V_m / P \in [0.05, 20.0]$, `C1` provenance completeness, `C1b` consensus dispersion bounds, and `C2` citation verification).

### 2. Three-Role Beta Separation & Cross-Sectional Pre-Screen
- **Decoupled Beta Estimators (`domain/compute/sectors.py`):**
  1. **Valuation Discount-Rate Beta ($\beta_{ke}$):** Vasicek (1973) cross-sectional shrinkage followed by Levi & Welch (2017) double shrinkage toward `1.0` ($\beta_{ke} = 1 + 0.35(\hat{\beta}_V - 1)$) so CAPM $k_e$ does not artificially inflate perpetuities on low-beta defensive stocks.
  2. **Risk & Exposure Beta ($\beta_{\text{Welch}}$):** Welch (2022, *Critical Finance Review*, `bswa`) slope-winsorized, exponentially weighted ($\text{half-life} = 84\text{d}$) estimator without artificial floors.
  3. **Portfolio Construction Beta:** Benchmark-relative sector and beta halfspaces during weight projection.
- **Continuous Empirical Bayes & Sector-Neutral Pre-Screen (`cli/commands/autopilot.py`):**
  - **Smith & Winkler (2006)** continuous Empirical Bayes shrinkage of Margin-of-Safety (MoS) across model dispersion.
  - **Ehsani, Harvey & Li (2023, *FAJ*)** 50/50 blend of absolute MoS and within-GICS-sector standardized $z$-score so technology and growth leaders compete fairly against structurally low-multiple sectors.
  - **Asness, Moskowitz & Pedersen (2013)** 12-1 month momentum filter to penalize value traps / falling knives, combined with **ProsusAI/finbert** headline sentiment scoring (`adapters/finbert_sentiment.py`).

### 3. Agentic LLM Pipeline & Fingerprint Verdict Cache
- **Structured Agent Pipeline (`domain/analyst`, `domain/debate`, `adapters/llm.py`):**
  - **Quick-Think Extractor:** Extracts cited qualitative catalysts and risks from 10-K/10-Q/8-K disclosures and news.
  - **Bull / Bear / Judge Debate (`--debate`) or Deep-Think Narrator:** Synthesizes thesis narrative, probability calibration, and directional verdict.
- **Content-Addressable Verdict Cache (`adapters/llm_verdict_cache.py`):** Hashes `(ticker, SEC filing dates, 5% log-price bucket, method estimates, news digest)` so unchanged fundamentals reuse validated `BUY`/`HOLD` evaluations while refreshing live prices (`SELL` verdicts are never cached).

### 4. Portfolio Construction, NOFX Runtime Guards & Execution
- **Residual Volatility Sizing (`app/agentic_portfolio.py`, `app/autopilot_runner.py`):** Sizes conviction inversely to **60-day idiosyncratic/residual volatility** $\sigma_{\varepsilon,i} = \sqrt{252 \cdot \mathrm{Var}(r_{i,t} - \hat{\beta}_i r_{\text{SPY},t})}$ (**Baker, Bradley & Wurgler 2011**), tilted around a $1/N$ base (**DeMiguel, Garlappi & Uppal 2009**).
- **NOFX Runtime Risk Guards (`domain/execution/nofx_guards.py`):**
  - **Lot-1 Anchored Volatility-Scaled Stop-Loss:** Individual stops scaled to $3\sigma_{20\text{d}} \in [-6\%, -12\%]$ anchored to the initial entry price so averaging never lowers the dollar stop level.
  - **Intraday High-Water Mark & Ornstein-Uhlenbeck Convergence Exits (`Leung & Li 2015`):** Tracks post-entry intraday highs (`IEX` snapshots + persisted SQLite peak memory) and tightens trailing giveback thresholds (`arm 3% / giveback 30%`) when price converges within `2%` of intrinsic fair value.
  - **Dykstra Alternating Convex Projection (`Boyle & Dykstra 1986; Roll 1992`):** Projects target weights onto the intersection of the simplex, per-asset caps (`10%`), locked `HOLD` slots, `2.5%` no-trade inertia bands, and SPY-relative GICS sector caps.
  - **Structural Safety & Kill Switch (`adapters/alpaca_config.py`, `adapters/kill_switch.py`):** Paper trading by default; live trading requires dual explicit opt-in (`ALPHALAB_ALPACA_ENDPOINT=live` + `--acknowledge-live`), deterministic `client_order_id` idempotency, sell-before-buy ordering, and an instant file-based `HALT` kill switch.

### 5. Statistical Certification & Continual Calibration
- **Backtest & Acid Certification (`domain/stats`):** Deflated Sharpe Ratio (**Bailey & López de Prado 2014**), Probability of Backtest Overfitting (**CSCV / PBO, Bailey et al. 2016**), Politis-Romano stationary bootstrap, and Mahalanobis turbulence index (**Kritzman & Li 2010**).
- **Forecast Journal & Wealth `e-process` (`domain/journal`):** Proper Brier score decomposition (reliability, resolution, uncertainty) and bounded betting-martingale wealth `e-process` (**Waudby-Smith & Ramdas 2024, *JRSS-B***) for anytime-valid alpha monitoring against `SPY`.

---

## Methodology (Step-by-Step Pipeline)

Every autonomous trading cycle and weekly research review follows a deterministic, seven-stage methodology designed to eliminate look-ahead bias, LLM hallucination, sector crowding, and data-snooping:

```text
[Step 1: PIT Ingestion & Valuation] ──► [Step 2: Empirical Bayes & Sector Pre-Screen]
                                                        │
[Step 4: Residual-Vol Sizing & Dykstra] ◄── [Step 3: Multi-Agent LLM Debate & Cache]
                  │
                  ▼
[Step 5: NOFX Runtime Guards & Alpaca] ──► [Step 6: Brier & Wealth e-process Journal]
                                                        │
                                                        ▼
                                     [Step 7: Weekly AlphaResearch Committee]
```

### Step 1 — Point-in-Time (PIT) Ingestion & Multi-Model Intrinsic Valuation
1. **Strict As-Of Filtering (`observed_at <= knowledge_date`):** Market bars (Alpaca IEX) and SEC EDGAR XBRL facts (`10-K`, `10-Q`, `8-K`) are queried strictly as known at the cycle timestamp. Any filing accepted after `knowledge_date` is invisible to the engine.
2. **Decoupled Cost of Equity ($k_e$):** Beta is estimated using **Vasicek (1973)** cross-sectional Bayesian shrinkage followed by **Levi & Welch (2017)** double shrinkage toward `1.0` ($\beta_{ke} = 1 + 0.35(\hat{\beta}_V - 1)$), preventing low-beta stocks from receiving artificially low discount rates.
3. **Ensemble Fair-Value Calculation:** Each stock is valued across applicable structural models (**2-Stage FCFF/FCFE DCF** with TTM flows and net buybacks, **DDM / Residual Income** for Financials/Utilities/REITs, **Graham Intrinsic Value**, and **EPV**), alongside **Piotroski 9-Point F-Score** and accounting red-flag checks.
4. **Hard Plausibility Gates (`C0`–`C2`):** Any valuation method violating unit-scale plausibility ($V_m / P \notin [0.05, 20.0]$) or missing provenance links is automatically discarded before consensus aggregation.

### Step 2 — Continuous Empirical Bayes & Sector-Neutral Pre-Screening (`Stage 1`)
1. **Empirical Bayes MoS Shrinkage (`Smith & Winkler 2006`):** Raw Margin of Safety ($\text{MoS}_i = (V_i - P_i)/P_i$) is shrunk toward the cross-sectional prior inversely to the dispersion across valuation models, penalizing noisy single-model outliers.
2. **Within-GICS Sector Standardization (`Ehsani, Harvey & Li 2023`):** To prevent structural value traps (e.g., Financials/Utilities permanently crowding out Technology), the ranking score blends **50% absolute shrunk MoS** and **50% within-sector standardized $z$-score** ($z_{\text{sector},i}$).
3. **Falling-Knife & Sentiment Overlay (`Asness, Moskowitz & Pedersen 2013`):** 12-1 month momentum and local **ProsusAI/finbert** headline sentiment penalize stocks in severe negative momentum or acute news distress.
4. **Candidate Selection:** All currently open positions (100% mandatory supervision) plus the top $K$ new candidates (`--candidate-cap`) advance to Stage 2.

### Step 3 — Contract-Gated Multi-Agent Evaluation (`Stage 2`)
1. **Content-Addressable Fingerprint Cache:** Before invoking the local LLM, Alphalab computes a SHA-256 fingerprint over `(ticker, SEC filing dates, 5% log-price bucket, valuation estimates, news digest)`. Unchanged `BUY`/`HOLD` evaluations are served from cache with live price re-validation (`SELL` verdicts are never cached).
2. **Numeric Boundary Invariant:** The LLM (`qwen3` via Ollama) receives a read-only `MarketContext` containing pre-computed valuation ranges, accounting findings, and SEC/news excerpts. Any numeric figure emitted by the LLM that does not trace to a certified provenance node is rejected by `domain/contract`.
3. **Quick-Think Extraction & Bull/Bear/Judge Synthesis:** Agents extract cited qualitative catalysts/risks and debate the thesis to emit a calibrated directional verdict (`BUY`, `HOLD`, or `SELL`) with conviction probability $p \in [0, 1]$.

### Step 4 — Portfolio Construction, Residual-Vol Sizing & Convex Projection
1. **Idiosyncratic Volatility Sizing (`Baker, Bradley & Wurgler 2011`):** Approved `BUY` positions are weighted around a $1/N$ baseline (**DeMiguel et al. 2009**) tilted by shrunk MoS, LLM conviction, and the inverse of **60-day residual volatility** ($\sigma_{\varepsilon,i} = \sqrt{252 \cdot \mathrm{Var}(r_{i,t} - \hat{\beta}_i r_{\text{SPY},t})}$) so high-beta growth stocks are not mechanically under-allocated.
2. **Dykstra Alternating Convex Projection (`Boyle & Dykstra 1986`):** Raw target weights are projected onto the intersection of closed convex constraints:
   - Long-only budget simplex ($\sum_i w_i \le 1$, $w_i \ge 0$) and per-position cap ($\le 10\%$).
   - **Locked `HOLD` Retention:** Open positions with an LLM `HOLD` verdict retain their current weight slot.
   - **GICS Sector Caps:** Dynamic SPY-relative sector ceilings (`max(18%, w_SPY_sector + 12%)`).
   - **No-Trade Inertia Band (`2.5%`):** Suppresses low-value rebalancing churn inside the neutral zone.

### Step 5 — NOFX Runtime Risk Guards & Idempotent Execution
1. **Lot-1 Anchored Volatility Stop-Loss:** Every open position carries a dynamic stop-loss scaled to $3\sigma_{20\text{d}} \in [-12\%, -6\%]$, anchored to its initial entry price so subsequent top-ups never lower the dollar protection floor.
2. **Intraday High-Water Mark & Ornstein-Uhlenbeck Exits (`Leung & Li 2015`):** Using real-time Alpaca `IEX` intraday highs (`dailyBar.h`) and persistent SQLite peak tracking, positions that arm above `+3%` and give back $\ge 30\%$ of peak gains (or converge within `2%` of intrinsic fair value) trigger an immediate `[DRAWDOWN_EXIT]`.
3. **Cooldown & Sell-Before-Buy Routing:** Exited symbols enter a 5-day cooldown (`execution.db`) to prevent whipsaw re-entry. Planned trades are routed with deterministic `client_order_id` keys, executing all `SELL` orders before `BUY` orders.

### Step 6 — Forecast Journal & Anytime-Valid Calibration
1. **Immutable Forecast Registration:** Every directional verdict is logged in `data/journal.db` with its entry price, horizon (21 trading days), and stated probability.
2. **Brier Decomposition & Betting Martingale (`Waudby-Smith & Ramdas 2024`):** Matured forecasts are settled against realized returns and `SPY`, tracking Brier skill score (reliability vs. resolution) and a nonnegative wealth `e-process` ($W_t$) that monitors statistical edge continuously without $p$-value peeking bias.

### Step 7 — Weekly `AlphaResearch` 3-Role Discernment Committee
1. **Multi-Source Discovery (`ScoutProposer`):** Once per week (`alphalab alpharesearch run`), the system scans local `reference/` repos, upstream GitHub commits (`NoFxAiOS/nofx`), recent `arXiv` (`q-fin.*`) / Semantic Scholar papers, and live portfolio postmortems from `execution.db` / `journal.db`.
2. **Adversarial Methodological Veto (`SkepticReviewer`):** Evaluates 6 failure modes without short-circuiting: anti-snooping graveyard deduplication, PIT safety, hexagonal purity, LLM numeric boundary compliance, NOFX guard preservation, and Occam's complexity penalty (`>200 LOC` without commensurate Sharpe gain).
3. **Statistical Certification (`EmpiricalGatekeeper`):** Increments the persistent `n_trials` counter and requires **Deflated Sharpe Ratio $\text{DSR} > 0.95$** (**Bailey & López de Prado 2014**), Politis-Romano paired stationary-bootstrap significance ($\Delta \text{Sharpe} > 0$), and non-degradation of Max Drawdown. Only proposals surviving all 3 roles are emitted into `data/alpharesearch/plans/YYYY-MM-DD-plan.md` for human-reviewed implementation.

---

## Architecture

Alphalab follows a strict **Hexagonal Architecture (Ports & Adapters)** enforced by AST architecture tests (`tests/unit/test_architecture.py`):

```text
src/alphalab/
├── domain/          # Pure domain logic (zero I/O, zero framework dependencies)
│   ├── analyst/     # Qualitative extraction & thesis contracts
│   ├── backtest/    # Walk-forward engine, signals, turbulence & shadow benchmark
│   ├── compute/     # DCF, DDM, Residual Income, Graham, Piotroski, GICS sectors & betas
│   ├── contract/    # Hard pre-verdict gates (C0 plausibility, C1 provenance, C1b, C2)
│   ├── costs/       # Spread, Almgren-Chriss impact & capacity models
│   ├── debate/      # Multi-agent Bull / Bear / Judge debate conductor
│   ├── execution/   # Trade planner, NOFX runtime risk guards & Dykstra projection
│   ├── experiment/  # Paired A/B evaluation arms & statistical power
│   ├── forecast/    # Forecast registration, integrity & probability calibration
│   ├── journal/     # Brier decomposition, settlement & Waudby-Smith/Ramdas e-process
│   ├── ledger/      # Hash-chained audit ledger
│   ├── pit/         # Point-in-time records & anti-leakage filters
│   ├── schedule/    # NYSE calendar, holidays & cadence specifications
│   ├── skeptic/     # Pre-research quantitative screening criteria
│   └── stats/       # Deflated Sharpe, PBO (CSCV), stationary bootstrap & gates
├── app/             # Application orchestrators (autopilot runner, ingest, pipeline, paper)
├── adapters/        # I/O implementations (Alpaca, SEC EDGAR, DuckDB/Parquet, Ollama, SQLite, FinBERT)
├── artifact/        # Gated EquityResearchArtifact models & persistence
├── report/          # Terminal rendering & provenance tree formatting
└── cli/             # Zero-dependency stdlib argparse CLI entrypoint & commands
```

---

## Quickstart

### 1. Prerequisites
- **Python 3.12+** and **[uv](https://docs.astral.sh/uv/)**
- **Alpaca Paper Trading API keys** ([app.alpaca.markets](https://app.alpaca.markets/))
- **Ollama** (optional, required only for `--llm` agentic mode; e.g. `qwen3` / `qwen2.5`)

### 2. Installation

```bash
git clone https://github.com/cve-zh00/alphalab.git
cd alphalab
make install
```

### 3. Environment Configuration

Copy `.env.example` to `.env` (which is git-ignored) and fill in your credentials:

```bash
cp .env.example .env
```

```dotenv
ALPHALAB_ALPACA_KEY_ID=your_alpaca_paper_key_id
ALPHALAB_ALPACA_SECRET_KEY=your_alpaca_paper_secret_key
ALPHALAB_ALPACA_ENDPOINT=paper
ALPHALAB_ALPACA_RATE_LIMIT_PER_MINUTE=150
ALPHALAB_EDGAR_USER_AGENT="Your Name your@email.com"
```

---

## CLI Usage

Verify your environment, database paths, Alpaca connectivity, and local LLM availability:

```bash
uv run alphalab doctor
```

### Single-Stock Equity Research

Run deterministic + LLM research on a ticker (or pass `--no-llm` for instant deterministic valuation, `--debate` for Bull/Bear/Judge multi-agent debate, `--provenance` for the full derivation tree):

```bash
# Live SEC EDGAR + LLM research with provenance tree
uv run alphalab research AAPL --source live --provenance

# Fast deterministic valuation (no Ollama required)
uv run alphalab research MSFT --source live --no-llm
```

### Autonomous NOFX Autopilot (Paper Trading)

Run a single cycle (dry-run by default; pass `--submit` to route orders to Alpaca Paper) or run the continuous autonomous daemon:

```bash
# Single dry-run cycle across the S&P 500 universe
uv run alphalab autopilot run --tickers SP500 --source live --llm

# Continuous autonomous loop submitting orders during NYSE market hours
uv run alphalab autopilot loop \
  --tickers SP500 \
  --source live \
  --llm \
  --submit \
  --require-market-open \
  --candidate-cap 10 \
  --max-positions 20 \
  --interval-seconds 3600

# Inspect current positions, runtime guards, ExposureReport (Welch beta / sector gaps) & order log
uv run alphalab autopilot status
```

> **Emergency Stop (Kill Switch):** Touching a `HALT` file in the repository root (`touch HALT`) immediately blocks all order submissions and aborts active execution cycles.

### Forecast Journal & Calibration

```bash
# Record a forward-dated prediction
uv run alphalab predict AAPL --source live --horizon-days 21

# Settle matured predictions against realized returns and SPY
uv run alphalab settle

# Inspect Brier calibration, skill score, and anytime-valid wealth e-process
uv run alphalab calibration
uv run alphalab journal
```

### Weekly AlphaResearch Committee & Discernment Pipeline

```bash
# Scan local reference/ repos, remote GitHub repos, arXiv q-fin papers & live postmortems,
# filter proposals through the 3-role committee (Scout -> Skeptic -> Empirical Gatekeeper),
# and emit a certified plan.md
uv run alphalab alpharesearch run

# View history of weekly committee runs, promoted proposals, and cumulative n_trials
uv run alphalab alpharesearch history
```

---

## Development & Testing

Alphalab enforces strict typing (`mypy --strict`), formatting/linting (`ruff`), property-based invariants (`hypothesis`), architectural boundary tests, and an **80%+ coverage gate** (currently **>91% coverage across 1,190+ tests**):

```bash
# Run lint, format check, mypy --strict, and full pytest suite with coverage
make check

# Run acceptance ("acid") certification tests
make acid-test
```
