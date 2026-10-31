# AGENTS.md — Alphalab Agentic & Quantitative Engineering Guide

This file defines the architectural invariants, quantitative engineering standards, available skills, and custom agents for AI coding assistants working in **Alphalab** (`https://github.com/cve-zh00/alphalab`).

---

## 1. Core System Philosophy

1. **"Code Calculates; the Model Argues and Judges" (FinRobot V2 Pattern):**
   - Language models **never** compute financial ratios, discount rates, or fair-value targets.
   - All quantitative magnitudes (`DCF`, `DDM`, `Residual Income`, `Graham`, `Piotroski F-Score`, `CAPM k_e`, consensus upside) are computed by deterministic Python operators returning `ProvenancedValue` trees in `src/alphalab/domain/compute/`.
   - The LLM (`src/alphalab/adapters/llm.py`) acts strictly as a qualitative extractor and Bull/Bear/Judge thesis narrator whose output must pass the deterministic `ContractGate` (`src/alphalab/domain/contract/`).

2. **"The Model Proposes; the Runtime Disposes" (NOFX Pattern):**
   - No LLM verdict directly routes an order to the broker or sizes a position.
   - Deterministic runtime risk guards (`src/alphalab/domain/execution/nofx_guards.py`) enforce final veto power over every trade: Lot-1 anchored volatility-scaled stop-losses, intraday high-water-mark trailing giveback with Ornstein-Uhlenbeck convergence tightening, losing top-up blocks, reentry cooldowns, `2.5%` no-trade inertia bands, and Dykstra alternating convex projection onto SPY-relative GICS sector caps.

3. **Anti-Leakage by Construction (Point-in-Time Integrity):**
   - Every fundamental fact from SEC EDGAR (`10-K`, `10-Q`, `8-K`) and market bar is filtered strictly by `observed_at <= knowledge_date` inside parameterized DuckDB queries (`src/alphalab/adapters/pit_store.py`).
   - Post-hoc statistical corrections (Deflated Sharpe Ratio, PBO) complement—but never replace—strict Point-in-Time data hygiene.

---

## 2. Hexagonal Architecture & Dependency Rules

Alphalab enforces a strict **Ports & Adapters (Hexagonal)** dependency graph validated on the AST by `tests/unit/test_architecture.py`:

```text
cli  ──►  app  ──►  domain  ◄──  adapters
 │         │          ▲
 └─────────┴──► artifact / report
```

- **`src/alphalab/domain/` (Pure Domain Core):**
  - **Zero I/O:** Never import `httpx`, `sqlite3`, `duckdb`, `pyarrow`, `pydantic_ai`, `os.environ`, filesystem paths, or network sockets.
  - **Zero Upward Imports:** `domain` must **never** import from `app`, `adapters`, `cli`, `artifact`, or `report`.
  - **Immutability:** Use `@dataclass(frozen=True)` for value objects, states, and results; prefer pure functions over in-place mutation.
- **`src/alphalab/app/` (Application Orchestration):**
  - Coordinates domain services via `Protocol` ports (`Broker`, `PriceProvider`, `ResearchFunction`, `PITStore`).
- **`src/alphalab/adapters/` (Infrastructure & External I/O):**
  - Owns wire formats, HTTP retries, rate limiters, SQLite/DuckDB persistence, Ollama/FinBERT inference, and `.env` configuration via `pydantic-settings`.
- **`src/alphalab/cli/` (Command-Line Interface):**
  - Stdlib `argparse` only; maps CLI flags to typed option dataclasses and renders output via `src/alphalab/report/`.

---

## 3. Quantitative & Risk Invariants (Do Not Break)

When modifying valuation, portfolio construction, or execution logic, preserve these tested invariants:

1. **Three-Role Beta Separation (`src/alphalab/domain/compute/sectors.py`):**
   - **$\beta_{ke}$ (`estimate_ke_beta_from_prices`):** Vasicek (1973) cross-sectional shrinkage + Levi & Welch (2017) double shrinkage toward `1.0` ($s = 0.35$) used **only** for CAPM $k_e$ so low-beta defensive stocks do not cluster at the $k_e = 6.5\%$ floor.
   - **$\beta_{\text{Welch}}$ (`estimate_welch_beta`):** Welch (2022, `bswa`) slope-winsorized ($[-2 r_m, +4 r_m]$), exponentially weighted ($\text{half-life} = 84\text{d}$) estimator **without clamp**, used for risk reporting (`ExposureReport`) and portfolio beta constraints.
   - **Construction Bands (`SPY_SECTOR_WEIGHTS`):** Used by Dykstra projection to bound active sector deviations relative to `SPY`.

2. **Contract Gates (`src/alphalab/domain/contract/`):**
   - **`contract_c0_unit_plausibility`:** Every individual valuation method $V_m$ must satisfy $0.05 \le V_m / P \le 20.0$; implausible per-method outliers are dropped prior to consensus aggregation.
   - **`contract_c1_provenance` & `contract_c1b_dispersion`:** Require complete derivation provenance and bounded inter-model dispersion.
   - **`contract_c2_citations`:** Every qualitative claim must cite an input document ID.

3. **Pre-Screen & Sizing (`src/alphalab/cli/commands/autopilot.py`, `src/alphalab/app/autopilot_runner.py`):**
   - **Continuous Empirical Bayes (`shrink_upside_empirical_bayes`):** Applied continuously to all `upside > 0` without threshold discontinuities.
   - **Intra-Sector MoS + 12-1 Momentum (`_apply_sector_and_momentum_adjustments`):** Blends `50/50` absolute MoS with within-GICS-sector standardized MoS (**Ehsani, Harvey & Li 2023**) and penalizes severe 12-1 momentum crashes (**Asness, Moskowitz & Pedersen 2013**).
   - **Residual Volatility Sizing (`_residual_volatilities_60d`):** Conviction is divided by 60-day annualized idiosyncratic volatility $\sigma_{\varepsilon,i} = \sqrt{252 \cdot \mathrm{Var}(r_{i,t} - \hat{\beta}_i r_{\text{SPY},t})}$ (**Baker, Bradley & Wurgler 2011**).

4. **Execution & Stop-Loss Safety (`src/alphalab/app/autopilot_runner.py`, `src/alphalab/domain/execution/nofx_guards.py`):**
   - **Pre-Evaluation Market-Open Sync (`_await_or_refresh_open`):** When `submit=True` and `require_market_open=True`, wait for the NYSE open *before* selecting/evaluating candidates and overlay live post-open prices (`_overlay_live_bars`).
   - **Lot-1 Stop Anchor & Intraday Peak Hygiene (`_build_position_risks`):** Anchor `entry_price` to `max(logged_entry_price, pos.avg_entry_price)` so top-ups never lower the dollar stop; exclude `b.high` on `b.on == entry_date` when `as_of == entry_date` or when the Lot-1 fill occurred after the morning open window (`>= 10:00 ET`).
   - **Losing Top-Up Block & Held-BUY Retention:** Never add to a losing position (`price_pnl_pct < 0`), and never trim an open position with a `BUY` verdict merely due to denominator dilution.

---

## 4. Skills & Custom Agents in This Repository

### Project Skills (`.claude/skills/`)
Invoke or consult these skills when working on their respective domains:

| Skill | Path | Trigger / Purpose |
|---|---|---|
| **`quant-financial-auditor`** | `.claude/skills/quant-financial-auditor/SKILL.md` | Full 7-pillar institutional quant & agentic audit backed by verified arXiv papers (`scripts/verify_citations.py`) and project anatomy scans (`scripts/scan_project_anatomy.py`). |
| **`pit-valuation-modeling`** | `.claude/skills/pit-valuation-modeling/SKILL.md` | Adding or modifying SEC EDGAR XBRL concepts, Point-in-Time DuckDB queries, DCF/DDM/Residual Income/Graham operators, Levi-Welch $\beta_{ke}$, or `ContractGate` clauses (`C0`–`C2`). |
| **`nofx-execution-and-risk`** | `.claude/skills/nofx-execution-and-risk/SKILL.md` | Modifying NOFX runtime risk guards, Dykstra alternating projection, Lot-1 stop-loss anchoring, Ornstein-Uhlenbeck convergence exits, or Alpaca order routing. |
| **`live-performance-benchmark`** | `.claude/skills/live-performance-benchmark/SKILL.md` | Auditing live Alpaca Paper/Live PnL from the true agentic inception baseline vs. ETFs (`SPY`, `QQQ`, `DIA`, `IWM`, `RSP`, `VTV`, `VUG`) and non-agentic baselines (`nofx_agentic`, `rank_ensemble`). |
| **`alpharesearch-committee`** | `.claude/skills/alpharesearch-committee/SKILL.md` | Running the weekly 3-role AlphaResearch committee (`alphalab alpharesearch run`) across `reference/`, GitHub, `arXiv q-fin`, and `execution.db` postmortems to emit a certified `plan.md`. |

### Custom Agents (`.github/agents/`)

| Agent | Path | Specialty |
|---|---|---|
| **`quant-auditor`** | `.github/agents/quant-auditor.agent.md` | Institutional 7-pillar methodology review, citation verification, and exposure/calibration diagnostics. |
| **`valuation-specialist`** | `.github/agents/valuation-specialist.agent.md` | Fundamental valuation math (`DCF`, `DDM`, `Residual Income`, `Graham`, `Piotroski`), PIT EDGAR ingestion, and `ContractGate` invariants. |
| **`execution-risk-guardian`** | `.github/agents/execution-risk-guardian.agent.md` | NOFX runtime risk guards, Dykstra convex projection, stop-loss mechanics, and Alpaca execution safety. |
| **`alpharesearch-committee`** | `.github/agents/alpharesearch-committee.agent.md` | Weekly R&D discovery & 3-role discernment pipeline (`ScoutProposer` → `SkepticReviewer` → `EmpiricalGatekeeper`). |

---

## 5. Development Workflow, Testing & Security

- **Quality Gates (Mandatory before committing):**
  ```bash
  make check      # Runs ruff check, ruff format --check, mypy src (--strict), and pytest (>=80% cov)
  make acid-test  # Runs acceptance certification suite
  ```
- **Security & Secret Hygiene:**
  - **Never** hardcode API keys, tokens, account IDs, or personal/corporate credentials in source code or tests.
  - All secrets are loaded via `pydantic-settings` (`SecretStr`) from environment variables (`ALPHALAB_ALPACA_*`, `ALPHALAB_EDGAR_USER_AGENT`) or `.env` (git-ignored).
  - Never commit runtime artifacts (`data/`, `*.db`, `*.sqlite`, `*.log`, `reference/`).
- **Git & Commit Conventions:**
  - Use **Conventional Commits**: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `perf:`, `style:`.
  - Keep commits atomic, tested, and authored under the repository owner's personal GitHub identity (`cve-zh00 <98620262+cve-zh00@users.noreply.github.com>`).
