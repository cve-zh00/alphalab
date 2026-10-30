# GitHub Copilot Instructions — Alphalab

You are working on **Alphalab**, an institutional-grade quantitative equity research and autonomous agentic trading system for the S&P 500 (`Python 3.12+`, `pydantic v2`, `pydantic-ai`, `duckdb`, `pyarrow`, `numpy`, `scipy`, `httpx`).

Always read and follow [`AGENTS.md`](../AGENTS.md) at the repository root. The rules below summarize the non-negotiable engineering constraints.

---

## 1. Architecture & Code Organization

- **Hexagonal Architecture (Ports & Adapters):**
  - `src/alphalab/domain/` contains pure domain logic and mathematical operators. It must **never** perform I/O or import from `app`, `adapters`, `cli`, `artifact`, or `report` (enforced by `tests/unit/test_architecture.py`).
  - `src/alphalab/app/` orchestrates use cases (`AutopilotRunner`, `LiveIngestOrchestrator`, `ResearchPipeline`, `DebatePipeline`) via `typing.Protocol` interfaces.
  - `src/alphalab/adapters/` implements external integrations (Alpaca REST/snapshots, SEC EDGAR XBRL, DuckDB/Parquet PIT store, SQLite journal/execution logs, Ollama LLM, FinBERT sentiment).
  - `src/alphalab/cli/` implements the stdlib `argparse` CLI and delegates to `src/alphalab/cli/commands/`.
- **Immutability & Type Safety:**
  - Use `@dataclass(frozen=True)` for domain value objects, configs, and results.
  - Maintain strict type annotations across all modules (`mypy --strict` must pass with zero errors).
  - Keep functions focused (<50 lines) and modules cohesive (<400 lines where practical).

---

## 2. Quantitative & Financial Safety Rules

1. **Numbers Come from Code, Never from LLMs:**
   - All valuation estimates (`DCF`, `DDM`, `Residual Income`, `Graham`, `EPV`, `Piotroski F-Score`) and discount rates (`CAPM k_e`) must be computed deterministically with full `ProvenancedValue` derivation trees.
   - Any LLM output must be validated through `ContractGate` (`C0` unit plausibility $V_m / P \in [0.05, 20.0]$, `C1` provenance, `C1b` model agreement, `C2` document citations).
2. **Strict Point-in-Time (PIT) Anti-Leakage:**
   - Every query to `ParquetPITStore` or `EdgarClient` must filter by `observed_at <= knowledge_date`.
   - Use parameterized DuckDB/SQLite queries (`?` placeholders) exclusively; never interpolate user or external strings into SQL.
3. **Three-Role Beta Separation (`domain/compute/sectors.py`):**
   - Use `estimate_ke_beta_from_prices` (Vasicek + Levi-Welch double shrinkage toward `1.0`) for CAPM cost of equity ($k_e$).
   - Use `estimate_welch_beta` (Welch 2022 `bswa` slope-winsorized, exponentially weighted, unclamped) for risk measurement (`ExposureReport`) and portfolio beta constraints.
4. **NOFX Runtime Risk Guards (`domain/execution/nofx_guards.py`, `app/autopilot_runner.py`):**
   - Preserve Lot-1 stop-loss anchoring (`max(logged_entry_price, pos.avg_entry_price)`), intraday peak hygiene (exclude `b.high` on `entry_date` when `as_of == entry_date` or fill was `>= 10:00 ET`), Ornstein-Uhlenbeck convergence giveback tightening, losing top-up blocks, `2.5%` no-trade bands, and Dykstra convex projection onto SPY-relative sector caps.
   - Never bypass the dual live-trading lock (`ALPHALAB_ALPACA_ENDPOINT=live` + explicit `acknowledge_live=True`) or the `HALT` kill switch.

---

## 3. Testing, Linting & Secret Hygiene

- **TDD & Verification:**
  - Write unit and property-based (`hypothesis`) tests for all new domain and application logic.
  - Run targeted `pytest` during iteration and verify with `.venv/bin/ruff check src tests` and `.venv/bin/pytest` (minimum coverage gate: **80%**, project standard **>90%**).
- **Zero Secrets in Code:**
  - Never commit `.env`, API keys, tokens, SQLite databases (`data/*.db`), logs (`data/*.log`), or cloned reference repositories (`reference/`).
  - Use `SecretStr` in `pydantic-settings` classes and unwrap secrets only inside `auth_headers()`.
