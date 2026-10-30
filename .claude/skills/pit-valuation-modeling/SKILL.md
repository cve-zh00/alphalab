---
name: pit-valuation-modeling
description: Guide and workflow for adding or modifying Point-in-Time (PIT) SEC EDGAR XBRL ingestion, GICS sector-routed fundamental valuation operators (DCF, DDM, Residual Income, Graham, Piotroski F-Score), three-role beta estimation (Levi-Welch ke beta and Welch 2022 bswa beta), and ContractGate plausibility clauses (C0-C2) in Alphalab. Use whenever working on src/alphalab/domain/compute/, src/alphalab/domain/pit/, src/alphalab/domain/contract/, src/alphalab/adapters/edgar*, or src/alphalab/cli/ingest.py.
---

# Point-in-Time Valuation & Contract Modeling Skill

Use this skill when extending, debugging, or auditing Alphalab's fundamental ingestion, intrinsic valuation operators, beta estimation, or contract gates.

---

## 1. Point-in-Time (PIT) Ingestion Rules (`domain/pit/`, `adapters/edgar*.py`, `adapters/pit_store.py`)

1. **Temporal Cutoff (`observed_at <= knowledge_date`):**
   - Every XBRL fact extracted from SEC EDGAR (`10-K`, `10-Q`) or disclosure (`8-K`) must carry both `as_of` (fiscal period end) and `observed_at` (SEC filing acceptance date).
   - Queries in `ParquetPITStore.query(ticker, field, knowledge_date)` bind `observed_at <= ?` in DuckDB SQL using `?` placeholders.
2. **Annual vs. Quarterly Flow Disambiguation (`adapters/edgar_live.py`, `cli/ingest.py`):**
   - For flow metrics (`revenue`, `operating_income`, `net_income`, `operating_cash_flow`, `capex`, `dividends_paid`), filter annual `10-K` facts by fiscal duration (`~350–380 days` via `period_start` to `as_of`) or construct Trailing Twelve Months (TTM) carefully so single-quarter facts never masquerade as full-year flows.
   - When a `10-K` restates prior years, disambiguate by `(as_of, observed_at)` so each fiscal year resolves to its latest PIT-known value as of `knowledge_date`.
3. **Raw Unadjusted Price for Valuation (`cli/ingest.py`):**
   - Always value per-share fundamentals against **raw unadjusted close** (`PriceBar.raw_close or PriceBar.close`), never dividend-back-adjusted close, to prevent historical dividend adjustments from distorting current price-to-value ratios.

---

## 2. Valuation Operators & GICS Sector Routing (`domain/compute/`)

1. **Provenanced Operators (`domain/compute/operators/`):**
   - Every operator (`dcf_value_per_share`, `ddm_value_per_share`, `residual_income_value_per_share`, `graham_value_per_share`, `multiples_value_per_share`) must return a `ProvenancedValue` with a complete derivation tree.
2. **Sector Profile Routing (`domain/compute/sectors.py`):**
   - `resolve_sector_profile(ticker)` maps S&P 500 symbols to their GICS sector (`FINANCIALS`, `REAL_ESTATE`, `UTILITIES`, `INFORMATION_TECHNOLOGY`, etc.) with sector-specific terminal growth, margin bounds, and method weights (`method_weights`).
   - Financials and REITs down-weight FCFF DCF and prioritize **Residual Income (Edwards-Bell-Ohlson)** and **DDM**.
3. **Three-Role Beta Separation (`domain/compute/sectors.py`):**
   - **Discount Rate ($\beta_{ke}$):** Always use `estimate_ke_beta_from_prices(asset_closes, benchmark_closes, sector_prior_beta=..., double_shrink_s=0.35)` for `compute_capm_cost_of_equity`. This applies Vasicek (1973) cross-sectional shrinkage followed by Levi & Welch (2017) double shrinkage toward `1.0` ($\beta_{ke} = 1 + 0.35(\hat{\beta}_V - 1)$), preventing low-beta defensives from clustering at the $k_e = 6.5\%$ floor.
   - **Risk Measurement ($\beta_{\text{Welch}}$):** Always use `estimate_welch_beta(asset_closes, benchmark_closes, halflife_days=84.0)` (Welch 2022 `bswa`, winsorized in $[-2 r_m, +4 r_m]$, unclamped) for portfolio risk (`ExposureReport`) and beta constraints.

---

## 3. Pre-Verdict Contract Gates (`domain/contract/`)

Before any `EquityResearchArtifact` can keep a directional verdict (`BUY` / `SELL`), it must pass `ContractGate`:
- **`C0` (`check_c0_unit_plausibility`):** Every individual valuation method $V_m \in \text{artifact.method\_values}$ must satisfy $0.05 \le V_m / P \le 20.0$. Furthermore, `cli/ingest.py` drops individual implausible method estimates *before* calling `consensus(...)` so a single broken XBRL tag cannot corrupt the consensus median/weighted mean.
- **`C1` (`check_c1_provenance`):** Headline fair value must match the root of the provenance derivation tree.
- **`C1b` (`check_c1b_model_agreement`):** Relative span `(upper - lower) / fair_value` must not exceed the dispersion threshold unless all 3+ valuation methods unanimously agree on `> 20%` upside.
- **`C2` (`check_c2_citations`):** Qualitative findings must reference valid input document IDs (`[DOC-...]`).

---

## 4. Verification Checklist

After modifying any file in this domain, run:
```bash
.venv/bin/pytest tests/unit/test_compute_*.py tests/unit/test_contract_*.py tests/unit/domain/compute/test_sectors.py tests/unit/test_fase6_institutional_synthesis.py -q
```
