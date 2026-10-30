---
applyTo: "src/alphalab/domain/**/*.py"
---

# Domain Layer Purity & Mathematical Rigor Instructions

When editing files under `src/alphalab/domain/`:

1. **Zero I/O & Zero Upward Imports:**
   - Never import from `alphalab.app`, `alphalab.adapters`, `alphalab.cli`, `alphalab.artifact`, or `alphalab.report`.
   - Never import network, filesystem, or database libraries (`httpx`, `sqlite3`, `duckdb`, `pyarrow`, `pydantic_ai`).
   - Verified automatically by `tests/unit/test_architecture.py`.

2. **Immutable Value Objects & Provenance:**
   - Decorate all domain models with `@dataclass(frozen=True)`.
   - Every valuation operator (`dcf.py`, `ddm.py`, `multiples.py`, `consensus.py`) must return a `ProvenancedValue` recording its exact formula, inputs, and child derivation nodes.
   - Raise domain-specific errors (`ComputationError`, `ContractViolation`) rather than returning `NaN`, `inf`, or silent `None` on invalid inputs.

3. **Numerical & Statistical Invariants:**
   - Guard against division by zero, non-positive share counts, and `discount_rate <= terminal_growth`.
   - Maintain the separation between `estimate_ke_beta_from_prices` (Levi-Welch double-shrunk $\beta_{ke}$ for CAPM $k_e$) and `estimate_welch_beta` (unclamped slope-winsorized `bswa` beta for risk).
   - Ensure `project_feasible_weights_dykstra` preserves locked weights (`locked_tickers`) and projects onto the intersection of simplex, individual caps, SPY-relative sector caps, and optional portfolio beta halfspaces.
