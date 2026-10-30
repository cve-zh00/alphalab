---
applyTo: "tests/**/*.py"
---

# Testing, Property Invariants & Coverage Instructions

When writing or updating tests under `tests/`:

1. **Deterministic & Offline Unit Tests:**
   - Unit (`tests/unit/`) and property (`tests/property/`) tests must run 100% offline using injected fake transports, deterministic clocks, and `tmp_path` SQLite/Parquet stores.
   - Live network tests belong exclusively in `tests/integration/` and must skip cleanly when credentials or local Ollama are absent.

2. **Arrange / Act / Assert & Property-Based Testing:**
   - Structure unit tests clearly with Arrange / Act / Assert.
   - Use `hypothesis` in `tests/property/` (and where appropriate in unit tests) to verify mathematical invariants across wide input domains (simplex conservation, Dykstra feasibility, Empirical Bayes monotonicity, e-process non-negativity, PIT cutoff monotonicity).

3. **Quality Gates:**
   - Maintain overall test coverage `>= 80%` (current baseline `> 91%`).
   - Ensure `ruff check src tests` and `mypy src` pass cleanly.
