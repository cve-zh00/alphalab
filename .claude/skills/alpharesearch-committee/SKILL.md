---
name: alpharesearch-committee
description: Runs and governs the weekly AlphaResearch 3-role discernment committee (`alphalab alpharesearch`). Scans local reference repos (`reference/`), remote GitHub repos (`NoFxAiOS/nofx`), recent quantitative finance papers (`arXiv q-fin`), and live portfolio postmortems (`execution.db` / `journal.db`), filters proposals through `ScoutProposer` -> `SkepticReviewer` -> `EmpiricalGatekeeper` (DSR > 0.95, PBO < 0.25, stationary block bootstrap CI), and outputs a certified `plan.md`.
---

# AlphaResearch Weekly Committee & Discernment Pipeline (`alphalab alpharesearch`)

## 1. Purpose & Core Philosophy

Not every paper, upstream repo commit, or backtest tweak deserves to reach production. The **AlphaResearch Committee** (`src/alphalab/domain/alpharesearch/`, `src/alphalab/app/alpharesearch_runner.py`, `src/alphalab/adapters/alpharesearch_sources.py`) implements a rigorous weekly discovery and discernment pipeline:

1. **4 Discovery Channels (`DiscoverySourceKind`):**
   - `local_repo`: Cloned repositories in `reference/` (`finrl`, `finrobot`, `ai-trader`, `hypergrok`, `FinRL-Meta`, `FinRL_Crypto`).
   - `remote_repo`: Upstream GitHub repositories (`NoFxAiOS/nofx`, `AI4Finance-Foundation/FinRL`, `AI4Finance-Foundation/FinRobot`, `HKUDS/AI-Trader`).
   - `academic_paper`: Recent working papers from **arXiv Quantitative Finance** (`q-fin.PM`, `q-fin.TR`, `q-fin.ST`).
   - `live_postmortem`: Empirical weekly postmortem of `data/execution.db` (stop-loss/drawdown exits, sector cap blocks) and `data/journal.db` (Brier calibration & wealth `e-process`).

2. **3-Role Adversarial Discernment (`3_roles_compact`):**
   - **Role 1 — `ScoutProposer`**: Formulates structured, falsifiable `ResearchProposal` hypotheses with explicit mathematical formulas, target subsystems, and affected files.
   - **Role 2 — `SkepticReviewer` (`run_skeptic_review`)**: Evaluates 6 non-short-circuiting criteria:
     1. `graveyard_dedup`: Rejects any proposal whose canonical `proposal_id` (`spec_hash`) was already rejected in a prior week.
     2. `pit_discipline`: Rejects any look-ahead risk or `filed_at > knowledge_date` violation.
     3. `llm_numeric_boundary`: Rejects any proposal allowing the LLM to emit free price targets, discount rates, or portfolio weights.
     4. `domain_purity`: Rejects any I/O coupling inside `src/alphalab/domain/`.
     5. `nofx_risk_preservation`: Rejects weakening of Lot-1 stop-loss anchoring, Ornstein-Uhlenbeck convergence exits, or Dykstra post-trade projection.
     6. `occam_and_turnover`: Rejects proposals exceeding `MAX_COMPLEXITY_LOC_DELTA` (`250` LOC) or increasing turnover by more than `+25%`.
   - **Role 3 — `EmpiricalGatekeeper` (`run_empirical_gatekeeper`)**:
     - Increments the honest trial counter `n_trials` in `data/alpharesearch.db`.
     - Certifies out-of-sample ablation return series via `certify_strategy` (**Deflated Sharpe Ratio** `DSR > 0.95`, hardened **PBO/CSCV**, and **Politis-Romano stationary block-bootstrap** Sharpe difference CI excluding zero) and verifies Max Drawdown non-degradation.

3. **Controlled Promotion via `plan.md`:**
   - Surviving proposals (`PROMOTED_TO_PLAN`) are rendered into `data/alpharesearch/plans/<YYYY-MM-DD>-plan.md` (or `--out plan.md`).
   - Vetoed proposals are logged in the **Graveyard Appendix** so they are never re-proposed.

---

## 2. CLI Commands

```bash
# Run the weekly committee across all 4 sources and write the certified plan
.venv/bin/python -m alphalab.cli alpharesearch run --ignore-watermarks

# Export the certified weekly plan directly to a custom path
.venv/bin/python -m alphalab.cli alpharesearch run --out data/alpharesearch/latest-plan.md

# Inspect past weekly committee runs and trial counts
.venv/bin/python -m alphalab.cli alpharesearch history
```
