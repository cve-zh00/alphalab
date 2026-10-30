---
name: nofx-execution-and-risk
description: Guide and workflow for modifying or debugging Alphalab's NOFX runtime risk guards, Lot-1 anchored volatility-scaled stop-losses, Ornstein-Uhlenbeck convergence giveback exits, Dykstra alternating convex weight projection, and Alpaca Paper/Live execution cycles. Use whenever working on src/alphalab/domain/execution/, src/alphalab/app/autopilot_runner.py, src/alphalab/app/agentic_portfolio.py, or src/alphalab/adapters/alpaca_broker.py.
---

# NOFX Runtime Risk Guards & Portfolio Execution Skill

Use this skill when modifying or diagnosing trade planning, runtime risk guards, portfolio weight projection, stop-loss mechanics, or live Alpaca autopilot execution.

---

## 1. Execution Cycle Order of Operations (`src/alphalab/app/autopilot_runner.py`)

In `AutopilotRunner.run_cycle(as_of=..., submit=...)`, preserve the exact institutional sequence:
1. **Broker Snapshot & Fill Sync:** Fetch `account()`, `positions()`, `clock()`, `open_orders()`, and call `_sync_broker_fills(...)` so `SqliteExecutionLog` records any newly filled orders before risk evaluation.
2. **Pre-Evaluation Market-Open Synchronization (`_await_or_refresh_open`):**
   - If `submit=True`, `require_market_open=True`, and `not clock.is_open`, check `rem = (clock.next_open - clock.timestamp).total_seconds()`.
   - When `0 < rem <= 300s`, sleep `rem + 15.0s` **before** selecting or evaluating candidates so the cycle runs with live post-open market state (`clock.is_open == True`).
3. **Live Bar Overlay (`_overlay_live_bars`):**
   - Overlay current broker/snapshot prices (`latest_prices`) onto `raw_bars` for `decision_date` so `live_builder`, `LlmVerdictCache` (`current_price_bucket`), momentum, and `OrderIntent.reference_price` use live post-open prices rather than stale $t-1$ closes.
4. **Residual Volatility Sizing (`_residual_volatilities_60d`):**
   - Compute 60-day annualized idiosyncratic volatility $\sigma_{\varepsilon,i} = \sqrt{252 \cdot \mathrm{Var}(r_{i,t} - \hat{\beta}_i r_{\text{SPY},t})}$ relative to `SPY` and pass to `conviction_volatility_targets`.

---

## 2. Position Risk & Stop-Loss Hygiene (`_build_position_risks`)

1. **Lot-1 Stop Anchor:**
   - Compute `entry_price = max(logged_entry_price, pos.avg_entry_price)` when both exist. Averaging down on a subsequent lot must **never** lower the dollar stop-loss level of the original entry.
2. **Entry-Day Intraday High Filtering (`_late_entry_fills`):**
   - Daily `PriceBar.high` (`b.high`) covers the full `09:30–16:00 ET` session without a timestamp.
   - For `b.on > entry_date`, include `b.high` in `peak_price`.
   - For `b.on == entry_date`, include `b.high` **only** if `as_of > entry_date` and the Lot-1 `FILLED` timestamp did not occur after the morning open window (`< 10:00 ET`). On the entry day itself (`as_of == entry_date`) or after a late fill, use `max(entry_price, b.close, current_price)` so pre-entry morning spikes never trigger false `[DRAWDOWN_EXIT]` liquidations.
3. **Volatility-Scaled Stop-Loss (`_vol_scaled_stop_pct`):**
   - Individual stop-loss is `clamp(3.0 * sigma_daily_20d, 0.06, 0.12)` (`-6%` floor, `-12%` ceiling).

---

## 3. NOFX Guard Disposition & Dykstra Projection (`src/alphalab/domain/execution/nofx_guards.py`)

1. **Open Position Evaluation (`_evaluate_open_positions`):**
   - Hard exits: `[STOP_LOSS_EXIT]`, `[DRAWDOWN_EXIT]`, or confirmed AI `[SELL]` (`sell_confirmations_required`).
   - **Ornstein-Uhlenbeck Convergence (`_effective_drawdown_thresholds`):** When `fair_value_upside_pct <= 2.0%` (Leung & Li 2015), tighten trailing drawdown thresholds from `(arm 5.0%, giveback 40.0%)` to `(arm 3.0%, giveback 30.0%)`.
   - **Losing Top-Up Block:** If `pos.price_pnl_pct < 0.0` and `proposed_w > retained_weight`, cap weight at `retained_weight` (`locked_tickers.add(ticker)`).
   - **Held-BUY Dilution Protection & No-Trade Band (`rebalance_band_abs = 0.025`):** Never trim an open `BUY` position below `retained_weight` (up to `max_position_weight`), and lock open positions when $|w_{\text{proposed}} - w_{\text{retained}}| < 2.5\%$.
2. **Dykstra Alternating Convex Projection (`project_feasible_weights_dykstra`):**
   - Projects post-trade target weights onto the intersection of:
     - Simplex ($\sum_i w_i = \min(1.0, \text{gross\_target})$, $w_i \ge 0$) with locked positions fixed,
     - Per-asset cap ($w_i \le 0.10$),
     - SPY-relative GICS sector caps ($w_s \le \max(\min(0.30, w_{\text{SPY},s} + 0.12), \text{locked\_sum}_s)$),
     - Optional portfolio Welch beta halfspaces ($\beta_{\min} \le \beta_p \le \beta_{\max}$).

---

## 4. Verification Checklist

```bash
.venv/bin/pytest tests/unit/test_nofx_guards.py tests/unit/domain/execution/test_nofx_guards.py tests/unit/test_autopilot_runner.py tests/unit/test_fase6_institutional_synthesis.py -q
```
