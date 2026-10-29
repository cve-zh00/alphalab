"""Unit tests for Fase 6 Institutional Synthesis (Live Execution, Risk & SPY Exposure).

Covers:
1. Three distinct roles of beta:
   - ``estimate_ke_beta_from_prices`` (Levi & Welch 2017 double shrinkage toward 1.0)
   - ``estimate_welch_beta`` (Welch 2022 CFR slope-winsorized decay-weighted unclamped beta)
   - ``SPY_SECTOR_WEIGHTS`` and Dykstra benchmark-relative sector & beta halfspaces
2. ``contract_c0_unit_plausibility`` (HARD gate & ingest filter for V_m / P in [0.05, 20.0])
3. Continuous Empirical Bayes shrinkage across upside = 1.0 (Smith & Winkler 2006)
4. Within-sector standardized MoS blending (Ehsani, Harvey & Li 2023) & 12-1 momentum filter
5. Residual volatility sizing invariance to market beta (Baker, Bradley & Wurgler 2011)
6. No-trade band, dilution protection on held BUYs, and blocking losing top-ups
7. Entry-day intraday high exclusion (b.on > entry_date) and Lot-1 stop-loss anchor
8. Post-trade Dykstra projection and freed sector headroom in the same cycle
9. Normalized bounded wealth e-process (Waudby-Smith & Ramdas 2024) and ExposureReport
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

from alphalab.adapters.alpaca_broker import BrokerPosition
from alphalab.adapters.execution_log import (
    ExecutionEvent,
    ExecutionEventKind,
    SqliteExecutionLog,
)
from alphalab.adapters.prices import PricePoint
from alphalab.app.agentic_portfolio import conviction_volatility_targets
from alphalab.app.autopilot_runner import (
    _build_exposure_report,
    _build_position_risks,
    _overlay_live_bars,
    _residual_volatilities_60d,
)
from alphalab.artifact import EquityResearchArtifact, PriceTarget, Verdict
from alphalab.cli.commands.autopilot import (
    _apply_sector_and_momentum_adjustments,
    _compute_momentum_12_1,
    _dispersion_adjusted_mos,
)
from alphalab.domain.backtest import PriceBar
from alphalab.domain.compute.sectors import (
    SPY_SECTOR_WEIGHTS,
    GicsSector,
    compute_capm_cost_of_equity,
    estimate_beta_from_prices,
    estimate_ke_beta_from_prices,
    estimate_welch_beta,
)
from alphalab.domain.contract import clauses, enforce_contract
from alphalab.domain.execution.nofx_guards import (
    PositionRiskSnapshot,
    apply_nofx_guards,
    project_feasible_weights_dykstra,
)
from alphalab.domain.journal.calibration import compute_wealth_eprocess
from alphalab.domain.types import Ticker


class TestThreeRolesOfBeta:
    def test_ke_beta_double_shrinkage_prevents_650bps_floor_pileup(self) -> None:
        rng = np.random.default_rng(42)
        spy_rets = rng.normal(0.0005, 0.01, size=252)
        # Defensive low-beta asset (true beta = 0.05)
        def_rets = 0.05 * spy_rets + rng.normal(0.0, 0.006, size=252)
        spy_closes = list(100.0 * np.cumprod(1.0 + np.insert(spy_rets, 0, 0.0)))
        def_closes = list(50.0 * np.cumprod(1.0 + np.insert(def_rets, 0, 0.0)))

        raw_vasicek = estimate_beta_from_prices(def_closes, spy_closes, default_beta=0.65)
        ke_beta = estimate_ke_beta_from_prices(
            def_closes, spy_closes, default_beta=0.65, double_shrink_s=0.35
        )
        assert raw_vasicek == pytest.approx(0.45)
        assert ke_beta == pytest.approx(1.0 + 0.35 * (0.45 - 1.0))
        # CAPM cost of equity no longer hits the 6.5% floor!
        assert compute_capm_cost_of_equity(ke_beta) > 0.075
        assert estimate_ke_beta_from_prices(def_closes, spy_closes, double_shrink_s=0.0) == 1.0

    def test_welch_bswa_beta_is_unclamped_and_winsorizes_outliers(self) -> None:
        rng = np.random.default_rng(7)
        spy_rets = rng.normal(0.0004, 0.01, size=252)
        # Near-zero beta REIT/staple with one massive idiosyncratic jump (+35% on an earnings day)
        reit_rets = 0.10 * spy_rets + rng.normal(0.0, 0.004, size=252)
        reit_rets[200] = 0.35
        spy_closes = list(100.0 * np.cumprod(1.0 + np.insert(spy_rets, 0, 0.0)))
        reit_closes = list(60.0 * np.cumprod(1.0 + np.insert(reit_rets, 0, 0.0)))

        b_welch = estimate_welch_beta(reit_closes, spy_closes, default_beta=0.85)
        # Unclamped below 0.45 so risk reporting sees true near-0.10 market sensitivity
        assert -0.10 < b_welch < 0.30


class TestUnitPlausibilityGateC0:
    def test_c0_flags_corrupt_method_value_and_withholds_point_target(self) -> None:
        art = EquityResearchArtifact(
            ticker="CORRUPT",
            verdict=Verdict.BUY,
            current_price=100.0,
            price_target=PriceTarget(value=130.0, unit="USD_per_share"),
            methods_used=("dcf", "pe", "pb"),
            method_values={"dcf": 140.0, "pe": 130.0, "pb": 2.0},  # 2.0 / 100 = 0.02 < 0.05!
            consensus_relative_span=0.20,
        )
        finding = clauses.check_c0_unit_plausibility(art)
        assert finding is not None
        assert finding.is_blocking()
        assert finding.check == "contract_c0_unit_plausibility"

        enforced = enforce_contract(art).artifact
        assert enforced.valuation_withheld is True
        assert enforced.price_target is None
        assert "[CONTRACT/contract_c0_unit_plausibility]" in enforced.contract_stamps

    def test_c0_passes_when_all_methods_within_plausible_ratio(self) -> None:
        art = EquityResearchArtifact(
            ticker="CLEAN",
            verdict=Verdict.BUY,
            current_price=100.0,
            price_target=PriceTarget(value=130.0, unit="USD_per_share"),
            methods_used=("dcf", "pe", "pb"),
            method_values={"dcf": 140.0, "pe": 130.0, "pb": 120.0},
            consensus_relative_span=0.15,
        )
        assert clauses.check_c0_unit_plausibility(art) is None


class TestContinuousEmpiricalBayesAndSectorMomentum:
    def test_dispersion_adjusted_mos_is_continuous_at_upside_one(self) -> None:
        def _art(target: float) -> EquityResearchArtifact:
            return EquityResearchArtifact(
                ticker="TEST",
                verdict=Verdict.BUY,
                current_price=100.0,
                price_target=PriceTarget(value=target, unit="USD_per_share"),
                methods_used=("dcf", "pe", "ev_ebitda"),
                consensus_relative_span=0.30,
            )

        left = _dispersion_adjusted_mos(_art(199.9))  # u = 0.999
        right = _dispersion_adjusted_mos(_art(200.1))  # u = 1.001
        assert right > left
        assert abs(right - left) < 0.01

    def test_sector_blending_and_momentum_crash_penalty(self) -> None:
        class _FakePrices:
            def series(self, ticker: Ticker) -> list[PricePoint]:
                start = date(2025, 1, 1)
                if ticker == Ticker("CRASH"):
                    # -40% crash from t-252 to t-21
                    return [
                        PricePoint(
                            on=date.fromordinal(start.toordinal() + i),
                            close=max(10.0, 100.0 - 0.20 * i),
                        )
                        for i in range(260)
                    ]
                # +15% steady uptrend
                return [
                    PricePoint(
                        on=date.fromordinal(start.toordinal() + i),
                        close=100.0 + 0.07 * i,
                    )
                    for i in range(260)
                ]

        ps = _FakePrices()
        as_of = date(2025, 12, 31)
        assert _compute_momentum_12_1(Ticker("CRASH"), ps, as_of) < -0.25
        assert _compute_momentum_12_1(Ticker("MSFT"), ps, as_of) > 0.05

        # O and REG are Real Estate; MSFT and AAPL are IT
        scored = [
            (2.0, -1, Ticker("O")),
            (1.8, -2, Ticker("REG")),
            (1.2, -3, Ticker("MSFT")),
            (0.9, -4, Ticker("AAPL")),
        ]
        adjusted = _apply_sector_and_momentum_adjustments(
            scored, price_source=ps, as_of=as_of, sector_blend=0.50
        )
        by_t = {t: s for s, _idx, t in adjusted}
        # MSFT is top of IT (z_sec = +1.0) + positive momentum -> boosted relative to raw 1.2
        assert by_t[Ticker("MSFT")] > 1.35


class TestResidualVolatilityAndGuardInvariants:
    def test_sizing_score_invariant_to_market_beta(self) -> None:
        rng = np.random.default_rng(123)
        n_days = 65
        spy_rets = rng.normal(0.0005, 0.012, size=n_days)
        idio_noise = rng.normal(0.0, 0.008, size=n_days)
        # Low-beta (0.25) and High-beta (1.55) stocks with IDENTICAL idiosyncratic noise
        low_b_rets = 0.25 * spy_rets + idio_noise
        high_b_rets = 1.55 * spy_rets + idio_noise

        def _to_bars(rets: np.ndarray) -> tuple[PriceBar, ...]:
            prices = 100.0 * np.cumprod(1.0 + np.insert(rets, 0, 0.0))
            return tuple(
                PriceBar(on=date.fromordinal(date(2026, 1, 1).toordinal() + i), close=float(p))
                for i, p in enumerate(prices)
            )

        bars = {
            Ticker("SPY"): _to_bars(spy_rets),
            Ticker("LOWB"): _to_bars(low_b_rets),
            Ticker("HIGHB"): _to_bars(high_b_rets),
        }
        as_of = bars[Ticker("SPY")][-1].on
        resid_vols = _residual_volatilities_60d(bars, as_of)
        assert resid_vols[Ticker("LOWB")] == pytest.approx(resid_vols[Ticker("HIGHB")], rel=1e-6)

        targets = conviction_volatility_targets(
            [Ticker("LOWB"), Ticker("HIGHB")],
            volatilities=resid_vols,
            slots=10,
        )
        assert targets[Ticker("LOWB")] == pytest.approx(targets[Ticker("HIGHB")], rel=1e-6)

    def test_held_buy_protected_from_dilution_and_losing_topup_blocked(self) -> None:
        hca = PositionRiskSnapshot(
            ticker=Ticker("HCA"),
            shares=10.0,
            entry_price=500.0,
            current_price=535.0,  # +7.0% winning position
            peak_price=540.0,
        )
        ctsh = PositionRiskSnapshot(
            ticker=Ticker("CTSH"),
            shares=50.0,
            entry_price=66.22,
            current_price=64.08,  # -3.23% losing position
            peak_price=66.22,
        )
        guarded = apply_nofx_guards(
            as_of=date(2026, 10, 29),
            proposed_targets={
                Ticker("HCA"): 0.048,  # diluted below 0.064 current weight
                Ticker("CTSH"): 0.072,  # model wants to top up losing position from 0.050
            },
            verdicts={Ticker("HCA"): Verdict.BUY, Ticker("CTSH"): Verdict.BUY},
            positions={Ticker("HCA"): hca, Ticker("CTSH"): ctsh},
            current_weights={Ticker("HCA"): 0.064, Ticker("CTSH"): 0.050},
            max_positions=20,
        )
        # HCA is NOT trimmed from 6.4% to 4.8%; CTSH is NOT topped up from 5.0% to 7.2%
        assert guarded.targets[Ticker("HCA")] == pytest.approx(0.064)
        assert guarded.targets[Ticker("CTSH")] == pytest.approx(0.050)


class TestEntryDayIntradayHighAndLot1StopAnchor:
    def test_entry_day_bar_high_excluded_and_lot1_anchors_stop_price(self, tmp_path: Path) -> None:
        log = SqliteExecutionLog(tmp_path / "exec.db")
        t_ctsh = Ticker("CTSH")
        # Lot 1 filled at $66.22 on 2026-10-28; Lot 2 topped up at $64.085 on 2026-10-29
        log.append(
            ExecutionEvent(
                cycle_id="c1",
                recorded_at=datetime(2026, 10, 28, 18, 5, tzinfo=UTC),
                kind=ExecutionEventKind.FILLED,
                ticker="CTSH",
                side="buy",
                shares=82.0,
                reference_price=66.22,
                client_order_id="cid1",
                broker_order_id="bid1",
                detail=None,
            )
        )
        log.append(
            ExecutionEvent(
                cycle_id="c2",
                recorded_at=datetime(2026, 10, 29, 13, 31, tzinfo=UTC),
                kind=ExecutionEventKind.FILLED,
                ticker="CTSH",
                side="buy",
                shares=35.0,
                reference_price=64.085,
                client_order_id="cid2",
                broker_order_id="bid2",
                detail=None,
            )
        )
        # Broker reports VWAP avg_entry_price = 65.58 (< Lot 1 $66.22)
        positions = {
            t_ctsh: BrokerPosition(
                ticker=t_ctsh,
                shares=117.0,
                price=64.50,
                avg_entry_price=65.58,
            )
        }
        # On entry_date (2026-10-28), morning high before fill was $68.10, close was $65.17
        bars = {t_ctsh: (PriceBar(on=date(2026, 10, 28), close=65.17, high=68.10, low=64.90),)}
        risks = _build_position_risks(positions, bars, log, as_of=date(2026, 10, 28))
        snap = risks[t_ctsh]
        # 1. Stop entry anchor stays at Lot 1 ($66.22), never lowered to VWAP ($65.58)
        assert snap.entry_price == pytest.approx(66.22)
        # 2. Pre-fill morning high ($68.10) on entry_date is NOT counted as post-entry peak
        assert snap.peak_price == pytest.approx(66.22)

    def test_overlay_live_bars_injects_post_open_price(self) -> None:
        t_ldos = Ticker("LDOS")
        raw_bars = {t_ldos: (PriceBar(on=date(2026, 10, 28), close=172.52, raw_close=172.52),)}
        updated = _overlay_live_bars(raw_bars, {t_ldos: 175.04}, as_of=date(2026, 10, 29))
        assert updated[t_ldos][-1].on == date(2026, 10, 29)
        assert updated[t_ldos][-1].close == pytest.approx(175.04)


class TestDykstraBenchmarkSectorAndBetaBands:
    def test_dykstra_enforces_spy_sector_bands_and_beta_halfspace(self) -> None:
        t = Ticker
        weights = {
            t("O"): 0.10,
            t("BXP"): 0.10,
            t("REG"): 0.10,
            t("MSFT"): 0.10,
            t("NVDA"): 0.10,
        }
        sectors = {
            t("O"): GicsSector.REAL_ESTATE.value,
            t("BXP"): GicsSector.REAL_ESTATE.value,
            t("REG"): GicsSector.REAL_ESTATE.value,
            t("MSFT"): GicsSector.INFORMATION_TECHNOLOGY.value,
            t("NVDA"): GicsSector.INFORMATION_TECHNOLOGY.value,
        }
        betas = {
            t("O"): 0.20,
            t("BXP"): 0.35,
            t("REG"): 0.30,
            t("MSFT"): 1.15,
            t("NVDA"): 1.45,
        }
        spy_bench = {s.value: w for s, w in SPY_SECTOR_WEIGHTS.items()}
        proj = project_feasible_weights_dykstra(
            weights,
            max_position_weight=0.10,
            sectors=sectors,
            benchmark_sector_weights=spy_bench,
            sector_active_band=0.12,
            betas=betas,
            min_portfolio_beta=0.80,
        )
        re_weight = sum(proj.get(x, 0.0) for x in (t("O"), t("BXP"), t("REG")))
        # Real Estate SPY weight is 2.3% + 12% band = 14.3% ceiling
        assert re_weight <= 0.143 + 1e-6
        port_beta = sum(w * betas[k] for k, w in proj.items())
        assert port_beta >= 0.80 * sum(proj.values()) - 1e-3

    def test_exited_position_frees_sector_headroom_in_same_cycle(self) -> None:
        t = Ticker
        # OLD_FIN triggers stop-loss (-9.5%), freeing its 25% Financials weight for NEW_FIN
        old_fin = PositionRiskSnapshot(
            ticker=t("OLD_FIN"),
            shares=10.0,
            entry_price=100.0,
            current_price=90.5,
            peak_price=100.0,
        )
        guarded = apply_nofx_guards(
            as_of=date(2026, 10, 29),
            proposed_targets={t("OLD_FIN"): 0.10, t("NEW_FIN"): 0.08},
            verdicts={t("OLD_FIN"): Verdict.BUY, t("NEW_FIN"): Verdict.BUY},
            positions={t("OLD_FIN"): old_fin},
            current_weights={t("OLD_FIN"): 0.25},
            sectors={t("OLD_FIN"): "financials", t("NEW_FIN"): "financials"},
            max_sector_weight=0.25,
            max_positions=20,
        )
        assert t("OLD_FIN") not in guarded.targets
        assert guarded.targets[t("NEW_FIN")] == pytest.approx(0.08)


class TestNormalizedWealthEProcessAndExposureReport:
    def test_normalized_wealth_eprocess_rejects_realistic_daily_alpha(self) -> None:
        rng = np.random.default_rng(2026)
        # +20 bps/day mean excess return with 1% daily tracking vol over 250 trading days
        excess = rng.normal(0.0020, 0.010, size=250)
        res = compute_wealth_eprocess(excess, bet_fraction_cap=0.5, return_bound=0.10)
        assert res.wealth >= 20.0
        assert res.reject_null_alpha_05 is True
        assert res.safe_lower_bound > 0.0

    def test_exposure_report_flags_low_beta_and_sector_overweight(self) -> None:
        rng = np.random.default_rng(99)
        spy_rets = rng.normal(0.0005, 0.01, size=100)
        reit_rets = 0.15 * spy_rets + rng.normal(0.0, 0.005, size=100)

        def _bars(rets: np.ndarray) -> tuple[PriceBar, ...]:
            px = 100.0 * np.cumprod(1.0 + np.insert(rets, 0, 0.0))
            return tuple(
                PriceBar(on=date.fromordinal(date(2026, 1, 1).toordinal() + i), close=float(p))
                for i, p in enumerate(px)
            )

        bars = {Ticker("SPY"): _bars(spy_rets), Ticker("O"): _bars(reit_rets)}
        as_of = bars[Ticker("SPY")][-1].on
        rep = _build_exposure_report(
            targets={Ticker("O"): 0.25},
            bars=bars,
            as_of=as_of,
            sectors={Ticker("O"): GicsSector.REAL_ESTATE.value},
        )
        assert rep.beta_welch < 0.30
        assert rep.active_sector_gaps_vs_spy["real_estate"] > 0.15
        assert len(rep.warnings) >= 2
