"""Unit tests for :mod:`alphalab.domain.compute.sectors`."""

from __future__ import annotations

import numpy as np
import pytest

from alphalab.domain.compute.sectors import (
    SECTOR_PROFILES,
    TANGIBLE_PB_SECTORS,
    GicsSector,
    compute_capm_cost_of_equity,
    compute_rolling_beta,
    estimate_beta_from_prices,
    resolve_sector_profile,
    shrink_peer_band,
    shrink_sector_profile,
    shrink_upside_empirical_bayes,
)


def _closes_from_returns(returns: np.ndarray, start: float = 100.0) -> list[float]:
    closes = start * np.cumprod(np.concatenate(([1.0], 1.0 + returns)))
    return [float(x) for x in closes]


class TestVasicekRollingBeta:
    def test_noisy_ols_beta_shrinks_more_toward_sector_prior_than_clean_beta(self) -> None:
        """Vasicek (1973) shrinks high-sampling-variance OLS betas toward default_beta."""
        n = 60
        t = np.arange(n, dtype=float)
        bench_ret = 0.001 + 0.012 * np.sin(t * 0.37)
        x_centered = bench_ret - np.mean(bench_ret)

        raw_ortho = np.cos(t * 0.83)
        raw_ortho = raw_ortho - np.mean(raw_ortho)
        proj = np.dot(raw_ortho, x_centered) / np.dot(x_centered, x_centered)
        ortho = raw_ortho - proj * x_centered

        target_ols_beta = 1.75
        clean_ret = target_ols_beta * bench_ret + 0.0005 * ortho
        noisy_ret = target_ols_beta * bench_ret + 0.035 * ortho

        bench_closes = _closes_from_returns(bench_ret)
        clean_closes = _closes_from_returns(clean_ret)
        noisy_closes = _closes_from_returns(noisy_ret)

        prior_beta = SECTOR_PROFILES[GicsSector.UTILITIES].beta  # 0.62

        beta_clean = compute_rolling_beta(
            clean_closes,
            bench_closes,
            default_beta=prior_beta,
            min_observations=30,
        )
        beta_noisy = compute_rolling_beta(
            noisy_closes,
            bench_closes,
            default_beta=prior_beta,
            min_observations=30,
        )

        # Both have the exact same OLS beta (~1.75), but noisy_closes has much higher
        # residual variance and therefore shrinks much closer to prior_beta (0.62).
        assert beta_clean == pytest.approx(target_ols_beta, abs=0.02)
        assert prior_beta < beta_noisy < beta_clean
        assert (target_ols_beta - beta_noisy) > 0.25

    def test_vasicek_matches_exact_precision_weighted_formula(self) -> None:
        """Verify exact agreement with the analytical Vasicek (1973) posterior."""
        n = 50
        rng = np.random.default_rng(42)
        bench_ret = rng.normal(0.0005, 0.01, size=n)
        asset_ret = 1.45 * bench_ret + rng.normal(0.0, 0.02, size=n)

        bench_closes = _closes_from_returns(bench_ret)
        asset_closes = _closes_from_returns(asset_ret)
        default_beta = 0.82

        var_m = float(np.var(bench_ret, ddof=1))
        cov_im = float(np.cov(asset_ret, bench_ret, ddof=1)[0, 1])
        raw_beta = cov_im / var_m
        var_i = float(np.var(asset_ret, ddof=1))
        resid_var = max(0.0, var_i - (raw_beta**2) * var_m)
        s_beta_sq = resid_var / (max(1, n - 2) * var_m)
        tau_sq = 0.09
        w_ols = tau_sq / (tau_sq + s_beta_sq)
        prior_beta = float(np.clip(default_beta, 0.45, 2.10))
        expected = float(np.clip(w_ols * raw_beta + (1.0 - w_ols) * prior_beta, 0.45, 2.10))

        actual = estimate_beta_from_prices(
            asset_closes,
            bench_closes,
            default_beta=default_beta,
            min_observations=30,
        )
        assert actual == pytest.approx(expected, rel=1e-9)
        assert compute_rolling_beta(
            asset_closes,
            bench_closes,
            default_beta=default_beta,
            min_observations=30,
        ) == pytest.approx(expected, rel=1e-9)

    def test_short_history_returns_clipped_prior_beta(self) -> None:
        short_beta = compute_rolling_beta(
            [100.0, 101.0, 102.0],
            [100.0, 100.5, 101.0],
            default_beta=0.68,
        )
        assert short_beta == pytest.approx(0.68)
        assert compute_rolling_beta([100.0, 101.0], default_beta=0.20) == pytest.approx(0.45)
        assert compute_rolling_beta([100.0, 101.0], default_beta=2.50) == pytest.approx(2.10)

    def test_resolve_sector_profile_and_capm_bounds(self) -> None:
        assert resolve_sector_profile("MSFT").sector is GicsSector.INFORMATION_TECHNOLOGY
        assert resolve_sector_profile("UNKNOWN_TICKER").sector is GicsSector.MARKET_DEFAULT
        assert compute_capm_cost_of_equity(0.10) == pytest.approx(0.065)
        assert compute_capm_cost_of_equity(3.00) == pytest.approx(0.145)


class TestShrinkUpsideEmpiricalBayes:
    def test_non_positive_upside_is_unchanged(self) -> None:
        assert shrink_upside_empirical_bayes(0.0, relative_span=0.50) == 0.0
        assert shrink_upside_empirical_bayes(-0.25, relative_span=0.80, methods_count=1) == -0.25

    def test_wide_span_shrinks_strongly_toward_prior_mean(self) -> None:
        raw_upside = 1.50
        tight = shrink_upside_empirical_bayes(raw_upside, relative_span=0.15, methods_count=3)
        wide = shrink_upside_empirical_bayes(raw_upside, relative_span=1.20, methods_count=3)

        # Tight span retains >85% of excess upside above prior_mean (0.10)
        assert tight > 1.25
        # Wide span shrinks strongly toward 0.10
        assert 0.10 < wide < 0.45
        assert wide < tight

    def test_single_method_shrinks_more_than_multi_method(self) -> None:
        raw_upside = 0.80
        multi = shrink_upside_empirical_bayes(raw_upside, relative_span=0.20, methods_count=2)
        single = shrink_upside_empirical_bayes(raw_upside, relative_span=0.20, methods_count=1)

        assert 0.10 < single < multi < raw_upside
        # Single method adds 0.25^2 = 0.0625 observation variance (equal to prior_var),
        # cutting shrinkage weight roughly in half.
        expected_single = 0.10 + (0.0625 / (0.0625 + 0.01 + 0.0625)) * (0.80 - 0.10)
        assert single == pytest.approx(expected_single)


class TestTangiblePbSectorsAndMultipleShrinkage:
    def test_pb_is_enabled_only_in_tangible_book_sectors(self) -> None:
        """Lev & Gu (2016); Arnott et al. (2021): disable P/B in intangible-heavy sectors."""
        intangible_sectors = {
            GicsSector.INFORMATION_TECHNOLOGY,
            GicsSector.COMMUNICATION_SERVICES,
            GicsSector.HEALTH_CARE,
            GicsSector.CONSUMER_DISCRETIONARY,
            GicsSector.CONSUMER_STAPLES,
            GicsSector.INDUSTRIALS,
        }
        for sec, profile in SECTOR_PROFILES.items():
            if sec in TANGIBLE_PB_SECTORS:
                assert len(profile.peer_pb) == 3, f"{sec} should have a 3-point P/B band"
            else:
                assert sec in intangible_sectors
                assert profile.peer_pb == (), f"{sec} should disable P/B due to intangible bias"

    def test_shrink_peer_band_preserves_disabled_or_small_sample_bands(self) -> None:
        prior = (22.0, 25.0, 28.0)
        assert shrink_peer_band((), [15.0, 18.0, 20.0, 22.0, 25.0, 30.0]) == ()
        # Fewer than min_peers=5 valid multiples -> returns prior unchanged
        assert shrink_peer_band(prior, [28.0, 30.0, 32.0, 35.0], min_peers=5) == prior
        # Non-positive or extreme outliers outside [min_valid, max_valid] are ignored
        assert (
            shrink_peer_band(
                prior,
                [-5.0, 0.0, 30.0, 32.0, 34.0, 500.0, float("nan")],
                min_peers=5,
                min_valid=2.0,
                max_valid=150.0,
            )
            == prior
        )
        assert shrink_peer_band(prior, [30.0] * 10, max_pit_weight=0.0) == prior

    def test_shrink_peer_band_empirical_bayes_weighting(self) -> None:
        prior = (20.0, 25.0, 30.0)
        # 15 valid observations at [30, 35, 40] -> N=15, kappa=15 -> w_eff = 0.30 * (15/30) = 0.15
        samples = [30.0] * 5 + [35.0] * 5 + [40.0] * 5
        shrunk = shrink_peer_band(
            prior,
            samples,
            max_pit_weight=0.30,
            prior_strength=15.0,
            min_peers=5,
        )
        assert shrunk == (
            round(0.85 * 20.0 + 0.15 * 30.0, 2),
            round(0.85 * 25.0 + 0.15 * 35.0, 2),
            round(0.85 * 30.0 + 0.15 * 40.0, 2),
        )

    def test_shrink_sector_profile_respects_sector_method_gates(self) -> None:
        it_prof = SECTOR_PROFILES[GicsSector.INFORMATION_TECHNOLOGY]
        shrunk_it = shrink_sector_profile(
            it_prof,
            sample_pe=[28.0, 30.0, 32.0, 34.0, 36.0, 38.0],
            sample_pb=[6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        )
        assert shrunk_it.peer_pe[1] > it_prof.peer_pe[1]
        assert shrunk_it.peer_pb == ()

        fin_prof = SECTOR_PROFILES[GicsSector.FINANCIALS]
        shrunk_fin = shrink_sector_profile(
            fin_prof,
            sample_pe=[12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            sample_ev_ebitda=[8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
            sample_pb=[1.4, 1.6, 1.8, 2.0, 2.2, 2.4],
        )
        assert shrunk_fin.peer_ev_ebitda == ()
        assert shrunk_fin.peer_pb[1] > fin_prof.peer_pb[1]
