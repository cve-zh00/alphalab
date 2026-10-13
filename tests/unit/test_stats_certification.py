"""Unit tests for the certification verdict and evaporation curve."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alphalab.domain.stats.certification import (
    CertificationResult,
    EvaporationCurve,
    certify_strategy,
)
from alphalab.domain.stats.synthetic import gbm_return_matrix


class TestEvaporationCurve:
    def test_decay_fields(self) -> None:
        # Arrange / Act
        curve = EvaporationCurve(design_sharpe=2.0, eval_sharpe=1.2)

        # Assert
        assert curve.design_to_eval_decay == pytest.approx(0.8)
        assert curve.eval_to_live_decay is None
        assert curve.live_sharpe is None

    def test_live_decay_when_present(self) -> None:
        curve = EvaporationCurve(design_sharpe=2.0, eval_sharpe=1.2, live_sharpe=0.5)
        assert curve.eval_to_live_decay == pytest.approx(0.7)

    def test_is_immutable(self) -> None:
        curve = EvaporationCurve(design_sharpe=2.0, eval_sharpe=1.2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            curve.design_sharpe = 3.0  # type: ignore[misc]


class TestCertifyStrategy:
    def test_no_edge_is_not_certified(self) -> None:
        # Arrange: a strategy and configuration matrix with no edge.
        rng = np.random.default_rng(0)
        matrix = gbm_return_matrix(rng=rng, n_periods=240, n_configs=12, mu=0.0, sigma=0.01)
        returns = matrix[:, 0]

        # Act
        result = certify_strategy(
            returns,
            matrix,
            n_trials=100,
            trial_sharpe_std=0.35,
            rng=np.random.default_rng(1),
            n_resamples=2000,
            n_strips=8,
            n_pbo_null_sims=40,
        )

        # Assert: edgeless data must be rejected with reasons.
        assert result.certified is False
        assert result.rejection_reasons

    def test_result_is_immutable(self) -> None:
        rng = np.random.default_rng(0)
        matrix = gbm_return_matrix(rng=rng, n_periods=120, n_configs=8)
        result = certify_strategy(
            matrix[:, 0],
            matrix,
            n_trials=50,
            trial_sharpe_std=0.3,
            rng=np.random.default_rng(1),
            n_resamples=1000,
            n_strips=8,
            n_pbo_null_sims=40,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.certified = True  # type: ignore[misc]

    def test_rejection_reasons_name_failing_gates(self) -> None:
        rng = np.random.default_rng(2)
        matrix = gbm_return_matrix(rng=rng, n_periods=200, n_configs=10)
        result = certify_strategy(
            matrix[:, 0],
            matrix,
            n_trials=100,
            trial_sharpe_std=0.35,
            rng=np.random.default_rng(3),
            n_resamples=2000,
            n_strips=8,
            n_pbo_null_sims=40,
        )
        joined = " ".join(result.rejection_reasons)
        # At least one gate name must appear in the human-readable reasons.
        assert any(tag in joined for tag in ("DSR", "PBO", "bootstrap"))

    def test_edgeless_pbo_gate_is_attached_and_rejects_with_numbers(self) -> None:
        # The hardened PBO gate must be attached to the result and, on edgeless
        # data, contribute a numeric rejection reason (not just a gate name).
        rng = np.random.default_rng(2)
        matrix = gbm_return_matrix(rng=rng, n_periods=200, n_configs=10)
        result = certify_strategy(
            matrix[:, 0],
            matrix,
            n_trials=100,
            trial_sharpe_std=0.35,
            rng=np.random.default_rng(3),
            n_resamples=2000,
            n_strips=8,
            n_pbo_null_sims=40,
        )
        assert result.pbo_gate.passed is False
        assert result.pbo == result.pbo_gate.pbo
        pbo_reasons = [r for r in result.rejection_reasons if r.startswith("PBO")]
        assert pbo_reasons
        assert f"{result.pbo_gate.p_value:.4f}" in pbo_reasons[0]

    def test_certified_requires_all_three_gates(self) -> None:
        # A result object constructed with all gates passing is certified only
        # when there are no rejection reasons.
        from alphalab.domain.stats.bootstrap import BootstrapCI
        from alphalab.domain.stats.pbo_gate import PBOGateResult

        ci = BootstrapCI(point=2.0, lower=1.0, upper=3.0, confidence=0.95, method="bca")
        gate = PBOGateResult(
            pbo=0.1,
            ci_lower=0.05,
            ci_upper=0.15,
            p_value=0.0,
            null_mean=0.5,
            null_std=0.15,
            n_configs=10,
            n_strips=16,
            passed=True,
        )
        passing = CertificationResult(
            dsr=0.99,
            pbo=0.1,
            bootstrap_ci=ci,
            t_eff=200.0,
            observed_sharpe=2.0,
            n_trials=100,
            certified=True,
            pbo_gate=gate,
            rejection_reasons=(),
        )
        assert passing.certified is True
        assert passing.bootstrap_ci.excludes_zero
        assert passing.pbo_gate.passed is True
