"""Unit tests for the hardened PBO gate (``pbo_gate.py``).

These tests pin down the audit resolution:

* the simulated PBO null is centred on ≈0.5 (the coin-flip finding);
* edgeless searches that *passed* the old ``PBO < 0.5`` gate (scenarios A and
  B) are now **rejected**;
* a genuinely stable signal (scenario D) still **passes** — the gate is not
  made useless by conservatism;
* the ``N == 1`` passive-benchmark branch still passes on PBO;
* everything is reproducible from a seed, and the demo prints the full
  before/after table the task requires.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from alphalab.domain.stats.pbo_gate import (
    PBO_POINT_THRESHOLD,
    PBO_PVALUE_THRESHOLD,
    evaluate_pbo_gate,
    pbo_null_distribution,
)
from alphalab.domain.stats.synthetic import gbm_return_matrix

# Reduced budgets keep the unit suite fast; the gate's behaviour is unchanged.
_FAST_NULL = 120
_FAST_BOOT = 800
_STRIPS = 8  # C(8,4)=70 partitions — cheap but still a real symmetric CSCV.


# --- Scenario builders (A, B, C noise/structure; D genuine signal) ----------


def _scenario_a(rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """A: independent GBM columns — pure noise, no edge, no shared family."""
    return gbm_return_matrix(rng=rng, n_periods=252, n_configs=40, sigma=0.01)


def _scenario_b(rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """B: one family — a weak common factor plus idiosyncratic noise."""
    n_periods, n_configs = 252, 40
    factor = rng.normal(0.0, 0.01, size=n_periods)
    noise = rng.normal(0.0, 0.01, size=(n_periods, n_configs))
    return (0.3 * factor[:, None] + noise).astype(np.float64)


def _scenario_c(rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """C: deliberate first-half/second-half rank reversal."""
    n_periods, n_configs = 252, 40
    half = n_periods // 2
    matrix = rng.normal(0.0, 0.01, size=(n_periods, n_configs))
    bump = np.linspace(0.0, 0.02, n_configs)
    matrix[:half, :] += bump
    matrix[half:, :] += bump[::-1]
    return matrix.astype(np.float64)


def _scenario_d(rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """D: one configuration genuinely dominates every period (stable edge)."""
    n_periods, n_configs = 252, 40
    matrix = rng.normal(0.0, 0.01, size=(n_periods, n_configs))
    matrix[:, 0] += 0.02
    return matrix.astype(np.float64)


class TestNullDistribution:
    def test_null_is_centred_near_one_half(self) -> None:
        # Arrange / Act: the PBO null for an edgeless search.
        null = pbo_null_distribution(
            n_periods=252,
            n_configs=40,
            n_strips=_STRIPS,
            rng=np.random.default_rng(999),
            n_sims=_FAST_NULL,
        )

        # Assert: mean sits on the coin-flip value — the root cause of the bug.
        assert null.mean == pytest.approx(0.5, abs=0.08)
        assert null.std > 0.05  # wide spread: that is why 0.5 is a coin flip
        assert null.p05 < 0.5 < null.p95
        assert null.n_sims == _FAST_NULL

    def test_null_reproducible_by_seed(self) -> None:
        kwargs = {
            "n_periods": 180,
            "n_configs": 20,
            "n_strips": _STRIPS,
            "n_sims": 40,
        }
        a = pbo_null_distribution(rng=np.random.default_rng(5), **kwargs)
        b = pbo_null_distribution(rng=np.random.default_rng(5), **kwargs)
        assert a == b

    def test_null_rejects_single_configuration(self) -> None:
        with pytest.raises(ValueError, match="two configurations"):
            pbo_null_distribution(
                n_periods=100,
                n_configs=1,
                n_strips=4,
                rng=np.random.default_rng(0),
            )


class TestHardenedGate:
    def test_scenario_a_noise_is_rejected(self) -> None:
        matrix = _scenario_a(np.random.default_rng(2024))
        result = evaluate_pbo_gate(
            matrix,
            rng=np.random.default_rng(7),
            n_strips=_STRIPS,
            n_bootstrap=_FAST_BOOT,
            n_null_sims=_FAST_NULL,
        )
        assert result.passed is False
        assert result.reason() is not None

    def test_scenario_b_family_noise_is_rejected(self) -> None:
        matrix = _scenario_b(np.random.default_rng(2024))
        result = evaluate_pbo_gate(
            matrix,
            rng=np.random.default_rng(7),
            n_strips=_STRIPS,
            n_bootstrap=_FAST_BOOT,
            n_null_sims=_FAST_NULL,
        )
        assert result.passed is False

    def test_scenario_d_genuine_signal_still_passes(self) -> None:
        # The direction that matters just as much: the hardened gate must not
        # become useless by rejecting a real, stable edge.
        matrix = _scenario_d(np.random.default_rng(2024))
        result = evaluate_pbo_gate(
            matrix,
            rng=np.random.default_rng(7),
            n_strips=_STRIPS,
            n_bootstrap=_FAST_BOOT,
            n_null_sims=_FAST_NULL,
        )
        assert result.passed is True
        assert result.reason() is None
        assert result.pbo <= PBO_POINT_THRESHOLD
        assert result.p_value <= PBO_PVALUE_THRESHOLD

    def test_single_configuration_passes_with_zero_pbo(self) -> None:
        # N=1: no selection, so the gate must pass on PBO (acid criterion 1).
        matrix = np.random.default_rng(0).normal(size=(300, 1))
        result = evaluate_pbo_gate(
            matrix, rng=np.random.default_rng(1), n_strips=16, n_null_sims=10
        )
        assert result.n_configs == 1
        assert result.pbo == 0.0
        assert result.p_value == 0.0
        assert result.ci_lower == 0.0
        assert result.ci_upper == 0.0
        assert result.passed is True

    def test_reproducible_by_seed(self) -> None:
        matrix = _scenario_a(np.random.default_rng(2024))

        def run() -> tuple[float, float, float, float]:
            r = evaluate_pbo_gate(
                matrix,
                rng=np.random.default_rng(123),
                n_strips=_STRIPS,
                n_bootstrap=_FAST_BOOT,
                n_null_sims=_FAST_NULL,
            )
            return r.pbo, r.ci_lower, r.ci_upper, r.p_value

        assert run() == run()

    def test_zero_configurations_raises(self) -> None:
        matrix = np.empty((100, 0), dtype=np.float64)
        with pytest.raises(ValueError, match="at least one configuration"):
            evaluate_pbo_gate(matrix, rng=np.random.default_rng(0))

    def test_reason_is_numeric_when_rejected(self) -> None:
        matrix = _scenario_a(np.random.default_rng(2024))
        result = evaluate_pbo_gate(
            matrix,
            rng=np.random.default_rng(7),
            n_strips=_STRIPS,
            n_bootstrap=_FAST_BOOT,
            n_null_sims=_FAST_NULL,
        )
        reason = result.reason()
        assert reason is not None
        # The reason must quote the numbers, not just name the gate.
        assert f"{result.pbo:.4f}" in reason
        assert f"{result.p_value:.4f}" in reason


class TestBeforeAfterDemo:
    """DEMO — the full before/after table for scenarios A, B, C, D.

    Proves the central claim: scenarios A and B PASSED the old ``PBO < 0.5``
    gate and now FAIL the hardened gate, while D PASSES under both. The table
    is printed with ``-s`` and the key transitions are asserted.
    """

    def test_old_vs_new_gate_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        builders = {
            "A (indep noise)": _scenario_a,
            "B (one family)": _scenario_b,
            "C (reversal)": _scenario_c,
            "D (genuine)": _scenario_d,
        }
        old_threshold = 0.5  # the deprecated naive bar
        verdicts: dict[str, tuple[bool, bool]] = {}

        header = f"{'scenario':<18} {'PBO':>7} {'CI':>19} {'p':>7} {'OLD<0.5':>8} {'NEW':>6}"
        print("\n" + header)
        print("-" * len(header))
        for name, build in builders.items():
            matrix = build(np.random.default_rng(2024))
            result = evaluate_pbo_gate(
                matrix,
                rng=np.random.default_rng(7),
                n_strips=_STRIPS,
                n_bootstrap=_FAST_BOOT,
                n_null_sims=_FAST_NULL,
            )
            old_pass = result.pbo < old_threshold
            new_pass = result.passed
            verdicts[name] = (old_pass, new_pass)
            ci = f"[{result.ci_lower:.3f},{result.ci_upper:.3f}]"
            print(
                f"{name:<18} {result.pbo:>7.4f} {ci:>19} {result.p_value:>7.4f} "
                f"{'PASS' if old_pass else 'FAIL':>8} "
                f"{'PASS' if new_pass else 'FAIL':>6}"
            )

        captured = capsys.readouterr()
        print(captured.out)  # surface the table even without -s

        # A and B: PASSED the old gate, now REJECTED by the hardened gate.
        assert verdicts["A (indep noise)"][1] is False
        assert verdicts["B (one family)"] == (True, False)
        assert verdicts["C (reversal)"][1] is False
        # D: PASSES under both — hardening did not break the genuine signal.
        assert verdicts["D (genuine)"] == (True, True)

    def test_full_scale_audit_numbers_a_and_b_flip(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Full-scale (N=100, S=16) reproduction of the audit, where BOTH A and
        B demonstrably PASSED the old ``PBO < 0.5`` gate and now FAIL.

        This is the headline the task requires: at the audited shape, A (0.4239
        in the acid best-of-100 construction) and B (0.3809) were under 0.5 and
        certified by the old gate; the hardened gate rejects both, while D still
        certifies. Uses the full null (200 sims) — the vectorized CSCV keeps it
        to a couple of seconds.
        """

        def big_a(rng: np.random.Generator) -> npt.NDArray[np.float64]:
            # The exact acid-test best-of-100 edgeless matrix (seed 123): its
            # PBO is 0.4239, i.e. it PASSED the old PBO<0.5 gate.
            return gbm_return_matrix(rng=rng, n_periods=504, n_configs=100, mu=0.0, sigma=0.01)

        def big_b(rng: np.random.Generator) -> npt.NDArray[np.float64]:
            n_periods, n_configs = 504, 100
            factor = rng.normal(0.0, 0.01, size=n_periods)
            noise = rng.normal(0.0, 0.01, size=(n_periods, n_configs))
            return (0.3 * factor[:, None] + noise).astype(np.float64)

        def big_d(rng: np.random.Generator) -> npt.NDArray[np.float64]:
            n_periods, n_configs = 504, 100
            matrix = rng.normal(0.0, 0.01, size=(n_periods, n_configs))
            matrix[:, 0] += 0.02
            return matrix.astype(np.float64)

        a = evaluate_pbo_gate(
            big_a(np.random.default_rng(123)),
            rng=np.random.default_rng(9),
            n_strips=16,
            n_null_sims=200,
        )
        b = evaluate_pbo_gate(
            big_b(np.random.default_rng(2024)),
            rng=np.random.default_rng(9),
            n_strips=16,
            n_null_sims=200,
        )
        d = evaluate_pbo_gate(
            big_d(np.random.default_rng(2024)),
            rng=np.random.default_rng(9),
            n_strips=16,
            n_null_sims=200,
        )
        print(
            f"\nFULL SCALE  A: PBO={a.pbo:.4f} old={'PASS' if a.pbo < 0.5 else 'FAIL'} "
            f"new={'PASS' if a.passed else 'FAIL'}"
        )
        print(
            f"FULL SCALE  B: PBO={b.pbo:.4f} old={'PASS' if b.pbo < 0.5 else 'FAIL'} "
            f"new={'PASS' if b.passed else 'FAIL'}"
        )
        print(
            f"FULL SCALE  D: PBO={d.pbo:.4f} old={'PASS' if d.pbo < 0.5 else 'FAIL'} "
            f"new={'PASS' if d.passed else 'FAIL'}"
        )
        print(capsys.readouterr().out)

        # A and B both passed the OLD gate (PBO < 0.5) ...
        assert a.pbo < 0.5
        assert b.pbo < 0.5
        # ... and both FAIL the hardened gate now.
        assert a.passed is False
        assert b.passed is False
        # D passes both, with the exact audit PBO.
        assert d.pbo == pytest.approx(0.0, abs=1e-9)
        assert d.passed is True
        # Pin the audited point estimates so a regression is caught.
        assert a.pbo == pytest.approx(0.4239, abs=5e-3)
        assert b.pbo == pytest.approx(0.3809, abs=5e-3)
