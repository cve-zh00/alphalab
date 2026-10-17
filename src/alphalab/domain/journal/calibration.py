"""Pure calibration metrics for the prediction journal.

If the system says "70% probability" it should be right about 70% of the
time. Declared confidence that is not calibrated is decorative, not tradable.
This module scores calibration from pairs of ``(declared_probability,
outcome)`` where ``outcome`` is ``1`` for a hit and ``0`` for a miss.

Everything here is pure (numpy only), deterministic, and frozen in its
outputs. Nothing performs I/O.

Metrics
-------
* **Brier score** - ``mean((p_i - o_i)^2)``. Lower is better; a 50/50 coin is
  0.25, a perfect predictor is 0, a perfectly anticorrelated one is 1.
* **ECE** - Expected Calibration Error: predictions are binned by declared
  probability and the accuracy/confidence gap is averaged, weighted by bin
  population.
* **Reliability diagram** - the per-bin data (mean confidence, observed
  accuracy, count) that shows *where* the system miscalibrates.
* **Overconfidence** - a *signed* diagnostic: positive means the system is
  systematically overconfident (confidence above accuracy), negative means
  underconfident.
* **Brier decomposition** - reliability, resolution and uncertainty, which
  separates "miscalibrated" from "no discriminative power".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DEFAULT_N_BINS",
    "MIN_SAMPLE_SIZE",
    "BrierDecomposition",
    "CalibrationReport",
    "ReliabilityBin",
    "brier_decomposition",
    "brier_score",
    "calibration_report",
    "expected_calibration_error",
    "overconfidence",
    "reliability_diagram",
    "winkler_score",
]

#: Default number of equal-width probability bins for ECE and the reliability
#: diagram. Ten bins ([0,0.1), [0.1,0.2), ... [0.9,1.0]) is the conventional
#: choice and keeps each bin interpretable as a 10-point confidence band.
DEFAULT_N_BINS = 10

#: Minimum number of settled predictions below which calibration is declared
#: **not conclusive**. With a handful of samples the bin estimates are noise;
#: 30 is the documented floor echoing the usual small-sample threshold.
MIN_SAMPLE_SIZE = 30


class ReliabilityBin(BaseModel):
    """One bin of a reliability diagram, as data (not a plot)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lower: float = Field(ge=0.0, le=1.0, description="Inclusive lower edge of the bin.")
    upper: float = Field(ge=0.0, le=1.0, description="Upper edge of the bin.")
    count: int = Field(ge=0, description="Number of predictions in the bin.")
    mean_confidence: float = Field(
        ge=0.0, le=1.0, description="Mean declared probability of the bin."
    )
    accuracy: float = Field(ge=0.0, le=1.0, description="Observed hit rate of the bin.")

    @property
    def gap(self) -> float:
        """Signed confidence-minus-accuracy gap (positive = overconfident)."""
        return self.mean_confidence - self.accuracy


class BrierDecomposition(BaseModel):
    """Murphy decomposition of the Brier score.

    ``brier = reliability - resolution + uncertainty``. Low reliability is
    good (well calibrated); high resolution is good (discriminates events);
    uncertainty is the irreducible base-rate variance ``o*(1-o)``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reliability: float = Field(ge=0.0, description="Calibration error term (lower is better).")
    resolution: float = Field(ge=0.0, description="Discrimination term (higher is better).")
    uncertainty: float = Field(ge=0.0, description="Irreducible base-rate variance.")


class CalibrationReport(BaseModel):
    """The full calibration readout for a set of settled predictions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_size: int = Field(ge=0, description="Number of settled predictions scored.")
    conclusive: bool = Field(
        description=f"False when sample_size < {MIN_SAMPLE_SIZE} (not informative)."
    )
    brier: float = Field(ge=0.0, le=1.0)
    ece: float = Field(ge=0.0, le=1.0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    overconfidence: float = Field(
        description="Signed sample-weighted confidence-minus-accuracy gap."
    )
    reliability: tuple[ReliabilityBin, ...] = Field(description="Per-bin reliability-diagram data.")
    decomposition: BrierDecomposition


def _as_arrays(
    probabilities: Sequence[float], outcomes: Sequence[bool | int]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate and coerce inputs to aligned float arrays.

    Raises:
        ValueError: On length mismatch, empty input, probabilities outside
            ``[0, 1]`` or outcomes outside ``{0, 1}``.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    outs = np.asarray(outcomes, dtype=np.float64)
    if probs.shape != outs.shape:
        raise ValueError("probabilities and outcomes must have equal length")
    if probs.ndim != 1:
        raise ValueError("probabilities and outcomes must be one-dimensional")
    if probs.size == 0:
        raise ValueError("at least one prediction is required")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if np.any((outs != 0.0) & (outs != 1.0)):
        raise ValueError("outcomes must be 0 or 1")
    return probs, outs


def brier_score(probabilities: Sequence[float], outcomes: Sequence[bool | int]) -> float:
    """Return the Brier score ``mean((p_i - o_i)^2)``.

    A perfect predictor scores 0, a 50/50 coin 0.25, a perfectly
    anticorrelated predictor 1.
    """
    probs, outs = _as_arrays(probabilities, outcomes)
    return float(np.mean((probs - outs) ** 2))


def _bin_edges(n_bins: int) -> npt.NDArray[np.float64]:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    return np.linspace(0.0, 1.0, n_bins + 1)


def _bin_index(probs: npt.NDArray[np.float64], n_bins: int) -> npt.NDArray[np.intp]:
    """Assign each probability to a bin in ``[0, n_bins - 1]``.

    Equal-width bins are half-open ``[lower, upper)`` except the last, which is
    closed so that ``p == 1.0`` lands in the top bin rather than overflowing.
    """
    idx = np.floor(probs * n_bins).astype(np.intp)
    return np.clip(idx, 0, n_bins - 1)


def reliability_diagram(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[ReliabilityBin, ...]:
    """Return the per-bin reliability data for a reliability diagram.

    Only non-empty bins are returned, each carrying its edges, population,
    mean declared confidence and observed accuracy. The populations across all
    returned bins sum to the sample size.
    """
    probs, outs = _as_arrays(probabilities, outcomes)
    edges = _bin_edges(n_bins)
    idx = _bin_index(probs, n_bins)
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                count=count,
                mean_confidence=float(np.mean(probs[mask])),
                accuracy=float(np.mean(outs[mask])),
            )
        )
    return tuple(bins)


def expected_calibration_error(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """Return the Expected Calibration Error over ``n_bins`` equal-width bins.

    ECE is the sample-weighted mean of ``|accuracy(bin) - confidence(bin)|``
    across populated bins. It lies in ``[0, 1]``.
    """
    bins = reliability_diagram(probabilities, outcomes, n_bins=n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        return 0.0
    weighted = sum(b.count * abs(b.accuracy - b.mean_confidence) for b in bins)
    return float(weighted / total)


def overconfidence(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
) -> float:
    """Return the signed overconfidence diagnostic ``mean(p) - mean(o)``.

    Positive means the system is systematically **overconfident** (declared
    confidence exceeds realized accuracy); negative means underconfident. This
    carries the direction of miscalibration that the unsigned ECE discards.
    """
    probs, outs = _as_arrays(probabilities, outcomes)
    return float(np.mean(probs) - np.mean(outs))


def brier_decomposition(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> BrierDecomposition:
    """Return the Murphy reliability/resolution/uncertainty decomposition.

    ``brier ≈ reliability - resolution + uncertainty`` (exact up to binning).
    Reliability is the calibration error (lower is better), resolution is the
    discriminative power (higher is better), uncertainty is the base-rate
    variance ``o*(1-o)``.
    """
    probs, outs = _as_arrays(probabilities, outcomes)
    n = probs.size
    base_rate = float(np.mean(outs))
    uncertainty = base_rate * (1.0 - base_rate)

    edges = _bin_edges(n_bins)
    idx = _bin_index(probs, n_bins)
    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(outs[mask]))
        reliability += count * (conf - acc) ** 2
        resolution += count * (acc - base_rate) ** 2
    _ = edges  # edges documented for clarity; binning handled by idx.
    return BrierDecomposition(
        reliability=reliability / n,
        resolution=resolution / n,
        uncertainty=uncertainty,
    )


def calibration_report(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int],
    *,
    n_bins: int = DEFAULT_N_BINS,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> CalibrationReport:
    """Assemble the full calibration report in one pass.

    The ``conclusive`` flag is ``False`` when fewer than ``min_sample_size``
    predictions are scored: the metrics are still computed and returned, but
    the caller is told not to trust them as informative.
    """
    probs, outs = _as_arrays(probabilities, outcomes)
    sample_size = int(probs.size)
    return CalibrationReport(
        sample_size=sample_size,
        conclusive=sample_size >= min_sample_size,
        brier=brier_score(probabilities, outcomes),
        ece=expected_calibration_error(probabilities, outcomes, n_bins=n_bins),
        hit_rate=float(np.mean(outs)),
        overconfidence=overconfidence(probabilities, outcomes),
        reliability=reliability_diagram(probabilities, outcomes, n_bins=n_bins),
        decomposition=brier_decomposition(probabilities, outcomes, n_bins=n_bins),
    )


def winkler_score(
    lowers: Sequence[float],
    uppers: Sequence[float],
    realized: Sequence[float],
    *,
    alpha: float = 0.20,
) -> float:
    """Return the mean Winkler interval score for central ``(1 - alpha)`` intervals.

    For interval ``[L, U]`` and realized value ``x``:
    ``W = (U - L) + (2/alpha)*(L - x)*I(x < L) + (2/alpha)*(x - U)*I(x > U)``.
    Lower is better (narrower intervals with fewer misses).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if not (len(lowers) == len(uppers) == len(realized)):
        raise ValueError("lowers, uppers, and realized must have equal length")
    if len(lowers) == 0:
        raise ValueError("cannot compute winkler_score on empty sequences")
    l_arr = np.asarray(lowers, dtype=np.float64)
    u_arr = np.asarray(uppers, dtype=np.float64)
    x_arr = np.asarray(realized, dtype=np.float64)
    if np.any(u_arr < l_arr):
        raise ValueError("upper bound must be >= lower bound")
    width = u_arr - l_arr
    below = np.maximum(0.0, l_arr - x_arr)
    above = np.maximum(0.0, x_arr - u_arr)
    scores = width + (2.0 / alpha) * (below + above)
    return float(np.mean(scores))

