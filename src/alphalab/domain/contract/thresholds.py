"""Calibrated thresholds for the output contract gate.

Every band and tolerance the contract clauses use lives here as a *named,
documented constant*. The gate is deliberately "dumb": it imports these
calibrated numbers rather than deriving them, and it verifies a result without
re-running the computation that produced it. A verifier that recomputes can
reproduce the very bug it is meant to catch; a verifier that checks a result
against an independent, calibrated band cannot.

All constants are module-level ``Final`` floats, immutable by convention.
No I/O, no state.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# C1 — upside/downside sanity band (target / current_price ratio)
# ---------------------------------------------------------------------------
# A single valuation method with no cross-check earns the *narrower* band: with
# nothing to corroborate it, a target more than 2x or less than 1/2x the live
# price is almost always a unit/scale blunder rather than a defensible call.
# Standard sell-side single-model targets rarely stray beyond ±100%.
SINGLE_METHOD_RATIO_K: Final[float] = 2.0

# When several methods corroborate the number, we tolerate a *wider* band
# (1/4x .. 4x): genuine deep-value or high-growth theses can defensibly imply a
# 4x re-rating, and the cross-method agreement is what earns that latitude.
MULTI_METHOD_RATIO_K: Final[float] = 4.0

# ---------------------------------------------------------------------------
# C1b — cross-method corroboration span
# ---------------------------------------------------------------------------
# ``consensus(...)`` already computes relative_span = (high-low)/|mid| across
# methods. When that dispersion exceeds this fraction the methods disagree too
# much for a single defensible point: the gate degrades to a range (keeps the
# range, drops the point) rather than publishing a false-precision number.
# 0.35 (35%) mirrors the compute layer's CONSENSUS_WIDE_SPAN_THRESHOLD so the
# contract and the aggregator share one calibration of "divergent".
METHOD_CORROBORATION_SPAN_K: Final[float] = 0.35

# ---------------------------------------------------------------------------
# C3 — conclusion-vs-headline drift
# ---------------------------------------------------------------------------
# The amount quoted in the conclusion must equal the headline target. A tiny
# relative slack absorbs rounding when the same number is rendered in two
# places (e.g. 150 vs 150.00); anything beyond it is an internal inconsistency
# and withholds the price. 0.1% is far below one cent on a $100 share.
NARRATIVE_DRIFT_TOLERANCE: Final[float] = 1e-3  # 0.1%

# ---------------------------------------------------------------------------
# Numeric scrubbing / matching precision
# ---------------------------------------------------------------------------
# Two decimal places is the natural precision for a USD-per-share price, so a
# withheld value is matched in free text at that granularity (e.g. "150",
# "150.0", "150.00"). The no-resurrection scan and the text scrubber use this
# to recognise the number regardless of trivial formatting.
PRICE_TEXT_DECIMALS: Final[int] = 2
