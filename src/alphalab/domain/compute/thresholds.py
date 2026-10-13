"""Calibrated thresholds and tolerances for the compute layer.

Every tolerance and magnitude bound used by the valuation operators and the
audit checks lives here as a *named, documented constant*. Scattering bare
numeric literals through the code would make the calibration invisible and
untraceable; the whole point of this layer is that a number can always be
explained. Each constant records *why* it has the value it does.

All constants are plain ``float`` / ``int`` module-level names (immutable by
convention — never rebind them). No I/O, no state.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# EV bridge reconciliation
# ---------------------------------------------------------------------------
# equity_value + net_debt should equal enterprise_value exactly in theory.
# In practice inputs are rounded to the cent / to millions, so we allow a
# small *relative* slack. 0.5% is tighter than any rounding a human analyst
# would introduce on a billion-dollar balance sheet, yet loose enough to not
# fire on cent-level rounding.
EV_BRIDGE_REL_TOLERANCE: Final[float] = 5e-3  # 0.5%

# ---------------------------------------------------------------------------
# Consensus / method dispersion
# ---------------------------------------------------------------------------
# Relative span = (max - min) / |mid|. When independent valuation methods
# disagree by more than this fraction, the point estimate is not trustworthy
# and the downstream contract gate (Task 7) must degrade to a range. 0.35
# (35%) is a standard equity-research rule of thumb: methods landing within a
# third of each other are "consistent", beyond that they are "divergent".
CONSENSUS_WIDE_SPAN_THRESHOLD: Final[float] = 0.35

# ---------------------------------------------------------------------------
# DCF / DDM growth-vs-discount guard
# ---------------------------------------------------------------------------
# The Gordon terminal value (and the DDM) require g < r strictly; as g -> r
# the value diverges to infinity. We additionally require a minimum spread so
# the result is numerically meaningful rather than merely finite. A 10 bps
# floor keeps the terminal multiple below ~1000x, which is already absurd but
# finite — tighter floors would reject legitimate low-spread utilities.
MIN_GROWTH_DISCOUNT_SPREAD: Final[float] = 1e-3  # 10 bps

# ---------------------------------------------------------------------------
# Magnitude sanity bounds (US equities)
# ---------------------------------------------------------------------------
# A share price must be positive. Berkshire A (~$600k) is the practical ceiling
# for a US-listed share; we pad to $1,000,000 to avoid false positives on
# extreme-but-real names while still catching unit/scale blunders.
MAX_PLAUSIBLE_PRICE_PER_SHARE: Final[float] = 1_000_000.0

# A positive, non-ridiculous P/E band. Loss-making or barely-profitable names
# can show enormous P/E; 1000x is already far beyond any defensible multiple
# and almost always signals a near-zero denominator or a unit error.
MAX_PLAUSIBLE_PE: Final[float] = 1_000.0

# EV/EBITDA rarely exceeds ~100x even for hyper-growth names; beyond that the
# EBITDA base is effectively zero and the multiple is meaningless.
MAX_PLAUSIBLE_EV_EBITDA: Final[float] = 100.0

# Net/operating margins are fractions in [-1, 1] for all but pathological
# cases. A margin outside [-100%, 100%] is structurally impossible for a going
# concern and indicates a sign or scale error.
MIN_PLAUSIBLE_MARGIN: Final[float] = -1.0
MAX_PLAUSIBLE_MARGIN: Final[float] = 1.0

# ---------------------------------------------------------------------------
# Provenance structural limits
# ---------------------------------------------------------------------------
# Hard ceiling on derivation-tree traversal depth. The DAG is acyclic by
# construction, but this is a defensive guard so a pathological input can never
# cause unbounded recursion. 256 levels is far deeper than any real valuation
# composition (a 3-method consensus is ~4 levels deep).
MAX_PROVENANCE_DEPTH: Final[int] = 256
