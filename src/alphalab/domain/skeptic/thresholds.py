"""Calibrated rejection thresholds for the deterministic skeptic.

Every bar the skeptic rejects against lives here as a *named, documented
``Final`` constant*. The skeptic is deliberately "dumb": it imports these
calibrated numbers rather than deriving them, so each rejection can always be
traced back to the exact criterion and the source the number came from.

Design thesis (``docs/00-research-dossier.md`` §0, §7): the skeptic is **code
with verifiable rejection criteria**, not an LLM persona. ``arXiv:2608.27734``
showed an oracle with deliberate leakage reaching Sharpe 34.7 while *passing*
Deflated Sharpe and PBO — so the defences must be **structural**, not
persuasive. Where the dossier states a concrete value (e.g. Harvey-Liu-Zhu
``t > 3``) we use it and cite it in the comment.

All constants are module-level ``Final`` values, immutable by convention.
No I/O, no state.
"""

from __future__ import annotations

from typing import Final

# Re-export the live DSR / PBO thresholds so the skeptic and the stats layer
# share one calibration (imported, never re-declared). Dossier §7: DSR > 0.95,
# PBO via CSCV in the red.
from alphalab.domain.stats import (
    DSR_THRESHOLD,
    PBO_POINT_THRESHOLD,
    PBO_PVALUE_THRESHOLD,
)

# ---------------------------------------------------------------------------
# S1 — Insufficient effective sample (T_eff)
# ---------------------------------------------------------------------------
# A Sharpe computed over a handful of *effective* (autocorrelation-adjusted)
# observations is noise. The dossier (§7) does not fix a numeric floor, so we
# derive one from the Harvey-Liu-Zhu significance bar used in S6: at the
# non-annualised Sharpe convention of this package, a per-period Sharpe needs
# roughly ``t = SR * sqrt(T_eff) > 3`` to clear HLZ. For a defensible
# single-digit per-period Sharpe (~0.1), that already needs hundreds of
# observations; we set a conservative structural floor of 60 effective
# observations below which *no* Sharpe is admissible regardless of magnitude.
# 60 ~ one quarter of daily data; anything thinner cannot support a backtest
# claim. T_eff (not raw T) is used via ``domain/stats/autocorrelation.py``.
MIN_EFFECTIVE_SAMPLE: Final[float] = 60.0

# ---------------------------------------------------------------------------
# S2 — Edge destroyed by costs
# ---------------------------------------------------------------------------
# "The criterion that kills the most strategies in the literature"
# (TradingAgents fell from Sharpe 0.43 gross to 0.22 net on frictions alone).
# We reject when the NET (per-period, non-annualised) Sharpe falls below this
# floor, OR when the sign of the edge flips under costs (handled in code). The
# floor is intentionally just above zero: a strategy whose entire Sharpe is
# eaten by frictions has no demonstrable edge net-of-costs. 0.02 per period is
# a token positive bar — below it the net result is indistinguishable from a
# costed random walk.
MIN_NET_SHARPE: Final[float] = 0.02

# ---------------------------------------------------------------------------
# S3 — Capacity exceeded
# ---------------------------------------------------------------------------
# The proposed size must stay strictly below the capacity ceiling where market
# impact consumes the whole edge (``costs.capacity_size``). We additionally
# demand a safety margin: trading right at the ceiling leaves zero net edge, so
# we require the proposed size to be at most this fraction of ``Q*``. 0.50
# (half the break-even size) keeps roughly half the gross edge after impact
# under the square-root law, a standard execution-desk rule of thumb.
MAX_CAPACITY_UTILISATION: Final[float] = 0.50

# ---------------------------------------------------------------------------
# S6 — Statistical significance (DSR, PBO gate, bootstrap CI, HLZ t-stat)
# ---------------------------------------------------------------------------
# Deflated Sharpe must clear the same bar the stats layer uses. Dossier §7
# item 8: "DSR ... umbral > 0.95." We import the live threshold from the stats
# layer rather than re-declaring it, so the two can never drift apart.
# (Re-exported below as ``DSR_THRESHOLD``.)

# Harvey, Liu & Zhu (2016, NBER w20592): with multiple testing, the
# significance bar for a *new* factor is a t-stat above ~3, not the classic 2.
# Dossier §7 item 10: "t > 3, no 2 (Harvey-Liu-Zhu)." The honest t-stat uses
# T_eff: ``t = observed_sharpe * sqrt(T_eff)``.
HARVEY_LIU_ZHU_T_STAT: Final[float] = 3.0

# ---------------------------------------------------------------------------
# S7 — Low cross-asset / cross-period robustness
# ---------------------------------------------------------------------------
# A real edge persists across assets and sub-periods; an artefact appears in
# one slice and vanishes elsewhere. We require that the fraction of slices
# (assets or sub-windows) whose per-slice Sharpe stays positive is at least
# this share. 0.60 (a clear majority) rejects edges that live in a minority of
# slices while tolerating the one-off bad regime every genuine strategy has.
MIN_ROBUST_SLICE_FRACTION: Final[float] = 0.60

# The minimum number of slices required before robustness can even be assessed.
# With fewer than two slices there is nothing to corroborate across, so a
# robustness claim is unfalsifiable and we decline to certify robustness.
MIN_SLICES_FOR_ROBUSTNESS: Final[int] = 2

__all__ = [
    "DSR_THRESHOLD",
    "HARVEY_LIU_ZHU_T_STAT",
    "MAX_CAPACITY_UTILISATION",
    "MIN_EFFECTIVE_SAMPLE",
    "MIN_NET_SHARPE",
    "MIN_ROBUST_SLICE_FRACTION",
    "MIN_SLICES_FOR_ROBUSTNESS",
    "PBO_POINT_THRESHOLD",
    "PBO_PVALUE_THRESHOLD",
]
