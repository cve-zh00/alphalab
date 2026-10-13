"""The deterministic skeptic — structural, auditable trade rejection.

Thesis (``docs/00-research-dossier.md`` §0, §7): a trustworthy skeptic is
**code with verifiable rejection criteria**, not an LLM persona debating the
bull. Two samples of the same prior are not independent validation; defences
against overfit and leakage must be *structural*, not persuasive. The reference
study ``arXiv:2608.27734`` showed a leakage oracle reaching Sharpe 34.7 while
*passing* Deflated Sharpe and PBO — proof that persuasion-based scepticism is
worthless.

This package therefore calls **no LLM**. Given a :class:`TradeCandidate` it
returns a :class:`SkepticVerdict` that accepts or rejects by auditable rules,
and every rejection records the exact criterion and the numbers that triggered
it. It is designed to be *measured*: the Phase-3 A/B experiment will decide
whether it improves out-of-sample drawdown and avoids edgeless trades
net-of-costs; if it does not, it is discarded as complexity without value.

Public surface:

* :func:`judge_candidate` — the aggregator (run all criteria, never raises).
* :class:`TradeCandidate`, :class:`SkepticVerdict`, :class:`DataProvenance`.
* The individual criterion functions, for targeted testing / reuse.
"""

from __future__ import annotations

from alphalab.domain.skeptic.aggregator import judge_candidate
from alphalab.domain.skeptic.candidate import (
    DataProvenance,
    SkepticVerdict,
    TradeCandidate,
)
from alphalab.domain.skeptic.criteria import (
    check_capacity,
    check_contradictory_data,
    check_cost_survival,
    check_cross_robustness,
    check_leakage,
    check_sample_sufficiency,
    check_statistical_significance,
    check_survivorship,
)

__all__ = [
    "DataProvenance",
    "SkepticVerdict",
    "TradeCandidate",
    "check_capacity",
    "check_contradictory_data",
    "check_cost_survival",
    "check_cross_robustness",
    "check_leakage",
    "check_sample_sufficiency",
    "check_statistical_significance",
    "check_survivorship",
    "judge_candidate",
]
