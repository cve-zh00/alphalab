"""The skeptic aggregator: run every criterion, return an auditable verdict.

Design decisions (documented here because the Phase-3 A/B experiment depends on
them):

* **No short-circuit.** Every criterion runs on every candidate. The verdict
  reports the *complete* list of failures so the experiment can measure how
  many and which criteria fired, not merely the first. This is strictly more
  expensive than bailing early and that cost is deliberate.
* **A single HARD failure rejects.** The skeptic's job is to protect capital;
  any one structural defect (leakage, survivorship, no edge net-of-costs,
  insufficient sample, failed significance, capacity breach, contradictory
  data) is sufficient to withhold acceptance. SOFT failures (currently only
  cross-asset robustness) are recorded and *do* block acceptance too, but are
  marked SOFT so a downstream consumer can distinguish "structurally broken"
  from "insufficiently corroborated". Acceptance requires *zero* failed
  criteria of either severity — a conservative stance appropriate to a gate
  whose false negatives (accepting a bad trade) are far costlier than its
  false positives.
* **Never raises.** Mirroring the contract gate in ``domain/contract/``, a
  degenerate candidate yields a rejection verdict with the failure stamped,
  never an exception. A skeptic that explodes is a skeptic that fails to
  protect. Each criterion is already total; the aggregator adds a defensive
  wrapper so even an unforeseen error becomes a HARD finding.

Pure: numpy only, no I/O, no global state. Randomness is always seeded by the
caller via an explicit ``numpy.random.Generator``.
"""

from __future__ import annotations

import numpy as np

from alphalab.domain.compute import AuditFinding, Severity
from alphalab.domain.skeptic.candidate import SkepticVerdict, TradeCandidate
from alphalab.domain.skeptic.criteria import (
    DETERMINISTIC_CRITERIA,
    check_statistical_significance,
)

__all__ = ["judge_candidate"]


def _safe_run(check: str, run: object, *args: object, **kwargs: object) -> AuditFinding:
    """Run a criterion, converting any explosion into a HARD finding.

    A criterion must never abort the verdict; an unforeseen error becomes a
    stamped HARD finding so the trade is withheld and the failure is visible.
    """
    try:
        result = run(*args, **kwargs)  # type: ignore[operator]
    except Exception as exc:  # defence in depth; criteria are already total
        return AuditFinding(
            check=f"{check}.raised",
            passed=False,
            severity=Severity.HARD,
            message=f"criterion raised {type(exc).__name__}: {exc}",
            values=(),
        )
    assert isinstance(result, AuditFinding)
    return result


def judge_candidate(
    candidate: TradeCandidate,
    *,
    rng: np.random.Generator,
) -> SkepticVerdict:
    """Run every rejection criterion and return an immutable verdict.

    Args:
        candidate: The trade/thesis on trial.
        rng: Explicit seeded generator; the statistical-significance criterion
            consumes it so the whole verdict is reproducible from one seed.

    Returns:
        A frozen :class:`SkepticVerdict`. ``accepted`` is ``True`` only when
        *no* criterion failed. ``rejection_criteria`` names each violated
        criterion with its numbers; ``findings`` always lists every criterion
        that ran, in a fixed order. Never raises.
    """
    findings: list[AuditFinding] = [
        _safe_run(criterion.__name__, criterion, candidate) for criterion in DETERMINISTIC_CRITERIA
    ]
    findings.append(
        _safe_run(
            "skeptic.significance",
            check_statistical_significance,
            candidate,
            rng=rng,
        )
    )

    rejection_criteria = tuple(f"{f.check}: {f.message}" for f in findings if not f.passed)
    return SkepticVerdict(
        accepted=not rejection_criteria,
        findings=tuple(findings),
        rejection_criteria=rejection_criteria,
    )
