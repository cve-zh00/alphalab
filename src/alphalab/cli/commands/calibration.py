"""The ``calibration`` command: read the journal and show its calibration.

Read-only. It opens the SQLite journal, pairs each settled prediction's
declared probability with its realized outcome, and runs the pure
:func:`calibration_report` to print Brier, ECE, hit rate, signed
overconfidence and the reliability diagram. If the journal is empty or missing
it says so plainly instead of failing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from alphalab.domain.journal import CalibrationReport, calibration_report
from alphalab.report.theme import Theme, theme_for


@dataclass(frozen=True)
class CalibrationOptions:
    """Resolved options for one ``calibration`` invocation."""

    db_path: Path
    as_json: bool
    no_color: bool


def run_calibration(opts: CalibrationOptions, *, out: TextIO, err: TextIO) -> int:
    """Execute the calibration command and return a process exit code."""
    if not opts.db_path.exists():
        print(
            f"error: no journal database at {opts.db_path}. "
            "Run some monthly cadences first to accumulate settled predictions.",
            file=err,
        )
        return 2
    try:
        pairs = _load_pairs(opts.db_path)
    except Exception as exc:  # pragma: no cover - defensive read boundary
        print(f"error: could not read journal: {type(exc).__name__}: {exc}", file=err)
        return 1

    if not pairs:
        print("No settled predictions in the journal yet; nothing to calibrate.", file=out)
        return 0

    probabilities = [p for p, _ in pairs]
    outcomes = [o for _, o in pairs]
    report = calibration_report(probabilities, outcomes)

    if opts.as_json:
        print(_as_json(report), file=out)
        return 0

    theme = theme_for(out, force_disable=opts.no_color)
    print(_format_report(report, theme), file=out)
    return 0


def _load_pairs(db_path: Path) -> list[tuple[float, int]]:
    """Return ``(probability, outcome)`` pairs for settled predictions."""
    from alphalab.adapters.journal_sqlite import SqliteJournal

    journal = SqliteJournal(db_path)
    try:
        predictions = {p.prediction_id: p for p in journal.list_predictions()}
        settlements = journal.list_settlements()
    finally:
        journal.close()
    pairs: list[tuple[float, int]] = []
    for settlement in settlements:
        prediction = predictions.get(settlement.prediction_id)
        if prediction is None:
            continue
        pairs.append((prediction.probability, 1 if settlement.outcome else 0))
    return pairs


def _format_report(report: CalibrationReport, theme: Theme) -> str:
    lines = [theme.bold("CALIBRATION REPORT")]
    status = "conclusive" if report.conclusive else "NOT conclusive (small sample)"
    style = "green" if report.conclusive else "yellow"
    lines.append(f"  Sample size: {report.sample_size} — {theme.paint(status, style)}")
    lines.append(f"  Brier score: {report.brier:.4f}  (lower is better; 0.25 = coin)")
    lines.append(f"  ECE: {report.ece:.4f}  (expected calibration error)")
    lines.append(f"  Hit rate: {report.hit_rate:.4f}")
    over = report.overconfidence
    label = "overconfident" if over > 0 else "underconfident" if over < 0 else "balanced"
    lines.append(f"  Overconfidence: {over:+.4f}  ({label})")
    lines.append(theme.bold("  Reliability diagram:"))
    if not report.reliability:
        lines.append("    (no populated bins)")
    for b in report.reliability:
        lines.append(
            f"    [{b.lower:.1f},{b.upper:.1f}) n={b.count:<4d} "
            f"conf={b.mean_confidence:.3f} acc={b.accuracy:.3f} gap={b.gap:+.3f}"
        )
    return "\n".join(lines)


def _as_json(report: CalibrationReport) -> str:
    payload = {
        "sample_size": report.sample_size,
        "conclusive": report.conclusive,
        "brier": report.brier,
        "ece": report.ece,
        "hit_rate": report.hit_rate,
        "overconfidence": report.overconfidence,
        "reliability": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "mean_confidence": b.mean_confidence,
                "accuracy": b.accuracy,
                "gap": b.gap,
            }
            for b in report.reliability
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)
