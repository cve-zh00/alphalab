"""DEMO: the journal measuring its own miscalibration.

This test registers ~30 predictions with declared probabilities, settles them
with deliberately constructed outcomes so the system is **overconfident**, and
produces the full calibration report. It is the proof that the journal can
detect its own miscalibration — the piece that makes the track record honest.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from alphalab.adapters.journal_sqlite import SqliteJournal
from alphalab.domain.journal import (
    Direction,
    Prediction,
    Settlement,
    TemplateReflector,
    calibration_report,
    due_predictions,
    settle_prediction,
)

# Thirty predictions, all declared at HIGH confidence (0.90) but only ~40% of
# them actually correct. A badly overconfident system by construction.
_N = 30
_DECLARED_P = 0.90
_HITS = 12  # 40% accuracy against 90% declared confidence.


@pytest.fixture
def journal(tmp_path: Path) -> Iterator[SqliteJournal]:
    with SqliteJournal(tmp_path / "demo.db") as j:
        yield j


def _make_predictions() -> list[Prediction]:
    return [
        Prediction(
            prediction_id=f"pred-{i:02d}",
            recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
            ticker="AAPL",
            direction=Direction.UP,
            probability=_DECLARED_P,
            horizon=date(2026, 2, 1),
            entry_price=150.0,
            knowledge_date=date(2026, 1, 1),
            llm_version="frontier-2026-09",
            llm_cutoff=date(2026, 9, 1),
        )
        for i in range(_N)
    ]


def _realized_for(index: int) -> float:
    """Construct realized returns so exactly ``_HITS`` predictions win.

    Predicted direction is UP; the first ``_HITS`` go up (win), the rest go
    down (loss). The outcome mix is therefore known exactly.
    """
    return 0.05 if index < _HITS else -0.05


class TestCalibrationDemo:
    def test_overconfidence_is_detected(self, journal: SqliteJournal) -> None:
        # Arrange: record 30 high-confidence predictions.
        predictions = _make_predictions()
        for pred in predictions:
            journal.record(pred)

        # Act: a later run settles every due prediction with known outcomes.
        due = due_predictions(
            journal.list_predictions(),
            settled_ids=journal.settled_ids(),
            as_of=date(2026, 3, 1),
        )
        reflector = TemplateReflector()
        for i, pred in enumerate(due):
            settlement: Settlement = settle_prediction(
                pred,
                realized_return=_realized_for(i),
                benchmark_return=0.0,
                cost_drag=0.001,
                settled_at=datetime(2026, 2, 1, tzinfo=UTC),
                as_of=date(2026, 3, 1),
            )
            journal.settle(settlement)
            journal.add_lesson(reflector.reflect(pred, settlement))

        # Build the calibration report from the settled record.
        settlements = journal.list_settlements()
        probs = [_DECLARED_P] * len(settlements)
        outcomes = [1 if s.outcome else 0 for s in settlements]
        report = calibration_report(probs, outcomes)

        # Assert: the system is measurably overconfident.
        assert report.sample_size == _N
        assert report.conclusive is True
        assert report.hit_rate == pytest.approx(_HITS / _N)  # 0.40
        # Brier for a constant p=0.9 predictor: mean((0.9-o)^2)
        #   hits:  (0.9-1)^2 = 0.01   (12 of them)
        #   misses:(0.9-0)^2 = 0.81   (18 of them)
        expected_brier = (_HITS * 0.01 + (_N - _HITS) * 0.81) / _N
        assert report.brier == pytest.approx(expected_brier)
        # All predictions sit in one bin (conf 0.9, acc 0.4) -> ECE = 0.5.
        assert report.ece == pytest.approx(abs(0.9 - (_HITS / _N)))
        # Signed diagnostic positive => OVERCONFIDENT.
        assert report.overconfidence > 0.0
        assert report.overconfidence == pytest.approx(0.9 - (_HITS / _N))
        # Reliability diagram: single populated bin summing to the sample.
        assert sum(b.count for b in report.reliability) == _N
        assert len(report.reliability) == 1

        # Emit the numbers for the record (visible with -s).
        print("\n=== CALIBRATION DEMO REPORT ===")
        print(f"sample_size       : {report.sample_size}")
        print(f"conclusive        : {report.conclusive}")
        print(f"brier             : {report.brier:.4f}")
        print(f"ece               : {report.ece:.4f}")
        print(f"hit_rate          : {report.hit_rate:.4f}")
        print(f"overconfidence    : {report.overconfidence:+.4f} (positive => overconfident)")
        print(
            "decomposition     : "
            f"reliability={report.decomposition.reliability:.4f} "
            f"resolution={report.decomposition.resolution:.4f} "
            f"uncertainty={report.decomposition.uncertainty:.4f}"
        )
        print("reliability diagram (bins):")
        for b in report.reliability:
            print(
                f"  [{b.lower:.1f},{b.upper:.1f}] n={b.count} "
                f"conf={b.mean_confidence:.3f} acc={b.accuracy:.3f} gap={b.gap:+.3f}"
            )
