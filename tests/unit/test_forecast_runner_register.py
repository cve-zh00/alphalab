"""Tests for registering research artifacts as forward predictions.

The registration half of the forecast runner is where the anti-leakage gate
lives, so these tests weigh heaviest on that invariant: a knowledge date the
model could already have seen must be refused before anything is written to the
append-only journal.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from alphalab.adapters.abstention_store import SqliteAbstentionStore
from alphalab.adapters.journal_sqlite import SqliteJournal
from alphalab.app.forecast_runner import LeakageError, register_forecast
from alphalab.artifact.models import EquityResearchArtifact, PriceTarget, Verdict
from alphalab.domain.forecast import Abstention
from alphalab.domain.journal import Direction, Prediction

_KD = date(2026, 10, 13)
_HORIZON = date(2027, 1, 13)
_NOW = datetime(2026, 10, 13, 22, 0, tzinfo=UTC)
_CUTOFF_BEFORE = date(2026, 1, 1)  # model trained well before the knowledge date
_CUTOFF_AFTER = date(2026, 12, 1)  # model trained *after* -> leakage


@pytest.fixture
def journal(tmp_path: Path) -> Iterator[SqliteJournal]:
    with SqliteJournal(tmp_path / "journal.db") as j:
        yield j


@pytest.fixture
def abstentions(tmp_path: Path) -> Iterator[SqliteAbstentionStore]:
    with SqliteAbstentionStore(tmp_path / "abstentions.db") as store:
        yield store


def _artifact(
    *,
    verdict: Verdict = Verdict.BUY,
    current_price: float = 200.0,
    target: float | None = 240.0,
    relative_span: float | None = 0.20,
    valuation_withheld: bool = False,
    knowledge_date: date | None = _KD,
    ticker: str = "AAPL",
) -> EquityResearchArtifact:
    price_target = PriceTarget(value=target, unit="USD/share") if target is not None else None
    return EquityResearchArtifact(
        ticker=ticker,
        knowledge_date=knowledge_date,
        verdict=verdict,
        price_target=price_target,
        current_price=current_price,
        consensus_relative_span=relative_span,
        valuation_withheld=valuation_withheld,
    )


class TestAntiLeakage:
    def test_knowledge_date_after_cutoff_is_allowed(
        self, journal: SqliteJournal, abstentions: SqliteAbstentionStore
    ) -> None:
        """A forecast dated after the model's cutoff is a genuine prediction."""
        result = register_forecast(
            _artifact(),
            horizon=_HORIZON,
            recorded_at=_NOW,
            journal=journal,
            abstentions=abstentions,
            llm_version="qwen3",
            llm_cutoff=_CUTOFF_BEFORE,
        )
        assert isinstance(result, Prediction)
        assert journal.get_prediction(result.prediction_id) is not None

    def test_knowledge_date_before_cutoff_is_refused(
        self, journal: SqliteJournal, abstentions: SqliteAbstentionStore
    ) -> None:
        """A knowledge date the model already saw is an oracle, and is rejected."""
        with pytest.raises(LeakageError, match="hindsight"):
            register_forecast(
                _artifact(),
                horizon=_HORIZON,
                recorded_at=_NOW,
                journal=journal,
                abstentions=abstentions,
                llm_version="qwen3",
                llm_cutoff=_CUTOFF_AFTER,
            )

    def test_knowledge_date_equal_to_cutoff_is_refused(
        self, journal: SqliteJournal, abstentions: SqliteAbstentionStore
    ) -> None:
        """The boundary is closed: equal dates are contaminated too."""
        with pytest.raises(LeakageError):
            register_forecast(
                _artifact(),
                horizon=_HORIZON,
                recorded_at=_NOW,
                journal=journal,
                abstentions=abstentions,
                llm_cutoff=_KD,
            )

    def test_refusal_writes_nothing(
        self, journal: SqliteJournal, abstentions: SqliteAbstentionStore
    ) -> None:
        """Leakage is caught *before* persistence; the record stays clean."""
        with pytest.raises(LeakageError):
            register_forecast(
                _artifact(),
                horizon=_HORIZON,
                recorded_at=_NOW,
                journal=journal,
                abstentions=abstentions,
                llm_cutoff=_CUTOFF_AFTER,
            )
        assert journal.list_predictions() == []
        assert abstentions.count() == 0

    def test_absent_cutoff_disables_the_gate(
        self, journal: SqliteJournal, abstentions: SqliteAbstentionStore
    ) -> None:
        """With no declared cutoff there is no leakage claim to enforce."""
        result = register_forecast(
            _artifact(),
            horizon=_HORIZON,
            recorded_at=_NOW,
            journal=journal,
            abstentions=abstentions,
            llm_cutoff=None,
        )
        assert isinstance(result, Prediction)


class TestVerdictMapping:
    def test_buy_becomes_an_upward_prediction(
        self, journal: SqliteJournal, abstentions: SqliteAbstentionStore
    ) -> None:
        result = register_forecast(
            _artifact(verdict=Verdict.BUY, target=240.0),
            horizon=_HORIZON,
            recorded_at=_NOW,
            journal=journal,
            abstentions=abstentions,
            llm_cutoff=_CUTOFF_BEFORE,
        )
        assert isinstance(result, Prediction)
        assert result.direction is Direction.UP
        assert result.entry_price == 200.0
        assert len(journal.list_predictions()) == 1
        assert abstentions.count() == 0

    def test_sell_becomes_a_downward_prediction(
        self, journal: SqliteJournal, abstentions: SqliteAbstentionStore
    ) -> None:
        result = register_forecast(
            _artifact(verdict=Verdict.SELL, target=160.0),
            horizon=_HORIZON,
            recorded_at=_NOW,
            journal=journal,
            abstentions=abstentions,
            llm_cutoff=_CUTOFF_BEFORE,
        )
        assert isinstance(result, Prediction)
        assert result.direction is Direction.DOWN

    def test_hold_abstains_and_persists_to_the_abstention_store(
        self, journal: SqliteJournal, abstentions: SqliteAbstentionStore
    ) -> None:
        """A HOLD has no directional claim; it is recorded as an abstention, not a prediction."""
        result = register_forecast(
            _artifact(verdict=Verdict.HOLD),
            horizon=_HORIZON,
            recorded_at=_NOW,
            journal=journal,
            abstentions=abstentions,
            llm_cutoff=_CUTOFF_BEFORE,
        )
        assert isinstance(result, Abstention)
        assert journal.list_predictions() == []
        stored = abstentions.list_abstentions()
        assert len(stored) == 1
        assert stored[0].ticker == "AAPL"
        assert stored[0].knowledge_date == _KD
        assert "directional" in stored[0].reason

    def test_withheld_valuation_still_predicts_at_base_rate(
        self, journal: SqliteJournal, abstentions: SqliteAbstentionStore
    ) -> None:
        """A gated name is scored at even odds, not dropped from the record."""
        result = register_forecast(
            _artifact(verdict=Verdict.BUY, target=None, valuation_withheld=True),
            horizon=_HORIZON,
            recorded_at=_NOW,
            journal=journal,
            abstentions=abstentions,
            llm_cutoff=_CUTOFF_BEFORE,
        )
        assert isinstance(result, Prediction)
        assert result.probability == pytest.approx(0.50)

    def test_nan_current_price_abstains_instead_of_crashing(
        self, journal: SqliteJournal, abstentions: SqliteAbstentionStore
    ) -> None:
        """Missing price at knowledge_date records an abstention rather than crashing."""
        result = register_forecast(
            _artifact(verdict=Verdict.BUY, current_price=float("nan"), target=240.0),
            horizon=_HORIZON,
            recorded_at=_NOW,
            journal=journal,
            abstentions=abstentions,
            llm_cutoff=_CUTOFF_BEFORE,
        )
        assert isinstance(result, Abstention)
        assert "current price unavailable" in result.reason
        assert journal.list_predictions() == []
        assert abstentions.count() == 1


class TestKnowledgeDateGuard:
    def test_missing_knowledge_date_is_rejected(
        self, journal: SqliteJournal, abstentions: SqliteAbstentionStore
    ) -> None:
        """A forecast with no knowledge date cannot be dated or leak-checked."""
        with pytest.raises(ValueError, match="knowledge_date"):
            register_forecast(
                _artifact(knowledge_date=None),
                horizon=_HORIZON,
                recorded_at=_NOW,
                journal=journal,
                abstentions=abstentions,
            )
