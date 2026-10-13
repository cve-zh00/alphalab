"""Unit tests for the pure analyst domain contracts.

Covers: immutability of the frozen models, the bounded confidence on
``QualitativeFinding``, the mandatory nature of every ``RunMetadata`` field
(a run without ``llm_cutoff`` cannot be built), and — the structural heart of
the phase — that ``ThesisNarrative`` has *no* price-target field in which a
model could emit a valuation number.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from alphalab.artifact.models import Verdict
from alphalab.domain.analyst import (
    Direction,
    ModelTier,
    QualitativeFinding,
    RunMetadata,
    ThesisNarrative,
)


def _finding() -> QualitativeFinding:
    return QualitativeFinding(
        category="regulatory_risk",
        direction=Direction.NEGATIVE,
        evidence_quote="New capital rules take effect in 2025.",
        source="10-K FY2023",
        confidence=0.8,
    )


def _metadata(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model_name": "qwen3.8:27b-mlx",
        "tier": ModelTier.DEEP,
        "llm_version": "qwen3.8-27b-mlx",
        "llm_cutoff": date(2024, 10, 1),
        "knowledge_date": date(2025, 1, 15),
        "dataset_hash": "abc123",
    }
    base.update(overrides)
    return base


class TestQualitativeFinding:
    def test_is_frozen(self) -> None:
        # Arrange
        finding = _finding()
        # Act / Assert
        with pytest.raises(ValidationError):
            finding.confidence = 0.1  # type: ignore[misc]

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -5.0])
    def test_confidence_is_bounded(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            QualitativeFinding(
                category="x",
                direction=Direction.NEUTRAL,
                evidence_quote="q",
                source="s",
                confidence=bad,
            )

    def test_rejects_extra_numeric_fields(self) -> None:
        # The schema forbids extras, so a model cannot smuggle a numeric
        # valuation field like ``price_target`` into a finding.
        with pytest.raises(ValidationError):
            QualitativeFinding(
                category="x",
                direction=Direction.NEUTRAL,
                evidence_quote="q",
                source="s",
                confidence=0.5,
                price_target=100.0,  # type: ignore[call-arg]
            )


class TestThesisNarrative:
    def test_has_no_price_target_field(self) -> None:
        # Structural guarantee: the narrative schema exposes only text + verdict.
        assert "price_target" not in ThesisNarrative.model_fields
        assert "value" not in ThesisNarrative.model_fields
        assert "fair_value" not in ThesisNarrative.model_fields
        assert set(ThesisNarrative.model_fields) == {
            "thesis",
            "risks",
            "catalysts",
            "verdict",
        }

    def test_rejects_injected_numeric_field(self) -> None:
        with pytest.raises(ValidationError):
            ThesisNarrative(
                thesis="t",
                risks="r",
                catalysts="c",
                verdict=Verdict.BUY,
                price_target=123.0,  # type: ignore[call-arg]
            )

    def test_is_frozen(self) -> None:
        narrative = ThesisNarrative(thesis="t", risks="r", catalysts="c", verdict=Verdict.HOLD)
        with pytest.raises(ValidationError):
            narrative.thesis = "mutated"  # type: ignore[misc]


class TestRunMetadata:
    def test_builds_with_all_fields(self) -> None:
        meta = RunMetadata(**_metadata())  # type: ignore[arg-type]
        assert meta.llm_cutoff == date(2024, 10, 1)
        assert meta.model_name == "qwen3.8:27b-mlx"

    def test_missing_cutoff_is_rejected(self) -> None:
        # The inviolable §6 rule: a run without a pinned cutoff cannot exist.
        fields = _metadata()
        del fields["llm_cutoff"]
        with pytest.raises(ValidationError):
            RunMetadata(**fields)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "missing",
        ["model_name", "llm_version", "llm_cutoff", "knowledge_date", "dataset_hash", "tier"],
    )
    def test_every_field_is_mandatory(self, missing: str) -> None:
        fields = _metadata()
        del fields[missing]
        with pytest.raises(ValidationError):
            RunMetadata(**fields)  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        meta = RunMetadata(**_metadata())  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            meta.dataset_hash = "other"  # type: ignore[misc]
