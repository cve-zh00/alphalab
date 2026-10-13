"""Unit tests for the experiment arm value types."""

from __future__ import annotations

import pytest

from alphalab.domain.experiment.arms import DecisionOutcome, ExperimentArm
from tests.unit._experiment_doubles import make_arm_result


def test_arm_result_n_vetoed_is_proposed_minus_executed() -> None:
    # Arrange
    result = make_arm_result("C", (0.01, 0.02, 0.03), n_proposed=5, n_executed=3)

    # Act / Assert
    assert result.n_vetoed == 2


def test_arm_result_is_frozen() -> None:
    # Arrange
    result = make_arm_result("A", (0.01, 0.02, 0.03))

    # Act / Assert
    with pytest.raises(Exception, match=r"frozen|Instance is frozen"):
        result.hit_rate = 0.0  # type: ignore[misc]


def test_experiment_arm_requires_nonempty_fields() -> None:
    # Arrange / Act / Assert
    with pytest.raises(Exception, match=r"at least 1 character|min_length|string_too_short"):
        ExperimentArm(name="", label="x", family="y")


def test_decision_outcome_defaults_have_no_probability() -> None:
    # Arrange / Act
    outcome = DecisionOutcome(proposed=True, executed=True, realized_return=0.01)

    # Assert
    assert outcome.predicted_prob is None
    assert outcome.was_winner is None


def test_decision_outcome_is_frozen() -> None:
    # Arrange
    outcome = DecisionOutcome(proposed=True, executed=False, realized_return=-0.02)

    # Act / Assert
    with pytest.raises(Exception, match=r"frozen|Instance is frozen"):
        outcome.executed = True  # type: ignore[misc]
