"""Unit tests for :mod:`alphalab.domain.types`."""

import pytest

from alphalab.domain.types import Side, Ticker


def test_ticker_is_usable_as_str() -> None:
    # Arrange
    raw = "AAPL"

    # Act
    ticker = Ticker(raw)

    # Assert
    assert ticker == raw


class TestSide:
    def test_members(self) -> None:
        # Arrange / Act / Assert
        assert Side.BUY.value == "buy"
        assert Side.SELL.value == "sell"

    @pytest.mark.parametrize(
        ("side", "expected"),
        [(Side.BUY, 1), (Side.SELL, -1)],
    )
    def test_sign(self, side: Side, expected: int) -> None:
        # Act
        result = side.sign

        # Assert
        assert result == expected

    @pytest.mark.parametrize(
        ("side", "expected"),
        [(Side.BUY, Side.SELL), (Side.SELL, Side.BUY)],
    )
    def test_opposite(self, side: Side, expected: Side) -> None:
        # Act
        result = side.opposite()

        # Assert
        assert result is expected
