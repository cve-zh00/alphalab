"""Price series source with a local CSV/Parquet-backed adapter.

Prices are kept behind the :class:`PriceSeriesSource` protocol so the rest of
the system never depends on any particular provider. ``yfinance`` is
deliberately *not* a dependency: it carries survivorship bias (delisted
tickers vanish) and is unsuitable for point-in-time research. For the acid
test, SPY is read from a local CSV if present.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

from alphalab.domain.types import Ticker


@dataclass(frozen=True)
class PricePoint:
    """A single close price on a given date."""

    on: date
    close: float
    raw_close: float | None = None


class PriceSeriesSource(Protocol):
    """Protocol for retrieving a close-price series for an instrument."""

    def series(self, ticker: Ticker) -> list[PricePoint]: ...


class CsvPriceSeriesSource:
    """Reads ``<ticker>.csv`` files (``date,close`` columns) from a directory."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def series(self, ticker: Ticker) -> list[PricePoint]:
        """Return the close series for ``ticker`` or ``[]`` if no file exists."""
        path = self._root / f"{ticker}.csv"
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [self._row_to_point(row) for row in reader]

    @staticmethod
    def _row_to_point(row: dict[str, str]) -> PricePoint:
        return PricePoint(
            on=datetime.strptime(row["date"], "%Y-%m-%d").date(),
            close=float(row["close"]),
        )
