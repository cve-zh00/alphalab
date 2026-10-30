"""Valuation uses the as-quoted close; trailing-stop peak uses the intraday high."""

from __future__ import annotations

from datetime import date

import pytest

from alphalab.adapters.prices import PricePoint
from alphalab.adapters.quotes import NoPriceAvailableError, current_valuation_price_at
from alphalab.domain.pit import KnowledgeDate
from alphalab.domain.types import Ticker

T = Ticker("DECK")


class _Src:
    def __init__(self, points: list[PricePoint]) -> None:
        self._points = points

    def series(self, ticker: Ticker) -> list[PricePoint]:
        return self._points


def test_prefers_raw_close_when_present() -> None:
    src = _Src([PricePoint(on=date(2024, 6, 3), close=10.0, raw_close=100.0)])
    assert current_valuation_price_at(src, T, KnowledgeDate(date(2024, 6, 3))) == 100.0


def test_falls_back_to_adjusted_close() -> None:
    src = _Src([PricePoint(on=date(2024, 6, 3), close=10.0)])
    assert current_valuation_price_at(src, T, KnowledgeDate(date(2024, 6, 3))) == 10.0


def test_never_reads_future_price() -> None:
    src = _Src([PricePoint(on=date(2024, 6, 5), close=10.0, raw_close=100.0)])
    with pytest.raises(NoPriceAvailableError):
        current_valuation_price_at(src, T, KnowledgeDate(date(2024, 6, 3)))


def test_peak_uses_intraday_high(tmp_path) -> None:
    from datetime import UTC, datetime

    from alphalab.adapters.alpaca_broker import BrokerPosition
    from alphalab.adapters.execution_log import (
        ExecutionEvent,
        ExecutionEventKind,
        SqliteExecutionLog,
    )
    from alphalab.app.autopilot_runner import _build_position_risks
    from alphalab.domain.backtest.types import PriceBar

    log = SqliteExecutionLog(tmp_path / "x.db")
    log.append(
        ExecutionEvent(
            cycle_id="c",
            recorded_at=datetime(2024, 6, 3, 15, tzinfo=UTC),
            kind=ExecutionEventKind.SUBMITTED,
            ticker="DECK",
            side="buy",
            shares=10.0,
            reference_price=100.0,
            client_order_id=None,
            broker_order_id=None,
            detail=None,
        )
    )
    pos = {T: BrokerPosition(ticker=T, shares=10.0, price=100.0, avg_entry_price=100.0)}
    bars = {
        T: [
            PriceBar(on=date(2024, 6, 3), close=101.0, high=110.0),
            PriceBar(on=date(2024, 6, 4), close=100.0),
        ]
    }
    risks = _build_position_risks(pos, bars, log, date(2024, 6, 4))
    assert risks[T].peak_price == 110.0


def test_peak_uses_live_intraday_high_and_persists_memory(tmp_path) -> None:
    from datetime import UTC, datetime

    from alphalab.adapters.alpaca_broker import BrokerPosition
    from alphalab.adapters.execution_log import (
        ExecutionEvent,
        ExecutionEventKind,
        SqliteExecutionLog,
    )
    from alphalab.app.autopilot_runner import _build_position_risks
    from alphalab.domain.backtest.types import PriceBar

    ual = Ticker("UAL")
    log = SqliteExecutionLog(tmp_path / "ual.db")
    log.append(
        ExecutionEvent(
            cycle_id="c",
            recorded_at=datetime(2024, 6, 3, 15, tzinfo=UTC),
            kind=ExecutionEventKind.SUBMITTED,
            ticker="UAL",
            side="buy",
            shares=10.0,
            reference_price=98.68,
            client_order_id=None,
            broker_order_id=None,
            detail=None,
        )
    )
    pos = {ual: BrokerPosition(ticker=ual, shares=10.0, price=101.98, avg_entry_price=98.68)}
    bars = {ual: [PriceBar(on=date(2024, 6, 3), close=100.0, high=102.75)]}
    memory: dict[tuple[Ticker, date], float] = {}

    # First cycle sees today's live intraday high $104.97 even though bars end yesterday
    r1 = _build_position_risks(
        pos,
        bars,
        log,
        date(2024, 6, 4),
        intraday_highs={ual: 104.97},
        peak_memory=memory,
    )
    assert r1[ual].peak_price == pytest.approx(104.97)
    assert r1[ual].profit_giveback_pct == pytest.approx(47.53577, rel=1e-3)

    # Next cycle: even if snapshot call fails (empty intraday_highs), memory retains $104.97
    r2 = _build_position_risks(
        pos,
        bars,
        log,
        date(2024, 6, 4),
        intraday_highs={},
        peak_memory=memory,
    )
    assert r2[ual].peak_price == pytest.approx(104.97)
