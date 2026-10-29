"""Unit tests for Fase 5 Quant Audit Addendum: FILLED ledger, LLM cache, and broker fills."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from alphalab.adapters.alpaca_broker import AlpacaBroker, BrokerFilledOrder, BrokerPosition
from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.edgar import HttpResponse
from alphalab.adapters.execution_log import (
    ExecutionEvent,
    ExecutionEventKind,
    SqliteExecutionLog,
)
from alphalab.adapters.llm_verdict_cache import (
    LlmVerdictCache,
    compute_request_fingerprint,
)
from alphalab.app.autopilot_runner import _build_position_risks, _latest_entries
from alphalab.app.inputs import AnalystRequest, ValuationBundle
from alphalab.artifact import EquityResearchArtifact, PriceTarget, Verdict
from alphalab.domain.analyst.contracts import RunMetadata
from alphalab.domain.analyst.tiers import ModelTier
from alphalab.domain.compute import ProvenancedValue
from alphalab.domain.types import Side, Ticker


class _FakeTransport:
    def __init__(self, responses: dict[str, HttpResponse]) -> None:
        self._responses = responses

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        for key, resp in self._responses.items():
            if url.endswith(key):
                return resp
        return HttpResponse(status_code=200, json=[])

    def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, object]
    ) -> HttpResponse:
        return HttpResponse(status_code=200, json={})


def _sample_request(*, price: float = 100.0, doc: str = "Q3 8-K beat") -> AnalystRequest:
    return AnalystRequest(
        ticker="HCA",
        knowledge_date=date(2026, 10, 28),
        current_price=price,
        documents=(doc,),
        valuation=ValuationBundle(
            estimates=(
                ProvenancedValue(
                    value=150.0,
                    unit="USD_per_share",
                    operator="dcf",
                    operator_version="1.0",
                    inputs=(),
                ),
            ),
            unit="USD_per_share",
        ),
        run_metadata=RunMetadata(
            model_name="qwen3",
            tier=ModelTier.DEEP,
            llm_version="1.0",
            llm_cutoff=date(2024, 10, 1),
            knowledge_date=date(2026, 10, 28),
            dataset_hash="edgar-pit-v1",
        ),
    )


def test_execution_log_append_fill_if_new_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "exec.db"
    with SqliteExecutionLog(db_path) as log:
        ev = ExecutionEvent(
            cycle_id="c1",
            recorded_at=datetime(2026, 10, 28, 15, 0, tzinfo=UTC),
            kind=ExecutionEventKind.FILLED,
            ticker="HCA",
            side="buy",
            shares=17.845,
            reference_price=463.06,
            client_order_id="coid-1",
            broker_order_id="boid-123",
            detail="broker fill synced",
        )
        assert log.append_fill_if_new(ev) is True
        assert log.append_fill_if_new(ev) is False
        assert log.count() == 1


def test_alpaca_broker_filled_orders_parses_closed_fills() -> None:
    settings = AlpacaSettings(api_key="k", api_secret="s")
    transport = _FakeTransport(
        {
            "/v2/orders?status=closed&limit=50": HttpResponse(
                status_code=200,
                json=[
                    {
                        "id": "boid-2",
                        "client_order_id": "coid-2",
                        "symbol": "PHM",
                        "side": "sell",
                        "status": "filled",
                        "filled_qty": "59.8545",
                        "filled_avg_price": "119.07",
                        "filled_at": "2026-10-28T18:55:10Z",
                    },
                    {
                        "id": "boid-1",
                        "client_order_id": "coid-1",
                        "symbol": "HCA",
                        "side": "buy",
                        "status": "filled",
                        "filled_qty": "17.845",
                        "filled_avg_price": "463.06",
                        "filled_at": "2026-10-28T15:06:33Z",
                    },
                    {
                        "id": "boid-3",
                        "symbol": "TFC",
                        "side": "buy",
                        "status": "canceled",
                        "filled_qty": "0",
                        "filled_avg_price": "0",
                    },
                ],
            )
        }
    )
    broker = AlpacaBroker(settings, transport=transport)
    fills = broker.filled_orders(limit=50)
    assert len(fills) == 2
    assert fills[0] == BrokerFilledOrder(
        ticker=Ticker("HCA"),
        side=Side.BUY,
        filled_qty=17.845,
        filled_avg_price=463.06,
        filled_at=datetime(2026, 10, 28, 15, 6, 33, tzinfo=UTC),
        client_order_id="coid-1",
        broker_order_id="boid-1",
    )
    assert fills[1].ticker == Ticker("PHM")
    assert fills[1].side is Side.SELL


def test_latest_entries_prefers_filled_events_over_stale_submitted(tmp_path: Path) -> None:
    db_path = tmp_path / "exec.db"
    with SqliteExecutionLog(db_path) as log:
        # Stale unfilled submission on Oct 15
        log.append(
            ExecutionEvent(
                cycle_id="c0",
                recorded_at=datetime(2026, 10, 15, 14, 0, tzinfo=UTC),
                kind=ExecutionEventKind.SUBMITTED,
                ticker="BKNG",
                side="buy",
                shares=36.7,
                reference_price=140.0,
                client_order_id="c0-bkng",
                broker_order_id="b0",
                detail="submitted",
            )
        )
        # Actual fill on Oct 28
        log.append_fill_if_new(
            ExecutionEvent(
                cycle_id="c1",
                recorded_at=datetime(2026, 10, 28, 14, 46, tzinfo=UTC),
                kind=ExecutionEventKind.FILLED,
                ticker="BKNG",
                side="buy",
                shares=36.7154,
                reference_price=147.80,
                client_order_id="c1-bkng",
                broker_order_id="b1",
                detail="broker fill synced",
            )
        )
        entries = _latest_entries(log, held_shares={Ticker("BKNG"): 36.7154})
        assert entries[Ticker("BKNG")] == (date(2026, 10, 28), pytest.approx(147.80))

        # And _build_position_risks attaches fair_value_upside_pct and ignores same-day live_high
        risks = _build_position_risks(
            {
                Ticker("BKNG"): BrokerPosition(
                    ticker=Ticker("BKNG"),
                    shares=36.7154,
                    price=148.19,
                    avg_entry_price=147.80,
                )
            },
            {},
            log,
            date(2026, 10, 28),
            intraday_highs={Ticker("BKNG"): 159.99},
            fair_value_upsides={Ticker("BKNG"): 42.5},
        )
        snap = risks[Ticker("BKNG")]
        assert snap.peak_price == pytest.approx(148.19)
        assert snap.fair_value_upside_pct == pytest.approx(42.5)


def test_llm_verdict_cache_reuses_same_dossier_and_invalidates_on_change(
    tmp_path: Path,
) -> None:
    cache = LlmVerdictCache(tmp_path / "llm_cache.json")
    req1 = _sample_request(price=100.0, doc="Q3 8-K beat")
    fp1 = compute_request_fingerprint(req1, llm_version="qwen3", debate=True)
    assert cache.get(fp1) is None

    art = EquityResearchArtifact(
        ticker="HCA",
        knowledge_date=date(2026, 10, 28),
        verdict=Verdict.BUY,
        current_price=100.0,
        price_target=PriceTarget(value=150.0, unit="USD_per_share"),
        methods_used=("dcf",),
        single_method_disclosed=True,
    )
    cache.put(fp1, art)

    # Small price move (+0.4%) stays in same bucket -> cache hit
    req_small_move = _sample_request(price=100.4, doc="Q3 8-K beat")
    fp_small = compute_request_fingerprint(req_small_move, llm_version="qwen3", debate=True)
    assert fp_small == fp1
    reloaded = LlmVerdictCache(tmp_path / "llm_cache.json").get(fp_small)
    assert reloaded is not None
    assert reloaded.verdict is Verdict.BUY

    # Large price move (+6%) or new 8-K changes fingerprint -> cache miss
    fp_big_move = compute_request_fingerprint(
        _sample_request(price=106.0, doc="Q3 8-K beat"),
        llm_version="qwen3",
        debate=True,
    )
    fp_new_doc = compute_request_fingerprint(
        _sample_request(price=100.0, doc="Q3 8-K beat + new FDA filing"),
        llm_version="qwen3",
        debate=True,
    )
    assert fp_big_move != fp1
    assert fp_new_doc != fp1
    assert cache.get(fp_big_move) is None
    assert cache.get(fp_new_doc) is None
