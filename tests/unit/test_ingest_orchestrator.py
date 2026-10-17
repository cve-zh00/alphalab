"""Unit tests for the ingest orchestrator (``build_request``), no network."""

from __future__ import annotations

from datetime import date

from alphalab.adapters.cik import CikResolver, CikSettings
from alphalab.adapters.edgar import EdgarClient, EdgarSettings, HttpResponse
from alphalab.adapters.edgar_live import FundamentalField
from alphalab.adapters.prices import PricePoint
from alphalab.app.ingest import (
    IngestResult,
    ValuationAssumptions,
    build_request,
)
from alphalab.domain.analyst.contracts import ModelTier, RunMetadata
from alphalab.domain.types import Ticker

# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class _FakeClock:
    def monotonic(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        return None


class _MapTransport:
    """Serves the CIK map then the companyfacts payload by URL."""

    def __init__(self, cik_map: dict[str, object], facts: dict[str, object]) -> None:
        self._cik_map = cik_map
        self._facts = facts
        self.calls: list[str] = []

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        self.calls.append(url)
        if "company_tickers" in url:
            return HttpResponse(200, self._cik_map)
        return HttpResponse(200, self._facts)


class _StubPriceSource:
    def __init__(self, points: list[PricePoint]) -> None:
        self._points = points

    def series(self, ticker: Ticker) -> list[PricePoint]:
        return list(self._points)


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #

_CIK_MAP: dict[str, object] = {
    "0": {"cik_str": 1, "ticker": "ACME", "title": "Acme Inc"},
}


def _usd(concept: str, entries: list[dict[str, object]]) -> dict[str, object]:
    return {concept: {"units": {"USD": entries}}}


def _usd_per_share(concept: str, entries: list[dict[str, object]]) -> dict[str, object]:
    return {concept: {"units": {"USD/shares": entries}}}


def _shares(concept: str, entries: list[dict[str, object]]) -> dict[str, object]:
    return {concept: {"units": {"shares": entries}}}


def _annual(
    concept_block: dict[str, object],
    concept: str,
    values: dict[str, float],
    unit: str = "USD",
    *,
    duration: bool = True,
) -> dict[str, object]:
    """Add annual facts for ``concept``, as EDGAR actually tags them.

    A *duration* fact (a flow accumulated over the year) carries ``start``; an
    *instant* fact (a balance-sheet stock) carries only ``end``. The extraction
    layer relies on exactly that distinction to tell a full year apart from a
    single quarter, so a fixture that omitted ``start`` would be testing a
    payload shape EDGAR never sends.
    """
    entries: list[dict[str, object]] = []
    for year, val in values.items():
        entry: dict[str, object] = {
            "end": f"{year}-12-31",
            "val": val,
            "filed": f"{year + 1}-02-15",
        }
        if duration:
            entry["start"] = f"{year}-01-01"
        entries.append(entry)
    builder = {"USD": _usd, "USD/shares": _usd_per_share, "shares": _shares}[unit]
    concept_block.update(builder(concept, entries))
    return concept_block


def _full_facts(*, with_dividends: bool) -> dict[str, object]:
    block: dict[str, object] = {}
    _annual(
        block, "NetCashProvidedByUsedInOperatingActivities", {2020: 100.0, 2021: 110.0, 2022: 120.0}
    )
    _annual(
        block, "PaymentsToAcquirePropertyPlantAndEquipment", {2020: 20.0, 2021: 22.0, 2022: 24.0}
    )
    _annual(block, "CommonStockSharesOutstanding", {2022: 10.0}, unit="shares", duration=False)
    _annual(block, "StockholdersEquity", {2022: 500.0}, duration=False)
    _annual(block, "LongTermDebtNoncurrent", {2022: 100.0}, duration=False)
    _annual(block, "CashAndCashEquivalentsAtCarryingValue", {2022: 50.0}, duration=False)
    _annual(block, "EarningsPerShareDiluted", {2022: 5.0}, unit="USD/shares")
    _annual(block, "OperatingIncomeLoss", {2022: 90.0})
    _annual(block, "DepreciationDepletionAndAmortization", {2022: 15.0})
    _annual(block, "Revenues", {2022: 300.0})
    _annual(block, "NetIncomeLoss", {2022: 60.0})
    _annual(block, "Assets", {2022: 800.0}, duration=False)
    _annual(block, "Liabilities", {2022: 300.0}, duration=False)
    if with_dividends:
        _annual(block, "CommonStockDividendsPerShareDeclared", {2022: 2.0}, unit="USD/shares")
    return {"facts": {"us-gaap": block}}


def _assumptions() -> ValuationAssumptions:
    return ValuationAssumptions(
        cost_of_equity=0.09,
        cost_of_debt=0.04,
        tax_rate=0.21,
        terminal_growth=0.02,
        dividend_growth=0.03,
        projection_years=5,
        peer_pe=(15.0, 18.0, 20.0),
    )


def _run_metadata() -> RunMetadata:
    return RunMetadata(
        model_name="test-model",
        tier=ModelTier.DEEP,
        llm_version="1.0",
        llm_cutoff=date(2023, 1, 1),
        knowledge_date=date(2023, 6, 30),
        dataset_hash="deadbeef",
    )


def _build(*, with_dividends: bool, tmp_path, price_points=None) -> IngestResult:
    transport = _MapTransport(_CIK_MAP, _full_facts(with_dividends=with_dividends))
    clock = _FakeClock()
    cik_resolver = CikResolver(
        CikSettings(user_agent="t t@e.com"),
        transport=transport,
        cache_path=tmp_path / "cik.json",
        clock=clock,
    )
    edgar_client = EdgarClient(
        EdgarSettings(user_agent="t t@e.com"), transport=transport, clock=clock
    )
    points = (
        price_points
        if price_points is not None
        else [
            PricePoint(on=date(2023, 6, 29), close=123.0),
            PricePoint(on=date(2023, 7, 3), close=999.0),  # future, must be ignored
        ]
    )
    return build_request(
        Ticker("ACME"),
        knowledge_date=date(2023, 6, 30),
        assumptions=_assumptions(),
        cik_resolver=cik_resolver,
        edgar_client=edgar_client,
        price_source=_StubPriceSource(points),
        pit_store_root=tmp_path / "store",
        run_metadata=_run_metadata(),
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestHappyPath:
    def test_produces_an_analyst_request(self, tmp_path) -> None:
        result = _build(with_dividends=True, tmp_path=tmp_path)
        assert result.request is not None
        assert result.request.ticker == "ACME"
        assert result.request.knowledge_date == date(2023, 6, 30)

    def test_current_price_never_from_the_future(self, tmp_path) -> None:
        result = _build(with_dividends=True, tmp_path=tmp_path)
        assert result.request is not None
        # 123.0 (2023-06-29), never 999.0 (2023-07-03).
        assert result.request.current_price == 123.0

    def test_valuation_bundle_has_multiple_methods(self, tmp_path) -> None:
        result = _build(with_dividends=True, tmp_path=tmp_path)
        assert result.request is not None
        operators = {e.operator for e in result.request.valuation.estimates}
        assert "dcf_value_per_share" in operators
        assert "ddm_value_per_share" in operators
        assert "pe_implied_value" in operators

    def test_features_resolved_through_the_gate_are_recorded(self, tmp_path) -> None:
        result = _build(with_dividends=True, tmp_path=tmp_path)
        # Every method that ran is reported; none silently assumed.
        assert "dcf" in result.methods_used
        assert "ddm" in result.methods_used
        assert "multiples_pe" in result.methods_used


class TestDegradation:
    def test_no_dividends_omits_ddm_and_records_it(self, tmp_path) -> None:
        result = _build(with_dividends=False, tmp_path=tmp_path)
        assert result.request is not None
        operators = {e.operator for e in result.request.valuation.estimates}
        assert "ddm_value_per_share" not in operators
        # Omission is explicit, not a zero.
        assert any("ddm" in reason for reason in result.omitted_methods)

    def test_missing_fundamentals_reported(self, tmp_path) -> None:
        result = _build(with_dividends=False, tmp_path=tmp_path)
        assert FundamentalField.DIVIDEND_PER_SHARE in result.missing_fields


class TestAsFiledRestatement:
    def test_knowledge_date_before_restatement_sees_original(self, tmp_path) -> None:
        # Arrange — NetIncomeLoss restated: original filed 2023-02-15 (60),
        # restated filed 2023-09-01 (40). We query as of 2023-06-30.
        block: dict[str, object] = {}
        _annual(
            block,
            "NetCashProvidedByUsedInOperatingActivities",
            {2020: 100.0, 2021: 110.0, 2022: 120.0},
        )
        _annual(
            block,
            "PaymentsToAcquirePropertyPlantAndEquipment",
            {2020: 20.0, 2021: 22.0, 2022: 24.0},
        )
        _annual(block, "CommonStockSharesOutstanding", {2022: 10.0}, unit="shares", duration=False)
        block.update(
            _usd(
                "NetIncomeLoss",
                [
                    {
                        "start": "2022-01-01",
                        "end": "2022-12-31",
                        "val": 60.0,
                        "filed": "2023-02-15",
                    },
                    {
                        "start": "2022-01-01",
                        "end": "2022-12-31",
                        "val": 40.0,
                        "filed": "2023-09-01",
                    },
                ],
            )
        )
        facts = {"facts": {"us-gaap": block}}
        transport = _MapTransport(_CIK_MAP, facts)
        clock = _FakeClock()
        cik_resolver = CikResolver(
            CikSettings(user_agent="t t@e.com"),
            transport=transport,
            cache_path=tmp_path / "cik.json",
            clock=clock,
        )
        edgar_client = EdgarClient(
            EdgarSettings(user_agent="t t@e.com"), transport=transport, clock=clock
        )
        from alphalab.adapters.pit_store import ParquetPITStore
        from alphalab.app.ingest import latest_knowable_value

        build_request(
            Ticker("ACME"),
            knowledge_date=date(2023, 6, 30),
            assumptions=_assumptions(),
            cik_resolver=cik_resolver,
            edgar_client=edgar_client,
            price_source=_StubPriceSource([PricePoint(on=date(2023, 6, 29), close=50.0)]),
            pit_store_root=tmp_path / "store",
            run_metadata=_run_metadata(),
        )

        # Assert — on 2023-06-30 only the original (60.0) was knowable.
        store = ParquetPITStore(root=tmp_path / "store")
        early = latest_knowable_value(
            store, Ticker("ACME"), FundamentalField.NET_INCOME, date(2023, 6, 30)
        )
        assert early == 60.0
        # A later knowledge date sees the restatement.
        late = latest_knowable_value(
            store, Ticker("ACME"), FundamentalField.NET_INCOME, date(2023, 12, 31)
        )
        assert late == 40.0


class TestMultiYearFiling:
    """One filing restates several years at once, and they tie on ``filed``.

    A 10-K carries the current year's figure alongside the two before it, all
    stamped with the same filing date. Ranking only by filing date leaves them
    tied and the winner falls out of whatever order the store returned — on
    real data that handed AMD its FY2023 diluted EPS of 0.53 instead of
    FY2025's 2.65: two years stale and less than half the right size.
    """

    def test_picks_the_latest_period_not_an_arbitrary_tie(self, tmp_path) -> None:
        # Arrange — three fiscal years, every one filed on the same day.
        block: dict[str, object] = {}
        block.update(
            _usd(
                "NetIncomeLoss",
                [
                    {
                        "start": f"{year}-01-01",
                        "end": f"{year}-12-31",
                        "val": val,
                        "filed": "2026-02-04",
                    }
                    for year, val in ((2023, 53.0), (2024, 100.0), (2025, 265.0))
                ],
            )
        )
        facts = {"facts": {"us-gaap": block}}
        transport = _MapTransport(_CIK_MAP, facts)
        clock = _FakeClock()
        cik_resolver = CikResolver(
            CikSettings(user_agent="t t@e.com"),
            transport=transport,
            cache_path=tmp_path / "cik.json",
            clock=clock,
        )
        edgar_client = EdgarClient(
            EdgarSettings(user_agent="t t@e.com"), transport=transport, clock=clock
        )
        from alphalab.adapters.pit_store import ParquetPITStore
        from alphalab.app.ingest import latest_knowable_value

        # Act
        build_request(
            Ticker("ACME"),
            knowledge_date=date(2026, 6, 30),
            assumptions=_assumptions(),
            cik_resolver=cik_resolver,
            edgar_client=edgar_client,
            price_source=_StubPriceSource([PricePoint(on=date(2026, 6, 29), close=50.0)]),
            pit_store_root=tmp_path / "store",
            run_metadata=_run_metadata(),
        )

        # Assert — the most recent fiscal year wins the tie, not the smallest
        # value or whichever row the store happened to yield first.
        store = ParquetPITStore(root=tmp_path / "store")
        resolved = latest_knowable_value(
            store, Ticker("ACME"), FundamentalField.NET_INCOME, date(2026, 6, 30)
        )
        assert resolved == 265.0


class TestInsufficientData:
    def test_no_methods_yields_no_request_but_a_clear_report(self, tmp_path) -> None:
        # Arrange — only a revenue fact, nothing valuation-usable.
        block: dict[str, object] = {}
        _annual(block, "Revenues", {2022: 300.0})
        facts = {"facts": {"us-gaap": block}}
        transport = _MapTransport(_CIK_MAP, facts)
        clock = _FakeClock()
        cik_resolver = CikResolver(
            CikSettings(user_agent="t t@e.com"),
            transport=transport,
            cache_path=tmp_path / "cik.json",
            clock=clock,
        )
        edgar_client = EdgarClient(
            EdgarSettings(user_agent="t t@e.com"), transport=transport, clock=clock
        )
        result = build_request(
            Ticker("ACME"),
            knowledge_date=date(2023, 6, 30),
            assumptions=_assumptions(),
            cik_resolver=cik_resolver,
            edgar_client=edgar_client,
            price_source=_StubPriceSource([PricePoint(on=date(2023, 6, 29), close=50.0)]),
            pit_store_root=tmp_path / "store",
            run_metadata=_run_metadata(),
        )
        assert result.request is None
        assert result.methods_used == ()
        assert result.omitted_methods  # explains why there is no number

    def test_ddm_skipped_when_payout_ratio_below_threshold(self, tmp_path) -> None:
        facts = _full_facts(with_dividends=True)
        transport = _MapTransport(_CIK_MAP, facts)
        clock = _FakeClock()
        cik_resolver = CikResolver(
            CikSettings(user_agent="t t@e.com"),
            transport=transport,
            cache_path=tmp_path / "cik.json",
            clock=clock,
        )
        edgar_client = EdgarClient(
            EdgarSettings(user_agent="t t@e.com"), transport=transport, clock=clock
        )
        # DPS=2.0, EPS=5.0 -> payout=40%; requiring 50% gates out DDM
        high_gate = _assumptions().model_copy(update={"min_payout_ratio": 0.50})
        result = build_request(
            Ticker("ACME"),
            knowledge_date=date(2023, 6, 30),
            assumptions=high_gate,
            cik_resolver=cik_resolver,
            edgar_client=edgar_client,
            price_source=_StubPriceSource([PricePoint(on=date(2023, 6, 29), close=50.0)]),
            pit_store_root=tmp_path / "store",
            run_metadata=_run_metadata(),
        )
        assert "ddm" not in result.methods_used
        assert any("payout ratio" in r for r in result.omitted_methods)
        assert result.request is not None
        # Verify PIT dossier qualitative text contains no digits (Contract Gate C2 safe)
        for text in result.request.documents:
            body = text.split("] ", 1)[-1]
            assert not any(ch.isdigit() for ch in body)

