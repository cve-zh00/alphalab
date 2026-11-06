"""Unit tests for the live EDGAR wiring seam (:mod:`alphalab.cli.ingest`).

No real network is touched: a fake ``Transport`` serves the SEC CIK map and a
``companyfacts`` payload, and the on-disk CIK cache / PIT store roots are
redirected to a temporary directory. These tests pin the three behaviours the
CLI owner is responsible for at this seam:

* a provisioned builder returns a real :class:`AnalystRequest`;
* a run that yields no usable valuation raises
  :class:`LiveIngestUnavailableError` carrying the *concrete* omission reasons
  (never an invented number);
* a missing SEC ``User-Agent`` raises :class:`LiveIngestConfigError` whose
  message names the exact environment variable to export, and the CLI maps that
  to exit code 2 with no stack trace.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import alphalab.cli.ingest as ingest_mod
from alphalab.adapters.edgar import HttpResponse
from alphalab.app.inputs import AnalystRequest
from alphalab.cli.commands.research import ResearchOptions, run_research
from alphalab.cli.ingest import (
    LiveIngestConfigError,
    LiveIngestUnavailableError,
    load_live_builder,
)

_USER_AGENT_ENV = "ALPHALAB_EDGAR_USER_AGENT"
_KNOWLEDGE_DATE = date(2023, 6, 30)


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class _FakeTransport:
    """Serves the CIK map then a ``companyfacts`` payload, keyed by URL.

    Mirrors the real :class:`~alphalab.adapters.edgar.Transport` protocol
    (``get`` only) so no socket is ever opened.
    """

    def __init__(self, cik_map: dict[str, object], facts: dict[str, object]) -> None:
        self._cik_map = cik_map
        self._facts = facts
        self.urls: list[str] = []

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        self.urls.append(url)
        if "company_tickers" in url:
            return HttpResponse(status_code=200, json=self._cik_map)
        return HttpResponse(status_code=200, json=self._facts)


_CIK_MAP: dict[str, object] = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}


def _usd(concept: str, entries: list[dict[str, object]]) -> dict[str, object]:
    return {concept: {"units": {"USD": entries}}}


def _shares(concept: str, entries: list[dict[str, object]]) -> dict[str, object]:
    return {concept: {"units": {"shares": entries}}}


def _annual_usd(
    concept: str, values: dict[int, float], *, shares: bool = False, duration: bool = True
) -> dict[str, object]:
    """Annual facts as EDGAR tags them: flows carry ``start``, instants do not.

    Extraction uses that span to tell a full year from a single quarter, so a
    flow fixture without ``start`` would describe a payload EDGAR never sends.
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
    return _shares(concept, entries) if shares else _usd(concept, entries)


def _full_facts() -> dict[str, object]:
    """A minimal but valuation-sufficient us-gaap block (DCF + P/E can run)."""
    block: dict[str, object] = {}
    block.update(
        _annual_usd(
            "NetCashProvidedByUsedInOperatingActivities",
            {2020: 100.0, 2021: 110.0, 2022: 120.0},
        )
    )
    block.update(
        _annual_usd(
            "PaymentsToAcquirePropertyPlantAndEquipment",
            {2020: 20.0, 2021: 22.0, 2022: 24.0},
        )
    )
    block.update(
        _annual_usd("CommonStockSharesOutstanding", {2022: 10.0}, shares=True, duration=False)
    )
    block.update(_annual_usd("StockholdersEquity", {2022: 500.0}, duration=False))
    block.update(
        {
            "EarningsPerShareDiluted": {
                "units": {
                    "USD/shares": [
                        {
                            "start": "2022-01-01",
                            "end": "2022-12-31",
                            "val": 5.0,
                            "filed": "2023-02-15",
                        },
                    ]
                }
            }
        }
    )
    return {"facts": {"us-gaap": block}}


def _empty_facts() -> dict[str, object]:
    """Only a revenue fact — nothing a valuation method can consume."""
    return {"facts": {"us-gaap": _annual_usd("Revenues", {2022: 300.0})}}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def with_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a valid identifying SEC User-Agent for the duration of a test."""
    monkeypatch.setenv(_USER_AGENT_ENV, "alphalab-tests tests@example.com")


@pytest.fixture
def redirected_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the CIK cache, PIT store and price source at a temp directory.

    Keeps every file this seam writes inside ``tmp_path`` so the test never
    touches the repo's real ``data/`` tree.
    """
    monkeypatch.setattr(ingest_mod, "_cik_cache_path", lambda: tmp_path / "cik.json")
    monkeypatch.setattr(ingest_mod, "_pit_store_root", lambda: tmp_path / "pit")
    monkeypatch.setattr(
        ingest_mod, "_price_source", lambda: ingest_mod.CsvPriceSeriesSource(tmp_path / "prices")
    )


def _install_transport(monkeypatch: pytest.MonkeyPatch, transport: _FakeTransport) -> None:
    """Replace the real httpx transport with ``transport`` (no network)."""
    monkeypatch.setattr(ingest_mod, "HttpxWriteTransport", lambda *a, **k: transport)


# --------------------------------------------------------------------------- #
# Tests: provisioned builder returns a real request
# --------------------------------------------------------------------------- #


class TestProvisionedBuilder:
    def test_returns_an_analyst_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
        with_user_agent: None,
        redirected_paths: None,
    ) -> None:
        # Arrange
        _install_transport(monkeypatch, _FakeTransport(_CIK_MAP, _full_facts()))

        # Act
        builder = load_live_builder()
        request = builder("AAPL", knowledge_date=_KNOWLEDGE_DATE)

        # Assert
        assert isinstance(request, AnalystRequest)
        assert request.ticker == "AAPL"
        assert request.knowledge_date == _KNOWLEDGE_DATE
        # At least one method ran on the real compute operators.
        assert request.valuation.estimates

    def test_uses_generic_uncalibrated_assumptions(
        self,
        monkeypatch: pytest.MonkeyPatch,
        with_user_agent: None,
        redirected_paths: None,
    ) -> None:
        # The seam must bind the documented generic assumptions, not leave them
        # unset; this guards the "generic, NOT calibrated" contract.
        assumptions = ingest_mod._generic_assumptions()
        assert assumptions.cost_of_equity == pytest.approx(0.09)
        assert assumptions.projection_years == 5
        assert assumptions.peer_pe  # a non-empty peer band so P/E can run


# --------------------------------------------------------------------------- #
# Tests: no usable request -> actionable, concrete error
# --------------------------------------------------------------------------- #


class TestNoUsableRequest:
    def test_raises_with_concrete_omission_reasons(
        self,
        monkeypatch: pytest.MonkeyPatch,
        with_user_agent: None,
        redirected_paths: None,
    ) -> None:
        # Arrange — EDGAR returns only revenue, so no method can run.
        _install_transport(monkeypatch, _FakeTransport(_CIK_MAP, _empty_facts()))
        builder = load_live_builder()

        # Act / Assert — the message must carry the real reason, not a stub.
        with pytest.raises(LiveIngestUnavailableError) as excinfo:
            builder("AAPL", knowledge_date=_KNOWLEDGE_DATE)
        message = str(excinfo.value)
        assert "dcf" in message  # a concrete omitted-method reason
        assert "AAPL" in message
        # Never an invented number: no price target leaks into the message.
        assert "price target" not in message.lower()


# --------------------------------------------------------------------------- #
# Tests: missing User-Agent -> actionable config error, exit 2, no stack trace
# --------------------------------------------------------------------------- #


class TestMissingUserAgent:
    def test_load_builder_raises_actionable_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — ensure the mandatory variable is absent.
        monkeypatch.delenv(_USER_AGENT_ENV, raising=False)

        # Act / Assert
        with pytest.raises(LiveIngestConfigError) as excinfo:
            load_live_builder()
        message = str(excinfo.value)
        assert _USER_AGENT_ENV in message
        assert "export" in message.lower()

    def test_research_command_exit_2_no_stack_trace(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import io

        # Arrange
        monkeypatch.delenv(_USER_AGENT_ENV, raising=False)
        opts = ResearchOptions(
            ticker="AAPL",
            knowledge_date=_KNOWLEDGE_DATE,
            source="live",
            use_llm=False,
            debate=False,
            as_json=False,
            show_provenance=False,
            no_color=True,
        )
        out, err = io.StringIO(), io.StringIO()

        # Act
        code = run_research(opts, out=out, err=err)

        # Assert — exit 2, actionable, and no Python traceback leaked.
        assert code == 2
        stderr = err.getvalue()
        assert _USER_AGENT_ENV in stderr
        assert "Traceback" not in stderr
        assert "Traceback" not in out.getvalue()

    def test_calibrated_assumptions_shrinks_toward_pit_sector_peers(self, tmp_path: Path) -> None:
        from alphalab.adapters.pit_store import ParquetPITStore
        from alphalab.domain.pit import PITRecord
        from alphalab.domain.types import Ticker

        pit_root = tmp_path / "pit"
        prices_dir = tmp_path / "prices"
        prices_dir.mkdir(parents=True, exist_ok=True)

        it_peers = ("AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM")
        for sym in it_peers:
            store = ParquetPITStore(root=pit_root / sym)
            store.write(
                [
                    PITRecord(
                        ticker=Ticker(sym),
                        field="eps_diluted",
                        value=5.0,
                        as_of=date(2022, 12, 31),
                        observed_at=date(2023, 2, 15),
                        source="SEC-EDGAR",
                    ),
                    # Future filing after _KNOWLEDGE_DATE must be ignored
                    PITRecord(
                        ticker=Ticker(sym),
                        field="eps_diluted",
                        value=1.0,
                        as_of=date(2023, 12, 31),
                        observed_at=date(2024, 2, 15),
                        source="SEC-EDGAR",
                    ),
                ]
            )
            (prices_dir / f"{sym}.csv").write_text(
                "date,close\n2023-06-29,175.0\n2024-06-29,500.0\n",
                encoding="utf-8",
            )

        price_source = ingest_mod.CsvPriceSeriesSource(prices_dir)
        assump = ingest_mod._calibrated_assumptions(
            "AAPL",
            knowledge_date=_KNOWLEDGE_DATE,
            price_source=price_source,
            pit_root=pit_root,
        )
        # IT disables P/B (intangible sector)
        assert assump.peer_pb == ()
        # Prior IT P/E median is 25.0; 6 peers at P/E = 175/5 = 35.0 pull median above 25.0
        # but stay well below 35.0 due to Empirical Bayes shrinkage
        assert 25.0 < assump.peer_pe[1] < 30.0


# --------------------------------------------------------------------------- #
# Tests: live module absent -> unavailable (not a config) error
# --------------------------------------------------------------------------- #


class TestLiveModuleAbsent:
    def test_missing_function_raises_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, with_user_agent: None
    ) -> None:
        # Arrange — the live module imports fine, but point the seam at a
        # function name it does not expose. This exercises the "present but no
        # build_request" branch without any dangerous global import patching.
        monkeypatch.setattr(ingest_mod, "_LIVE_FUNCTION", "does_not_exist")

        # Act / Assert
        with pytest.raises(LiveIngestUnavailableError) as excinfo:
            load_live_builder()
        assert "does_not_exist" in str(excinfo.value)


def test_caching_edgar_client_writes_and_reads_cache(tmp_path: Path) -> None:
    from alphalab.adapters.edgar import EdgarSettings
    from alphalab.domain.types import Ticker

    transport = _FakeTransport(_CIK_MAP, _full_facts())
    client = ingest_mod._CachingEdgarClient(
        EdgarSettings(user_agent="test test@example.com"),
        transport=transport,
        cache_dir=tmp_path / "edgar_cache",
    )
    first = client.company_facts(Ticker("AAPL"), cik="320193")
    second = client.company_facts(Ticker("AAPL"), cik="0000320193")
    assert first == second
    assert len(transport.urls) == 1
