"""Unit tests for the agentic compose seams that are testable without network.

The adapter constructors themselves need credentials/LLM, so those are
monkeypatched to inert doubles; the point here is to prove the *wiring* — that
each builder returns a callable of the right shape and threads provenance and
horizons through — not to reach the network.
"""

from __future__ import annotations

from datetime import date

import pytest

import alphalab.cli.agentic_compose as compose
from alphalab.app.agentic_portfolio import RunProvenance
from alphalab.artifact.models import EquityResearchArtifact, PriceTarget, Verdict
from alphalab.domain.backtest import PriceBar
from alphalab.domain.forecast import Abstention
from alphalab.domain.journal import Prediction
from alphalab.domain.types import Ticker


def test_bars_price_source_maps_bars_to_points() -> None:
    source = compose._BarsPriceSource(
        {
            Ticker("AAA"): (
                PriceBar(on=date(2025, 1, 1), close=10.0, volume=1.0),
                PriceBar(on=date(2025, 1, 2), close=11.0, volume=1.0),
            )
        }
    )
    series = source.series(Ticker("AAA"))
    assert [p.close for p in series] == [10.0, 11.0]
    assert source.series(Ticker("MISSING")) == []


def test_run_provenance_reads_deep_tier_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import alphalab.adapters.llm_config as llm_config

    class _Settings:
        deep_version = "v-deep"
        deep_cutoff = date(2024, 10, 1)
        deep_model = "m-deep"

    monkeypatch.setattr(llm_config, "LLMSettings", lambda: _Settings())
    prov = compose.run_provenance(date(2025, 1, 2))
    assert prov.llm_version == "v-deep"
    assert prov.model_name == "m-deep"
    assert prov.llm_cutoff == date(2024, 10, 1)


def test_build_pipeline_runner_debate_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: patch the debate adapter + pipeline so no LLM is constructed.
    import alphalab.adapters.debate_llm as debate_llm
    import alphalab.app.debate_pipeline as debate_pipeline

    artifact = _artifact()

    class _Pipeline:
        def __init__(self, **_kw: object) -> None:
            pass

        def run(self, _request: object) -> object:
            return type("R", (), {"artifact": artifact})()

    monkeypatch.setattr(debate_llm, "OllamaDebateAdapter", lambda *_a, **_k: object())
    monkeypatch.setattr(debate_llm, "bull_provider", lambda a: a)
    monkeypatch.setattr(debate_llm, "bear_provider", lambda a: a)
    monkeypatch.setattr(debate_pipeline, "DebatePipeline", _Pipeline)

    # Act
    runner = compose.build_pipeline_runner(analyst=False)
    result = runner(object())

    # Assert
    assert result is artifact


def test_build_pipeline_runner_analyst_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    import alphalab.adapters.llm as llm
    import alphalab.app.pipeline as pipeline

    artifact = _artifact()

    class _Pipeline:
        def __init__(self, **_kw: object) -> None:
            pass

        def run(self, _request: object) -> object:
            return type("R", (), {"artifact": artifact})()

    monkeypatch.setattr(llm, "OllamaLLMAdapter", lambda *_a, **_k: object())
    monkeypatch.setattr(pipeline, "AnalystPipeline", _Pipeline)

    runner = compose.build_pipeline_runner(analyst=True)
    assert runner(object()) is artifact


def test_build_forecast_sink_threads_horizon_and_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    import alphalab.adapters.abstention_store as abst
    import alphalab.adapters.journal_sqlite as jrnl
    import alphalab.app.forecast_runner as runner

    captured: dict[str, object] = {}

    class _Store:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def close(self) -> None:
            pass

    def _register(artifact: object, **kwargs: object) -> Prediction | Abstention:
        captured.update(kwargs)
        return Abstention(ticker="AAA", knowledge_date=date(2025, 1, 2), reason="x")

    monkeypatch.setattr(jrnl, "SqliteJournal", _Store)
    monkeypatch.setattr(abst, "SqliteAbstentionStore", _Store)
    monkeypatch.setattr(runner, "register_forecast", _register)

    import pathlib

    prov = RunProvenance(
        llm_version="v",
        llm_cutoff=date(2024, 10, 1),
        knowledge_date=date(2025, 1, 2),
        model_name="m",
    )
    sink, _journal, _abstentions = compose.build_forecast_sink(
        journal_db=pathlib.Path(str(tmp_path)) / "j.db",
        abstentions_db=pathlib.Path(str(tmp_path)) / "a.db",
        horizon=date(2025, 1, 23),
        provenance=prov,
    )
    sink(_artifact())

    assert captured["horizon"] == date(2025, 1, 23)
    assert captured["llm_version"] == "v"
    assert captured["llm_cutoff"] == date(2024, 10, 1)


def test_build_request_closure_binds_assumptions_and_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: patch EDGAR/CIK/metadata and capture the build_request call.
    import alphalab.adapters.cik as cik
    import alphalab.adapters.edgar as edgar
    import alphalab.cli.paper_transport as transport_mod
    from alphalab.app.ingest import IngestResult

    captured: dict[str, object] = {}

    monkeypatch.setattr(transport_mod, "HttpxWriteTransport", lambda *_a, **_k: object())
    monkeypatch.setattr(edgar, "EdgarClient", lambda *_a, **_k: "edgar")
    monkeypatch.setattr(edgar, "EdgarSettings", lambda: object())
    monkeypatch.setattr(cik, "CikResolver", lambda *_a, **_k: "resolver")
    monkeypatch.setattr(cik, "CikSettings", lambda: object())
    monkeypatch.setattr(compose, "_run_metadata", lambda kd: "meta")

    def _fake_build_request(ticker: object, **kwargs: object) -> IngestResult:
        captured.update(kwargs)
        captured["ticker"] = ticker
        return IngestResult(
            request=None, methods_used=(), omitted_methods=(), missing_fields=frozenset()
        )

    monkeypatch.setattr(compose, "build_request", _fake_build_request)

    # Act
    builder = compose.build_request_closure({}, knowledge_date=date(2025, 1, 2))
    builder(Ticker("AAA"), knowledge_date=date(2025, 1, 2))

    # Assert: the explicit assumptions and bound adapters flow into build_request.
    assert captured["assumptions"] is compose.ASSUMPTIONS
    assert captured["edgar_client"] == "edgar"
    assert captured["cik_resolver"] == "resolver"
    assert captured["run_metadata"] == "meta"


def test_build_candidates_screens_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: patch the network fetch to return a flat, liquid panel.
    import pathlib

    def _fake_fetch(prices_dir: object, *, as_of: object) -> dict:  # type: ignore[type-arg]
        return {
            Ticker("AAA"): tuple(
                PriceBar(on=date(2024, 1, 1), close=10.0, volume=1_000_000.0) for _ in range(70)
            ),
            Ticker("BBB"): tuple(
                PriceBar(on=date(2024, 1, 1), close=2.0, volume=1_000_000.0) for _ in range(70)
            ),
        }

    monkeypatch.setattr(compose, "_fetch_bars", _fake_fetch)

    # Act
    candidates, bars = compose.build_candidates(
        pathlib.Path("x"), max_candidates=5, as_of=date(2024, 4, 1)
    )

    # Assert: AAA clears the screen, the sub-$5 BBB is dropped.
    assert [str(c.ticker) for c in candidates] == ["AAA"]
    assert Ticker("AAA") in bars


def _artifact() -> EquityResearchArtifact:
    return EquityResearchArtifact(
        ticker="AAA",
        knowledge_date=date(2025, 1, 2),
        verdict=Verdict.BUY,
        price_target=PriceTarget(value=120.0, unit="USD_per_share"),
        current_price=100.0,
        methods_used=("dcf",),
        method_values={"dcf": 120.0},
        consensus_relative_span=0.1,
    )
