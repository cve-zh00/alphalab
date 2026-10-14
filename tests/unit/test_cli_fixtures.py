"""Unit tests for the research LLM path hint, doctor probes, and fixtures.

These cover the branches that need injected doubles: the LLM path when Ollama
is unreachable (must give an actionable --no-llm hint, no stack trace), the
doctor Ollama "up" branch (mocked transport), and the fixture request builder.
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pytest

from alphalab.cli import main
from alphalab.cli.commands import research as research_cmd
from alphalab.cli.commands.doctor import DoctorOptions, run_doctor
from alphalab.cli.fixtures import (
    FixtureNotFoundError,
    available_tickers,
    build_fixture_request,
)


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


class TestResearchLlmPath:
    def test_llm_failure_is_actionable_and_hints_no_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the LLM pipeline raises, the user gets a hint, not a traceback."""

        def _boom(*_a: object, **_k: object) -> object:
            raise ConnectionError("connection refused")

        # Replace the analyst pipeline runner so no real Ollama call happens.
        monkeypatch.setattr(research_cmd, "_run_analyst", _boom)
        code, _out, err = _run(["research", "AAPL"])  # no --no-llm → LLM path
        assert code == 1
        assert "analysis failed" in err
        assert "--no-llm" in err
        assert "Traceback" not in err


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class TestDoctorOllamaProbe:
    def test_reports_ollama_up_with_models(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import httpx

        payload = {"models": [{"name": "qwen3.8:27b-mlx"}, {"model": "qwen3-vl:latest"}]}
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(200, payload))
        out, err = io.StringIO(), io.StringIO()
        run_doctor(
            DoctorOptions(db_path=tmp_path / "none.db", as_json=False, no_color=True),
            out=out,
            err=err,
        )
        text = out.getvalue()
        assert "up at" in text
        assert "qwen3.8:27b-mlx" in text

    def test_reports_ollama_non_200(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import httpx

        monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(503, {}))
        out, err = io.StringIO(), io.StringIO()
        run_doctor(
            DoctorOptions(db_path=tmp_path / "none.db", as_json=False, no_color=True),
            out=out,
            err=err,
        )
        assert "responded 503" in out.getvalue()


class TestFixtureBuilder:
    def test_builds_request_from_fixture(self) -> None:
        result = build_fixture_request("AAPL", knowledge_date=date(2024, 6, 1))
        req = result.request
        assert req.ticker == "AAPL"
        assert req.current_price > 0.0
        assert len(req.valuation.estimates) == 4
        assert req.valuation.unit == "USD_per_share"
        assert "NOT real SEC EDGAR" in result.source

    def test_unknown_ticker_raises_with_available_list(self) -> None:
        with pytest.raises(FixtureNotFoundError) as exc:
            build_fixture_request("ZZZZ", knowledge_date=date(2024, 6, 1))
        assert "no price fixture" in str(exc.value)

    def test_available_tickers_includes_known_names(self) -> None:
        tickers = available_tickers()
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_available_tickers_empty_for_missing_dir(self, tmp_path: Path) -> None:
        assert available_tickers(tmp_path / "nope") == ()
