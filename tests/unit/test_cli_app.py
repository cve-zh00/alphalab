"""Unit tests for the CLI entrypoint: argv parsing, dispatch, exit codes.

All tests invoke ``main([...])`` with injected ``out``/``err`` streams, no
network and no LLM. The ``--no-llm`` path is exercised end-to-end against the
bundled price fixtures.
"""

from __future__ import annotations

import io
import json

import pytest

from alphalab.cli import main


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


class TestTopLevel:
    def test_no_command_prints_help_and_exits_zero(self) -> None:
        code, out, _ = _run([])
        assert code == 0
        assert "research" in out
        assert "doctor" in out

    def test_help_flag_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--help"], out=io.StringIO(), err=io.StringIO())
        assert exc.value.code == 0

    def test_unknown_command_is_usage_error(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["frobnicate"], out=io.StringIO(), err=io.StringIO())
        assert exc.value.code == 2


class TestResearchNoLlm:
    def test_runs_offline_against_fixtures(self) -> None:
        """--no-llm must work with no Ollama and no network."""
        code, out, err = _run(["research", "AAPL", "--no-llm"])
        assert code == 0, err
        assert "EQUITY RESEARCH" in out
        assert "AAPL" in out
        assert "VERDICT" in out
        assert "NOT financial advice" in out

    def test_discloses_fixture_data_source(self) -> None:
        _, out, _ = _run(["research", "AAPL", "--no-llm"])
        assert "DATA SOURCE" in out
        assert "NOT real SEC EDGAR" in out

    def test_json_output_is_parseable(self) -> None:
        code, out, _ = _run(["research", "AAPL", "--no-llm", "--json"])
        assert code == 0
        payload = json.loads(out)
        assert payload["ticker"] == "AAPL"
        assert payload["verdict"] in {"BUY", "HOLD", "SELL"}
        assert "data_source" in payload
        assert payload["disclaimer"].startswith("Technical research")

    def test_debate_pipeline_runs_offline(self) -> None:
        code, out, err = _run(["research", "AAPL", "--no-llm", "--debate"])
        assert code == 0, err
        assert "VERDICT" in out

    def test_provenance_flag_adds_derivation(self) -> None:
        _, out, _ = _run(["research", "AAPL", "--no-llm", "--provenance"])
        assert "PROVENANCE" in out

    def test_lowercase_ticker_is_accepted(self) -> None:
        code, out, _ = _run(["research", "aapl", "--no-llm"])
        assert code == 0
        assert "AAPL" in out

    def test_unknown_ticker_is_actionable_error(self) -> None:
        code, _out, err = _run(["research", "NOTATICKER", "--no-llm"])
        assert code == 2
        assert "no price fixture" in err
        assert "Available" in err

    def test_bad_date_is_usage_error(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(
                ["research", "AAPL", "--knowledge-date", "nope"],
                out=io.StringIO(),
                err=io.StringIO(),
            )
        assert exc.value.code == 2


class TestResearchLlmErrorPath:
    def test_live_source_without_wiring_is_handled(self) -> None:
        """--source live is not provisioned yet: actionable error, no stack trace."""
        code, _out, err = _run(["research", "AAPL", "--source", "live", "--no-llm"])
        assert code == 1
        assert "fixture" in err.lower()
        assert "Traceback" not in err
