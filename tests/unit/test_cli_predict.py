"""Unit tests for the ``predict`` command: register vs abstain, both exit 0.

Every test runs with the deterministic ``--no-llm`` narrator against the
bundled price fixtures: no network, no Ollama, no 60s wait. The two outcomes a
user must be able to tell apart — a *registered* dated prediction and a
*deliberate abstention* — are both successes (exit 0) with visibly different
output, and the anti-leakage refusal is a distinct, actionable exit 2.

The deterministic narrator derives its verdict from the (parity-calibrated)
fixture numbers, so it yields HOLD — a genuine abstention. ``--assume-verdict``
overrides that *only on the offline path* so the registration flow can be
exercised without an LLM; the default demo keeps the honest numbers-derived
verdict.
"""

from __future__ import annotations

import io
import json
from datetime import date, timedelta
from pathlib import Path

from alphalab.adapters.journal_sqlite import SqliteJournal
from alphalab.cli import main
from alphalab.cli.commands.predict import DEFAULT_HORIZON_DAYS

#: Far-future knowledge date, so the anti-leakage gate never fires in tests.
_KNOWLEDGE_DATE = date(2099, 1, 1)


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def _db_flags(tmp_path: Path) -> list[str]:
    return [
        "--db",
        str(tmp_path / "journal.db"),
        "--abstentions-db",
        str(tmp_path / "abstentions.db"),
    ]


class TestPredictRegisters:
    def test_bullish_fixture_registers_a_prediction(self, tmp_path: Path) -> None:
        """A BUY artifact with a far-future knowledge date registers, exit 0."""
        code, out, err = _run(
            [
                "predict",
                "AAPL",
                "--no-llm",
                "--assume-verdict",
                "buy",
                "--horizon-days",
                "90",
                "--knowledge-date",
                "2099-01-01",
                *_db_flags(tmp_path),
            ]
        )
        assert code == 0, err
        assert "REGISTERED" in out
        assert "AAPL" in out
        # This case passes --horizon-days explicitly, so the expected date is
        # derived from that override rather than from the default.
        assert (_KNOWLEDGE_DATE + timedelta(days=90)).isoformat() in out
        journal = SqliteJournal(tmp_path / "journal.db")
        try:
            assert len(journal.list_predictions()) == 1
        finally:
            journal.close()

    def test_registration_shows_probability_and_direction(self, tmp_path: Path) -> None:
        _, out, _ = _run(
            [
                "predict",
                "AAPL",
                "--no-llm",
                "--assume-verdict",
                "buy",
                "--knowledge-date",
                "2099-01-01",
                *_db_flags(tmp_path),
            ]
        )
        assert "probability" in out.lower()
        assert "up" in out.lower()

    def test_prints_db_location(self, tmp_path: Path) -> None:
        _, out, _ = _run(
            [
                "predict",
                "AAPL",
                "--no-llm",
                "--assume-verdict",
                "buy",
                "--knowledge-date",
                "2099-01-01",
                *_db_flags(tmp_path),
            ]
        )
        assert str(tmp_path / "journal.db") in out

    def test_json_is_parseable_and_marks_registered(self, tmp_path: Path) -> None:
        code, out, _ = _run(
            [
                "predict",
                "AAPL",
                "--no-llm",
                "--assume-verdict",
                "buy",
                "--knowledge-date",
                "2099-01-01",
                "--json",
                *_db_flags(tmp_path),
            ]
        )
        assert code == 0
        payload = json.loads(out)
        assert payload["status"] == "registered"
        assert payload["ticker"] == "AAPL"
        assert 0.0 <= payload["probability"] <= 1.0
        assert payload["direction"] == "up"
        # Derived from the constant, not hardcoded: the default horizon is a
        # tuning decision that changes as calibration needs dictate, and a
        # literal date here would fail the build for a legitimate change.
        expected = _KNOWLEDGE_DATE + timedelta(days=DEFAULT_HORIZON_DAYS)
        assert payload["horizon"] == expected.isoformat()


class TestPredictAbstains:
    def test_leakage_is_refused_with_exit_two(self, tmp_path: Path) -> None:
        """A knowledge date on/before the model cutoff is hindsight, not a forecast."""
        code, _out, err = _run(
            [
                "predict",
                "AAPL",
                "--no-llm",
                "--assume-verdict",
                "buy",
                "--knowledge-date",
                "2000-01-01",
                "--llm-cutoff",
                "2024-01-01",
                *_db_flags(tmp_path),
            ]
        )
        assert code == 2
        assert "leak" in err.lower() or "hindsight" in err.lower()
        journal = SqliteJournal(tmp_path / "journal.db")
        try:
            assert journal.list_predictions() == []
        finally:
            journal.close()

    def test_hold_verdict_abstains_and_is_not_an_error(self, tmp_path: Path) -> None:
        """The default fixture verdict is HOLD: abstains, exit 0, distinct output."""
        from alphalab.adapters.abstention_store import SqliteAbstentionStore

        code, out, err = _run(
            [
                "predict",
                "AAPL",
                "--no-llm",
                "--knowledge-date",
                "2099-01-01",
                *_db_flags(tmp_path),
            ]
        )
        assert code == 0, err
        assert "ABSTAINED" in out
        assert "reason" in out.lower()
        store = SqliteAbstentionStore(tmp_path / "abstentions.db")
        try:
            assert store.count() == 1
        finally:
            store.close()

    def test_abstention_json_marks_abstained(self, tmp_path: Path) -> None:
        code, out, _ = _run(
            [
                "predict",
                "AAPL",
                "--no-llm",
                "--knowledge-date",
                "2099-01-01",
                "--json",
                *_db_flags(tmp_path),
            ]
        )
        assert code == 0
        payload = json.loads(out)
        assert payload["status"] == "abstained"
        assert payload["reason"]


class TestPredictMultipleTickers:
    def test_two_tickers_both_processed(self, tmp_path: Path) -> None:
        code, out, err = _run(
            [
                "predict",
                "AAPL",
                "MSFT",
                "--no-llm",
                "--knowledge-date",
                "2099-01-01",
                *_db_flags(tmp_path),
            ]
        )
        assert code == 0, err
        assert out.count("REGISTERED") + out.count("ABSTAINED") == 2

    def test_unknown_ticker_is_actionable_error(self, tmp_path: Path) -> None:
        code, _out, err = _run(
            [
                "predict",
                "NOTATICKER",
                "--no-llm",
                "--knowledge-date",
                "2099-01-01",
                *_db_flags(tmp_path),
            ]
        )
        assert code == 2
        assert "no price fixture" in err


class TestPredictToleratesUnusableTickers:
    """One unusable name must not silence the rest of the batch.

    Across a real universe many tickers have no usable fundamentals (ADRs,
    funds, non-standard filers). Aborting on the first one made a mass run
    impossible: it reported a single failure and never looked at the other 344.
    """

    def test_bad_ticker_first_does_not_abort_the_batch(self, tmp_path: Path) -> None:
        code, out, err = _run(
            [
                "predict",
                "NOTATICKER",
                "AAPL",
                "--no-llm",
                "--knowledge-date",
                "2099-01-01",
                *_db_flags(tmp_path),
            ]
        )
        assert code == 0, err
        assert "no price fixture" in err
        assert out.count("REGISTERED") + out.count("ABSTAINED") == 1
        assert "SKIPPED" in out

    def test_census_counts_usable_and_skipped(self, tmp_path: Path) -> None:
        _, out, _ = _run(
            [
                "predict",
                "NOTATICKER",
                "AAPL",
                "MSFT",
                "--no-llm",
                "--knowledge-date",
                "2099-01-01",
                *_db_flags(tmp_path),
            ]
        )
        assert "2/3 usable; 1 skipped." in out

    def test_skips_are_visible_in_json(self, tmp_path: Path) -> None:
        code, out, _ = _run(
            [
                "predict",
                "NOTATICKER",
                "AAPL",
                "--no-llm",
                "--knowledge-date",
                "2099-01-01",
                "--json",
                *_db_flags(tmp_path),
            ]
        )
        assert code == 0
        rows = json.loads(out)["results"]
        by_status = {row["status"]: row for row in rows}
        assert by_status["skipped"]["ticker"] == "NOTATICKER"
        assert by_status["skipped"]["reason"]

    def test_a_run_with_nothing_usable_still_fails(self, tmp_path: Path) -> None:
        """Tolerating skips must not turn a wholly empty run into a success."""
        code, _out, err = _run(
            [
                "predict",
                "NOTATICKER",
                "ALSONOTATICKER",
                "--no-llm",
                "--knowledge-date",
                "2099-01-01",
                *_db_flags(tmp_path),
            ]
        )
        assert code == 2
        assert err.count("no price fixture") == 2

    def test_leakage_still_aborts_the_whole_batch(self, tmp_path: Path) -> None:
        """Leakage is about the invocation, not one ticker: every name would fail."""
        code, _out, err = _run(
            [
                "predict",
                "AAPL",
                "MSFT",
                "--no-llm",
                "--assume-verdict",
                "buy",
                "--knowledge-date",
                "2000-01-01",
                "--llm-cutoff",
                "2024-01-01",
                *_db_flags(tmp_path),
            ]
        )
        assert code == 2
        # Reported once, for the first ticker — the batch stopped there rather
        # than reprinting the same misconfiguration for MSFT.
        assert err.count("refused to register") == 1
        assert "MSFT" not in err
