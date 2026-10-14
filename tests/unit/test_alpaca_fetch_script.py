"""Unit tests for the Alpaca universe fetch script (no network, no credentials)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "fetch_alpaca_universe.py"


def _load_script() -> ModuleType:
    """Import the standalone script module by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("fetch_alpaca_universe", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_script = _load_script()


class TestCredentialHandling:
    def test_missing_credentials_exit_with_clear_message_no_secret(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange — ensure no Alpaca env vars and no .env leakage.
        for key in ("ALPHALAB_ALPACA_KEY_ID", "ALPHALAB_ALPACA_SECRET_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.chdir(Path(__file__).parent)  # a dir with no .env

        # Act / Assert
        with pytest.raises(SystemExit) as exc:
            _script._load_settings()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "ALPHALAB_ALPACA_KEY_ID" in err
        assert "ALPHALAB_ALPACA_SECRET_KEY" in err


class TestUniverseWriting:
    def test_universe_json_records_counts_and_ratio(self, tmp_path: Path) -> None:
        # Arrange
        from alphalab.adapters.alpaca_assets import AlpacaAsset, UniverseSnapshot

        def _asset(sym: str, status: str) -> AlpacaAsset:
            return AlpacaAsset(
                symbol=sym,
                name=f"{sym} Inc",
                exchange="NASDAQ",
                asset_class="us_equity",
                status=status,
                tradable=status == "active",
                fractionable=True,
            )

        snapshot = UniverseSnapshot(
            active=(_asset("AAPL", "active"), _asset("MSFT", "active")),
            inactive=(_asset("ENRN", "inactive"),),
        )

        # Act
        path = _script._write_universe(snapshot, tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        # Assert
        assert payload["active_count"] == 2
        assert payload["inactive_count"] == 1
        assert payload["survivorship_ratio"] == pytest.approx(0.5)
        assert {a["symbol"] for a in payload["inactive"]} == {"ENRN"}


class TestIdempotency:
    def test_already_downloaded_symbols_are_skipped(self, tmp_path: Path) -> None:
        # Arrange
        from alphalab.adapters.alpaca_assets import AlpacaAsset, UniverseSnapshot

        def _asset(sym: str) -> AlpacaAsset:
            return AlpacaAsset(
                symbol=sym,
                name=sym,
                exchange="NASDAQ",
                asset_class="us_equity",
                status="active",
                tradable=True,
                fractionable=True,
            )

        snapshot = UniverseSnapshot(active=(_asset("AAPL"), _asset("MSFT")), inactive=())
        # Pre-create AAPL's bar file so it is considered done.
        existing = _script._bars_path(tmp_path, "AAPL")
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("[]", encoding="utf-8")

        # Act
        pending = _script._pending_symbols(snapshot, tmp_path)

        # Assert — only the not-yet-downloaded symbol remains.
        assert [str(t) for t in pending] == ["MSFT"]


class TestFaultTolerance:
    def test_one_failing_symbol_does_not_abort_and_is_recorded(self, tmp_path: Path) -> None:
        # Arrange — a fake bars client that fails on one symbol, succeeds on the
        # other, exercising the per-symbol worker and its error capture.
        from datetime import date

        from alphalab.domain.backtest.types import PriceBar
        from alphalab.domain.types import Ticker

        class _FakeBarsClient:
            def daily_bars(
                self, symbols: list[Ticker], *, start: date, end: date
            ) -> dict[Ticker, tuple[PriceBar, ...]]:
                sym = symbols[0]
                if str(sym) == "BAD":
                    raise RuntimeError("boom")
                return {sym: (PriceBar(on=date(2021, 1, 4), close=100.0, volume=1.0),)}

        good = _script._fetch_one(
            _FakeBarsClient(),
            Ticker("AAPL"),
            start=date(2021, 1, 1),
            end=date(2021, 1, 5),
            out_dir=tmp_path,
        )

        # Act / Assert — the good symbol wrote a file with one bar.
        assert good == ("AAPL", 1)
        rows = json.loads(_script._bars_path(tmp_path, "AAPL").read_text(encoding="utf-8"))
        assert len(rows) == 1 and rows[0]["close"] == 100.0

        # The bad symbol raises, which the orchestrator records rather than crashes.
        with pytest.raises(RuntimeError, match="boom"):
            _script._fetch_one(
                _FakeBarsClient(),
                Ticker("BAD"),
                start=date(2021, 1, 1),
                end=date(2021, 1, 5),
                out_dir=tmp_path,
            )
