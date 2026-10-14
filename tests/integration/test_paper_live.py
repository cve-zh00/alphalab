"""Real end-to-end paper test against the live Alpaca **paper** API (opt-in).

Skipped unless both ``ALPHALAB_ALPACA_LIVE_TESTS=1`` and valid credentials are
present. It is strictly read-only: it reads the account, positions and clock,
and runs **one dry-run cycle** (``submit=False``), so it never places an order.
Run with::

    ALPHALAB_ALPACA_LIVE_TESTS=1 \\
        ALPHALAB_ALPACA_KEY_ID=... ALPHALAB_ALPACA_SECRET_KEY=... \\
        uv run pytest tests/integration/test_paper_live.py -v -m alpaca_live --no-cov
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from alphalab.adapters.alpaca_broker import AlpacaBroker
from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.execution_log import SqliteExecutionLog
from alphalab.adapters.kill_switch import KillSwitch
from alphalab.app.paper_runner import PaperRunner
from alphalab.cli.paper_transport import HttpxWriteTransport
from alphalab.domain.backtest import PriceBar
from alphalab.domain.types import Ticker

pytestmark = pytest.mark.alpaca_live

_LIVE = os.environ.get("ALPHALAB_ALPACA_LIVE_TESTS") == "1"


def _settings_or_skip() -> AlpacaSettings:
    if not _LIVE:
        pytest.skip("set ALPHALAB_ALPACA_LIVE_TESTS=1 to run")
    try:
        return AlpacaSettings()  # type: ignore[call-arg]  # from env/.env
    except ValidationError:
        pytest.skip("Alpaca credentials not set (ALPHALAB_ALPACA_KEY_ID/SECRET_KEY)")


class _StaticPrices:
    """A tiny deterministic panel so the dry run has something to plan against."""

    def recent_bars(self) -> Mapping[Ticker, Sequence[PriceBar]]:
        from datetime import date, timedelta

        def series(level: float) -> tuple[PriceBar, ...]:
            return tuple(
                PriceBar(on=date(2023, 1, 1) + timedelta(days=i), close=level + i * 0.1)
                for i in range(300)
            )

        return {Ticker("SPY"): series(400.0)}


def test_live_paper_dry_run_reads_account_and_sends_nothing(tmp_path: Path) -> None:
    # Arrange — a halt file would be irrelevant here since we never submit.
    settings = _settings_or_skip()
    broker = AlpacaBroker(
        settings,
        transport=HttpxWriteTransport(),
        kill_switch=KillSwitch(halt_path=tmp_path / "HALT", env={}),
    )
    log = SqliteExecutionLog(tmp_path / "exec.db")
    runner = PaperRunner(broker=broker, prices=_StaticPrices(), execution_log=log)

    # Act — dry run: submit defaults to False.
    result = runner.run_cycle(submit=False)
    log.close()

    # Assert — we read a real account and placed no order.
    assert result.account.equity > 0.0
    assert not result.submitted
    assert result.orders_reached_broker == 0
    print(
        f"\n[alpaca-paper] equity={result.account.equity:.2f} "
        f"market_open={result.market_open} "
        f"shadow_spy_return={result.shadow.total_return if result.shadow else 'n/a'}"
    )
