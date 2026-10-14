"""The hidden ``_run-cadence`` command: what launchd actually fires.

``schedule generate`` writes one launchd plist per calendar cadence, each
invoking ``alphalab _run-cadence <cadence>``. This module is the receiving end:
it maps a cadence name onto the *new* user-facing commands so the recurring
work runs itself once the agents are installed.

* **daily** → ``settle``: liquidate every due prediction. This is what makes
  the forward clock advance, and it is pure arithmetic over stored prices, so
  it spends no trial budget and is safe to run unattended every day.
* **weekly** → ``predict`` over a configurable watchlist: research each name and
  record a dated forecast (or a stated abstention).
* **monthly** → ``calibration``: score the accumulated settled record, honestly
  declaring when the sample is still too small to conclude anything.

It is a thin composition root: it builds the resolved options and delegates to
the same runners the interactive CLI uses, so there is exactly one code path
per command. It never runs ``launchctl`` and never shells out.
"""

from __future__ import annotations

from datetime import date
from typing import TextIO

from alphalab.cli.commands.calibration import CalibrationOptions, run_calibration
from alphalab.cli.commands.predict import DEFAULT_HORIZON_DAYS, PredictOptions, run_predict
from alphalab.cli.commands.settle import SettleOptions, run_settle
from alphalab.cli.forecast_paths import (
    default_abstention_db,
    default_journal_db,
    default_prices_dir,
)

#: The weekly deep-dive watchlist. Deliberately small — the weekly cadence is
#: the only one that spends the trial budget, so the list is short and the
#: offline deterministic narrator is used so an unattended run never blocks on
#: a 60s LLM call or a missing Ollama. Override by editing this constant at the
#: composition root; the cadence itself stays agnostic.
_DEFAULT_WATCHLIST = ("AAPL", "MSFT", "NVDA")


def run_cadence(cadence: str, *, out: TextIO, err: TextIO, no_color: bool = False) -> int:
    """Dispatch a launchd-fired cadence to the matching user-facing command."""
    today = date.today()
    if cadence == "daily":
        return run_settle(
            SettleOptions(
                db_path=default_journal_db(),
                prices_dir=default_prices_dir(),
                as_of=today,
                as_json=False,
                no_color=no_color,
            ),
            out=out,
            err=err,
        )
    if cadence == "weekly":
        return run_predict(
            PredictOptions(
                tickers=_DEFAULT_WATCHLIST,
                knowledge_date=today,
                horizon_days=DEFAULT_HORIZON_DAYS,
                source="fixture",
                use_llm=False,
                debate=False,
                as_json=False,
                no_color=no_color,
                db_path=default_journal_db(),
                abstentions_path=default_abstention_db(),
            ),
            out=out,
            err=err,
        )
    if cadence == "monthly":
        return run_calibration(
            CalibrationOptions(db_path=default_journal_db(), as_json=False, no_color=no_color),
            out=out,
            err=err,
        )
    print(f"error: unknown cadence {cadence!r} (expected daily/weekly/monthly).", file=err)
    return 2
