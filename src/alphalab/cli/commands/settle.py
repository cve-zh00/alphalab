"""The ``settle`` command: liquidate every due prediction, honestly.

Settlement is what makes the forward clock *advance*: it turns a dated thesis
whose horizon has passed into a scored :class:`~alphalab.domain.journal.Settlement`,
pricing the realized and benchmark moves from the local close series. It is the
daily workhorse of the cadence.

Honest degradation (dossier §0): when the name or the benchmark has no close
within tolerance of the horizon, the prediction is **not** settled — it is
reported as skipped with the concrete reason. No price is invented, so a gap in
the data shows up as a gap in the record, never as a silent fabricated hit.

Three counts are always reported side by side so none can hide the others:
settled, still-pending (due later), and could-not-price (with reasons). This is
orchestration only; the pricing and the hit decision live in the engine
(:func:`alphalab.app.forecast_runner.settle_due_predictions`) and the domain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TextIO

from alphalab.adapters.journal_sqlite import SqliteJournal
from alphalab.adapters.prices import CsvPriceSeriesSource
from alphalab.app.forecast_runner import SettlementRun, settle_due_predictions
from alphalab.domain.backtest import ProportionalCostModel
from alphalab.domain.costs import CostSpec
from alphalab.domain.journal import due_predictions
from alphalab.report.theme import Theme, theme_for

#: A modest, non-zero default cost. Five basis points of half-spread keeps the
#: net-alpha column honest without pretending to model a specific venue.
_DEFAULT_HALF_SPREAD_BPS = 5.0


@dataclass(frozen=True)
class SettleOptions:
    """Resolved options for one ``settle`` invocation."""

    db_path: Path
    prices_dir: Path
    as_of: date
    as_json: bool
    no_color: bool


def run_settle(opts: SettleOptions, *, out: TextIO, err: TextIO) -> int:
    """Settle every due prediction from the local price series. Exit 0 on success."""
    if not opts.db_path.exists():
        print(
            f"error: no journal database at {opts.db_path}. "
            "Run 'alphalab predict ...' first to record something to settle.",
            file=err,
        )
        return 2

    journal = SqliteJournal(opts.db_path)
    try:
        pending = _pending_count(journal, opts.as_of)
        run = settle_due_predictions(
            journal=journal,
            prices=CsvPriceSeriesSource(opts.prices_dir),
            as_of=opts.as_of,
            settled_at=datetime.now(),
            cost_model=ProportionalCostModel(CostSpec(half_spread_bps=_DEFAULT_HALF_SPREAD_BPS)),
        )
    finally:
        journal.close()

    if opts.as_json:
        print(_as_json(run, pending, opts), file=out)
        return 0
    theme = theme_for(out, force_disable=opts.no_color)
    print(_format(run, pending, opts, theme), file=out)
    return 0


def _pending_count(journal: SqliteJournal, as_of: date) -> int:
    """Count predictions recorded but not yet due at ``as_of`` (still open)."""
    settled = set(journal.settled_ids())
    due_now = {
        p.prediction_id
        for p in due_predictions(
            journal.list_predictions(), settled_ids=journal.settled_ids(), as_of=as_of
        )
    }
    pending = 0
    for prediction in journal.list_predictions():
        if prediction.prediction_id in settled:
            continue
        if prediction.prediction_id in due_now:
            continue
        pending += 1
    return pending


def _format(run: SettlementRun, pending: int, opts: SettleOptions, theme: Theme) -> str:
    lines = [theme.bold("SETTLEMENT PASS")]
    lines.append(f"  as of {opts.as_of} — prices from {opts.prices_dir}")
    lines.append(
        f"  {theme.paint(f'settled {len(run.settled)}', 'green')}, "
        f"{pending} still pending, "
        f"{theme.paint(f'{len(run.skipped)} could not be priced', 'yellow')}"
    )
    for outcome in run.settled:
        pred = outcome.prediction
        settlement = outcome.settlement
        hit = theme.paint("HIT", "green") if settlement.outcome else theme.paint("MISS", "red")
        lines.append(
            f"    {pred.ticker} [{hit}] alpha_net={settlement.alpha_net:+.4f} "
            f"(horizon {pred.horizon})"
        )
    for skip in run.skipped:
        lines.append(f"    {skip.prediction.ticker} [skipped]: {skip.reason}")
    lines.append(theme.dim(f"  journal: {opts.db_path}"))
    return "\n".join(lines)


def _as_json(run: SettlementRun, pending: int, opts: SettleOptions) -> str:
    payload = {
        "journal": str(opts.db_path),
        "as_of": opts.as_of.isoformat(),
        "settled": len(run.settled),
        "skipped": len(run.skipped),
        "pending": pending,
        "settlements": [
            {
                "ticker": o.prediction.ticker,
                "horizon": o.prediction.horizon.isoformat(),
                "outcome": o.settlement.outcome,
                "alpha_gross": o.settlement.alpha_gross,
                "alpha_net": o.settlement.alpha_net,
            }
            for o in run.settled
        ],
        "unpriced": [{"ticker": s.prediction.ticker, "reason": s.reason} for s in run.skipped],
    }
    return json.dumps(payload, indent=2, sort_keys=True)
