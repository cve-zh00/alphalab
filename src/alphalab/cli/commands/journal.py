"""The ``journal`` command: list the accumulating forward record, read-only.

Where ``calibration`` scores the record, this command simply *shows* it: every
prediction with its state (open or settled, and if settled, hit or miss), plus
how many names the system has declined. It is the window a user looks through
to watch the forward clock accumulate rows between settlements — nothing here
mutates anything.

Open and settled are listed together, newest-horizon last, so the growing
history reads top to bottom. An empty or missing journal is reported plainly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from alphalab.adapters.abstention_store import SqliteAbstentionStore
from alphalab.adapters.journal_sqlite import SqliteJournal
from alphalab.domain.journal import Prediction, Settlement
from alphalab.report.theme import Theme, theme_for


@dataclass(frozen=True)
class JournalOptions:
    """Resolved options for one ``journal`` invocation."""

    db_path: Path
    abstentions_path: Path | None
    as_json: bool
    no_color: bool


@dataclass(frozen=True)
class _Row:
    """A prediction paired with its settlement, if any (open when ``None``)."""

    prediction: Prediction
    settlement: Settlement | None


def run_journal(opts: JournalOptions, *, out: TextIO, err: TextIO) -> int:
    """List open and settled predictions. Read-only; exit 0 on success."""
    if not opts.db_path.exists():
        print(
            f"error: no journal database at {opts.db_path}. "
            "Run 'alphalab predict ...' first to record a forecast.",
            file=err,
        )
        return 2

    journal = SqliteJournal(opts.db_path)
    try:
        rows = _load_rows(journal)
    finally:
        journal.close()
    abstention_count = _count_abstentions(opts.abstentions_path)

    if opts.as_json:
        print(_as_json(rows, abstention_count, opts), file=out)
        return 0
    theme = theme_for(out, force_disable=opts.no_color)
    print(_format(rows, abstention_count, theme), file=out)
    return 0


def _load_rows(journal: SqliteJournal) -> list[_Row]:
    settlements = {s.prediction_id: s for s in journal.list_settlements()}
    predictions = sorted(journal.list_predictions(), key=lambda p: (p.horizon, p.ticker))
    return [_Row(prediction=p, settlement=settlements.get(p.prediction_id)) for p in predictions]


def _count_abstentions(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    store = SqliteAbstentionStore(path)
    try:
        return store.count()
    finally:
        store.close()


def _format(rows: list[_Row], abstention_count: int, theme: Theme) -> str:
    lines = [theme.bold("PREDICTION JOURNAL")]
    if not rows:
        lines.append("  No predictions recorded yet; the journal is empty.")
    open_rows = [r for r in rows if r.settlement is None]
    settled_rows = [r for r in rows if r.settlement is not None]
    lines.append(
        f"  {len(open_rows)} open, {len(settled_rows)} settled, {abstention_count} abstained."
    )
    for row in rows:
        lines.append(_format_row(row, theme))
    return "\n".join(lines)


def _format_row(row: _Row, theme: Theme) -> str:
    pred = row.prediction
    head = (
        f"    {pred.ticker} {pred.direction.value} p={pred.probability:.3f} horizon={pred.horizon}"
    )
    if row.settlement is None:
        return f"{head}  [{theme.paint('OPEN', 'blue')}]"
    settled = row.settlement
    verdict = theme.paint("HIT", "green") if settled.outcome else theme.paint("MISS", "red")
    status = theme.paint("SETTLED", "bold")
    return f"{head}  [{status} {verdict}] alpha_net={settled.alpha_net:+.4f}"


def _as_json(rows: list[_Row], abstention_count: int, opts: JournalOptions) -> str:
    predictions = [_row_json(row) for row in rows]
    payload = {
        "journal": str(opts.db_path),
        "open_count": sum(1 for r in rows if r.settlement is None),
        "settled_count": sum(1 for r in rows if r.settlement is not None),
        "abstained_count": abstention_count,
        "predictions": predictions,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _row_json(row: _Row) -> dict[str, object]:
    pred = row.prediction
    data: dict[str, object] = {
        "ticker": pred.ticker,
        "direction": pred.direction.value,
        "probability": pred.probability,
        "horizon": pred.horizon.isoformat(),
        "knowledge_date": pred.knowledge_date.isoformat(),
        "state": "settled" if row.settlement is not None else "open",
    }
    if row.settlement is not None:
        data["outcome"] = row.settlement.outcome
        data["alpha_net"] = row.settlement.alpha_net
    return data
