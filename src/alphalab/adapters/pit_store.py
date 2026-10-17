"""Partitioned Parquet store with point-in-time reads via DuckDB.

The store has exactly one read method, :meth:`ParquetPITStore.query`, and it
requires a ``knowledge_date``. There is deliberately no public method that
returns rows without a temporal cut-off, so a careless caller cannot bypass
the anti-leakage filter. The ``observed_at <= knowledge_date`` predicate is
applied inside the SQL (as a bound parameter), not by the caller.

All value filters use DuckDB ``?`` placeholders — never string interpolation —
per the BICE anti-injection guideline. The only interpolated token is the
Parquet glob path, which is derived from a validated store root.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from alphalab.domain.pit import KnowledgeDate, PITRecord
from alphalab.domain.types import Ticker

# Allow-listed physical column names. Query building never accepts a column
# name from the caller; this frozenset is the single source of truth.
_COLUMNS = ("ticker", "field", "value", "as_of", "observed_at", "source", "period_start")

_SELECT_SQL = (
    "SELECT ticker, field, value, as_of, observed_at, source, period_start "
    "FROM read_parquet(?) "
    "WHERE ticker = ? AND field = ? AND observed_at <= ? "
    "ORDER BY observed_at, value"
)

_SHARED_CON = duckdb.connect(database=":memory:")


class ParquetPITStore:
    """Append-only PIT store backed by Parquet files under a root directory."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._part = self._root / "facts.parquet"
        self._con = _SHARED_CON

    def write(self, records: Iterable[PITRecord]) -> None:
        """Persist records as Parquet, merging and deduplicating with any existing partition."""
        new_rows = [self._to_row(r) for r in records]
        if not new_rows:
            return
        if self._part.exists():
            existing = pq.read_table(self._part)
            ex_cols = existing.to_pydict()
            ex_keys = {
                (
                    ex_cols["ticker"][i],
                    ex_cols["field"][i],
                    ex_cols["value"][i],
                    ex_cols["as_of"][i],
                    ex_cols["observed_at"][i],
                    ex_cols["source"][i],
                    ex_cols["period_start"][i],
                )
                for i in range(existing.num_rows)
            }
            novel_rows = [
                r
                for r in new_rows
                if (
                    r["ticker"],
                    r["field"],
                    r["value"],
                    r["as_of"],
                    r["observed_at"],
                    r["source"],
                    r["period_start"],
                )
                not in ex_keys
            ]
            if not novel_rows:
                return
            table = pa.concat_tables([existing, self._rows_to_table(novel_rows)])
        else:
            table = self._rows_to_table(new_rows)
        pq.write_table(table, self._part)

    def query(
        self,
        *,
        ticker: Ticker,
        field: str,
        knowledge_date: KnowledgeDate,
    ) -> list[PITRecord]:
        """Return records for ``ticker``/``field`` knowable at ``knowledge_date``.

        The temporal predicate is enforced in SQL, so no row observed after
        the knowledge date can ever be returned.
        """
        if not self._part.exists():
            return []
        glob = self._parquet_glob()
        params = [glob, str(ticker), field, knowledge_date.value]
        cursor = self._con.execute(_SELECT_SQL, params)
        fetched = cursor.fetchall()
        return [self._to_record(row) for row in fetched]

    def _parquet_glob(self) -> str:
        """Return the validated Parquet path used by ``read_parquet``.

        The path is derived from the store root we created, never from caller
        input, so interpolating it into the DuckDB call cannot inject SQL.
        """
        resolved = self._part.resolve()
        root_resolved = self._root.resolve()
        if os.path.commonpath([resolved, root_resolved]) != str(root_resolved):
            raise ValueError(f"partition path escapes store root: {resolved}")
        return str(resolved)

    @staticmethod
    def _to_row(record: PITRecord) -> dict[str, object]:
        return {
            "ticker": str(record.ticker),
            "field": record.field,
            "value": record.value,
            "as_of": record.as_of,
            "observed_at": record.observed_at,
            "source": record.source,
            "period_start": record.period_start,
        }

    @staticmethod
    def _rows_to_table(rows: list[dict[str, object]]) -> pa.Table:
        schema = pa.schema(
            [
                ("ticker", pa.string()),
                ("field", pa.string()),
                ("value", pa.float64()),
                ("as_of", pa.date32()),
                ("observed_at", pa.date32()),
                ("source", pa.string()),
                ("period_start", pa.date32()),
            ]
        )
        columns = {name: [row[name] for row in rows] for name in _COLUMNS}
        return pa.table(columns, schema=schema)

    @staticmethod
    def _to_record(row: tuple[object, ...]) -> PITRecord:
        ticker, field, value, as_of, observed_at, source, period_start = row
        return PITRecord(
            ticker=Ticker(str(ticker)),
            field=str(field),
            value=float(value),  # type: ignore[arg-type]
            as_of=_as_date(as_of),
            observed_at=_as_date(observed_at),
            source=None if source is None else str(source),
            period_start=None if period_start is None else _as_date(period_start),
        )


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    raise TypeError(f"expected date from store, got {type(value)!r}")
