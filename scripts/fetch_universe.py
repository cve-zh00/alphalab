#!/usr/bin/env python
"""Download a mechanically-sampled US equity universe.

Why this script exists
----------------------
The committed 48-name fixture was assembled from memory in 2025, so its
composition embeds hindsight: equal-weighting it beat SPY by +7.47% a year with
no signal at all. No statistical gate can detect that, because the returns are
genuine — the bias is in *which names were chosen*.

This script removes the author from the selection. Tickers are drawn by a
seeded pseudo-random sample over the SEC's full company list, an ordering that
has nothing to do with performance, so the resulting universe is reproducible
and auditable but not curated.

Remaining bias, stated plainly: the price source does not serve most delisted
names, so survivors are still over-represented. This narrows the bias; it does
not eliminate it. Only historical index membership with delisted terminal
values would.

Usage::

    ALPHALAB_EDGAR_USER_AGENT="you you@example.com" \
        uv run python scripts/fetch_universe.py --candidates 600 --out data/prices
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from pathlib import Path

_SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
_CHART = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
_SAMPLE_SEED = 42
_MIN_ROWS = 1500


def _fetch(url: str, *, user_agent: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload: bytes = response.read()
        return payload


def _sec_tickers(user_agent: str) -> list[str]:
    """Every ticker the SEC lists, sorted for determinism."""
    payload = json.loads(_fetch(_SEC_TICKERS, user_agent=user_agent))
    tickers = {
        str(entry["ticker"]).upper()
        for entry in payload.values()
        if isinstance(entry, dict) and entry.get("ticker")
    }
    return sorted(t for t in tickers if t.isalpha() and len(t) <= 5)


def _sample(tickers: list[str], n: int) -> list[str]:
    """Seeded sample, so the universe is reproducible from the seed alone."""
    import random

    rng = random.Random(_SAMPLE_SEED)
    return rng.sample(tickers, min(n, len(tickers)))


def _download_prices(ticker: str, user_agent: str) -> list[tuple[str, float, float]]:
    """Daily total-return closes and volumes, or an empty list when unavailable.

    Uses ``adjclose`` rather than ``quote.close``. Yahoo's raw close is already
    split-adjusted but *not* dividend-adjusted, so a raw series understates the
    return of any dividend payer and makes payers and non-payers
    non-comparable in a cross-sectional ranking.
    """
    url = _CHART.format(ticker=ticker) + "?range=20y&interval=1d"
    try:
        payload = json.loads(_fetch(url, user_agent=user_agent))
    except Exception:
        return []
    try:
        result = payload["chart"]["result"][0]
        stamps = result["timestamp"]
        closes = result["indicators"]["adjclose"][0]["adjclose"]
        volumes = result["indicators"]["quote"][0]["volume"]
    except (KeyError, IndexError, TypeError):
        return []

    import datetime as dt

    rows: list[tuple[str, float, float]] = []
    for stamp, close, volume in zip(stamps, closes, volumes, strict=False):
        if close is None or close <= 0:
            continue
        day = dt.datetime.fromtimestamp(stamp, tz=dt.UTC).date().isoformat()
        rows.append((day, float(close), float(volume or 0.0)))
    return rows


def _write(path: Path, rows: list[tuple[str, float, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "close", "volume"])
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=600)
    parser.add_argument("--out", type=Path, default=Path("data/prices"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)

    user_agent = os.environ.get("ALPHALAB_EDGAR_USER_AGENT")
    if not user_agent:
        print("set ALPHALAB_EDGAR_USER_AGENT to identify yourself", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    universe = _sample(_sec_tickers(user_agent), args.candidates)
    print(f"sampled {len(universe)} candidates from the SEC list (seed {_SAMPLE_SEED})")

    pending = [t for t in universe if not (args.out / f"{t}.csv").exists()]
    already = len(universe) - len(pending)
    print(f"{already} already on disk, fetching {len(pending)} with {args.workers} workers")

    kept = _fetch_all(pending, args.out, user_agent, args.workers)
    total = already + kept
    print(f"done: {total} tickers with >= {_MIN_ROWS} rows in {args.out}")
    return 0


def _fetch_all(pending: list[str], out: Path, user_agent: str, workers: int) -> int:
    """Download concurrently; a sequential pass is far too slow for 600 names.

    Failures are counted, not raised: a universe definition that collapses
    because one ticker 404s would not be reproducible.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    kept = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_prices, ticker, user_agent): ticker for ticker in pending}
        for future in as_completed(futures):
            ticker = futures[future]
            done += 1
            try:
                rows = future.result()
            except Exception:
                rows = []
            if len(rows) >= _MIN_ROWS:
                _write(out / f"{ticker}.csv", rows)
                kept += 1
            if done % 50 == 0:
                print(f"  {done}/{len(pending)} fetched, {kept} kept")
    return kept


if __name__ == "__main__":
    raise SystemExit(main())
