"""Download the de-biased Alpaca universe and its adjusted daily bars to disk.

This script closes the project's largest declared debt — survivorship bias — by
pulling the US-equity universe *including delisted names* and the dividend/split
adjusted history for each. Everything lands under ``data/alpaca/`` (already
gitignored via ``data/``).

Properties
----------
* **Credentials from the environment only.** ``AlpacaSettings`` reads
  ``ALPHALAB_ALPACA_KEY_ID`` / ``ALPHALAB_ALPACA_SECRET_KEY``; if they are
  missing the script exits with a clear message and never prints the secret.
* **Concurrent.** Bars are fetched with a :class:`ThreadPoolExecutor`; a prior
  sequential pass took ~20 minutes and was unacceptable. Alpaca's own rate
  limiter inside the client still bounds the request rate.
* **Idempotent.** A symbol whose output file already exists is skipped, so a
  re-run resumes rather than re-downloads.
* **Fault tolerant.** A symbol that fails is logged and recorded; it does not
  abort the rest of the run.

Read-only by construction: it touches ``/v2/assets`` and ``/v2/stocks/bars``
only. There is no code path here that creates, sends, modifies, or cancels an
order.

Usage::

    ALPHALAB_ALPACA_KEY_ID=... ALPHALAB_ALPACA_SECRET_KEY=... \\
        uv run python scripts/fetch_alpaca_universe.py --start 2016-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError

from alphalab.adapters.alpaca_assets import AlpacaAssetsClient, UniverseSnapshot
from alphalab.adapters.alpaca_bars import AlpacaBarsClient, DataFeed
from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.edgar import HttpResponse
from alphalab.domain.types import Ticker

_DEFAULT_OUT = Path("data/alpaca")
_DEFAULT_WORKERS = 8


class HttpxTransport:
    """Real HTTP transport, shaped like the shared EDGAR ``Transport`` protocol."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        response = self._client.get(url, headers=headers)
        payload: object = {}
        if response.status_code == 200:
            payload = response.json()
        # HttpResponse.json is typed dict, but Alpaca's assets endpoint returns a
        # list; the adapters accept ``object`` and narrow at runtime.
        return HttpResponse(status_code=response.status_code, json=payload)  # type: ignore[arg-type]


def _load_settings() -> AlpacaSettings:
    """Build settings from the environment, or exit with an actionable message."""
    try:
        return AlpacaSettings()  # type: ignore[call-arg]  # values come from env/.env
    except ValidationError:
        sys.stderr.write(
            "error: missing Alpaca credentials. Set ALPHALAB_ALPACA_KEY_ID and "
            "ALPHALAB_ALPACA_SECRET_KEY in the environment or an untracked .env.\n"
        )
        raise SystemExit(2) from None


def _write_universe(snapshot: UniverseSnapshot, out_dir: Path) -> Path:
    """Persist the universe and return the path written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "universe.json"
    payload = {
        "active_count": snapshot.active_count,
        "inactive_count": snapshot.inactive_count,
        "survivorship_ratio": snapshot.survivorship_ratio,
        "active": [asdict(a) for a in snapshot.active],
        "inactive": [asdict(a) for a in snapshot.inactive],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _bars_path(out_dir: Path, symbol: str) -> Path:
    return out_dir / "bars" / f"{symbol}.json"


def _fetch_one(
    client: AlpacaBarsClient, ticker: Ticker, *, start: date, end: date, out_dir: Path
) -> tuple[str, int]:
    """Fetch and persist bars for one symbol; return (symbol, bar_count)."""
    series = client.daily_bars([ticker], start=start, end=end).get(ticker, ())
    path = _bars_path(out_dir, str(ticker))
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"on": b.on.isoformat(), "close": b.close, "volume": b.volume} for b in series]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return str(ticker), len(series)


def _pending_symbols(snapshot: UniverseSnapshot, out_dir: Path) -> list[Ticker]:
    """All US-equity tickers whose bar file does not yet exist (idempotency)."""
    pending: list[Ticker] = []
    for asset in snapshot.all_assets:
        if not _bars_path(out_dir, asset.symbol).exists():
            pending.append(asset.ticker)
    return pending


def _run(
    *,
    start: date,
    end: date,
    out_dir: Path,
    feed: DataFeed,
    workers: int,
) -> int:
    settings = _load_settings()
    transport = HttpxTransport()
    assets_client = AlpacaAssetsClient(settings, transport=transport)
    bars_client = AlpacaBarsClient(settings, transport=transport, feed=feed)

    snapshot = assets_client.fetch_universe()
    universe_path = _write_universe(snapshot, out_dir)
    sys.stdout.write(
        f"universe: active={snapshot.active_count} inactive={snapshot.inactive_count} "
        f"ratio={snapshot.survivorship_ratio:.3f} -> {universe_path}\n"
    )

    pending = _pending_symbols(snapshot, out_dir)
    sys.stdout.write(f"fetching bars for {len(pending)} symbols with {workers} workers\n")

    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one, bars_client, t, start=start, end=end, out_dir=out_dir): t
            for t in pending
        }
        for future in as_completed(futures):
            symbol = str(futures[future])
            try:
                _, count = future.result()
            except Exception as exc:  # record the failure and continue
                failures[symbol] = f"{type(exc).__name__}: {exc}"
                sys.stderr.write(f"  FAIL {symbol}: {failures[symbol]}\n")
            else:
                sys.stdout.write(f"  ok   {symbol}: {count} bars\n")

    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        sys.stderr.write(f"{len(failures)} symbols failed; see {out_dir / 'failures.json'}\n")
    return len(failures)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the de-biased Alpaca universe and bars.")
    parser.add_argument("--start", type=_parse_date, default=_parse_date("2016-01-01"))
    parser.add_argument("--end", type=_parse_date, default=date.today())
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--feed", type=DataFeed, choices=list(DataFeed), default=DataFeed.IEX)
    parser.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return _run(
        start=args.start,
        end=args.end,
        out_dir=args.out,
        feed=args.feed,
        workers=args.workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
