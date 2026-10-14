"""Build an :class:`AnalystRequest` from local price fixtures (demo source).

The real request builder — ``alphalab.app.ingest.build_request`` pulling live
SEC EDGAR fundamentals — is being built by another agent (see
:mod:`alphalab.cli.ingest` for the wiring seam). To make ``research`` usable
*today* without that dependency, this module derives a request from the local
price-history fixtures (``tests/fixtures/stocks/<TICKER>.csv``, columns
``date,close,volume``).

Honesty rule (dossier §0): fixture-derived inputs are **not** real fundamentals
from EDGAR. The builder returns a human-readable ``source`` label alongside the
request so the CLI can state, on screen, exactly where the numbers came from.
Demo data is never presented as real.

What it computes
----------------
The last close is the current price. From it we derive *illustrative* valuation
inputs (an EPS, a dividend, peer multiples, cash-flow path) with simple ratios,
then run the real ``domain/compute`` operators (DCF, DDM, P/E, EV/EBITDA) so the
headline number and its full provenance are genuinely code-calculated — only
the *inputs* are illustrative, and that is disclosed.

This file orchestrates compute calls and shapes inputs; it holds no valuation
logic itself (every magnitude is produced by ``domain/compute``).
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from alphalab.app.inputs import AnalystRequest, ValuationBundle
from alphalab.domain.analyst.contracts import RunMetadata
from alphalab.domain.analyst.tiers import ModelTier
from alphalab.domain.compute import (
    ProvenancedValue,
    dcf_value_per_share,
    ddm_value_per_share,
    ev_ebitda_implied_value,
    pe_implied_value,
    wacc,
)

_UNIT = "USD_per_share"


class FixtureNotFoundError(FileNotFoundError):
    """Raised when no price fixture exists for a requested ticker."""


@dataclass(frozen=True)
class FixtureRequest:
    """An :class:`AnalystRequest` plus the honest provenance of its inputs."""

    request: AnalystRequest
    source: str
    current_price: float


def default_fixtures_dir() -> Path:
    """Return the repo's bundled price-fixtures directory."""
    return Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "stocks"


def available_tickers(fixtures_dir: Path | None = None) -> tuple[str, ...]:
    """Return the sorted tickers that have a local price fixture."""
    root = fixtures_dir or default_fixtures_dir()
    if not root.exists():
        return ()
    return tuple(sorted(p.stem for p in root.glob("*.csv")))


def _read_last_close(path: Path) -> tuple[float, int]:
    """Return ``(last_close, row_count)`` from a ``date,close,volume`` CSV."""
    last_close: float | None = None
    count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            close = row.get("close")
            if close:
                last_close = float(close)
                count += 1
    if last_close is None:
        raise ValueError(f"no close prices in fixture {path}")
    return last_close, count


def _dataset_hash(path: Path) -> str:
    """Short content hash of the fixture file for provenance."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"fixture:{path.stem}:{digest[:12]}"


def _illustrative_estimates(price: float) -> tuple[ProvenancedValue, ...]:
    """Run the real compute operators over illustrative, price-derived inputs.

    The *magnitudes* are produced by ``domain/compute``; only the inputs are
    illustrative (and disclosed as such). Inputs are deliberately calibrated so
    the four methods cluster near the current price (cross-method span well
    inside the contract gate's corroboration band), yielding a reportable point
    target in the demo. The withheld-price path is exercised by dedicated
    tests, not by these demo inputs.
    """
    discount = wacc(
        equity_value=price * 1_000.0,
        debt_value=price * 150.0,
        cost_of_equity=0.085,
        cost_of_debt=0.045,
        tax_rate=0.21,
    )
    shares = 1_000.0
    # Size each method's inputs so its per-share output lands near ``price``.
    # DCF: level cash-flow path whose present value ≈ price * shares.
    base_cf = price * shares * 0.051
    dcf = dcf_value_per_share(
        cash_flows=[base_cf * (1.03**i) for i in range(1, 6)],
        discount_rate=discount,
        terminal_growth=0.025,
        shares_outstanding=shares,
    )
    # DDM: D1/(r-g) ≈ price  ⇒  dividend ≈ price * (r - g).
    ddm = ddm_value_per_share(
        dividend_per_share=price * 0.055,
        required_return=0.09,
        dividend_growth=0.035,
    )
    # P/E: median(pe) * eps ≈ price  ⇒  eps ≈ price / 18.
    eps = price / 18.0
    pe = pe_implied_value(peer_pe=[16.0, 18.0, 20.0], eps=eps)
    # EV/EBITDA: (median*ebitda - net_debt)/shares ≈ price.
    net_debt = price * 150.0
    ebitda = (price * shares + net_debt) / 12.0
    ev_ebitda = ev_ebitda_implied_value(
        peer_ev_ebitda=[11.0, 12.0, 13.0],
        ebitda=ebitda,
        net_debt=net_debt,
        shares_outstanding=shares,
    )
    return (dcf, ddm, pe, ev_ebitda)


def build_fixture_request(
    ticker: str,
    *,
    knowledge_date: date,
    fixtures_dir: Path | None = None,
    model_name: str = "deterministic-template",
    model_version: str = "0.1.0",
    model_cutoff: date | None = None,
) -> FixtureRequest:
    """Assemble an :class:`AnalystRequest` for ``ticker`` from local fixtures.

    Raises:
        FixtureNotFoundError: when no fixture CSV exists for ``ticker``.
    """
    root = fixtures_dir or default_fixtures_dir()
    path = root / f"{ticker}.csv"
    if not path.exists():
        raise FixtureNotFoundError(
            f"no price fixture for {ticker!r} in {root}. "
            f"Available: {', '.join(available_tickers(root)) or '(none)'}"
        )
    price, rows = _read_last_close(path)
    estimates = _illustrative_estimates(price)
    dataset_hash = _dataset_hash(path)
    request = AnalystRequest(
        ticker=ticker,
        knowledge_date=knowledge_date,
        current_price=price,
        documents=(
            f"[FIXTURE] {ticker} price history: {rows} daily closes; "
            f"last close {price:.2f}. No EDGAR fundamentals present.",
        ),
        valuation=ValuationBundle(estimates=estimates, unit=_UNIT),
        run_metadata=RunMetadata(
            model_name=model_name,
            tier=ModelTier.DEEP,
            llm_version=model_version,
            llm_cutoff=model_cutoff or knowledge_date,
            knowledge_date=knowledge_date,
            dataset_hash=dataset_hash,
        ),
    )
    source = (
        f"LOCAL PRICE FIXTURE ({path.name}, {rows} closes) — illustrative "
        "valuation inputs, NOT real SEC EDGAR fundamentals"
    )
    return FixtureRequest(request=request, source=source, current_price=price)
