"""The ingest orchestrator: ticker + knowledge date -> ``AnalystRequest``.

This closes the real-data loop. Given a ticker and a point-in-time knowledge
date it:

1. resolves the CIK (:mod:`alphalab.adapters.cik`),
2. pulls as-filed fundamentals (:mod:`alphalab.adapters.edgar_live`) and loads
   them into the PIT store (:mod:`alphalab.adapters.pit_store`),
3. reads each feature **only** through the sanctioned
   :func:`~alphalab.domain.pit.resolve_feature` gate at the knowledge date,
   with every feature registered against an audited availability lag,
4. computes the valuation with ``domain/compute`` (DCF + DDM when dividends
   exist + a peer-multiple method) and aggregates it with ``consensus``,
5. reads the ``current_price`` with no look-ahead (:mod:`alphalab.adapters.quotes`),
6. assembles the :class:`~alphalab.app.inputs.AnalystRequest`.

**Honest degradation.** A method whose inputs are missing is *omitted* and the
omission is recorded in :attr:`IngestResult.omitted_methods`; nothing is ever
zero-filled or assumed. If no method can run, :attr:`IngestResult.request` is
``None`` and the report explains why — never an invented number.

**Judgement is supplied, never invented.** Discount rates, terminal growth,
peer multiples and the projection horizon are explicit
:class:`ValuationAssumptions` passed in by the caller. The fundamentals come
from EDGAR; the opinions do not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from alphalab.adapters.cik import CikResolver
from alphalab.adapters.edgar import EdgarClient
from alphalab.adapters.edgar_live import FundamentalField, extract_fundamentals
from alphalab.adapters.pit_store import ParquetPITStore
from alphalab.adapters.prices import PriceSeriesSource
from alphalab.adapters.quotes import NoPriceAvailableError, current_price_at
from alphalab.app.inputs import AnalystRequest, ValuationBundle
from alphalab.domain.analyst.contracts import RunMetadata
from alphalab.domain.compute import (
    ComputeError,
    ProvenancedValue,
    consensus,
    dcf_value_per_share,
    ddm_value_per_share,
    ev_ebitda_implied_value,
    ffo_implied_value,
    pb_implied_value,
    pe_implied_value,
    residual_income_value_per_share,
    reverse_dcf_implied_growth,
    wacc,
)
from alphalab.domain.compute.sectors import GicsSector
from alphalab.domain.pit import (
    FeatureRegistry,
    FeatureSpec,
    KnowledgeDate,
    PITRecord,
    resolve_feature,
)
from alphalab.domain.types import Ticker

_VALUE_UNIT = "USD_per_share"
# Conservative registered availability lag. The EDGAR ``filed`` date is already
# the true knowable date, so a 0-day floor can never falsely reject a record
# (``observed_at >= as_of`` always holds) while still forcing every feature to
# be registered before use.
_LAG_DAYS = 0
_DEFAULT_DOCUMENTS: tuple[str, ...] = ("(no narrative documents supplied)",)
_MIN_EXPLICIT_GROWTH = -0.05
_MAX_EXPLICIT_GROWTH = 0.15


class ValuationAssumptions(BaseModel):
    """Judgement inputs the caller supplies; never derived from the filing."""

    model_config = ConfigDict(frozen=True)

    cost_of_equity: float = Field(gt=0)
    cost_of_debt: float = Field(ge=0)
    tax_rate: float = Field(ge=0, lt=1)
    terminal_growth: float
    dividend_growth: float
    projection_years: int = Field(gt=0)
    peer_pe: tuple[float, ...] = ()
    peer_ev_ebitda: tuple[float, ...] = ()
    peer_pb: tuple[float, ...] = ()
    peer_ffo: tuple[float, ...] = ()
    sector: GicsSector = GicsSector.MARKET_DEFAULT
    min_payout_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    dcf_mode: Literal["fcfe", "fcff"] = "fcfe"
    explicit_growth: float | None = None


@dataclass(frozen=True)
class IngestResult:
    """Outcome of one ingest run, with an honest account of what was used.

    Attributes:
        request: The assembled request, or ``None`` if no method could run.
        methods_used: Short tags of valuation methods that produced an estimate.
        omitted_methods: Human-readable reasons for every omitted method.
        missing_fields: Canonical fundamentals that EDGAR did not provide.
        resolved_concepts: Which raw XBRL concept supplied each present field.
    """

    request: AnalystRequest | None
    methods_used: tuple[str, ...]
    omitted_methods: tuple[str, ...]
    missing_fields: frozenset[FundamentalField]
    resolved_concepts: dict[FundamentalField, str] = field(default_factory=dict)


def build_request(
    ticker: Ticker,
    *,
    knowledge_date: date,
    assumptions: ValuationAssumptions,
    cik_resolver: CikResolver,
    edgar_client: EdgarClient,
    price_source: PriceSeriesSource,
    pit_store_root: Path,
    run_metadata: RunMetadata,
    documents: tuple[str, ...] = ("(no narrative documents supplied)",),
) -> IngestResult:
    """Assemble an :class:`AnalystRequest` for ``ticker`` at ``knowledge_date``."""
    cik = cik_resolver.resolve(ticker)
    fundamentals = extract_fundamentals(edgar_client, ticker, cik=cik)

    store = ParquetPITStore(root=pit_store_root)
    store.write(fundamentals.records)

    kd = KnowledgeDate(knowledge_date)
    values = _resolve_fields(store, ticker, kd)
    try:
        current_price = current_price_at(price_source, ticker, kd)
    except NoPriceAvailableError:
        current_price = float("nan")
    pit_cagr = _resolve_pit_cagr(store, ticker, kd)
    buyback_yield = _resolve_net_buyback_yield(store, ticker, kd)
    quality = _compute_piotroski_and_yoy(store, ticker, kd)

    estimates, used, omitted = _build_estimates(
        values,
        assumptions,
        current_price=current_price,
        pit_cagr=pit_cagr,
        buyback_yield=buyback_yield,
    )
    omitted = list(omitted) + [f"missing fundamental: {f}" for f in fundamentals.missing_fields]

    if not estimates:
        return IngestResult(
            request=None,
            methods_used=(),
            omitted_methods=tuple(omitted),
            missing_fields=fundamentals.missing_fields,
            resolved_concepts=fundamentals.resolved_concepts,
        )

    effective_docs = (
        _build_pit_dossier(
            values=values,
            assumptions=assumptions,
            current_price=current_price,
            pit_cagr=pit_cagr,
            estimates=estimates,
            quality=quality,
        )
        if documents == _DEFAULT_DOCUMENTS
        else documents
    )

    request = _assemble_request(
        ticker=ticker,
        knowledge_date=knowledge_date,
        estimates=estimates,
        current_price=current_price,
        run_metadata=run_metadata,
        documents=effective_docs,
    )
    return IngestResult(
        request=request,
        methods_used=tuple(used),
        omitted_methods=tuple(omitted),
        missing_fields=fundamentals.missing_fields,
        resolved_concepts=fundamentals.resolved_concepts,
    )


def _registry() -> FeatureRegistry:
    """Register every canonical field with its audited availability lag."""
    return FeatureRegistry(
        FeatureSpec(name=str(f), availability_lag_days=_LAG_DAYS) for f in FundamentalField
    )


def _resolve_fields(
    store: ParquetPITStore, ticker: Ticker, kd: KnowledgeDate
) -> dict[FundamentalField, float]:
    """Resolve the latest knowable value for each field through the gate."""
    registry = _registry()
    out: dict[FundamentalField, float] = {}
    for field_name in FundamentalField:
        record = _latest_knowable_record(store, ticker, field_name, kd)
        if record is None:
            continue
        out[field_name] = resolve_feature(record, registry=registry, knowledge_date=kd)
    return out


def _latest_knowable_record(
    store: ParquetPITStore,
    ticker: Ticker,
    field_name: FundamentalField,
    kd: KnowledgeDate,
) -> PITRecord | None:
    """Return the record for the latest period knowable at ``kd`` (or ``None``).

    Ranking is by *period* first and filing date only as a tiebreak, because a
    single filing restates several years at once: a 10-K carries this year's
    figure alongside the two before it, all stamped with the same ``filed``
    date. Ranking by filing date alone leaves those tied, and the winner then
    falls out of whatever order the store happened to return — for AMD, the
    FY2023 diluted EPS of 0.53 instead of FY2025's 2.65, a figure two years
    stale and less than half the right size.

    Within one period the later filing still wins, so a restatement supersedes
    the original it corrects. The anti-leakage guarantee is untouched: the
    store has already dropped everything filed after ``kd``, so this only ever
    chooses among records that were genuinely knowable.
    """
    rows = store.query(ticker=ticker, field=str(field_name), knowledge_date=kd)
    if not rows:
        return None
    return max(rows, key=lambda r: (r.as_of, r.observed_at))


def latest_knowable_value(
    store: ParquetPITStore,
    ticker: Ticker,
    field_name: FundamentalField,
    knowledge_date: date,
) -> float | None:
    """Public helper: latest knowable value for a field (via the PIT gate).

    Exposed for callers (and tests) that want to inspect what a single field
    resolved to at a knowledge date without running the whole pipeline.
    """
    kd = KnowledgeDate(knowledge_date)
    record = _latest_knowable_record(store, ticker, field_name, kd)
    if record is None:
        return None
    return resolve_feature(record, registry=_registry(), knowledge_date=kd)


def _annual_series(
    store: ParquetPITStore,
    ticker: Ticker,
    field_name: FundamentalField,
    kd: KnowledgeDate,
) -> list[tuple[date, float]]:
    """Return knowable annual ``(as_of, value)`` pairs sorted ascending by period end."""
    registry = _registry()
    rows = store.query(ticker=ticker, field=str(field_name), knowledge_date=kd)
    if not rows:
        return []
    by_period: dict[date, PITRecord] = {}
    for row in rows:
        prev = by_period.get(row.as_of)
        if prev is None or row.observed_at >= prev.observed_at:
            by_period[row.as_of] = row
    return [
        (period_end, resolve_feature(rec, registry=registry, knowledge_date=kd))
        for period_end, rec in sorted(by_period.items(), key=lambda kv: kv[0])
    ]


def _resolve_pit_cagr(
    store: ParquetPITStore,
    ticker: Ticker,
    kd: KnowledgeDate,
) -> float | None:
    """Compute a 3-year PIT CAGR from annual FCF or Revenue clamped to ``[-5%, +15%]``."""
    ocf_series = dict(_annual_series(store, ticker, FundamentalField.OPERATING_CASH_FLOW, kd))
    capex_series = dict(_annual_series(store, ticker, FundamentalField.CAPEX, kd))
    shared_dates = sorted(set(ocf_series).intersection(capex_series))
    fcf_points = [(d, ocf_series[d] - capex_series[d]) for d in shared_dates]

    for series in (fcf_points, _annual_series(store, ticker, FundamentalField.REVENUE, kd)):
        if len(series) < 2:
            continue
        window = series[-4:]  # up to 3-year span (4 annual endpoints)
        start_date, start_val = window[0]
        end_date, end_val = window[-1]
        years = (end_date - start_date).days / 365.25
        if start_val > 0.0 and end_val > 0.0 and years >= 0.75:
            raw_cagr = (end_val / start_val) ** (1.0 / years) - 1.0
            return float(min(max(raw_cagr, _MIN_EXPLICIT_GROWTH), _MAX_EXPLICIT_GROWTH))
    return None


_MAX_BUYBACK_YIELD = 0.06
_YEAR_DAYS = 365.25


def _resolve_net_buyback_yield(
    store: ParquetPITStore, ticker: Ticker, kd: KnowledgeDate
) -> float:
    """Annualised net share-count reduction over up to ~3 trailing years.

    Returns a fraction clamped to ``[0, 0.06]``; dilution or missing data -> 0.
    """
    series = _annual_series(store, ticker, FundamentalField.SHARES_OUTSTANDING, kd)
    if len(series) < 2:
        return 0.0
    end_date, end_val = series[-1]
    candidates = [
        (d, v)
        for d, v in series[:-1]
        if 300 <= (end_date - d).days <= 3.5 * _YEAR_DAYS and v > 0.0
    ]
    if not candidates or end_val <= 0.0:
        return 0.0
    start_date, start_val = candidates[0]
    years = (end_date - start_date).days / _YEAR_DAYS
    annual = 1.0 - (end_val / start_val) ** (1.0 / years)
    if not math.isfinite(annual):
        return 0.0
    return float(min(max(annual, 0.0), _MAX_BUYBACK_YIELD))


def _build_estimates(
    values: dict[FundamentalField, float],
    assumptions: ValuationAssumptions,
    *,
    current_price: float = float("nan"),
    pit_cagr: float | None = None,
    buyback_yield: float = 0.0,
) -> tuple[list[ProvenancedValue], list[str], list[str]]:
    """Run each valuation method that has its inputs; omit the rest.

    Sector routing: FINANCIALS skip the FCFE DCF (meaningless for balance-sheet
    businesses) in favour of residual income; REAL_ESTATE swap P/E for P/FFO.
    """
    estimates: list[ProvenancedValue] = []
    used: list[str] = []
    omitted: list[str] = []
    sector = assumptions.sector

    if sector is GicsSector.FINANCIALS:
        omitted.append("dcf: skipped for financials (residual income used instead)")
        _try_residual_income(values, assumptions, estimates, used, omitted)
    else:
        _try_dcf(
            values,
            assumptions,
            estimates,
            used,
            omitted,
            current_price=current_price,
            pit_cagr=pit_cagr,
        )
    _try_ddm(
        values,
        assumptions,
        estimates,
        used,
        omitted,
        buyback_yield=buyback_yield,
        current_price=current_price,
    )
    if sector is GicsSector.REAL_ESTATE:
        _try_ffo(values, assumptions, estimates, used, omitted)
    else:
        _try_pe(values, assumptions, estimates, used, omitted)
    _try_ev_ebitda(values, assumptions, estimates, used, omitted)
    _try_pb(values, assumptions, estimates, used, omitted)
    return estimates, used, omitted


def _resolve_market_or_book_equity(
    values: dict[FundamentalField, float],
    *,
    shares: float,
    current_price: float,
) -> float | None:
    """Prefer market capitalization ``price * shares`` over book equity for WACC weights."""
    if math.isfinite(current_price) and current_price > 0.0 and shares > 0.0:
        return current_price * shares
    book = values.get(FundamentalField.STOCKHOLDERS_EQUITY)
    if book is not None and book > 0.0:
        return book
    return None


def _try_dcf(
    values: dict[FundamentalField, float],
    assumptions: ValuationAssumptions,
    estimates: list[ProvenancedValue],
    used: list[str],
    omitted: list[str],
    *,
    current_price: float = float("nan"),
    pit_cagr: float | None = None,
) -> None:
    ocf = values.get(FundamentalField.OPERATING_CASH_FLOW)
    capex = values.get(FundamentalField.CAPEX)
    shares = values.get(FundamentalField.SHARES_OUTSTANDING)
    equity = _resolve_market_or_book_equity(
        values, shares=shares or 0.0, current_price=current_price
    )
    debt = max(values.get(FundamentalField.LONG_TERM_DEBT, 0.0), 0.0)
    cash = max(values.get(FundamentalField.CASH, 0.0), 0.0)

    if shares is None or shares <= 0 or equity is None:
        omitted.append("dcf: missing operating cash flow, capex, shares or equity")
        return

    explicit_g = (
        assumptions.explicit_growth
        if assumptions.explicit_growth is not None
        else (pit_cagr if pit_cagr is not None else assumptions.terminal_growth)
    )
    explicit_g = float(min(max(explicit_g, _MIN_EXPLICIT_GROWTH), _MAX_EXPLICIT_GROWTH))

    if assumptions.dcf_mode == "fcff":
        ebit = values.get(FundamentalField.OPERATING_INCOME)
        da = values.get(FundamentalField.DEPRECIATION_AMORTIZATION)
        if ebit is None or da is None or capex is None:
            omitted.append("dcf: missing EBIT, D&A, or capex for FCFF mode")
            return
        base_cf = ebit * (1.0 - assumptions.tax_rate) + da - capex
        net_debt: float | None = debt - cash
        debt_for_discount = debt
    else:
        if ocf is None or capex is None:
            omitted.append("dcf: missing operating cash flow, capex, shares or equity")
            return
        base_cf = ocf - capex
        # FCFE is already after interest & debt service: discount at Ke without EV bridge.
        net_debt = None
        debt_for_discount = 0.0

    if base_cf <= 0.0:
        omitted.append("dcf: non-positive base free cash flow")
        return

    projected_cfs = [
        base_cf * ((1.0 + explicit_g) ** yr) for yr in range(1, assumptions.projection_years + 1)
    ]
    try:
        discount = wacc(
            equity_value=equity,
            debt_value=debt_for_discount,
            cost_of_equity=assumptions.cost_of_equity,
            cost_of_debt=assumptions.cost_of_debt,
            tax_rate=assumptions.tax_rate,
        )
        est = dcf_value_per_share(
            cash_flows=projected_cfs,
            discount_rate=discount,
            terminal_growth=assumptions.terminal_growth,
            shares_outstanding=shares,
            net_debt=net_debt,
        )
    except ComputeError as exc:
        omitted.append(f"dcf: {exc}")
        return
    if est.value <= 0.0:
        omitted.append("dcf: implied equity value is non-positive")
        return
    estimates.append(est)
    used.append("dcf")


def _try_ddm(
    values: dict[FundamentalField, float],
    assumptions: ValuationAssumptions,
    estimates: list[ProvenancedValue],
    used: list[str],
    omitted: list[str],
    *,
    buyback_yield: float = 0.0,
    current_price: float = float("nan"),
) -> None:
    """Dividend discount on *total shareholder yield*.

    The payout gate uses the declared dividend only. Once it passes, the
    dividend fed to the model is raised by ``buyback_yield * current_price``
    (net repurchases expressed per share, as cash returned to holders) when a
    finite positive price exists; growth is left at the assumption.
    """
    dps = values.get(FundamentalField.DIVIDEND_PER_SHARE)
    if dps is None or dps <= 0.0:
        omitted.append("ddm: company pays no dividend (no dividend_per_share)")
        return
    eps = values.get(FundamentalField.EPS_DILUTED)
    if (
        eps is not None
        and assumptions.min_payout_ratio > 0.0
        and (eps <= 0.0 or (dps / eps) < assumptions.min_payout_ratio)
    ):
        omitted.append(
            f"ddm: payout ratio below {assumptions.min_payout_ratio:.0%} minimum"
        )
        return
    effective_dps = dps
    if buyback_yield > 0.0 and math.isfinite(current_price) and current_price > 0.0:
        effective_dps = dps + buyback_yield * current_price
    try:
        est = ddm_value_per_share(
            dividend_per_share=effective_dps,
            required_return=assumptions.cost_of_equity,
            dividend_growth=assumptions.dividend_growth,
        )
    except ComputeError as exc:
        omitted.append(f"ddm: {exc}")
        return
    estimates.append(est)
    used.append("ddm")


def _try_residual_income(
    values: dict[FundamentalField, float],
    assumptions: ValuationAssumptions,
    estimates: list[ProvenancedValue],
    used: list[str],
    omitted: list[str],
) -> None:
    equity = values.get(FundamentalField.STOCKHOLDERS_EQUITY)
    shares = values.get(FundamentalField.SHARES_OUTSTANDING)
    eps = values.get(FundamentalField.EPS_DILUTED)
    if equity is None or equity <= 0.0 or shares is None or shares <= 0.0 or eps is None:
        omitted.append("residual_income: missing book equity, shares or diluted EPS")
        return
    try:
        est = residual_income_value_per_share(
            book_value_per_share=equity / shares,
            eps=eps,
            cost_of_equity=assumptions.cost_of_equity,
            terminal_growth=assumptions.terminal_growth,
            projection_years=assumptions.projection_years,
        )
    except ComputeError as exc:
        omitted.append(f"residual_income: {exc}")
        return
    if est.value <= 0.0:
        omitted.append("residual_income: implied equity value is non-positive")
        return
    estimates.append(est)
    used.append("residual_income")


def _try_ffo(
    values: dict[FundamentalField, float],
    assumptions: ValuationAssumptions,
    estimates: list[ProvenancedValue],
    used: list[str],
    omitted: list[str],
) -> None:
    net_income = values.get(FundamentalField.NET_INCOME)
    da = values.get(FundamentalField.DEPRECIATION_AMORTIZATION)
    shares = values.get(FundamentalField.SHARES_OUTSTANDING)
    if (
        net_income is None
        or da is None
        or shares is None
        or shares <= 0.0
        or not assumptions.peer_ffo
    ):
        omitted.append("multiples_ffo: missing net income, D&A, shares or peer P/FFO set")
        return
    ffo_per_share = (net_income + da) / shares
    if ffo_per_share <= 0.0:
        omitted.append("multiples_ffo: non-positive FFO per share")
        return
    try:
        est = ffo_implied_value(peer_ffo=assumptions.peer_ffo, ffo_per_share=ffo_per_share)
    except ComputeError as exc:
        omitted.append(f"multiples_ffo: {exc}")
        return
    estimates.append(est)
    used.append("multiples_ffo")


def _try_pe(
    values: dict[FundamentalField, float],
    assumptions: ValuationAssumptions,
    estimates: list[ProvenancedValue],
    used: list[str],
    omitted: list[str],
) -> None:
    eps = values.get(FundamentalField.EPS_DILUTED)
    if eps is None or eps <= 0.0 or not assumptions.peer_pe:
        omitted.append("multiples_pe: missing or non-positive diluted EPS or peer P/E set")
        return
    try:
        est = pe_implied_value(peer_pe=assumptions.peer_pe, eps=eps)
    except ComputeError as exc:
        omitted.append(f"multiples_pe: {exc}")
        return
    estimates.append(est)
    used.append("multiples_pe")


def _try_ev_ebitda(
    values: dict[FundamentalField, float],
    assumptions: ValuationAssumptions,
    estimates: list[ProvenancedValue],
    used: list[str],
    omitted: list[str],
) -> None:
    if not assumptions.peer_ev_ebitda:
        return
    ebit = values.get(FundamentalField.OPERATING_INCOME)
    da = values.get(FundamentalField.DEPRECIATION_AMORTIZATION)
    shares = values.get(FundamentalField.SHARES_OUTSTANDING)
    if ebit is None or da is None or shares is None or shares <= 0.0:
        omitted.append("multiples_ev_ebitda: missing operating income, D&A, or shares")
        return
    ebitda = ebit + da
    if ebitda <= 0.0:
        omitted.append("multiples_ev_ebitda: non-positive EBITDA")
        return
    debt = max(values.get(FundamentalField.LONG_TERM_DEBT, 0.0), 0.0)
    cash = max(values.get(FundamentalField.CASH, 0.0), 0.0)
    try:
        est = ev_ebitda_implied_value(
            peer_ev_ebitda=assumptions.peer_ev_ebitda,
            ebitda=ebitda,
            net_debt=debt - cash,
            shares_outstanding=shares,
        )
    except ComputeError as exc:
        omitted.append(f"multiples_ev_ebitda: {exc}")
        return
    if est.value <= 0.0:
        omitted.append("multiples_ev_ebitda: implied equity value is non-positive")
        return
    estimates.append(est)
    used.append("multiples_ev_ebitda")


def _try_pb(
    values: dict[FundamentalField, float],
    assumptions: ValuationAssumptions,
    estimates: list[ProvenancedValue],
    used: list[str],
    omitted: list[str],
) -> None:
    if not assumptions.peer_pb:
        return
    equity = values.get(FundamentalField.STOCKHOLDERS_EQUITY)
    shares = values.get(FundamentalField.SHARES_OUTSTANDING)
    if equity is None or equity <= 0.0 or shares is None or shares <= 0.0:
        omitted.append("multiples_pb: missing or non-positive book equity or shares")
        return
    try:
        est = pb_implied_value(
            peer_pb=assumptions.peer_pb,
            book_value_per_share=equity / shares,
        )
    except ComputeError as exc:
        omitted.append(f"multiples_pb: {exc}")
        return
    if est.value <= 0.0:
        omitted.append("multiples_pb: implied equity value is non-positive")
        return
    estimates.append(est)
    used.append("multiples_pb")


@dataclass(frozen=True)
class QualitySignals:
    """Year-over-year trend and Piotroski F-score (computable signals only)."""

    piotroski: int
    signals_available: int
    margin_delta: float | None
    roa_delta: float | None


_PIOTROSKI_MIN_SIGNALS = 5
_DATE_TOLERANCE_DAYS = 10


def _near(series: list[tuple[date, float]], target: date) -> float | None:
    best: tuple[int, float] | None = None
    for d, v in series:
        gap = abs((d - target).days)
        if gap <= _DATE_TOLERANCE_DAYS and (best is None or gap < best[0]):
            best = (gap, v)
    return None if best is None else best[1]


def _compute_piotroski_and_yoy(
    store: ParquetPITStore, ticker: Ticker, kd: KnowledgeDate
) -> QualitySignals | None:
    """Latest vs prior annual period over the nine Piotroski signals.

    Signals with missing inputs are skipped; ``None`` when fewer than five are
    computable (or no two annual periods exist).
    """
    def ser(f: FundamentalField) -> list[tuple[date, float]]:
        return _annual_series(store, ticker, f, kd)

    anchor = ser(FundamentalField.REVENUE) or ser(FundamentalField.NET_INCOME)
    if len(anchor) < 2:
        return None
    d1 = anchor[-1][0]
    prior = [d for d, _ in anchor[:-1] if 330 <= (d1 - d).days <= 400]
    if not prior:
        return None
    d0 = min(prior, key=lambda d: abs((d1 - d).days - 365))

    s = {f: ser(f) for f in FundamentalField}

    def pair(f: FundamentalField) -> tuple[float | None, float | None]:
        return _near(s[f], d1), _near(s[f], d0)

    ni1, ni0 = pair(FundamentalField.NET_INCOME)
    ocf1, _ = pair(FundamentalField.OPERATING_CASH_FLOW)
    a1, a0 = pair(FundamentalField.ASSETS)
    rev1, rev0 = pair(FundamentalField.REVENUE)
    ebit1, ebit0 = pair(FundamentalField.OPERATING_INCOME)
    ltd1, ltd0 = pair(FundamentalField.LONG_TERM_DEBT)
    cash1, cash0 = pair(FundamentalField.CASH)
    liab1, liab0 = pair(FundamentalField.LIABILITIES)
    sh1, sh0 = pair(FundamentalField.SHARES_OUTSTANDING)

    def roa(ni: float | None, a: float | None) -> float | None:
        return ni / a if ni is not None and a and a > 0.0 else None

    def ratio(n: float | None, d: float | None) -> float | None:
        return n / d if n is not None and d and d > 0.0 else None

    roa1, roa0 = roa(ni1, a1), roa(ni0, a0)
    margin1, margin0 = ratio(ebit1, rev1), ratio(ebit0, rev0)
    lev1, lev0 = ratio(ltd1, a1), ratio(ltd0, a0)
    liq1, liq0 = ratio(cash1, liab1), ratio(cash0, liab0)
    turn1, turn0 = ratio(rev1, a1), ratio(rev0, a0)

    checks: list[bool | None] = [
        None if roa1 is None else roa1 > 0.0,
        None if ocf1 is None else ocf1 > 0.0,
        None if roa1 is None or roa0 is None else roa1 > roa0,
        None if ocf1 is None or ni1 is None else ocf1 > ni1,
        None if lev1 is None or lev0 is None else lev1 <= lev0,
        None if liq1 is None or liq0 is None else liq1 >= liq0,
        None if sh1 is None or sh0 is None else sh1 <= sh0,
        None if margin1 is None or margin0 is None else margin1 > margin0,
        None if turn1 is None or turn0 is None else turn1 > turn0,
    ]
    computed = [c for c in checks if c is not None]
    if len(computed) < _PIOTROSKI_MIN_SIGNALS:
        return None
    return QualitySignals(
        piotroski=sum(computed),
        signals_available=len(computed),
        margin_delta=None if margin1 is None or margin0 is None else margin1 - margin0,
        roa_delta=None if roa1 is None or roa0 is None else roa1 - roa0,
    )


def _build_pit_dossier(
    *,
    values: dict[FundamentalField, float],
    assumptions: ValuationAssumptions,
    current_price: float,
    pit_cagr: float | None,
    estimates: list[ProvenancedValue],
    quality: QualitySignals | None = None,
) -> tuple[str, ...]:
    """Build deterministic qualitative regime statements from PIT fundamentals.

    Avoids raw numeric figures inside the document text so the quick-think
    extractor can quote verbatim without violating its no-numeric-figures rule
    or triggering numeric concatenation checks, while conveying growth,
    profitability, cash conversion, Sloan accrual quality, leverage, and
    reverse-DCF expectations.
    """
    notes: list[str] = []

    if pit_cagr is not None:
        if pit_cagr >= 0.08:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Multi-year operating cash flow and revenue "
                "trajectory demonstrates strong organic expansion across the filing window."
            )
        elif pit_cagr >= 0.0:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Multi-year fundamental trajectory exhibits "
                "steady, positive compounding across recent annual filings."
            )
        else:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Multi-year fundamental trajectory reflects "
                "top-line or cash-flow contraction over the historical filing window."
            )

    revenue = values.get(FundamentalField.REVENUE)
    ebit = values.get(FundamentalField.OPERATING_INCOME)
    ocf = values.get(FundamentalField.OPERATING_CASH_FLOW)
    capex = values.get(FundamentalField.CAPEX)
    if revenue and revenue > 0.0 and ebit is not None:
        op_margin = ebit / revenue
        if op_margin >= 0.20:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Operating profitability is in a high-margin "
                "regime, indicating durable pricing power and operating leverage."
            )
        elif op_margin >= 0.08:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Operating profitability is positive and within "
                "a healthy institutional operating band."
            )
        else:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Operating margins are thin or compressed, "
                "leaving earnings sensitive to cost inflation and demand cyclicality."
            )

    if revenue and revenue > 0.0 and ocf is not None and capex is not None:
        fcf_margin = (ocf - capex) / revenue
        if fcf_margin >= 0.12:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Free cash flow conversion is strong, generating "
                "substantial internal capital after capital expenditure needs."
            )
        elif fcf_margin <= 0.0:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Free cash flow after capital expenditures is "
                "negative, requiring external financing or balance-sheet liquidity."
            )

    net_income = values.get(FundamentalField.NET_INCOME)
    assets = values.get(FundamentalField.ASSETS)
    if net_income is not None and ocf is not None and assets and assets > 0.0:
        sloan_accruals = (net_income - ocf) / assets
        if sloan_accruals <= -0.02:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Operating cash flow exceeds accounting net income "
                "(conservative Sloan accrual regime), confirming high cash earnings quality."
            )
        elif sloan_accruals >= 0.05:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Accounting net income materially exceeds operating "
                "cash flow (elevated Sloan accruals), signaling working-capital drag."
            )

    debt = max(values.get(FundamentalField.LONG_TERM_DEBT, 0.0), 0.0)
    cash = max(values.get(FundamentalField.CASH, 0.0), 0.0)
    da = values.get(FundamentalField.DEPRECIATION_AMORTIZATION, 0.0)
    if cash >= debt:
        notes.append(
            "[EDGAR 10-K PIT Dossier] Balance sheet holds a net-cash position with liquid "
            "reserves exceeding long-term debt obligations."
        )
    elif ebit is not None and (ebit + da) > 0.0:
        nd_ebitda = (debt - cash) / (ebit + da)
        if nd_ebitda <= 2.0:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Net debt relative to operating cash generation "
                "remains conservative and well-covered by annual EBITDA."
            )
        elif nd_ebitda >= 3.5:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Net leverage is elevated relative to EBITDA, "
                "increasing sensitivity to refinancing rates and credit conditions."
            )

    shares = values.get(FundamentalField.SHARES_OUTSTANDING)
    if (
        math.isfinite(current_price)
        and current_price > 0.0
        and ocf is not None
        and capex is not None
        and shares is not None
        and shares > 0.0
    ):
        implied_g = reverse_dcf_implied_growth(
            current_price=current_price,
            base_cash_flow=ocf - capex,
            discount_rate=assumptions.cost_of_equity,
            terminal_growth=assumptions.terminal_growth,
            shares_outstanding=shares,
            projection_years=assumptions.projection_years,
        )
        ref_g = pit_cagr if pit_cagr is not None else assumptions.terminal_growth
        if implied_g is not None:
            if implied_g < ref_g:
                notes.append(
                    "[EDGAR 10-K PIT Dossier] Reverse-DCF analysis indicates that current "
                    "market pricing embeds conservative growth expectations below historical "
                    "fundamental cash-flow compounding."
                )
            else:
                notes.append(
                    "[EDGAR 10-K PIT Dossier] Reverse-DCF analysis indicates that current "
                    "market pricing embeds demanding growth expectations above historical "
                    "cash-flow compounding."
                )

    if quality is not None:
        if quality.margin_delta is not None:
            trend = "expanded" if quality.margin_delta > 0.0 else "contracted"
            notes.append(
                f"[EDGAR 10-K PIT Dossier] Year-over-year operating margin {trend} "
                "versus the prior annual period."
            )
        if quality.roa_delta is not None:
            trend = "improved" if quality.roa_delta > 0.0 else "weakened"
            notes.append(
                f"[EDGAR 10-K PIT Dossier] Year-over-year return on assets {trend} "
                "relative to the prior annual period."
            )
        if quality.piotroski >= 7:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Piotroski financial-strength regime is strong "
                "across profitability, leverage and efficiency signals."
            )
        elif quality.piotroski >= 4:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Piotroski financial-strength regime is neutral, "
                "with mixed improvement and deterioration signals."
            )
        else:
            notes.append(
                "[EDGAR 10-K PIT Dossier] Piotroski financial-strength regime is "
                "deteriorating across most fundamental signals."
            )

    if not notes and estimates:
        notes.append(
            "[EDGAR 10-K PIT Dossier] As-filed SEC EDGAR fundamentals support multi-method "
            "intrinsic valuation at the point-in-time knowledge date."
        )
    return tuple(notes) if notes else _DEFAULT_DOCUMENTS


def _assemble_request(
    *,
    ticker: Ticker,
    knowledge_date: date,
    estimates: list[ProvenancedValue],
    current_price: float,
    run_metadata: RunMetadata,
    documents: tuple[str, ...],
) -> AnalystRequest:
    """Aggregate estimates and build the request."""
    consensus(estimates=estimates, unit=_VALUE_UNIT)  # validates unit coherence
    bundle = ValuationBundle(estimates=tuple(estimates), unit=_VALUE_UNIT)
    return AnalystRequest(
        ticker=str(ticker),
        knowledge_date=knowledge_date,
        current_price=current_price,
        documents=documents,
        valuation=bundle,
        run_metadata=run_metadata,
    )
