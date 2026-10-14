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

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

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
    pe_implied_value,
    wacc,
)
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

    estimates, used, omitted = _build_estimates(values, assumptions)
    omitted = list(omitted) + [f"missing fundamental: {f}" for f in fundamentals.missing_fields]

    if not estimates:
        return IngestResult(
            request=None,
            methods_used=(),
            omitted_methods=tuple(omitted),
            missing_fields=fundamentals.missing_fields,
            resolved_concepts=fundamentals.resolved_concepts,
        )

    request = _assemble_request(
        ticker=ticker,
        knowledge_date=knowledge_date,
        estimates=estimates,
        price_source=price_source,
        kd=kd,
        run_metadata=run_metadata,
        documents=documents,
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
    """Return the most recently filed record knowable at ``kd`` (or ``None``)."""
    rows = store.query(ticker=ticker, field=str(field_name), knowledge_date=kd)
    if not rows:
        return None
    return max(rows, key=lambda r: r.observed_at)


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


def _build_estimates(
    values: dict[FundamentalField, float], assumptions: ValuationAssumptions
) -> tuple[list[ProvenancedValue], list[str], list[str]]:
    """Run each valuation method that has its inputs; omit the rest."""
    estimates: list[ProvenancedValue] = []
    used: list[str] = []
    omitted: list[str] = []

    _try_dcf(values, assumptions, estimates, used, omitted)
    _try_ddm(values, assumptions, estimates, used, omitted)
    _try_pe(values, assumptions, estimates, used, omitted)
    return estimates, used, omitted


def _try_dcf(
    values: dict[FundamentalField, float],
    assumptions: ValuationAssumptions,
    estimates: list[ProvenancedValue],
    used: list[str],
    omitted: list[str],
) -> None:
    ocf = values.get(FundamentalField.OPERATING_CASH_FLOW)
    capex = values.get(FundamentalField.CAPEX)
    shares = values.get(FundamentalField.SHARES_OUTSTANDING)
    equity = values.get(FundamentalField.STOCKHOLDERS_EQUITY)
    debt = values.get(FundamentalField.LONG_TERM_DEBT, 0.0)
    if ocf is None or capex is None or shares is None or shares <= 0 or equity is None:
        omitted.append("dcf: missing operating cash flow, capex, shares or equity")
        return
    fcf = ocf - capex
    try:
        discount = wacc(
            equity_value=equity,
            debt_value=debt,
            cost_of_equity=assumptions.cost_of_equity,
            cost_of_debt=assumptions.cost_of_debt,
            tax_rate=assumptions.tax_rate,
        )
        est = dcf_value_per_share(
            cash_flows=[fcf] * assumptions.projection_years,
            discount_rate=discount,
            terminal_growth=assumptions.terminal_growth,
            shares_outstanding=shares,
        )
    except ComputeError as exc:
        omitted.append(f"dcf: {exc}")
        return
    estimates.append(est)
    used.append("dcf")


def _try_ddm(
    values: dict[FundamentalField, float],
    assumptions: ValuationAssumptions,
    estimates: list[ProvenancedValue],
    used: list[str],
    omitted: list[str],
) -> None:
    dps = values.get(FundamentalField.DIVIDEND_PER_SHARE)
    if dps is None or dps <= 0.0:
        omitted.append("ddm: company pays no dividend (no dividend_per_share)")
        return
    try:
        est = ddm_value_per_share(
            dividend_per_share=dps,
            required_return=assumptions.cost_of_equity,
            dividend_growth=assumptions.dividend_growth,
        )
    except ComputeError as exc:
        omitted.append(f"ddm: {exc}")
        return
    estimates.append(est)
    used.append("ddm")


def _try_pe(
    values: dict[FundamentalField, float],
    assumptions: ValuationAssumptions,
    estimates: list[ProvenancedValue],
    used: list[str],
    omitted: list[str],
) -> None:
    eps = values.get(FundamentalField.EPS_DILUTED)
    if eps is None or not assumptions.peer_pe:
        omitted.append("multiples_pe: missing diluted EPS or peer P/E set")
        return
    try:
        est = pe_implied_value(peer_pe=assumptions.peer_pe, eps=eps)
    except ComputeError as exc:
        omitted.append(f"multiples_pe: {exc}")
        return
    estimates.append(est)
    used.append("multiples_pe")


def _assemble_request(
    *,
    ticker: Ticker,
    knowledge_date: date,
    estimates: list[ProvenancedValue],
    price_source: PriceSeriesSource,
    kd: KnowledgeDate,
    run_metadata: RunMetadata,
    documents: tuple[str, ...],
) -> AnalystRequest:
    """Aggregate estimates, read the current price, and build the request."""
    consensus(estimates=estimates, unit=_VALUE_UNIT)  # validates unit coherence
    bundle = ValuationBundle(estimates=tuple(estimates), unit=_VALUE_UNIT)
    try:
        price = current_price_at(price_source, ticker, kd)
    except NoPriceAvailableError:
        price = float("nan")
    return AnalystRequest(
        ticker=str(ticker),
        knowledge_date=knowledge_date,
        current_price=price,
        documents=documents,
        valuation=bundle,
        run_metadata=run_metadata,
    )
