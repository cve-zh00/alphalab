"""Canonical fundamentals extraction layered over the EDGAR adapter.

The raw :class:`~alphalab.adapters.edgar.EdgarClient` returns one
:class:`~alphalab.domain.pit.PITRecord` per XBRL fact, keyed by the *raw* XBRL
concept name (e.g. ``NetIncomeLoss``). Those names are inconsistent between
filers, so this module maps them onto a small set of **canonical** fields the
``domain/compute`` operators actually consume.

Two properties are preserved deliberately:

* **As-filed.** Every filing is kept. When the same ``as_of`` appears in several
  filings (a restatement) all records survive; the PIT store decides which was
  visible at a given knowledge date. We never deduplicate to "latest".
* **Honest absence.** Each canonical field has an *ordered* list of candidate
  XBRL concepts. We take the first concept that has any facts and emit its
  records under the canonical name. If no candidate is present the field is
  reported in :attr:`FundamentalRecords.missing_fields` — never fabricated,
  never zero-filled.
* **Annual flows only.** A *flow* field is kept only when the fact spans a full
  year. The compute operators consume annual figures, but EDGAR publishes a
  filer's quarterly facts too, and those are filed more recently than the last
  10-K — so "most recently filed" silently resolves to a three-month number.
  Measured on ABT: diluted EPS came through as 0.61 (one quarter) where the
  annual figure is near 3.30, and dividends per share as 0.59 against an annual
  2.36. Every valuation method then landed roughly an order of magnitude below
  the traded price, the output contract gate withheld the implausible figure,
  and the forecast fell back to even odds. A flow with no annual fact is
  reported *missing* rather than quietly backfilled from a quarter: a stated
  absence costs one name, a 4x understatement poisons the whole comparison.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from alphalab.adapters.edgar import EdgarClient
from alphalab.domain.pit import PITRecord
from alphalab.domain.types import Ticker


class FundamentalField(StrEnum):
    """Canonical fundamental field names used across the compute layer."""

    REVENUE = "revenue"
    NET_INCOME = "net_income"
    EPS_DILUTED = "eps_diluted"
    STOCKHOLDERS_EQUITY = "stockholders_equity"
    ASSETS = "assets"
    LIABILITIES = "liabilities"
    CASH = "cash"
    LONG_TERM_DEBT = "long_term_debt"
    OPERATING_INCOME = "operating_income"
    DEPRECIATION_AMORTIZATION = "depreciation_amortization"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    CAPEX = "capex"
    SHARES_OUTSTANDING = "shares_outstanding"
    DIVIDEND_PER_SHARE = "dividend_per_share"


# Ordered XBRL concept fallbacks per canonical field. The first concept that
# has any facts for the filer wins; concepts are tried left to right.
CANONICAL_CONCEPTS: dict[FundamentalField, tuple[str, ...]] = {
    FundamentalField.REVENUE: (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ),
    FundamentalField.NET_INCOME: (
        "NetIncomeLoss",
        "ProfitLoss",
    ),
    FundamentalField.EPS_DILUTED: (
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
    ),
    FundamentalField.STOCKHOLDERS_EQUITY: (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    FundamentalField.ASSETS: ("Assets",),
    FundamentalField.LIABILITIES: ("Liabilities",),
    FundamentalField.CASH: (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
    FundamentalField.LONG_TERM_DEBT: (
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ),
    FundamentalField.OPERATING_INCOME: ("OperatingIncomeLoss",),
    FundamentalField.DEPRECIATION_AMORTIZATION: (
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ),
    FundamentalField.OPERATING_CASH_FLOW: (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    FundamentalField.CAPEX: (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    FundamentalField.SHARES_OUTSTANDING: (
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ),
    FundamentalField.DIVIDEND_PER_SHARE: (
        "CommonStockDividendsPerShareDeclared",
        "CommonStockDividendsPerShareCashPaid",
    ),
}


#: Canonical fields accumulated over a span rather than measured at an instant.
#: These are the ones that must be taken from an annual period; everything else
#: is a balance-sheet stock, which has no duration and is read as filed.
FLOW_FIELDS: frozenset[FundamentalField] = frozenset(
    {
        FundamentalField.REVENUE,
        FundamentalField.NET_INCOME,
        FundamentalField.EPS_DILUTED,
        FundamentalField.OPERATING_INCOME,
        FundamentalField.DEPRECIATION_AMORTIZATION,
        FundamentalField.OPERATING_CASH_FLOW,
        FundamentalField.CAPEX,
        FundamentalField.DIVIDEND_PER_SHARE,
    }
)

#: Inclusive day-count bounds for "a full year". A 52/53-week filer reports 364
#: or 371 days and a calendar-year filer 365 or 366, so the window is widened to
#: tolerate transition periods while staying far clear of a half-year (~182) and
#: a quarter (~91) — the two it exists to exclude.
_ANNUAL_MIN_DAYS = 300
_ANNUAL_MAX_DAYS = 400


def _is_annual(record: PITRecord) -> bool:
    """Whether ``record`` covers a full year (instants are never annual flows)."""
    days = record.period_days
    return days is not None and _ANNUAL_MIN_DAYS <= days <= _ANNUAL_MAX_DAYS


@dataclass(frozen=True)
class FundamentalRecords:
    """Canonicalised, as-filed fundamentals plus an honest absence report.

    Attributes:
        records: As-filed PIT records keyed by canonical field name. Multiple
            records may share an ``as_of`` (restatements) — all are kept.
        missing_fields: Canonical fields for which no candidate concept had any
            facts. These are explicitly absent, not zero.
        resolved_concepts: Which raw XBRL concept supplied each present field,
            for provenance and reporting.
    """

    records: tuple[PITRecord, ...]
    missing_fields: frozenset[FundamentalField]
    resolved_concepts: dict[FundamentalField, str]


def extract_fundamentals(client: EdgarClient, ticker: Ticker, *, cik: str) -> FundamentalRecords:
    """Fetch company facts and canonicalise them into usable fundamentals."""
    raw = client.company_facts(ticker, cik=cik)
    by_concept: dict[str, list[PITRecord]] = defaultdict(list)
    for record in raw:
        by_concept[record.field].append(record)

    records: list[PITRecord] = []
    missing: list[FundamentalField] = []
    resolved: dict[FundamentalField, str] = {}

    for field, concepts in CANONICAL_CONCEPTS.items():
        keep = _is_annual if field in FLOW_FIELDS else _any_record
        chosen = _first_present(concepts, by_concept, keep=keep)
        if chosen is None:
            missing.append(field)
            continue
        resolved[field] = chosen
        records.extend(_relabel(rec, field) for rec in by_concept[chosen] if keep(rec))

    return FundamentalRecords(
        records=tuple(records),
        missing_fields=frozenset(missing),
        resolved_concepts=resolved,
    )


def _any_record(record: PITRecord) -> bool:
    """Accept every record — the rule for stocks, which have no duration."""
    return True


def _first_present(
    concepts: tuple[str, ...],
    by_concept: dict[str, list[PITRecord]],
    *,
    keep: Callable[[PITRecord], bool],
) -> str | None:
    """First candidate concept with at least one *usable* fact, or ``None``.

    Usability is part of the test rather than a filter applied afterwards. A
    filer may tag its preferred concept only quarterly while a fallback concept
    carries the annual figure; picking the first concept with *any* fact and
    only then demanding an annual one would discard that fallback and report
    the field missing when it is in fact available.
    """
    for concept in concepts:
        if any(keep(record) for record in by_concept.get(concept, ())):
            return concept
    return None


def _relabel(record: PITRecord, field: FundamentalField) -> PITRecord:
    """Return a copy of ``record`` keyed by the canonical ``field`` name."""
    return record.model_copy(update={"field": str(field)})
