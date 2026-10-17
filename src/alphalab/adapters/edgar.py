"""SEC EDGAR fundamentals adapter (XBRL ``companyfacts``), as-filed.

Design notes:

* The HTTP call sits behind the :class:`Transport` protocol so tests use a
  double and never touch the network.
* The identifying ``User-Agent`` is read from configuration
  (:class:`EdgarSettings`, pydantic-settings) — never hardcoded, never a
  secret in source. SEC requires a descriptive agent.
* Requests are throttled with :class:`RateLimiter` (~10 req/s) and retried
  with exponential backoff on throttling/5xx responses.
* Each XBRL fact becomes one :class:`PITRecord` keyed by its ``filed`` date,
  so a restatement is a *new* record and the original is never overwritten.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from alphalab.domain.pit import PITRecord
from alphalab.domain.types import Ticker

_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_SOURCE = "SEC-EDGAR"
_8K_ITEM_LABELS: dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements or Restatements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure or Election of Directors or Principal Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Material Corporate Events",
    "9.01": "Financial Statements and Exhibits",
}


@dataclass(frozen=True)
class HttpResponse:
    """Minimal transport-agnostic HTTP response."""

    status_code: int
    json: dict[str, object]


class Transport(Protocol):
    """Protocol for the one HTTP call the adapter makes."""

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse: ...


class Clock(Protocol):
    """Protocol for time, so throttling/backoff are deterministic in tests."""

    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


@dataclass
class _SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class EdgarSettings(BaseSettings):
    """Configuration for the EDGAR adapter.

    ``user_agent`` must be supplied via environment (``ALPHALAB_EDGAR_...``)
    or construction; it is required and carries no default so the service
    cannot run anonymously against SEC.
    """

    model_config = SettingsConfigDict(env_prefix="ALPHALAB_EDGAR_")

    user_agent: str
    max_rps: float = Field(default=10.0, gt=0)


@dataclass
class RateLimiter:
    """Simple spacing-based limiter enforcing a maximum requests-per-second."""

    max_rps: float
    clock: Clock = field(default_factory=_SystemClock)
    _last: float | None = field(default=None, init=False)

    def acquire(self) -> None:
        min_interval = 1.0 / self.max_rps
        now = self.clock.monotonic()
        if self._last is not None:
            elapsed = now - self._last
            wait = min_interval - elapsed
            if wait > 0:
                self.clock.sleep(wait)
                now = self.clock.monotonic()
        self._last = now


class EdgarClient:
    """Fetches and parses as-filed XBRL fundamentals from SEC EDGAR."""

    def __init__(
        self,
        settings: EdgarSettings,
        *,
        transport: Transport,
        clock: Clock | None = None,
        max_retries: int = 5,
        base_backoff: float = 0.5,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._clock = clock or _SystemClock()
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._limiter = RateLimiter(max_rps=settings.max_rps, clock=self._clock)

    def company_facts(self, ticker: Ticker, *, cik: str) -> list[PITRecord]:
        """Return all as-filed fundamental records for ``ticker``."""
        url = _BASE_URL.format(cik=cik.zfill(10))
        payload = self._get_with_retry(url)
        return list(self._parse_facts(ticker, payload))

    def recent_filing_disclosures(
        self,
        ticker: Ticker,
        *,
        cik: str,
        knowledge_date: date,
        limit: int = 5,
    ) -> tuple[str, ...]:
        """Return up to ``limit`` PIT 8-K/10-Q/10-K disclosures on or before ``knowledge_date``."""
        if limit <= 0:
            return ()
        url = _SUBMISSIONS_URL.format(cik=cik.zfill(10))
        payload = self._get_with_retry(url)
        return _parse_submissions(payload, knowledge_date=knowledge_date, limit=limit)

    def _get_with_retry(self, url: str) -> dict[str, object]:
        headers = {"User-Agent": self._settings.user_agent}
        last_status: int | None = None
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            response = self._transport.get(url, headers=headers)
            if response.status_code == 200:
                return response.json
            last_status = response.status_code
            if response.status_code not in _RETRYABLE_STATUS:
                break
            if attempt < self._max_retries:
                self._clock.sleep(self._base_backoff * (2**attempt))
        raise RuntimeError(f"EDGAR request failed (status={last_status}) for {url}")

    def _parse_facts(self, ticker: Ticker, payload: dict[str, object]) -> list[PITRecord]:
        records: list[PITRecord] = []
        facts = _as_dict(payload.get("facts"))
        for taxonomy in facts.values():
            for concept_name, concept in _as_dict(taxonomy).items():
                records.extend(self._parse_concept(ticker, concept_name, _as_dict(concept)))
        return records

    def _parse_concept(
        self, ticker: Ticker, concept_name: str, concept: dict[str, object]
    ) -> list[PITRecord]:
        records: list[PITRecord] = []
        units = _as_dict(concept.get("units"))
        for entries in units.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                record = self._parse_entry(ticker, concept_name, _as_dict(entry))
                if record is not None:
                    records.append(record)
        return records

    def _parse_entry(
        self, ticker: Ticker, concept_name: str, entry: dict[str, object]
    ) -> PITRecord | None:
        """Build a :class:`PITRecord` from one XBRL fact, or ``None`` if unusable.

        EDGAR occasionally publishes *forward-looking* facts whose period ends
        **after** the filing date (observed with ``CommonStockSharesAuthorized``
        and ``DisposalGroupNotDiscontinuedOperationGainLossOnDisposal``: ``end``
        in a quarter that had not closed when the filing went out). Those facts
        violate the ``observed_at >= as_of`` domain invariant, which exists to
        guarantee a datum is never attributed to a period it precedes.

        Such facts are **skipped**, never coerced: shifting ``as_of`` to the
        filing date would misstate the period, and relaxing the invariant would
        weaken the anti-leakage guarantee for every other record. They are rare
        (4 of ~30k facts for KO) and confined to non-core concepts.

        A *duration* fact (revenue, cash flow — anything accumulated over a
        span) carries ``start`` alongside ``end``; an *instant* fact (assets,
        shares outstanding) carries only ``end``. That distinction is recorded
        in ``period_start`` rather than dropped, because without it a quarterly
        and an annual figure are indistinguishable downstream and the most
        recently filed one wins — which for any flow is the quarter.
        """
        end = entry.get("end")
        filed = entry.get("filed")
        val = entry.get("val")
        if not isinstance(end, str) or not isinstance(filed, str):
            return None
        if not isinstance(val, int | float):
            return None
        as_of = _parse_date(end)
        observed_at = _parse_date(filed)
        if observed_at < as_of:
            return None
        start = entry.get("start")
        period_start = _parse_date(start) if isinstance(start, str) else None
        if period_start is not None and period_start > as_of:
            return None
        return PITRecord(
            ticker=ticker,
            field=concept_name,
            value=float(val),
            as_of=as_of,
            observed_at=observed_at,
            source=_SOURCE,
            period_start=period_start,
        )


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_submissions(
    payload: dict[str, object],
    *,
    knowledge_date: date,
    limit: int,
) -> tuple[str, ...]:
    recent = _as_dict(_as_dict(payload.get("filings")).get("recent"))
    forms = recent.get("form")
    filing_dates = recent.get("filingDate")
    items_list = recent.get("items")
    if not isinstance(forms, list) or not isinstance(filing_dates, list):
        return ()
    raw_items = items_list if isinstance(items_list, list) else []
    disclosures: list[str] = []
    for idx, (form_raw, fdate_raw) in enumerate(zip(forms, filing_dates, strict=False)):
        if len(disclosures) >= limit:
            break
        if not isinstance(form_raw, str) or not isinstance(fdate_raw, str):
            continue
        if form_raw not in {"8-K", "10-Q", "10-K"}:
            continue
        try:
            filed_on = _parse_date(fdate_raw)
        except ValueError:
            continue
        if filed_on > knowledge_date:
            continue
        raw_item = raw_items[idx] if idx < len(raw_items) else ""
        item_str = raw_item if isinstance(raw_item, str) else ""
        disclosures.append(_format_filing_disclosure(form_raw, filed_on, item_str))
    return tuple(disclosures)


def _format_filing_disclosure(form: str, filed_on: date, items_csv: str) -> str:
    prefix = f"[SEC EDGAR {form} filed {filed_on.isoformat()}]"
    if form == "8-K":
        codes = [c.strip() for c in items_csv.split(",") if c.strip() and c.strip() != "9.01"]
        labels = [_8K_ITEM_LABELS.get(code, f"Item {code}") for code in codes]
        if labels:
            return f"{prefix} Current Report disclosures: {'; '.join(labels)}."
        return f"{prefix} Current Report (Form 8-K) filed with the SEC."
    if form == "10-Q":
        return f"{prefix} Quarterly Report (Form 10-Q) filed with the SEC."
    return f"{prefix} Annual Report (Form 10-K) filed with the SEC."

