"""Wiring seam for the live SEC EDGAR ingest (built by another agent).

Another agent owns ``alphalab.app.ingest.build_request``, which pulls **real**
fundamentals from SEC EDGAR. That function has a rich dependency surface — it
needs valuation ``assumptions``, a ``cik_resolver``, an ``edgar_client`` (with
a configured SEC ``User-Agent``), a ``price_source``, a point-in-time store
root, and ``run_metadata`` — and returns a *result* object (``.request``), not
an ``AnalystRequest`` directly::

    build_request(
        ticker, *, knowledge_date, assumptions, cik_resolver, edgar_client,
        price_source, pit_store_root, run_metadata, documents=...,
    ) -> IngestResult   # .request: AnalystRequest | None

This module is the **single, documented place** that live path is bound. It is
the composition root for ``--source live``: it provisions the EDGAR/CIK HTTP
adapters (identifying ``User-Agent`` read from the environment, never
hardcoded), a disk-backed CIK cache, the local price series source, a
point-in-time store on disk, and the (deliberately generic) valuation
judgement. Everything downstream — the pipeline, the contract gate, the
report — is untouched.

The seam:

* :class:`RequestBuilder` is the callable shape the CLI depends on
  (``(ticker, *, knowledge_date) -> AnalystRequest``). The fixture builder
  satisfies it today; the closure returned by :func:`load_live_builder` with
  the live ``build_request`` dependencies pre-bound satisfies it for live.
* :func:`load_live_builder` lazily imports the live module and binds its
  adapters. When the module is absent it raises
  :class:`LiveIngestUnavailableError`; when the required SEC ``User-Agent`` is
  not configured it raises :class:`LiveIngestConfigError` with an actionable,
  stack-trace-free message naming the environment variable to export.

Why the ``User-Agent`` is mandatory with no default
----------------------------------------------------
Both :class:`~alphalab.adapters.cik.CikSettings` and
:class:`~alphalab.adapters.edgar.EdgarSettings` require ``user_agent`` and
carry **no** default (verified: ``cik.py:48-50``, ``edgar.py:72-74``). The SEC
Fair Access policy requires an identifiable ``User-Agent`` on every request;
anonymous automation is throttled/blocked. Making it mandatory at the settings
boundary means this resolver can never accidentally hammer the SEC
unidentified — a missing agent fails *before* a single socket opens.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import ValidationError

from alphalab.adapters.alpaca_config import AlpacaSettings
from alphalab.adapters.alpaca_news import AlpacaNewsClient
from alphalab.adapters.cik import CikResolver, CikSettings
from alphalab.adapters.edgar import EdgarClient, EdgarSettings, _as_dict
from alphalab.adapters.edgar_live import CANONICAL_CONCEPTS
from alphalab.adapters.finbert_sentiment import FinBertSentimentScorer
from alphalab.adapters.pit_store import ParquetPITStore  # noqa: F401  (surfaces import errors early)
from alphalab.adapters.prices import CsvPriceSeriesSource, PriceSeriesSource
from alphalab.app.ingest import IngestResult, ValuationAssumptions
from alphalab.app.inputs import AnalystRequest
from alphalab.cli.forecast_paths import project_root
from alphalab.cli.paper_transport import HttpxWriteTransport
from alphalab.domain.analyst.contracts import RunMetadata
from alphalab.domain.analyst.tiers import ModelTier
from alphalab.domain.compute.sectors import (
    compute_capm_cost_of_equity,
    estimate_ke_beta_from_prices,
    resolve_sector_profile,
)
from alphalab.domain.pit import PITRecord
from alphalab.domain.types import Ticker

_CANDIDATE_CONCEPT_NAMES: frozenset[str] = frozenset(
    concept for concepts in CANONICAL_CONCEPTS.values() for concept in concepts
)

#: Where the live ingest lives and what it exposes.
_LIVE_MODULE = "alphalab.app.ingest"
_LIVE_FUNCTION = "build_request"

#: Environment variable (``ALPHALAB_EDGAR_`` prefix + ``user_agent``) the SEC
#: adapters read their mandatory identifying agent from.
_USER_AGENT_ENV = "ALPHALAB_EDGAR_USER_AGENT"

#: Illustrative, uncalibrated model identity for the ingest run's provenance.
_MODEL_NAME = "live-edgar-ingest"
_MODEL_VERSION = "0.1.0"


class RequestBuilder(Protocol):
    """Builds an :class:`AnalystRequest` for a ticker at a knowledge date.

    The fixture builder conforms to this shape today; the live closure from
    :func:`load_live_builder` conforms for ``--source live``.
    """

    def __call__(self, ticker: str, *, knowledge_date: date) -> AnalystRequest: ...


class _LiveBuildRequest(Protocol):
    """Typed shape of :func:`alphalab.app.ingest.build_request`.

    The live module is imported lazily (an older checkout may lack it), so its
    attribute arrives untyped. Casting the resolved attribute to this protocol
    restores full type-checking of the call without forcing a module-level
    import of the live path.
    """

    def __call__(
        self,
        ticker: Ticker,
        *,
        knowledge_date: date,
        assumptions: ValuationAssumptions,
        cik_resolver: CikResolver,
        edgar_client: EdgarClient,
        price_source: PriceSeriesSource,
        pit_store_root: Path,
        run_metadata: RunMetadata,
    ) -> IngestResult: ...


class LiveIngestUnavailableError(RuntimeError):
    """Raised when ``--source live`` is requested but cannot be provisioned.

    Covers a non-importable live module *and* a run that produced no usable
    request (no valuation method could run). In the latter case the message
    carries the concrete reasons from ``IngestResult.omitted_methods`` — never
    a generic placeholder — so the user learns exactly which inputs EDGAR did
    not supply.
    """


class LiveIngestConfigError(RuntimeError):
    """Raised when the live path is mis-/un-configured (e.g. no ``User-Agent``).

    Distinct from :class:`LiveIngestUnavailableError` so the CLI can map a
    configuration mistake to exit code 2 (user must fix their environment)
    rather than exit 1 (the live path is simply not available).
    """


def _generic_assumptions() -> ValuationAssumptions:
    """Return *generic, uncalibrated* valuation judgement for the live path.

    These are deliberately **not** calibrated per issuer. They are plausible
    long-run defaults for a large, mature US equity: an ~9% cost of equity, a
    4% after-tax cost of debt at a 21% federal rate, 2.5% terminal growth, 3%
    dividend growth over a 5-year horizon, and a conservative market-wide peer
    P/E band.
    """
    return ValuationAssumptions(
        cost_of_equity=0.09,
        cost_of_debt=0.04,
        tax_rate=0.21,
        terminal_growth=0.025,
        dividend_growth=0.03,
        projection_years=5,
        peer_pe=(15.0, 18.0, 20.0),
    )


def _calibrated_assumptions(
    ticker: str,
    *,
    knowledge_date: date,
    price_source: PriceSeriesSource,
) -> ValuationAssumptions:
    """Build sector- and beta-calibrated valuation assumptions for ``ticker``."""
    profile = resolve_sector_profile(ticker)
    asset_points = [
        p.close for p in price_source.series(Ticker(ticker.upper())) if p.on <= knowledge_date
    ]
    bench_points = [p.close for p in price_source.series(Ticker("SPY")) if p.on <= knowledge_date]
    beta = estimate_ke_beta_from_prices(
        asset_points,
        bench_points if bench_points else None,
        default_beta=profile.beta,
    )
    ke = compute_capm_cost_of_equity(beta)
    return ValuationAssumptions(
        cost_of_equity=ke,
        cost_of_debt=0.045,
        tax_rate=0.21,
        terminal_growth=0.025,
        dividend_growth=0.03,
        projection_years=5,
        peer_pe=profile.peer_pe,
        peer_ev_ebitda=profile.peer_ev_ebitda,
        peer_pb=profile.peer_pb,
        peer_ffo=profile.peer_ffo,
        sector=profile.sector,
        min_payout_ratio=0.25,
        dcf_mode="fcfe",
    )


def _data_dir() -> Path:
    """Working-data directory, mirroring :mod:`alphalab.cli.forecast_paths`."""
    return project_root() / "data"


def _cik_cache_path() -> Path:
    """On-disk CIK-map cache so a batch ingest hits the SEC map at most once."""
    return _data_dir() / "cik_cache.json"


def _pit_store_root() -> Path:
    """Point-in-time store root, alongside the other working-data stores."""
    return _data_dir() / "pit"


def _edgar_cache_dir() -> Path:
    """On-disk SEC companyfacts cache so S&P 500 batch runs avoid redundant SEC hits."""
    return _data_dir() / "edgar_cache"


class _CachingEdgarClient(EdgarClient):
    """Disk-backed ``companyfacts`` cache around :class:`EdgarClient`."""

    def __init__(
        self,
        settings: EdgarSettings,
        *,
        transport: Any,
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__(settings, transport=transport)
        self._cache_dir = cache_dir
        self._facts_cache: dict[tuple[str, str], list[PITRecord]] = {}

    def company_facts(self, ticker: Ticker, *, cik: str) -> list[PITRecord]:
        key = (str(ticker), cik)
        cached = self._facts_cache.get(key)
        if cached is not None:
            return cached
        records = super().company_facts(ticker, cik=cik)
        if self._cache_dir is not None:
            self._facts_cache[key] = records
        return records

    def _parse_facts(self, ticker: Ticker, payload: dict[str, object]) -> list[PITRecord]:
        if self._cache_dir is None:
            return super()._parse_facts(ticker, payload)
        records: list[PITRecord] = []
        facts = _as_dict(payload.get("facts"))
        for taxonomy in facts.values():
            for concept_name, concept in _as_dict(taxonomy).items():
                if concept_name not in _CANDIDATE_CONCEPT_NAMES:
                    continue
                records.extend(self._parse_concept(ticker, concept_name, _as_dict(concept)))
        return records

    def _get_with_retry(self, url: str) -> dict[str, object]:
        filename = url.rsplit("/", 1)[-1]
        is_cik_json = filename.startswith("CIK") and filename.endswith(".json")
        cache_name = f"submissions_{filename}" if "/submissions/" in url else filename
        if self._cache_dir is not None and is_cik_json:
            path = self._cache_dir / cache_name
            if path.is_file():
                return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        data = super()._get_with_retry(url)
        if self._cache_dir is not None and is_cik_json:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_dir / cache_name
            path.write_text(json.dumps(data), encoding="utf-8")
        return data


def _price_source() -> PriceSeriesSource:
    """Local close-series source (the sanctioned ``PriceSeriesSource``).

    Reuses :class:`~alphalab.adapters.prices.CsvPriceSeriesSource` over the
    repo's ``data/prices`` directory — the same series the settler prices
    against. When a ticker has no local series the price resolves to ``NaN``
    (no look-ahead, no fabrication); the valuation methods still run on EDGAR
    fundamentals and the report states the price is unavailable.
    """
    return CsvPriceSeriesSource(_data_dir() / "prices")


def _run_metadata(knowledge_date: date) -> RunMetadata:
    """Provenance for one live ingest run.

    The dataset hash is a stable tag for the live-EDGAR source at this
    knowledge date; the real per-filing provenance lives inside each
    ``ProvenancedValue`` the compute layer stamps.
    """
    return RunMetadata(
        model_name=_MODEL_NAME,
        tier=ModelTier.DEEP,
        llm_version=_MODEL_VERSION,
        llm_cutoff=knowledge_date,
        knowledge_date=knowledge_date,
        dataset_hash=f"live-edgar:{knowledge_date.isoformat()}",
    )


def _load_settings() -> tuple[CikSettings, EdgarSettings]:
    """Load the SEC adapter settings, or raise an actionable config error.

    ``user_agent`` is mandatory and read from ``ALPHALAB_EDGAR_USER_AGENT``. A
    pydantic ``ValidationError`` here means the operator forgot it; we translate
    that into a message that names the exact variable to export, so the user
    never has to read a stack trace to learn what to do.
    """
    try:
        return CikSettings(), EdgarSettings()  # type: ignore[call-arg]
    except ValidationError as exc:
        raise LiveIngestConfigError(
            f"the live SEC EDGAR path needs an identifying User-Agent that is not set. "
            f"Export {_USER_AGENT_ENV} with a contactable string, e.g.\n"
            f'    export {_USER_AGENT_ENV}="alphalab-research you@example.com"\n'
            "The SEC Fair Access policy requires it; anonymous requests are blocked. "
            "Or re-run with '--source fixture' to use local price fixtures."
        ) from exc


def _supplemental_documents(
    ticker: Ticker,
    *,
    knowledge_date: date,
    cik_resolver: CikResolver,
    edgar_client: EdgarClient,
    news_client: AlpacaNewsClient | None,
) -> tuple[str, ...]:
    """Gather point-in-time SEC 8-K/10-Q/10-K disclosures and Alpaca News articles."""
    extra: list[str] = []
    with contextlib.suppress(Exception):
        cik = cik_resolver.resolve(ticker)
        extra.extend(
            edgar_client.recent_filing_disclosures(
                ticker,
                cik=cik,
                knowledge_date=knowledge_date,
                limit=5,
            )
        )
    if news_client is not None:
        with contextlib.suppress(Exception):
            extra.extend(
                news_client.recent_documents(
                    ticker,
                    knowledge_date=knowledge_date,
                    limit=5,
                )
            )
    return tuple(extra)


def _maybe_default_news_client(transport: Any) -> AlpacaNewsClient | None:
    """Provision an :class:`AlpacaNewsClient` when Alpaca credentials exist."""
    try:
        settings = AlpacaSettings()  # type: ignore[call-arg]
    except ValidationError:
        return None
    return AlpacaNewsClient(settings, transport=transport)


def _build_live_request(
    build: _LiveBuildRequest,
    ticker: str,
    *,
    knowledge_date: date,
    cik_resolver: CikResolver,
    edgar_client: EdgarClient,
    price_source: PriceSeriesSource | None = None,
    include_news: bool = False,
    news_client: AlpacaNewsClient | None = None,
    sentiment_scorer: FinBertSentimentScorer | None = None,
) -> AnalystRequest:
    """Call the live ``build_request`` and unwrap its result honestly.

    Propagates the concrete omission reasons when no method could run, so the
    user sees *why* there is no request rather than an invented number.
    """
    resolved_prices = price_source or _price_source()
    sym = Ticker(ticker)
    result = build(
        sym,
        knowledge_date=knowledge_date,
        assumptions=_calibrated_assumptions(
            ticker,
            knowledge_date=knowledge_date,
            price_source=resolved_prices,
        ),
        cik_resolver=cik_resolver,
        edgar_client=edgar_client,
        price_source=resolved_prices,
        pit_store_root=_pit_store_root() / ticker.upper(),
        run_metadata=_run_metadata(knowledge_date),
    )
    if result.request is None:
        reasons = "; ".join(result.omitted_methods) or "no valuation method had its inputs"
        raise LiveIngestUnavailableError(
            f"EDGAR produced no usable valuation for {ticker!r} at {knowledge_date.isoformat()}: "
            f"{reasons}. Re-run with '--source fixture' to use local price fixtures."
        )
    if not include_news:
        return result.request
    extra_docs = _supplemental_documents(
        sym,
        knowledge_date=knowledge_date,
        cik_resolver=cik_resolver,
        edgar_client=edgar_client,
        news_client=news_client,
    )
    if not extra_docs:
        return result.request
    if news_client is not None:
        with contextlib.suppress(Exception):
            scorer = sentiment_scorer or FinBertSentimentScorer()
            note = scorer.score_documents(extra_docs).regime_note()
            extra_docs = (note, *extra_docs)
    return result.request.model_copy(update={"documents": (*result.request.documents, *extra_docs)})


def load_news_sentiment_fn(
    *,
    news_client: AlpacaNewsClient | None = None,
    sentiment_scorer: FinBertSentimentScorer | None = None,
) -> Callable[[Ticker, date], float | None] | None:
    """Return ``(ticker, as_of) -> net news sentiment``, or ``None`` without Alpaca credentials."""
    client = news_client or _maybe_default_news_client(HttpxWriteTransport())
    if client is None:
        return None
    scorer = sentiment_scorer or FinBertSentimentScorer()

    def _net(ticker: Ticker, as_of: date) -> float | None:
        docs = client.recent_documents(ticker, knowledge_date=as_of, limit=5)
        if not docs:
            return None
        return scorer.score_documents(docs).net_score

    return _net


def load_live_builder(
    *,
    price_source: PriceSeriesSource | None = None,
    include_news: bool = True,
    news_client: AlpacaNewsClient | None = None,
    sentiment_scorer: FinBertSentimentScorer | None = None,
) -> RequestBuilder:
    """Return a live SEC EDGAR request builder, or raise an actionable error.

    The import is deliberately lazy and local: the live module may be absent in
    an older checkout. Settings are loaded eagerly so a missing ``User-Agent``
    fails fast — before any network — with a clear pointer to the variable to
    export.

    Raises:
        LiveIngestUnavailableError: the live module/function is not importable.
        LiveIngestConfigError: the mandatory SEC ``User-Agent`` is not set.
    """
    try:
        module = __import__(_LIVE_MODULE, fromlist=[_LIVE_FUNCTION])
    except ImportError as exc:
        raise LiveIngestUnavailableError(
            f"live SEC EDGAR ingest module is not importable ({exc}). "
            "Re-run with '--source fixture' to use local price fixtures."
        ) from exc
    build = getattr(module, _LIVE_FUNCTION, None)
    if build is None:
        raise LiveIngestUnavailableError(
            f"{_LIVE_MODULE} exists but exposes no '{_LIVE_FUNCTION}'. "
            "Re-run with '--source fixture' to use local price fixtures."
        )
    live_build = cast(_LiveBuildRequest, build)

    cik_settings, edgar_settings = _load_settings()
    transport = HttpxWriteTransport()
    is_real_transport = isinstance(HttpxWriteTransport, type)
    cache_dir = _edgar_cache_dir() if is_real_transport else None
    cik_resolver = CikResolver(cik_settings, transport=transport, cache_path=_cik_cache_path())
    edgar_client = _CachingEdgarClient(edgar_settings, transport=transport, cache_dir=cache_dir)
    effective_include_news = include_news and (is_real_transport or news_client is not None)
    resolved_news_client = (
        news_client
        if news_client is not None
        else (_maybe_default_news_client(transport) if effective_include_news else None)
    )
    request_cache: dict[tuple[str, date], AnalystRequest] = {}
    resolved_scorer = sentiment_scorer or FinBertSentimentScorer()

    def _builder(ticker: str, *, knowledge_date: date) -> AnalystRequest:
        key = (ticker.upper(), knowledge_date)
        cached = request_cache.get(key)
        if cached is not None:
            return cached
        request = _build_live_request(
            live_build,
            ticker,
            knowledge_date=knowledge_date,
            cik_resolver=cik_resolver,
            edgar_client=edgar_client,
            price_source=price_source,
            include_news=effective_include_news,
            news_client=resolved_news_client,
            sentiment_scorer=resolved_scorer,
        )
        request_cache[key] = request
        return request

    return _builder
