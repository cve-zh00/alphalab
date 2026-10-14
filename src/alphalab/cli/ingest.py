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

Fully composing those adapters (EDGAR credentials, a CIK map, a PIT store on
disk) is infrastructure this CLI owner does not provision; it is the live path
the scheduler/composition-root will wire. So this module is the **single,
documented place** that path is bound, and until it is provisioned
``--source live`` fails *gracefully* with an actionable message rather than a
stack trace.

The seam:

* :class:`RequestBuilder` is the callable shape the CLI depends on
  (``(ticker, *, knowledge_date) -> AnalystRequest``). The fixture builder
  satisfies it today; a thin closure over the live ``build_request`` with its
  dependencies pre-bound will satisfy it tomorrow.
* :func:`load_live_builder` lazily imports the live module. When the module is
  absent *or* present-but-not-yet-provisioned with real adapters, it raises
  :class:`LiveIngestUnavailableError` pointing the user at ``--source
  fixture``. Wiring the real adapters is a one-function change here (see the
  TODO), with nothing else in the CLI moving.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from alphalab.app.inputs import AnalystRequest

#: Where the live ingest lives and what it exposes.
_LIVE_MODULE = "alphalab.app.ingest"
_LIVE_FUNCTION = "build_request"


class RequestBuilder(Protocol):
    """Builds an :class:`AnalystRequest` for a ticker at a knowledge date.

    The fixture builder conforms to this shape today; a closure that pre-binds
    the live ``build_request`` dependencies will conform tomorrow.
    """

    def __call__(self, ticker: str, *, knowledge_date: date) -> AnalystRequest: ...


class LiveIngestUnavailableError(RuntimeError):
    """Raised when ``--source live`` is requested but not yet provisioned."""


def load_live_builder() -> RequestBuilder:
    """Return a live EDGAR request builder, or raise an actionable error.

    The import is deliberately lazy and local: the live module may be absent
    (older checkout) or present but requiring real adapters this CLI does not
    provision. Either way the user gets a clear pointer to ``--source
    fixture``.
    """
    try:
        module = __import__(_LIVE_MODULE, fromlist=[_LIVE_FUNCTION])
    except ImportError as exc:
        raise LiveIngestUnavailableError(
            f"live SEC EDGAR ingest module is not importable ({exc}). "
            "Re-run with '--source fixture' to use local price fixtures."
        ) from exc
    if getattr(module, _LIVE_FUNCTION, None) is None:
        raise LiveIngestUnavailableError(
            f"{_LIVE_MODULE} exists but exposes no '{_LIVE_FUNCTION}'. "
            "Re-run with '--source fixture' to use local price fixtures."
        )
    # TODO(cli): once EDGAR credentials, a CIK resolver, a price source and a
    # PIT store root are provisioned at the composition root, bind them here:
    #
    #     def _builder(ticker: str, *, knowledge_date: date) -> AnalystRequest:
    #         result = module.build_request(
    #             ticker, knowledge_date=knowledge_date,
    #             assumptions=..., cik_resolver=..., edgar_client=...,
    #             price_source=..., pit_store_root=..., run_metadata=...,
    #         )
    #         if result.request is None:
    #             raise LiveIngestUnavailableError("EDGAR returned no usable request")
    #         return result.request
    #     return _builder
    #
    # Until then the live path is not provisioned by this CLI owner.
    raise LiveIngestUnavailableError(
        f"{_LIVE_MODULE}.{_LIVE_FUNCTION} is present but the CLI does not yet "
        "provision its live dependencies (EDGAR client/User-Agent, CIK resolver, "
        "price source, PIT store, valuation assumptions). "
        "Re-run with '--source fixture' to use local price fixtures today."
    )
