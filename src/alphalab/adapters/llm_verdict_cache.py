"""Persistent canonical LLM verdict cache keyed by PIT dossier fingerprint.

Under Moulton's (1990) design effect $N_{\\text{eff}} = n / (1 + (n-1)\\rho)$ and
Wald's (1945) Beta-Binomial SPRT saturation, polling the same LLM hourly on
unchanged point-in-time EDGAR fundamentals and news ($\\text{ICC} \\approx 0.85$)
yields almost zero incremental statistical information while inflating false
entries under a 1-BUY disjunction.

This adapter caches the canonical :class:`~alphalab.domain.report.artifact.EquityResearchArtifact`
for each ``(ticker, knowledge_date, llm_version, debate, documents_hash,
valuation_hash, price_bucket)`` tuple so intraday cycles reuse the canonical
daily verdict unless new 8-K filings/news arrive or the price moves beyond the
configured bucket (default 3%).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path

from alphalab.app.inputs import AnalystRequest
from alphalab.artifact import EquityResearchArtifact, PriceTarget

__all__ = [
    "DEFAULT_VERDICT_CACHE_PATH",
    "LlmVerdictCache",
    "compute_request_fingerprint",
]

DEFAULT_VERDICT_CACHE_PATH = Path("data/llm_verdict_cache.json")
_MAX_CACHE_ENTRIES = 2000


def compute_request_fingerprint(
    request: AnalystRequest,
    *,
    llm_version: str = "",
    debate: bool = True,
    price_bucket_pct: float = 3.0,
) -> str:
    """Compute a deterministic SHA-256 fingerprint for an ``AnalystRequest``."""
    bucket_step = max(0.005, price_bucket_pct / 100.0)
    safe_price = max(float(request.current_price), 0.01)
    price_bucket = round(math.log(safe_price) / bucket_step)
    estimates = sorted(
        (str(e.operator), round(float(e.value), 4))
        for e in request.valuation.estimates
    )
    payload = {
        "ticker": str(request.ticker).upper(),
        "knowledge_date": request.knowledge_date.isoformat(),
        "llm_version": str(llm_version),
        "debate": bool(debate),
        "price_bucket": price_bucket,
        "estimates": estimates,
        "documents": list(request.documents),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class LlmVerdictCache:
    """JSON-backed cache of canonical LLM research artifacts."""

    def __init__(self, cache_path: str | Path = DEFAULT_VERDICT_CACHE_PATH) -> None:
        self._path = Path(cache_path)
        self._entries: dict[str, dict[str, object]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with contextlib.suppress(Exception):
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._entries = {
                    str(k): v for k, v in raw.items() if isinstance(v, dict)
                }

    def get(self, fingerprint: str) -> EquityResearchArtifact | None:
        """Return the cached artifact for ``fingerprint``, or ``None``."""
        entry = self._entries.get(fingerprint)
        if entry is None:
            return None
        with contextlib.suppress(Exception):
            hydrated = dict(entry)
            pt = hydrated.get("price_target")
            if isinstance(pt, dict) and "value" in pt and "unit" in pt:
                hydrated["price_target"] = PriceTarget(
                    value=float(pt["value"]),
                    unit=str(pt["unit"]),
                )
            return EquityResearchArtifact.model_validate(hydrated)
        return None

    def put(self, fingerprint: str, artifact: EquityResearchArtifact) -> None:
        """Persist ``artifact`` under ``fingerprint``."""
        with contextlib.suppress(Exception):
            dumped = artifact.model_dump(mode="json", exclude={"price_target"})
            if isinstance(dumped, dict):
                if artifact.price_target is not None:
                    dumped["price_target"] = {
                        "value": float(artifact.price_target.value),
                        "unit": str(artifact.price_target.unit),
                    }
                else:
                    dumped["price_target"] = None
                self._entries[fingerprint] = dumped
                if len(self._entries) > _MAX_CACHE_ENTRIES:
                    excess = len(self._entries) - _MAX_CACHE_ENTRIES
                    for old_key in list(self._entries.keys())[:excess]:
                        self._entries.pop(old_key, None)
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self._path.with_suffix(".tmp")
                tmp_path.write_text(
                    json.dumps(self._entries, separators=(",", ":")),
                    encoding="utf-8",
                )
                tmp_path.replace(self._path)
