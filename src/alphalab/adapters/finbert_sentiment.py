"""Headline sentiment sensor (FinBERT optional, deterministic lexicon default)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CACHE_PATH = Path("data/finbert_cache.json")
_ENV_FLAG = "ALPHALAB_FINBERT"
_MODEL_NAME = "ProsusAI/finbert"
_SEVERE_THRESHOLD = -0.75
_SKEW_THRESHOLD = 0.2

_POSITIVE = frozenset(
    {
        "beats", "beat", "raises", "raised", "upgrade", "upgraded", "record", "surge",
        "surges", "soars", "growth", "profit", "strong", "outperform", "approval",
        "approved", "buyback", "dividend", "expands", "exceeds", "bullish", "wins",
        "rally", "rebound", "upside",
    }
)  # fmt: skip
_NEGATIVE = frozenset(
    {
        "misses", "missed", "miss", "cuts", "cut", "downgrade", "downgraded", "lawsuit",
        "sues", "probe", "investigation", "recall", "bankruptcy", "warning", "warns",
        "plunge", "plunges", "falls", "slump", "fraud", "layoffs", "default", "weak",
        "bearish", "subpoena", "delisting", "restatement", "loss", "losses", "decline",
    }
)  # fmt: skip
_SEVERE = frozenset(
    {"bankruptcy", "fraud", "subpoena", "delisting", "restatement", "default", "recall"}
)  # fmt: skip
_NEGATORS = frozenset({"not", "no", "without", "never", "fails", "failed"})
_TOKEN = re.compile(r"[a-z']+")


class SentimentSummary(BaseModel):
    """Aggregate sentiment over a set of documents."""

    model_config = ConfigDict(frozen=True)

    net_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    worst_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    count: int = 0

    def regime_note(self) -> str:
        """Qualitative, digit-free sentence for LLM documents."""
        if self.count == 0:
            return "[FinBERT News Sentiment] No recent headlines were available to assess."
        if self.worst_score <= _SEVERE_THRESHOLD:
            return (
                "[FinBERT News Sentiment] Recent headlines contain a severe negative "
                "catalyst that deserves scrutiny before trusting any valuation upside."
            )
        if self.net_score >= _SKEW_THRESHOLD:
            return "[FinBERT News Sentiment] Recent headlines skew constructive."
        if self.net_score <= -_SKEW_THRESHOLD:
            return "[FinBERT News Sentiment] Recent headlines skew cautious to negative."
        return "[FinBERT News Sentiment] Recent headlines are largely neutral noise."


def lexicon_score(text: str) -> float:
    """Deterministic per-document score in [-1, 1]."""
    tokens = _TOKEN.findall(text.lower())
    pos = neg = 0
    severe = False
    for i, tok in enumerate(tokens):
        negated = any(t in _NEGATORS for t in tokens[max(0, i - 2) : i])
        if tok in _POSITIVE:
            neg, pos = (neg + 1, pos) if negated else (neg, pos + 1)
        elif tok in _NEGATIVE:
            if negated:
                pos += 1
            else:
                neg += 1
                severe = severe or tok in _SEVERE
    total = pos + neg
    if total == 0:
        return 0.0
    score = (pos - neg) / total
    if severe:
        score = min(score, -0.9)
    return max(-1.0, min(1.0, score))


class FinBertSentimentScorer:
    """Scores documents; FinBERT only when opted in and importable."""

    def __init__(self, *, cache_path: Path | str | None = DEFAULT_CACHE_PATH) -> None:
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[str, float] | None = None
        self._pipe: Any = None
        self._backend: str | None = None
        self._lock = Lock()

    # -- backend -------------------------------------------------------
    def _resolve_backend(self) -> str:
        if self._backend is not None:
            return self._backend
        self._backend = "lexicon"
        if os.environ.get(_ENV_FLAG) == "1":
            with contextlib.suppress(Exception):
                import torch  # type: ignore[import-not-found]
                from transformers import pipeline  # type: ignore[import-not-found]

                device = "mps" if torch.backends.mps.is_available() else "cpu"
                try:
                    pipe = pipeline(
                        "text-classification", model=_MODEL_NAME, top_k=None, device=device,
                        model_kwargs={"local_files_only": True},
                    )
                except Exception:
                    pipe = pipeline(
                        "text-classification", model=_MODEL_NAME, top_k=None, device=device
                    )
                self._pipe = pipe
                self._backend = "finbert"
        return self._backend

    def _finbert_score(self, text: str) -> float:
        out = self._pipe([text[:1500]], truncation=True, max_length=256)[0]
        probs = {str(d["label"]).lower(): float(d["score"]) for d in out}
        return max(-1.0, min(1.0, probs.get("positive", 0.0) - probs.get("negative", 0.0)))

    def _score_one(self, text: str, backend: str) -> float:
        if backend == "finbert":
            try:
                return self._finbert_score(text)
            except Exception:
                return lexicon_score(text)
        return lexicon_score(text)

    # -- cache ---------------------------------------------------------
    def _load_cache(self) -> dict[str, float]:
        if self._cache is not None:
            return self._cache
        cache: dict[str, float] = {}
        if self._cache_path is not None:
            with contextlib.suppress(Exception):
                raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    cache = {
                        str(k): float(v)
                        for k, v in raw.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    }
        self._cache = cache
        return cache

    def _save_cache(self, cache: dict[str, float]) -> None:
        if self._cache_path is None:
            return
        with contextlib.suppress(Exception):
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._cache_path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(cache, fh)
                os.replace(tmp, self._cache_path)
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)

    # -- public --------------------------------------------------------
    def score_documents(self, docs: Sequence[str]) -> SentimentSummary:
        """Never raises; returns a neutral empty summary on empty/failed input."""
        try:
            texts = [d for d in docs if isinstance(d, str) and d.strip()]
            if not texts:
                return SentimentSummary()
            with self._lock:
                backend = self._resolve_backend()
                cache = self._load_cache()
                dirty = False
                scores: list[float] = []
                for text in texts:
                    key = hashlib.sha256(text.encode("utf-8")).hexdigest() + ":" + backend
                    if key not in cache:
                        cache[key] = self._score_one(text, backend)
                        dirty = True
                    scores.append(cache[key])
                if dirty:
                    self._save_cache(cache)
            pos = sum(1 for s in scores if s > 0.1)
            neg = sum(1 for s in scores if s < -0.1)
            return SentimentSummary(
                net_score=max(-1.0, min(1.0, sum(scores) / len(scores))),
                positive_count=pos,
                neutral_count=len(scores) - pos - neg,
                negative_count=neg,
                worst_score=max(-1.0, min(1.0, min(scores))),
                count=len(scores),
            )
        except Exception:
            return SentimentSummary()
