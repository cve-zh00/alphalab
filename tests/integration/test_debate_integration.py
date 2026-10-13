"""Real end-to-end debate integration test (opt-in, skipped by default).

Runs ONE full 1-round debate against the real Ollama models and verifies the
three roles return correctly-typed domain contracts via the ``ToolOutput`` path
(the MLX tag ignores the JSON schema, so ``NativeOutput`` would fail). Skipped
unless BOTH hold:

* ``ALPHALAB_LLM_TESTS=1`` is set, and
* the Ollama server at the configured ``base_url`` answers ``/models``.

Run it explicitly with::

    ALPHALAB_LLM_TESTS=1 uv run pytest -m llm -v --no-cov

``max_rounds=1`` keeps it to two debater turns plus one deep-think ruling; the
27B judge still takes tens of seconds, which is exactly why this is never part
of the default suite.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest

from alphalab.adapters.debate_llm import (
    OllamaDebateAdapter,
    bear_provider,
    bull_provider,
)
from alphalab.adapters.llm_config import LLMSettings
from alphalab.artifact.models import Verdict
from alphalab.domain.debate.conductor import conduct_debate
from alphalab.domain.debate.contracts import DebateTranscript, JudgeRuling

_ENABLED = os.environ.get("ALPHALAB_LLM_TESTS") == "1"


def _server_reachable(base_url: str) -> bool:
    """True if the Ollama OpenAI-compatible endpoint answers ``/models``."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


_SETTINGS = LLMSettings()
_REACHABLE = _ENABLED and _server_reachable(_SETTINGS.base_url)

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not _ENABLED,
        reason="real LLM test disabled; set ALPHALAB_LLM_TESTS=1 to enable",
    ),
    pytest.mark.skipif(
        _ENABLED and not _REACHABLE,
        reason=f"Ollama server not reachable at {_SETTINGS.base_url}",
    ),
]


def test_full_one_round_debate_returns_typed_contracts() -> None:
    # Arrange: real two-tier adapter (quick bull/bear, deep judge).
    adapter = OllamaDebateAdapter(_SETTINGS)
    evidence = (
        "SYNTH Corp reaffirmed full-year revenue guidance and announced a new "
        "product launch for next quarter.",
        "Regulators opened an inquiry into SYNTH Corp's data practices.",
    )
    summary = (("consensus_mid", 42.0), ("current_price", 40.0))

    # Act: a complete 1-round debate (bull, bear, then the deep-think judge).
    transcript = conduct_debate(
        ticker="SYNTH",
        evidence=evidence,
        valuation_summary=summary,
        max_rounds=1,
        bull=bull_provider(adapter),
        bear=bear_provider(adapter),
        judge=adapter,
    )

    # Assert: correctly-typed domain contracts came back from the real models.
    assert isinstance(transcript, DebateTranscript)
    assert len(transcript.turns) == 2
    assert transcript.turns[0].argument
    assert transcript.turns[1].argument
    ruling = transcript.ruling
    assert isinstance(ruling, JudgeRuling)
    assert isinstance(ruling.verdict, Verdict)
    assert ruling.rationale
