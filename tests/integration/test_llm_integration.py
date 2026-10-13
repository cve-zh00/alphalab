"""Real end-to-end LLM integration test (opt-in, skipped by default).

This test actually hits a running Ollama server and verifies that the
``ToolOutput`` path returns correctly-typed domain contracts from the real
models. It is skipped unless BOTH conditions hold:

* ``ALPHALAB_LLM_TESTS=1`` is set in the environment, and
* the Ollama server at the configured ``base_url`` answers its ``/models``
  endpoint.

Run it explicitly with::

    ALPHALAB_LLM_TESTS=1 uv run pytest -m llm -v --no-cov

The deep (27B) model takes tens of seconds per call, which is exactly why this
is never part of the default suite.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest

from alphalab.adapters.llm import OllamaLLMAdapter
from alphalab.adapters.llm_config import LLMSettings
from alphalab.artifact.models import Verdict
from alphalab.domain.analyst.contracts import (
    ExtractionRequest,
    NarrationRequest,
    QualitativeFinding,
    ThesisNarrative,
)
from alphalab.domain.analyst.direction import Direction

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


def test_tool_output_returns_typed_narrative() -> None:
    # Arrange
    adapter = OllamaLLMAdapter(_SETTINGS)
    findings = (
        QualitativeFinding(
            category="management_guidance",
            direction=Direction.POSITIVE,
            evidence_quote="Management raised full-year revenue guidance.",
            source="Q3 earnings call",
            confidence=0.8,
        ),
        QualitativeFinding(
            category="regulatory_risk",
            direction=Direction.NEGATIVE,
            evidence_quote="A new antitrust probe was opened.",
            source="Reuters",
            confidence=0.6,
        ),
    )
    request = NarrationRequest(
        ticker="SYNTH",
        findings=findings,
        valuation_summary=(("consensus_mid", 42.0), ("current_price", 40.0)),
    )

    # Act: real deep-think model via ToolOutput.
    narrative = adapter.narrate(request)

    # Assert: we got the correctly-typed domain contract back.
    assert isinstance(narrative, ThesisNarrative)
    assert isinstance(narrative.verdict, Verdict)
    assert narrative.thesis
    assert narrative.risks
    assert narrative.catalysts


def test_tool_output_returns_typed_findings() -> None:
    # Arrange
    adapter = OllamaLLMAdapter(_SETTINGS)
    request = ExtractionRequest(
        ticker="SYNTH",
        documents=(
            "SYNTH Corp reported that management reaffirmed its full-year "
            "guidance and announced a new product launch for next quarter. "
            "Separately, regulators opened an inquiry into its data practices.",
        ),
    )

    # Act: real quick-think model via ToolOutput.
    findings = adapter.extract(request)

    # Assert: a tuple of correctly-typed findings with bounded confidence.
    assert isinstance(findings, tuple)
    assert all(isinstance(f, QualitativeFinding) for f in findings)
    assert all(0.0 <= f.confidence <= 1.0 for f in findings)
