"""Local pytest configuration for the LLM integration tests.

This ``conftest.py`` lives inside the integration test directory this agent
owns, so it registers the custom ``llm`` marker *without* touching the shared
``pyproject.toml``. Registering it here keeps ``-m llm`` selectable and avoids
the "unknown marker" warning for these tests.

The real-LLM test is skipped by default; it runs only when
``ALPHALAB_LLM_TESTS=1`` is set in the environment (see the test module).
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``llm`` marker locally for this test package."""
    config.addinivalue_line(
        "markers",
        "llm: real end-to-end test that hits a running Ollama server "
        "(opt-in via ALPHALAB_LLM_TESTS=1).",
    )
    config.addinivalue_line(
        "markers",
        "edgar_live: real end-to-end test that hits the live SEC EDGAR API "
        "(opt-in via ALPHALAB_LIVE_TESTS=1).",
    )
