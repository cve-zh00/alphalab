"""Unit tests for the LLM adapter — configuration and metadata only (no LLM).

These tests construct the adapter and exercise everything that does not touch
the network: settings resolution from env, structural conformance to the
domain ports, and the mandatory, config-driven ``RunMetadata`` (the cutoff is
read from config, never invented).
"""

from __future__ import annotations

from datetime import date

import pytest

from alphalab.adapters.llm import OllamaLLMAdapter
from alphalab.adapters.llm_config import LLMSettings
from alphalab.domain.analyst.ports import ExtractionProvider, NarrationProvider
from alphalab.domain.analyst.tiers import ModelTier


class TestLLMSettings:
    def test_defaults_point_at_local_ollama(self) -> None:
        settings = LLMSettings()
        assert settings.base_url == "http://localhost:11434/v1"
        assert settings.api_key == "ollama"  # placeholder, not a secret
        assert settings.deep_model == "qwen3.8:27b-mlx"
        assert settings.quick_model == "qwen3-vl:latest"

    def test_env_overrides_base_url_and_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHALAB_LLM_BASE_URL", "http://example:9999/v1")
        monkeypatch.setenv("ALPHALAB_LLM_DEEP_MODEL", "custom-deep")
        monkeypatch.setenv("ALPHALAB_LLM_DEEP_CUTOFF", "2023-05-01")
        settings = LLMSettings()
        assert settings.base_url == "http://example:9999/v1"
        assert settings.deep_model == "custom-deep"
        assert settings.deep_cutoff == date(2023, 5, 1)

    def test_timeout_must_be_positive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHALAB_LLM_TIMEOUT_SECONDS", "-1")
        with pytest.raises(ValueError):
            LLMSettings()

    def test_settings_are_frozen(self) -> None:
        settings = LLMSettings()
        with pytest.raises(ValueError):
            settings.base_url = "x"  # type: ignore[misc]


class TestAdapterConformance:
    def test_satisfies_both_ports(self) -> None:
        adapter = OllamaLLMAdapter()
        assert isinstance(adapter, ExtractionProvider)
        assert isinstance(adapter, NarrationProvider)


class TestRunMetadata:
    def test_deep_tier_uses_deep_config(self) -> None:
        adapter = OllamaLLMAdapter()
        meta = adapter.run_metadata(
            tier=ModelTier.DEEP,
            knowledge_date=date(2025, 1, 15),
            dataset_hash="h1",
        )
        assert meta.model_name == "qwen3.8:27b-mlx"
        assert meta.tier is ModelTier.DEEP
        assert meta.llm_cutoff == date(2024, 10, 1)
        assert meta.dataset_hash == "h1"

    def test_quick_tier_uses_quick_config(self) -> None:
        adapter = OllamaLLMAdapter()
        meta = adapter.run_metadata(
            tier=ModelTier.QUICK,
            knowledge_date=date(2025, 1, 15),
            dataset_hash="h2",
        )
        assert meta.model_name == "qwen3-vl:latest"
        assert meta.tier is ModelTier.QUICK

    def test_rejects_non_date_knowledge_date(self) -> None:
        adapter = OllamaLLMAdapter()
        with pytest.raises(TypeError):
            adapter.run_metadata(
                tier=ModelTier.DEEP,
                knowledge_date="2025-01-15",  # not a date
                dataset_hash="h",
            )

    def test_cutoff_is_never_none(self) -> None:
        # Dossier §6: a run can never silently drop the cutoff.
        adapter = OllamaLLMAdapter()
        meta = adapter.run_metadata(
            tier=ModelTier.DEEP,
            knowledge_date=date(2025, 1, 15),
            dataset_hash="h",
        )
        assert meta.llm_cutoff is not None

    def test_run_with_retry_recovers_from_transient_http_500(self) -> None:
        adapter = OllamaLLMAdapter()
        calls: list[object] = []

        class _FakeResult:
            output = "ok"

        class _FakeAgent:
            def run_sync(self, prompt: str, model_settings: object = None) -> _FakeResult:
                calls.append(model_settings)
                if len(calls) == 1:
                    raise RuntimeError("XML syntax error on line 13: element <parameter>")
                return _FakeResult()

        out = adapter._run_with_retry(_FakeAgent(), "prompt")  # type: ignore[arg-type]
        assert out == "ok"
        assert len(calls) == 2
        assert calls[0] is None
        assert calls[1] is not None
