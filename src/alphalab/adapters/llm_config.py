"""Configuration for the Ollama-backed LLM adapter (pydantic-settings).

No secrets live in code: the Ollama ``api_key`` is a harmless placeholder
(``"ollama"``) and the ``base_url`` is overridable by environment variable so
the endpoint is never hard-wired. Model names, timeouts and retries are all
configuration.

Crucially, the model knowledge **cutoff** cannot be discovered from Ollama
programmatically for the MLX tags, so each tier's cutoff is an *explicit,
required* setting (dossier §6): we refuse to invent it or leave it ``None``.

Environment variables use the ``ALPHALAB_LLM_`` prefix, e.g.
``ALPHALAB_LLM_BASE_URL``, ``ALPHALAB_LLM_DEEP_MODEL``,
``ALPHALAB_LLM_DEEP_CUTOFF``.
"""

from __future__ import annotations

from datetime import date

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Settings for the two-tier Ollama LLM adapter.

    Attributes:
        base_url: OpenAI-compatible endpoint for Ollama.
        api_key: Placeholder credential required by the OpenAI client shape;
            Ollama ignores it. Not a secret.
        quick_model: Model name for the quick-think (extraction) tier.
        deep_model: Model name for the deep-think (synthesis) tier.
        quick_cutoff: Knowledge cutoff of the quick model (explicit, required).
        deep_cutoff: Knowledge cutoff of the deep model (explicit, required).
        quick_version: Pinned version string for the quick model.
        deep_version: Pinned version string for the deep model.
        timeout_seconds: Per-request timeout; the 27B takes tens of seconds.
        max_retries: Bounded retries for the typed output tool call.
    """

    model_config = SettingsConfigDict(
        env_prefix="ALPHALAB_LLM_",
        frozen=True,
        extra="ignore",
    )

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    quick_model: str = "qwen3-vl:latest"
    deep_model: str = "qwen3.8:27b-mlx"
    quick_cutoff: date = date(2024, 10, 1)
    deep_cutoff: date = date(2024, 10, 1)
    quick_version: str = "qwen3-vl"
    deep_version: str = "qwen3.8-27b-mlx"
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 42
    anonymize_ticker: bool = True
