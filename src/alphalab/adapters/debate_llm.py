"""Ollama-backed debate adapter implementing the Bull / Bear / Judge ports.

This mirrors ``adapters/llm.py`` exactly — same PydanticAI ``ToolOutput`` path,
same OpenAI-compatible Ollama provider, same ``ALPHALAB_LLM_`` configuration
(:class:`~alphalab.adapters.llm_config.LLMSettings`) — and adds the three
debate roles without touching the existing analyst adapter's public API.

Why ``ToolOutput`` and not ``NativeOutput``: the ``qwen3.8:27b-mlx`` tag
ignores the JSON schema and answers in prose, so ``NativeOutput`` fails with
"Exceeded maximum output retries". Tool-calling reliably coerces the typed
shape. This was verified empirically against the running server (see
``adapters/llm.py``).

Tier assignment (TradingAgents quick-/deep-think split):

* Bull and Bear → **quick** (``qwen3-vl:latest``): fast debaters.
* Judge → **deep** (``qwen3.8:27b-mlx``): the arbiter that reads the whole
  transcript.

The adapter never produces a valuation number: none of the debate contracts
has a numeric valuation field, so the model has nowhere to put one.
"""

from __future__ import annotations

from pydantic_ai import Agent, ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from alphalab.adapters.llm_config import LLMSettings
from alphalab.domain.debate.contracts import BearCase, BullCase, JudgeRuling
from alphalab.domain.debate.ports import BearRequest, BullRequest, JudgeRequest

_BULL_SYSTEM = (
    "You are the BULL analyst in a structured investment debate. Argue the "
    "bullish case for the given ticker from the shared evidence, rebutting the "
    "bear's prior points when present. State your thesis and catalysts in prose, "
    "cite verbatim evidence, and give a bounded confidence in [0,1]. The "
    "valuation numbers were computed by trusted code and are context only. Never "
    "invent, estimate, or state any price target, valuation, ratio, or figure."
)

_BEAR_SYSTEM = (
    "You are the BEAR analyst in a structured investment debate. Argue the "
    "bearish case for the given ticker from the shared evidence, rebutting the "
    "bull's prior points when present. State your thesis and risks in prose, "
    "cite verbatim evidence, and give a bounded confidence in [0,1]. The "
    "valuation numbers were computed by trusted code and are context only. Never "
    "invent, estimate, or state any price target, valuation, ratio, or figure."
)

_JUDGE_SYSTEM = (
    "You are the impartial JUDGE resolving an investment debate. Read the full "
    "ordered transcript of bull and bear turns and issue a directional verdict "
    "(BUY/HOLD/SELL). Explain your reasoning in prose and name which bull points "
    "and which bear points weighed in your decision. The valuation numbers were "
    "computed by trusted code and are context only. Never invent, alter, or "
    "restate any price target or numeric valuation."
)


class OllamaDebateAdapter:
    """Concrete two-tier LLM provider for the Bull / Bear / Judge debate.

    Implements :class:`~alphalab.domain.debate.ports.BullProvider`,
    :class:`~alphalab.domain.debate.ports.BearProvider`, and
    :class:`~alphalab.domain.debate.ports.JudgeProvider`. One instance owns all
    three agents so the composition root wires a single object.
    """

    __slots__ = ("_bear_agent", "_bull_agent", "_judge_agent", "_settings")

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self._settings = settings or LLMSettings()
        model_settings = ModelSettings(timeout=self._settings.timeout_seconds)
        self._bull_agent: Agent[None, BullCase] = Agent(
            self._build_model(self._settings.quick_model),
            output_type=ToolOutput(BullCase),
            system_prompt=_BULL_SYSTEM,
            retries=self._settings.max_retries,
            model_settings=model_settings,
        )
        self._bear_agent: Agent[None, BearCase] = Agent(
            self._build_model(self._settings.quick_model),
            output_type=ToolOutput(BearCase),
            system_prompt=_BEAR_SYSTEM,
            retries=self._settings.max_retries,
            model_settings=model_settings,
        )
        self._judge_agent: Agent[None, JudgeRuling] = Agent(
            self._build_model(self._settings.deep_model),
            output_type=ToolOutput(JudgeRuling),
            system_prompt=_JUDGE_SYSTEM,
            retries=self._settings.max_retries,
            model_settings=model_settings,
        )

    def _build_model(self, model_name: str) -> OpenAIChatModel:
        """Build an OpenAI-compatible model pointed at the Ollama endpoint."""
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(
                base_url=self._settings.base_url,
                api_key=self._settings.api_key,
            ),
        )

    def argue_bull(self, request: BullRequest) -> BullCase:
        """Bull case via the quick-think agent (``BullProvider.argue``)."""
        return self._bull_agent.run_sync(_bull_prompt(request)).output

    def argue_bear(self, request: BearRequest) -> BearCase:
        """Bear case via the quick-think agent (``BearProvider.argue``)."""
        return self._bear_agent.run_sync(_bear_prompt(request)).output

    def rule(self, request: JudgeRequest) -> JudgeRuling:
        """Arbiter ruling via the deep-think agent (``JudgeProvider.rule``)."""
        return self._judge_agent.run_sync(_judge_prompt(request)).output


def _numbers_block(valuation_summary: tuple[tuple[str, float], ...]) -> str:
    return "\n".join(f"- {name}: {value}" for name, value in valuation_summary)


def _bull_prompt(request: BullRequest) -> str:
    evidence = "\n".join(f"- {e}" for e in request.evidence)
    rebuttal = "\n".join(f"- {a}" for a in request.opponent_arguments)
    return (
        f"Ticker: {request.ticker}\n\n"
        f"Computed valuation context (do not alter):\n"
        f"{_numbers_block(request.valuation_summary)}\n\n"
        f"Shared evidence:\n{evidence}\n\n"
        f"Bear's prior public arguments to rebut:\n{rebuttal or '- (none yet)'}"
    )


def _bear_prompt(request: BearRequest) -> str:
    evidence = "\n".join(f"- {e}" for e in request.evidence)
    rebuttal = "\n".join(f"- {a}" for a in request.opponent_arguments)
    return (
        f"Ticker: {request.ticker}\n\n"
        f"Computed valuation context (do not alter):\n"
        f"{_numbers_block(request.valuation_summary)}\n\n"
        f"Shared evidence:\n{evidence}\n\n"
        f"Bull's prior public arguments to rebut:\n{rebuttal or '- (none yet)'}"
    )


def _judge_prompt(request: JudgeRequest) -> str:
    transcript = "\n".join(f"{t.turn_number}. [{t.role.value}] {t.argument}" for t in request.turns)
    return (
        f"Ticker: {request.ticker}\n\n"
        f"Computed valuation context (do not alter):\n"
        f"{_numbers_block(request.valuation_summary)}\n\n"
        f"Debate transcript (in order):\n{transcript}"
    )


class _BullAdapter:
    """Thin ``BullProvider`` view over :class:`OllamaDebateAdapter`."""

    __slots__ = ("_adapter",)

    def __init__(self, adapter: OllamaDebateAdapter) -> None:
        self._adapter = adapter

    def argue(self, request: BullRequest) -> BullCase:
        return self._adapter.argue_bull(request)


class _BearAdapter:
    """Thin ``BearProvider`` view over :class:`OllamaDebateAdapter`."""

    __slots__ = ("_adapter",)

    def __init__(self, adapter: OllamaDebateAdapter) -> None:
        self._adapter = adapter

    def argue(self, request: BearRequest) -> BearCase:
        return self._adapter.argue_bear(request)


def bull_provider(adapter: OllamaDebateAdapter) -> _BullAdapter:
    """Return a ``BullProvider`` bound to ``adapter`` (``.argue`` dispatch)."""
    return _BullAdapter(adapter)


def bear_provider(adapter: OllamaDebateAdapter) -> _BearAdapter:
    """Return a ``BearProvider`` bound to ``adapter`` (``.argue`` dispatch)."""
    return _BearAdapter(adapter)
