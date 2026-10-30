"""Ollama-backed LLM adapter implementing the analyst ports.

This is the only module that knows about PydanticAI. It implements
:class:`~alphalab.domain.analyst.ports.ExtractionProvider` and
:class:`~alphalab.domain.analyst.ports.NarrationProvider` using PydanticAI
agents with **``ToolOutput``** structured output.

Why ``ToolOutput`` and not ``NativeOutput``: the ``qwen3.8:27b-mlx`` tag
ignores both Ollama's native ``format`` and the OpenAI-compatible
``response_format`` JSON schema and answers in prose, so ``NativeOutput``
fails with ``Exceeded maximum output retries``. Tool-calling works, so
``ToolOutput`` reliably coerces the model into the typed shape. This was
verified empirically against the running server.

Two tiers (TradingAgents split): the quick model drives extraction, the deep
model drives synthesis. Both are plain Ollama models reached through the
OpenAI-compatible provider; the placeholder ``api_key`` is not a secret.

The adapter never produces a valuation number: the extraction schema has no
free numeric field and the narration schema has no price-target field, so the
model has nowhere to put one.
"""

from __future__ import annotations

from datetime import date

from pydantic_ai import Agent, ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from alphalab.adapters.llm_config import LLMSettings
from alphalab.domain.analyst.contracts import (
    ExtractionRequest,
    NarrationRequest,
    QualitativeFinding,
    RunMetadata,
    ThesisNarrative,
)
from alphalab.domain.analyst.tiers import ModelTier

_EXTRACTION_SYSTEM = (
    "You are a financial analyst assistant. Call the final_result tool "
    "immediately with NO prose outside the tool call. Extract up to 3 concise "
    "qualitative facts (both positive and negative) from the source text for "
    "the ticker (when shown as 'COMPANY_X', treat it as an intentionally "
    "anonymized equity identifier for blind evaluation). Quote a short verbatim "
    "phrase for evidence_quote (<15 words) and name its source. Report ONLY "
    "qualitative facts without numbers."
)

_NARRATION_SYSTEM = (
    "You are a senior equity research arbiter. Call the final_result tool "
    "immediately with NO prose outside the tool call. From the supplied "
    "findings and pre-computed valuation context for the ticker (when shown as "
    "'COMPANY_X', treat it as an intentionally anonymized equity identifier for "
    "blind evaluation), state thesis, risks, and catalysts in 1 short sentence "
    "each (<15 words) and issue a directional verdict (BUY/HOLD/SELL). Weigh "
    "both valuation and fundamental quality: if findings report negative free "
    "cash flow, compressed margins, or elevated net leverage, issue HOLD or "
    "SELL even when consensus_mid exceeds current_price. Do NOT invent, alter, "
    "or restate any price target or numeric valuation."
)


class OllamaLLMAdapter:
    """Concrete two-tier LLM provider for extraction and narration.

    Implements both domain ports. One adapter instance owns both tier agents so
    the composition root wires a single object. The returned
    :class:`RunMetadata` for a given tier is built from configuration — the
    model's knowledge cutoff is explicit config, never guessed.
    """

    __slots__ = ("_deep_agent", "_quick_agent", "_settings")

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self._settings = settings or LLMSettings()
        model_settings = ModelSettings(
            timeout=self._settings.timeout_seconds,
            temperature=self._settings.temperature,
            seed=self._settings.seed,
            extra_body={"reasoning_effort": "none"},
        )
        self._quick_agent: Agent[None, tuple[QualitativeFinding, ...]] = Agent(
            self._build_model(self._settings.quick_model),
            output_type=ToolOutput(tuple[QualitativeFinding, ...]),
            system_prompt=_EXTRACTION_SYSTEM,
            retries=self._settings.max_retries,
            model_settings=model_settings,
        )
        self._deep_agent: Agent[None, ThesisNarrative] = Agent(
            self._build_model(self._settings.deep_model),
            output_type=ToolOutput(ThesisNarrative),
            system_prompt=_NARRATION_SYSTEM,
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

    def extract(self, request: ExtractionRequest) -> tuple[QualitativeFinding, ...]:
        """Extract qualitative findings via the quick-think agent."""
        prompt = self._extraction_prompt(request, anonymize=self._settings.anonymize_ticker)
        return tuple(self._run_with_retry(self._quick_agent, prompt))

    def narrate(self, request: NarrationRequest) -> ThesisNarrative:
        """Synthesise the thesis narrative via the deep-think agent."""
        prompt = self._narration_prompt(request, anonymize=self._settings.anonymize_ticker)
        return self._run_with_retry(self._deep_agent, prompt)

    def _run_with_retry(self, agent: Agent[None, object], prompt: str) -> object:  # type: ignore[type-var]
        last_exc: Exception | None = None
        attempts = max(1, self._settings.max_retries + 1)
        for attempt in range(attempts):
            try:
                if attempt == 0:
                    return agent.run_sync(prompt).output
                retry_settings = ModelSettings(
                    timeout=self._settings.timeout_seconds,
                    temperature=max(0.2, self._settings.temperature),
                    seed=(self._settings.seed or 42) + attempt,
                    extra_body={"reasoning_effort": "none"},
                )
                return agent.run_sync(prompt, model_settings=retry_settings).output
            except Exception as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    def run_metadata(
        self,
        *,
        tier: ModelTier,
        knowledge_date: object,
        dataset_hash: str,
    ) -> RunMetadata:
        """Build first-class run metadata for the given tier from config.

        The knowledge cutoff is read from configuration (explicit per dossier
        §6), never invented. ``knowledge_date`` is the PIT date of the run.
        """
        if not isinstance(knowledge_date, date):
            raise TypeError("knowledge_date must be a datetime.date")
        if tier is ModelTier.DEEP:
            name, version, cutoff = (
                self._settings.deep_model,
                self._settings.deep_version,
                self._settings.deep_cutoff,
            )
        else:
            name, version, cutoff = (
                self._settings.quick_model,
                self._settings.quick_version,
                self._settings.quick_cutoff,
            )
        return RunMetadata(
            model_name=name,
            tier=tier,
            llm_version=version,
            llm_cutoff=cutoff,
            knowledge_date=knowledge_date,
            dataset_hash=dataset_hash,
        )

    @staticmethod
    def _extraction_prompt(request: ExtractionRequest, *, anonymize: bool = False) -> str:
        label = "COMPANY_X" if anonymize else request.ticker
        joined = "\n\n---\n\n".join(
            doc.replace("<", "(").replace(">", ")") for doc in request.documents
        )
        return f"Ticker: {label}\n\nSource documents:\n{joined}"

    @staticmethod
    def _narration_prompt(request: NarrationRequest, *, anonymize: bool = False) -> str:
        label = "COMPANY_X" if anonymize else request.ticker
        facts = "\n".join(
            f"- [{f.category}/{f.direction.value}] "
            f"{f.evidence_quote.replace('<', '(').replace('>', ')')} ({f.source})"
            for f in request.findings
        )
        numbers = "\n".join(f"- {name}: {value}" for name, value in request.valuation_summary)
        return (
            f"Ticker: {label}\n\n"
            f"Computed valuation context (do not alter):\n{numbers}\n\n"
            f"Qualitative findings:\n{facts}"
        )
