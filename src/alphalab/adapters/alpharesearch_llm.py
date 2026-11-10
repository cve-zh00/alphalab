"""Ollama-backed LLM deliberation adapter for the 3-role AlphaResearch committee."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelSettings, ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from alphalab.adapters.llm_config import LLMSettings
from alphalab.domain.alpharesearch.contracts import (
    CommitteeDeliberationNote,
    DiscoverySignal,
    ResearchProposal,
)

_COMMITTEE_SYSTEM_PROMPT = (
    "Eres el Comité Cuantitativo de Discernimiento AlphaResearch (ScoutProposer + SkepticReviewer) "
    "para un sistema institucional de renta variable S&P 500 Point-in-Time (PIT).\n"
    "Tu rol es evaluar señales de descubrimiento recientes (papers arXiv q-fin, repositorios "
    "cuantitativos y postmortems empíricos de ejecución/calibración) y auditar cada propuesta "
    "candidata ANTES de que pase al EmpiricalGatekeeper.\n"
    "REGLAS ESTRICTAS:\n"
    "1. 'El código calcula; el modelo argumenta y juzga': jamás inventes precios objetivo, "
    "tasas de descuento ni retornos simulados; las series de ablación provienen exclusivamente "
    "del panel histórico Point-in-Time.\n"
    "2. Para cada propuesta (identificada por su proposal_index), emite en español "
    "técnico y conciso:\n"
    "   - scout_rationale: justificación cuantitativa y vínculo con las señales "
    "descubiertas (1-2 frases).\n"
    "   - skeptic_audit_note: crítica adversarial de riesgo metodológico (fuga PIT, sobreajuste, "
    "frontera numérica LLM, rotación o preservación de guardas NOFX Lot-1/Dykstra) (1-2 frases).\n"
    "   - recommended_safeguard: salvaguarda determinista de implementación o test "
    "unitario exigido (1 frase).\n"
    "/no_think"
)


class _LLMProposalDeliberationItem(BaseModel):
    """Structured LLM output item for a single candidate proposal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_index: int = Field(ge=0)
    scout_rationale: str = Field(min_length=8)
    skeptic_audit_note: str = Field(min_length=8)
    recommended_safeguard: str = Field(min_length=5)


class _LLMCommitteeBatchOutput(BaseModel):
    """Structured batch deliberation returned by the local Ollama model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: list[_LLMProposalDeliberationItem] = Field(default_factory=list)


class OllamaAlphaResearchCommittee:
    """Executes Role 1 (ScoutProposer) + Role 2 (SkepticReviewer) LLM deliberation via Ollama."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self._settings = settings or LLMSettings()
        provider = OpenAIProvider(
            base_url=self._settings.base_url,
            api_key=self._settings.api_key,
        )
        model = OpenAIChatModel(
            self._settings.deep_model,
            provider=provider,
        )
        model_settings = ModelSettings(
            timeout=self._settings.timeout_seconds,
            temperature=self._settings.temperature,
            seed=self._settings.seed,
            extra_body={"reasoning_effort": "none"},
        )
        self._agent: Agent[None, _LLMCommitteeBatchOutput] = Agent(
            model,
            output_type=ToolOutput(_LLMCommitteeBatchOutput, strict=True),
            system_prompt=_COMMITTEE_SYSTEM_PROMPT,
            retries=self._settings.max_retries,
            model_settings=model_settings,
        )

    @property
    def model_name(self) -> str:
        """Configured deep deliberation model identifier."""
        return self._settings.deep_model

    def _is_server_reachable(self) -> bool:
        """Fast connectivity probe so offline runs fail open immediately."""
        base_root = self._settings.base_url.removesuffix("/v1").rstrip("/")
        try:
            resp = httpx.get(f"{base_root}/api/tags", timeout=1.5)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def deliberate(
        self,
        *,
        signals: Sequence[DiscoverySignal],
        proposals: Sequence[ResearchProposal],
    ) -> dict[int, CommitteeDeliberationNote]:
        """Run structured LLM deliberation across the discovered signals and candidate proposals."""
        if not proposals or not self._is_server_reachable():
            return {}

        signal_lines = [
            f"- [{sig.source_kind.value}] {sig.source_name}: {sig.title} | {sig.summary[:220]}"
            for sig in signals[:8]
        ]
        proposal_lines = [
            (
                f"[{idx}] subsystem={prop.subsystem.value} | title={prop.title}\n"
                f"    formula={prop.mathematical_formulation}\n"
                f"    hypothesis={prop.falsifiable_hypothesis[:220]}\n"
                f"    flags: uses_pit={prop.pit_safe}, "
                f"llm_emits_numbers={not prop.llm_numeric_boundary_safe}, "
                f"weakens_nofx={not prop.preserves_nofx_guards}, "
                f"loc_delta={prop.estimated_loc_delta}, "
                f"turnover_delta={prop.ablation_evidence.turnover_delta_pct:.2f}"
            )
            for idx, prop in enumerate(proposals)
        ]
        prompt = (
            "SEÑALES DESCUBIERTAS EN ESTE CICLO:\n"
            + ("\n".join(signal_lines) if signal_lines else "- Sin señales externas nuevas.")
            + "\n\nPROPUESTAS CANDIDATAS A EVALUAR:\n"
            + "\n".join(proposal_lines)
            + "\n\nDevuelve un item por cada proposal_index (0 .. "
            + f"{len(proposals) - 1})."
        )
        try:
            result = self._agent.run_sync(prompt)
        except Exception:
            return {}

        notes: dict[int, CommitteeDeliberationNote] = {}
        for item in result.output.items:
            if 0 <= item.proposal_index < len(proposals):
                notes[item.proposal_index] = CommitteeDeliberationNote(
                    proposal_index=item.proposal_index,
                    scout_rationale=item.scout_rationale.strip(),
                    skeptic_audit_note=item.skeptic_audit_note.strip(),
                    recommended_safeguard=item.recommended_safeguard.strip(),
                )
        return notes
