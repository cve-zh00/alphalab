"""The ``research`` command: run a pipeline and render its artifact.

Orchestration only. It wires a data source (fixture today, live EDGAR when the
other agent lands it) to a pipeline (:class:`AnalystPipeline` or
:class:`DebatePipeline`) with either the real Ollama adapter or the offline
template providers (``--no-llm``), then hands the resulting artifact to the
report formatter. No valuation logic lives here — every number comes from the
pipeline, which gets them from ``domain/compute``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import TextIO

from alphalab.app.debate_pipeline import DebatePipeline
from alphalab.app.inputs import AnalystRequest
from alphalab.app.pipeline import AnalystPipeline
from alphalab.artifact.models import EquityResearchArtifact
from alphalab.artifact.store import InMemoryArtifactStore
from alphalab.cli.deterministic import (
    TemplateBearProvider,
    TemplateBullProvider,
    TemplateExtractionProvider,
    TemplateJudgeProvider,
    TemplateNarrationProvider,
)
from alphalab.cli.fixtures import FixtureNotFoundError, build_fixture_request
from alphalab.cli.ingest import LiveIngestUnavailableError, load_live_builder
from alphalab.domain.analyst.contracts import RunMetadata
from alphalab.report import format_artifact, theme_for


@dataclass(frozen=True)
class ResearchOptions:
    """Resolved options for one ``research`` invocation."""

    ticker: str
    knowledge_date: date
    source: str
    use_llm: bool
    debate: bool
    as_json: bool
    show_provenance: bool
    no_color: bool


def run_research(opts: ResearchOptions, *, out: TextIO, err: TextIO) -> int:
    """Execute the research command and return a process exit code."""
    try:
        artifact, request, source_label = build_research(opts)
    except FixtureNotFoundError as exc:
        print(f"error: {exc}", file=err)
        return 2
    except LiveIngestUnavailableError as exc:
        print(f"error: {exc}", file=err)
        return 1
    except Exception as exc:  # boundary: never leak a stack trace to the user
        report_pipeline_error(exc, use_llm=opts.use_llm, err=err)
        return 1

    if opts.as_json:
        print(_as_json(artifact, source_label, request.run_metadata), file=out)
        return 0

    theme = theme_for(out, force_disable=opts.no_color)
    print(
        format_artifact(
            artifact,
            theme=theme,
            run_metadata=request.run_metadata,
            data_source=source_label,
            show_provenance=opts.show_provenance,
        ),
        file=out,
    )
    return 0


def _build_request(opts: ResearchOptions) -> tuple[AnalystRequest, str]:
    """Build the request for the chosen source and return it with its label."""
    if opts.source == "live":
        builder = load_live_builder()
        request = builder(opts.ticker, knowledge_date=opts.knowledge_date)
        return request, "LIVE SEC EDGAR fundamentals"
    fixture = build_fixture_request(opts.ticker, knowledge_date=opts.knowledge_date)
    return fixture.request, fixture.source


def build_research(opts: ResearchOptions) -> tuple[EquityResearchArtifact, AnalystRequest, str]:
    """Run the research pipeline and return ``(artifact, request, source_label)``.

    Shared by ``research`` (which renders the artifact) and ``predict`` (which
    turns it into a dated forecast). Raising collaborators —
    :class:`FixtureNotFoundError`, :class:`LiveIngestUnavailableError`, or a
    pipeline failure — are left for the caller to translate into exit codes,
    so both commands keep a single, consistent error policy.
    """
    request, source_label = _build_request(opts)
    artifact = _run_pipeline(request, use_llm=opts.use_llm, debate=opts.debate)
    return artifact, request, source_label


def _run_pipeline(
    request: AnalystRequest, *, use_llm: bool, debate: bool
) -> EquityResearchArtifact:
    """Run the selected pipeline and return the gated artifact."""
    store = InMemoryArtifactStore()
    if debate:
        return _run_debate(request, store, use_llm=use_llm)
    return _run_analyst(request, store, use_llm=use_llm)


def _run_analyst(
    request: AnalystRequest, store: InMemoryArtifactStore, *, use_llm: bool
) -> EquityResearchArtifact:
    if use_llm:
        from alphalab.adapters.llm import OllamaLLMAdapter

        adapter = OllamaLLMAdapter()
        pipeline = AnalystPipeline(extractor=adapter, narrator=adapter, store=store)
    else:
        pipeline = AnalystPipeline(
            extractor=TemplateExtractionProvider(),
            narrator=TemplateNarrationProvider(),
            store=store,
        )
    return pipeline.run(request).artifact


def _run_debate(
    request: AnalystRequest, store: InMemoryArtifactStore, *, use_llm: bool
) -> EquityResearchArtifact:
    if use_llm:
        from alphalab.adapters.debate_llm import (
            OllamaDebateAdapter,
            bear_provider,
            bull_provider,
        )

        adapter = OllamaDebateAdapter()
        pipeline = DebatePipeline(
            bull=bull_provider(adapter),
            bear=bear_provider(adapter),
            judge=adapter,
            store=store,
        )
    else:
        pipeline = DebatePipeline(
            bull=TemplateBullProvider(),
            bear=TemplateBearProvider(),
            judge=TemplateJudgeProvider(),
            store=store,
        )
    return pipeline.run(request).artifact


def report_pipeline_error(exc: Exception, *, use_llm: bool, err: TextIO) -> None:
    """Print an actionable, stack-trace-free error message."""
    print(f"error: analysis failed: {type(exc).__name__}: {exc}", file=err)
    if use_llm:
        print(
            "hint: the LLM path needs Ollama at http://localhost:11434. "
            "Re-run with --no-llm for an offline template report.",
            file=err,
        )


def _as_json(
    artifact: EquityResearchArtifact,
    source_label: str,
    run_metadata: RunMetadata,
) -> str:
    """Serialise the artifact to machine-readable JSON."""
    target = artifact.price_target
    payload: dict[str, object] = {
        "ticker": artifact.ticker,
        "knowledge_date": (
            artifact.knowledge_date.isoformat() if artifact.knowledge_date else None
        ),
        "verdict": artifact.verdict.value,
        "data_source": source_label,
        "price_target": None if target is None else {"value": target.value, "unit": target.unit},
        "current_price": artifact.current_price,
        "price_range": (
            None
            if artifact.price_range is None
            else {
                "low": artifact.price_range.low,
                "mid": artifact.price_range.mid,
                "high": artifact.price_range.high,
            }
        ),
        "methods_used": list(artifact.methods_used),
        "method_values": dict(artifact.method_values),
        "consensus_relative_span": artifact.consensus_relative_span,
        "valuation_withheld": artifact.valuation_withheld,
        "contract_stamps": list(artifact.contract_stamps),
        "narrative": dict(artifact.narrative_slots),
        "run_metadata": {
            "model_name": run_metadata.model_name,
            "tier": run_metadata.tier.value,
            "llm_version": run_metadata.llm_version,
            "llm_cutoff": run_metadata.llm_cutoff.isoformat(),
            "knowledge_date": run_metadata.knowledge_date.isoformat(),
            "dataset_hash": run_metadata.dataset_hash,
        },
        "disclaimer": "Technical research output, NOT financial advice.",
    }
    return json.dumps(payload, indent=2, sort_keys=True)
