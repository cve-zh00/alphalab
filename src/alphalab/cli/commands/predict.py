"""The ``predict`` command: run research and record a *dated* forecast.

This is the user-facing face of the forward-evaluation engine
(:func:`alphalab.app.forecast_runner.register_forecast`). It runs the same
research pipeline as ``research`` (via :func:`build_research`), then hands the
gated artifact to the engine, which either records a scoreable
:class:`~alphalab.domain.journal.Prediction` in the journal or a stated
:class:`~alphalab.domain.forecast.Abstention` in the abstention store.

Three honest invariants surface here:

* An **abstention is not a failure.** A HOLD, or a verdict that contradicts its
  own valuation, is a deliberate refusal to make a scoreable claim — exit 0,
  with the reason shown. Hiding it would flatter the eventual hit rate.
* The **anti-leakage refusal** (``knowledge_date <= llm_cutoff``) is correct
  behaviour, not a bug: a claim the model could already have seen is hindsight,
  not a forecast, and recording it would irreversibly contaminate calibration.
  It is reported as an actionable usage error (exit 2), never a stack trace.
* A **ticker without usable data is skipped, not fatal.** Across a real universe
  many names have no standard EDGAR companyfacts, and one of them must not
  silence the rest of the batch. Skips are reported individually and counted at
  the end; only an error about the invocation itself (leakage) aborts the run,
  and a run where *nothing* was usable still exits non-zero.

No valuation logic lives here; every number comes from the pipeline, every
dated row from the engine. This module only orchestrates and renders.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TextIO

from alphalab.adapters.abstention_store import SqliteAbstentionStore
from alphalab.adapters.journal_sqlite import SqliteJournal
from alphalab.app.forecast_runner import LeakageError, register_forecast
from alphalab.artifact.models import EquityResearchArtifact, Verdict
from alphalab.cli.commands.research import ResearchOptions, build_research, report_pipeline_error
from alphalab.cli.fixtures import FixtureNotFoundError
from alphalab.cli.ingest import LiveIngestConfigError, LiveIngestUnavailableError
from alphalab.domain.forecast import Abstention
from alphalab.domain.journal import Prediction
from alphalab.report.theme import Theme, theme_for

#: Default forward horizon, in calendar days.
#:
#: Lowered from 90 to 21 for a measured reason. Calibration needs roughly 30
#: settled predictions before :func:`calibration_report` will call itself
#: conclusive. At a 90-day horizon that is years away; at 21 days — about a
#: trading month — the same evidence accrues more than four times faster.
#:
#: This is the only genuinely fast feedback loop available. Detecting whether a
#: *strategy* beats zero is bounded by elapsed time, not by effort: at the
#: out-of-sample Sharpe measured here (0.17 to 0.27) it would take over a
#: century. Calibration asks a different and far cheaper question — when this
#: system says 60%, is it right 60% of the time? — and that question is
#: answerable in months.
#:
#: Shorter still would be worse: under about three weeks, single-name moves are
#: dominated by noise and the resolution would measure luck rather than
#: judgement.
DEFAULT_HORIZON_DAYS = 21


@dataclass(frozen=True)
class PredictOptions:
    """Resolved options for one ``predict`` invocation."""

    tickers: tuple[str, ...]
    knowledge_date: date
    horizon_days: int
    source: str
    use_llm: bool
    debate: bool
    as_json: bool
    no_color: bool
    db_path: Path
    abstentions_path: Path
    llm_cutoff: date | None = None
    assume_verdict: Verdict | None = None


@dataclass(frozen=True)
class _Result:
    """One ticker's outcome: a registered prediction or a recorded abstention."""

    ticker: str
    prediction: Prediction | None
    abstention: Abstention | None


@dataclass(frozen=True)
class _Skipped:
    """A ticker whose data was unusable: already reported, batch continues.

    One name without usable fundamentals says nothing about the next one, so it
    must not silence the rest of the batch. The reason is kept so the run ends
    with a visible census of what was skipped and why — an omission the user can
    see is honest; one that vanishes is not.
    """

    ticker: str
    reason: str
    exit_code: int


@dataclass(frozen=True)
class _Abort:
    """A reported failure that invalidates the whole run, carrying its exit code.

    Reserved for errors about the *invocation* rather than about one ticker —
    an anti-leakage refusal, say, which would reject every name identically.
    Continuing would just reprint the same misconfiguration once per ticker.
    """

    exit_code: int


_VERDICTS: dict[str, Verdict] = {
    "buy": Verdict.BUY,
    "sell": Verdict.SELL,
    "hold": Verdict.HOLD,
}


def options_from_args(args: argparse.Namespace, *, no_color: bool) -> PredictOptions:
    """Translate parsed ``predict`` args into a resolved options object.

    Lives next to the options it builds so the CLI surface→options mapping is
    close to the dataclass, keeping the argparse wiring in
    :mod:`alphalab.cli.app` thin. ``--assume-verdict`` is honoured only on the
    offline path by :func:`run_predict`.
    """
    assume = _VERDICTS.get(args.assume_verdict) if args.assume_verdict else None
    return PredictOptions(
        tickers=tuple(t.upper() for t in args.ticker),
        knowledge_date=args.knowledge_date,
        horizon_days=args.horizon_days,
        source=args.source,
        use_llm=not args.no_llm,
        debate=args.debate,
        as_json=args.json,
        no_color=no_color,
        db_path=args.db,
        abstentions_path=args.abstentions_db,
        llm_cutoff=args.llm_cutoff,
        assume_verdict=assume,
    )


def run_predict(opts: PredictOptions, *, out: TextIO, err: TextIO) -> int:
    """Run research for each ticker and record a dated forecast or abstention."""
    opts.db_path.parent.mkdir(parents=True, exist_ok=True)
    opts.abstentions_path.parent.mkdir(parents=True, exist_ok=True)
    journal = SqliteJournal(opts.db_path)
    abstentions = SqliteAbstentionStore(opts.abstentions_path)
    results: list[_Result] = []
    skipped: list[_Skipped] = []
    try:
        for ticker in opts.tickers:
            outcome = _predict_one(ticker, opts, journal=journal, abstentions=abstentions, err=err)
            if isinstance(outcome, _Abort):
                return outcome.exit_code
            if isinstance(outcome, _Skipped):
                skipped.append(outcome)
                continue
            results.append(outcome)
    finally:
        journal.close()
        abstentions.close()

    if opts.as_json:
        print(_as_json(results, skipped, opts), file=out)
    else:
        theme = theme_for(out, force_disable=opts.no_color)
        print(_format(results, skipped, opts, theme), file=out)
    # A run that produced nothing at all is a failure, and reports the exit code
    # of the first thing that went wrong — so a single-ticker invocation keeps
    # exactly the exit code it had before batches became fault-tolerant.
    if not results and skipped:
        return skipped[0].exit_code
    return 0


def _predict_one(
    ticker: str,
    opts: PredictOptions,
    *,
    journal: SqliteJournal,
    abstentions: SqliteAbstentionStore,
    err: TextIO,
) -> _Result | _Skipped | _Abort:
    """Produce and persist one forecast, or an already-reported skip/abort."""
    research_opts = ResearchOptions(
        ticker=ticker,
        knowledge_date=opts.knowledge_date,
        source=opts.source,
        use_llm=opts.use_llm,
        debate=opts.debate,
        as_json=False,
        show_provenance=False,
        no_color=opts.no_color,
    )
    try:
        artifact = _artifact_for(research_opts, opts.assume_verdict)
    except FixtureNotFoundError as exc:
        print(f"error: {exc}", file=err)
        return _Skipped(ticker=ticker, reason=str(exc), exit_code=2)
    except LiveIngestConfigError as exc:
        print(f"error: {exc}", file=err)
        return _Abort(exit_code=2)
    except LiveIngestUnavailableError as exc:
        print(f"error: {exc}", file=err)
        return _Skipped(ticker=ticker, reason=str(exc), exit_code=1)
    except Exception as exc:  # boundary: never leak a stack trace
        report_pipeline_error(exc, use_llm=opts.use_llm, err=err)
        return _Skipped(ticker=ticker, reason=str(exc), exit_code=1)

    horizon = opts.knowledge_date + timedelta(days=opts.horizon_days)
    try:
        recorded = register_forecast(
            artifact,
            horizon=horizon,
            recorded_at=datetime.now(),
            journal=journal,
            abstentions=abstentions,
            llm_version="deterministic-template" if not opts.use_llm else None,
            llm_cutoff=opts.llm_cutoff,
        )
    except LeakageError as exc:
        print(f"error: refused to register {ticker}: {exc}", file=err)
        print(
            "hint: pick a --knowledge-date strictly after --llm-cutoff. A claim the "
            "model could already have seen is hindsight, not a forecast.",
            file=err,
        )
        return _Abort(exit_code=2)
    if isinstance(recorded, Abstention):
        return _Result(ticker=ticker, prediction=None, abstention=recorded)
    return _Result(ticker=ticker, prediction=recorded, abstention=None)


def _artifact_for(research_opts: ResearchOptions, assume: Verdict | None) -> EquityResearchArtifact:
    """Build the research artifact, optionally overriding the offline verdict.

    ``assume`` is honoured only on the deterministic (``--no-llm``) path, so the
    registration flow can be exercised without an LLM; it never overrides a real
    model's judgment.
    """
    artifact, _request, _source = build_research(research_opts)
    if assume is not None and not research_opts.use_llm:
        return artifact.model_copy(update={"verdict": assume})
    return artifact


def _format(
    results: list[_Result], skipped: list[_Skipped], opts: PredictOptions, theme: Theme
) -> str:
    lines = [theme.bold("FORECAST  (forward-dated, append-only)")]
    for result in results:
        lines.extend(_format_one(result, theme))
    for skip in skipped:
        badge = theme.paint("SKIPPED", "bold", "red")
        lines.append(f"  [{badge}] {skip.ticker}: {skip.reason}")
    if skipped:
        total = len(results) + len(skipped)
        lines.append(theme.dim(f"  {len(results)}/{total} usable; {len(skipped)} skipped."))
    lines.append(theme.dim(f"  journal: {opts.db_path}"))
    lines.append(theme.dim(f"  abstentions: {opts.abstentions_path}"))
    return "\n".join(lines)


def _format_one(result: _Result, theme: Theme) -> list[str]:
    if result.prediction is not None:
        pred = result.prediction
        badge = theme.paint("REGISTERED", "bold", "green")
        return [
            f"  [{badge}] {pred.ticker}: direction={pred.direction.value} "
            f"probability={pred.probability:.3f}",
            f"            settles {pred.horizon} (knowledge {pred.knowledge_date}).",
        ]
    abst = result.abstention
    assert abst is not None
    badge = theme.paint("ABSTAINED", "bold", "yellow")
    return [f"  [{badge}] {abst.ticker}: reason: {abst.reason}"]


def _as_json(results: list[_Result], skipped: list[_Skipped], opts: PredictOptions) -> str:
    rows = [_row(result) for result in results]
    rows.extend(
        {"status": "skipped", "ticker": skip.ticker, "reason": skip.reason} for skip in skipped
    )
    payload: dict[str, object] = {
        "journal": str(opts.db_path),
        "abstentions": str(opts.abstentions_path),
    }
    if len(rows) == 1:
        payload.update(rows[0])
    payload["results"] = rows
    return json.dumps(payload, indent=2, sort_keys=True)


def _row(result: _Result) -> dict[str, object]:
    if result.prediction is not None:
        pred = result.prediction
        return {
            "status": "registered",
            "ticker": pred.ticker,
            "direction": pred.direction.value,
            "probability": pred.probability,
            "horizon": pred.horizon.isoformat(),
            "knowledge_date": pred.knowledge_date.isoformat(),
        }
    abst = result.abstention
    assert abst is not None
    return {"status": "abstained", "ticker": abst.ticker, "reason": abst.reason}
