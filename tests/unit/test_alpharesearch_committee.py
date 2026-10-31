"""Unit and CLI integration tests for the AlphaResearch 3-role weekly committee."""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import numpy as np

from alphalab.adapters.alpharesearch_store import SQLiteAlphaResearchStore
from alphalab.app.alpharesearch_runner import synthesize_proposals_from_signals
from alphalab.cli.app import main
from alphalab.domain.alpharesearch import (
    AblationEvidence,
    DiscoverySignal,
    DiscoverySourceKind,
    ProposalDecisionStatus,
    ResearchProposal,
    TargetSubsystem,
    judge_proposal,
    run_skeptic_review,
)


def _sample_signal() -> DiscoverySignal:
    return DiscoverySignal(
        signal_id="sig-1",
        source_kind=DiscoverySourceKind.ACADEMIC_PAPER,
        source_name="arXiv q-fin",
        title="Sloan Accrual Quality Filter in PIT Equity Valuation",
        reference_uri="https://arxiv.org/abs/2501.00001",
        observed_at=date(2026, 10, 31),
        watermark_token="2501.00001",
        summary="Penalizes extreme accrual ratios in point-in-time valuation.",
        quantitative_mechanism="AccrualRatio = (NI - CFO) / Assets",
        suggested_subsystem=TargetSubsystem.PRESCREEN,
    )


def test_skeptic_reviewer_vetoes_all_six_failure_modes_without_short_circuit() -> None:
    bad_proposal = ResearchProposal.create(
        title="Unbounded LLM Intraday Target with Future Close",
        subsystem=TargetSubsystem.VALUATION,
        source_signals=(_sample_signal(),),
        problem_statement="Test all failure modes simultaneously.",
        mathematical_formulation="Uses future_close and bypass pit to weight assets.",
        falsifiable_hypothesis="In-sample Sharpe increases.",
        affected_files=("src/alphalab/domain/compute/dcf.py",),
        implementation_todos=("Add I/O call inside domain",),
        pit_safe=False,
        domain_pure=False,
        llm_numeric_boundary_safe=False,
        preserves_nofx_guards=False,
        estimated_loc_delta=400,
        expected_sharpe_gain=0.01,
        ablation_evidence=AblationEvidence(turnover_delta_pct=50.0),
    )
    critique = run_skeptic_review(
        bad_proposal,
        rejected_proposal_ids=frozenset({bad_proposal.proposal_id}),
    )
    assert not critique.accepted
    assert len(critique.findings) == 6
    assert len(critique.veto_reasons) == 6
    assert critique.risk_score == 1.0


def test_empirical_gatekeeper_promotes_strong_challenger_and_rejects_noise() -> None:
    proposals = synthesize_proposals_from_signals((_sample_signal(),), seed=42)
    assert len(proposals) == 4

    rng = np.random.default_rng(42)
    # Proposal 0: Sloan accrual quality filter -> strong positive alpha -> PROMOTED_TO_PLAN
    d0 = judge_proposal(proposals[0], prior_n_trials=0, rng=rng.spawn(1)[0])
    assert d0.status is ProposalDecisionStatus.PROMOTED_TO_PLAN
    assert d0.promoted
    assert d0.empirical_scorecard.dsr is not None and d0.empirical_scorecard.dsr > 0.95
    assert (
        d0.empirical_scorecard.sharpe_delta is not None
        and d0.empirical_scorecard.sharpe_delta > 0.0
    )
    assert d0.empirical_scorecard.n_trials_total == 1

    # Proposal 2: Free LLM price target -> vetoed by Role 2 (SkepticReviewer)
    d2 = judge_proposal(proposals[2], prior_n_trials=1, rng=rng.spawn(1)[0])
    assert d2.status is ProposalDecisionStatus.REJECTED_BY_SKEPTIC
    assert not d2.promoted

    # Proposal 3: 3-day reversal with negative net alpha -> vetoed by Role 3 (EmpiricalGatekeeper)
    d3 = judge_proposal(proposals[3], prior_n_trials=2, rng=rng.spawn(1)[0])
    assert d3.status is ProposalDecisionStatus.REJECTED_BY_EMPIRICAL_GATE
    assert not d3.promoted
    assert d3.empirical_scorecard.n_trials_total == 3


def test_empirical_gatekeeper_holds_proposal_without_evidence() -> None:
    prop = ResearchProposal.create(
        title="Purely Conceptual Idea Without Backtest or Invariants",
        subsystem=TargetSubsystem.CALIBRATION,
        source_signals=(_sample_signal(),),
        problem_statement="Need better calibration.",
        mathematical_formulation="p_cal = p_raw",
        falsifiable_hypothesis="Improves Brier score.",
        affected_files=("src/alphalab/domain/journal/calibration.py",),
        implementation_todos=("Investigate calibration.",),
        ablation_evidence=AblationEvidence(),
    )
    decision = judge_proposal(prop, prior_n_trials=5, rng=np.random.default_rng(7))
    assert decision.status is ProposalDecisionStatus.HOLD_FOR_EVIDENCE
    assert decision.empirical_scorecard.n_trials_total == 6


def test_runner_and_cli_end_to_end_persists_plan_and_graveyard(tmp_path: Path) -> None:
    db_path = tmp_path / "alpharesearch.db"
    plans_dir = tmp_path / "plans"
    out_plan = tmp_path / "custom-plan.md"
    exec_db = tmp_path / "execution.db"

    # Create a minimal execution.db so LivePortfolioPostmortemScanner emits a signal
    import sqlite3

    with sqlite3.connect(exec_db) as conn:
        conn.execute(
            "CREATE TABLE position_events "
            "(id INTEGER PRIMARY KEY, event_type TEXT, occurred_at TEXT)"
        )
        conn.execute(
            "INSERT INTO position_events (event_type, occurred_at) "
            "VALUES ('DRAWDOWN_EXIT', '2026-10-30T14:33:00Z')"
        )
        conn.execute("CREATE TABLE cycle_runs (id INTEGER PRIMARY KEY, started_at TEXT)")
        conn.execute("INSERT INTO cycle_runs (started_at) VALUES ('2026-10-30T14:33:00Z')")

    out = io.StringIO()
    err = io.StringIO()
    rc = main(
        [
            "alpharesearch",
            "run",
            "--as-of",
            "2026-10-31",
            "--sources",
            "postmortem",
            "--db",
            str(db_path),
            "--execution-db",
            str(exec_db),
            "--plans-dir",
            str(plans_dir),
            "--out",
            str(out_plan),
        ],
        out=out,
        err=err,
    )
    assert rc == 0, err.getvalue()
    assert out_plan.is_file()
    md_text = out_plan.read_text(encoding="utf-8")
    assert "## Plan de Implementación (Todos Promovidos)" in md_text
    assert "## Apéndice de Discernimiento: Propuestas Vetadas (Graveyard)" in md_text
    assert "rejected_by_skeptic" in md_text
    assert "rejected_by_empirical_gate" in md_text

    # Verify history subcommand
    hist_out = io.StringIO()
    rc_hist = main(
        ["alpharesearch", "history", "--db", str(db_path)],
        out=hist_out,
        err=err,
    )
    assert rc_hist == 0
    assert "2026-10-31" in hist_out.getvalue()

    # Verify Graveyard anti-snooping memory in SQLite store
    store = SQLiteAlphaResearchStore(db_path)
    assert len(store.rejected_proposal_ids()) == 2
    assert store.total_trials_count() == 4


def test_source_scanners_local_remote_and_academic_with_offline_resilience(
    tmp_path: Path, monkeypatch: object
) -> None:
    from alphalab.adapters.alpharesearch_sources import (
        AcademicPaperScanner,
        LocalReferenceRepoScanner,
        RemoteGitHubRepoScanner,
    )

    # 1. LocalReferenceRepoScanner on the workspace reference directory
    local_scanner = LocalReferenceRepoScanner(Path("reference"))
    local_signals = local_scanner.scan(as_of=date(2026, 10, 31), watermarks={})
    assert len(local_signals) >= 1
    assert local_signals[0].source_kind is DiscoverySourceKind.LOCAL_REPO

    # 2. RemoteGitHubRepoScanner with mocked HTTP response and fallback
    class _FakeResp:
        def __init__(self, status_code: int, payload: object, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self) -> object:
            return self._payload

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> _FakeClient:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def get(self, url: str, **kwargs: object) -> _FakeResp:
            if "api.github.com" in url:
                return _FakeResp(
                    200,
                    [
                        {
                            "sha": "abc1234567890",
                            "commit": {
                                "message": "feat: tighten drawdown guard\nDetails",
                                "committer": {"date": "2026-10-30T12:00:00Z"},
                            },
                        }
                    ],
                )
            atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
              <entry>
                <id>http://arxiv.org/abs/2610.99999v1</id>
                <published>2026-10-29T00:00:00Z</published>
                <title>Deflated Sharpe Ratio and Portfolio Shrinkage</title>
                <summary>Empirical Bayes shrinkage with transaction cost bands.</summary>
              </entry>
            </feed>"""
            return _FakeResp(200, {}, text=atom_xml)

    import httpx

    monkeypatch.setattr(httpx, "Client", _FakeClient)  # type: ignore[attr-defined]

    remote_scanner = RemoteGitHubRepoScanner(repos=("NoFxAiOS/nofx",))
    remote_signals = remote_scanner.scan(as_of=date(2026, 10, 31), watermarks={})
    assert len(remote_signals) == 1
    assert remote_signals[0].source_kind is DiscoverySourceKind.REMOTE_REPO

    paper_scanner = AcademicPaperScanner()
    paper_signals = paper_scanner.scan(as_of=date(2026, 10, 31), watermarks={})
    assert len(paper_signals) >= 1
    assert paper_signals[0].source_kind is DiscoverySourceKind.ACADEMIC_PAPER

