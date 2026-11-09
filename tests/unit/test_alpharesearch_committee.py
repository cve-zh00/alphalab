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

    # 1. LocalReferenceRepoScanner on a temporary reference directory with a git repo
    ref_dir = tmp_path / "reference"
    repo_dir = ref_dir / "finrobot"
    repo_dir.mkdir(parents=True)
    import subprocess

    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    (repo_dir / "README.md").write_text("# FinRobot\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "feat: add multi-agent valuation"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    local_scanner = LocalReferenceRepoScanner(ref_dir)
    local_signals = local_scanner.scan(as_of=date(2026, 10, 31), watermarks={})
    assert len(local_signals) >= 1
    assert local_signals[0].source_kind is DiscoverySourceKind.LOCAL_REPO

    # 2. RemoteGitHubRepoScanner & AcademicPaperScanner with mocked httpx.get
    class _FakeResp:
        def __init__(self, status_code: int, payload: object, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self) -> object:
            return self._payload

    def _fake_get(url: str, **kwargs: object) -> _FakeResp:
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

    monkeypatch.setattr(httpx, "get", _fake_get)  # type: ignore[attr-defined]

    remote_scanner = RemoteGitHubRepoScanner(repos=("NoFxAiOS/nofx",))
    remote_signals = remote_scanner.scan(as_of=date(2026, 10, 31), watermarks={})
    assert len(remote_signals) == 1
    assert remote_signals[0].source_kind is DiscoverySourceKind.REMOTE_REPO

    paper_scanner = AcademicPaperScanner()
    paper_signals = paper_scanner.scan(as_of=date(2026, 10, 31), watermarks={})
    assert len(paper_signals) >= 1
    assert paper_signals[0].source_kind is DiscoverySourceKind.ACADEMIC_PAPER


def test_remote_scanner_skips_non_quant_commits_like_telegram_links(
    monkeypatch: object,
) -> None:
    import httpx

    from alphalab.adapters.alpharesearch_sources import RemoteGitHubRepoScanner

    class _FakeResp:
        def __init__(self, status_code: int, payload: object) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> object:
            return self._payload

    def _fake_get(url: str, **kwargs: object) -> _FakeResp:
        assert "per_page=10" in url
        return _FakeResp(
            200,
            [
                {
                    "sha": "65e9037000000000",
                    "commit": {
                        "message": "fix: restore Telegram community links without bot integration"
                    },
                },
                {
                    "sha": "a949bf9111111111",
                    "commit": {
                        "message": "refactor: tighten drawdown and stop-loss risk guard parameters"
                    },
                },
            ],
        )

    monkeypatch.setattr(httpx, "get", _fake_get)  # type: ignore[attr-defined]
    scanner = RemoteGitHubRepoScanner(repos=("NoFxAiOS/nofx",))
    signals = scanner.scan(as_of=date(2026, 11, 6), watermarks={})
    assert len(signals) == 1
    assert signals[0].watermark_token == "a949bf9111111111"
    assert "Telegram" not in signals[0].title


def test_postmortem_scanner_reads_execution_events_and_settlements_and_panel_enforces_pit(
    tmp_path: Path,
) -> None:
    import sqlite3

    import pyarrow as pa
    import pyarrow.parquet as pq

    from alphalab.adapters.alpharesearch_sources import (
        LivePortfolioPostmortemScanner,
        load_empirical_ablation_panel,
    )

    exec_db = tmp_path / "execution.db"
    journal_db = tmp_path / "journal.db"
    pit_dir = tmp_path / "pit"
    (pit_dir / "AAPL").mkdir(parents=True)
    (pit_dir / "TRAP").mkdir(parents=True)

    with sqlite3.connect(exec_db) as conn:
        conn.execute(
            "CREATE TABLE execution_events ("
            "event_id TEXT PRIMARY KEY, cycle_id TEXT, recorded_at TEXT, "
            "kind TEXT, ticker TEXT, side TEXT, shares REAL, reference_price REAL, "
            "client_order_id TEXT, broker_order_id TEXT, detail TEXT)"
        )
        conn.execute(
            "INSERT INTO execution_events VALUES "
            "('e1', 'c1', '2026-10-20T14:00:00Z', 'filled', 'AAPL', 'BUY', 10, 200.0, "
            "'co1', 'bo1', 'filled')"
        )
        conn.execute(
            "INSERT INTO execution_events VALUES "
            "('e2', 'c2', '2026-10-25T14:00:00Z', 'rejected', 'TRAP', 'SELL', 10, 90.0, "
            "'co2', 'bo2', 'stop_loss_exit triggered')"
        )
        conn.execute(
            "INSERT INTO execution_events VALUES "
            "('e3', 'c3', '2026-10-28T14:00:00Z', 'rejected', 'MSFT', 'SELL', 5, 400.0, "
            "'co3', 'bo3', 'drawdown_exit triggered')"
        )

    with sqlite3.connect(journal_db) as conn:
        conn.execute(
            "CREATE TABLE predictions ("
            "prediction_id TEXT PRIMARY KEY, recorded_at TEXT, ticker TEXT, "
            "direction TEXT, probability REAL, horizon INTEGER, entry_price REAL, "
            "knowledge_date TEXT, llm_version TEXT, llm_cutoff TEXT)"
        )
        conn.execute(
            "CREATE TABLE settlements ("
            "prediction_id TEXT PRIMARY KEY, settled_at TEXT, realized_return REAL, "
            "benchmark_return REAL, alpha_gross REAL, alpha_net REAL, outcome INTEGER)"
        )
        for i in range(180):
            is_good = (i % 3) != 0
            tkr = "AAPL" if is_good else "TRAP"
            pid = f"pred-{i:03d}"
            prob = 0.68 if is_good else 0.72
            alpha_g = 0.032 if is_good else -0.028
            alpha_n = 0.030 if is_good else -0.031
            outcome = 1 if is_good else 0
            conn.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, 'LONG', ?, 21, 100.0, '2026-10-15', "
                "'v1', '2026-01-01')",
                (pid, f"2026-10-15T10:{i % 60:02d}:00Z", tkr, prob),
            )
            conn.execute(
                "INSERT INTO settlements VALUES (?, '2026-11-05T16:00:00Z', ?, 0.01, ?, ?, ?)",
                (pid, alpha_g + 0.01, alpha_g, alpha_n, outcome),
            )

    # Write PIT facts for AAPL (strong cash flow, negative accruals) and
    # TRAP (high accruals + post-knowledge_date leaked restatement that must be ignored)
    aapl_tbl = pa.table(
        {
            "field": [
                "assets",
                "net_income",
                "operating_cash_flow",
                "capex",
                "stockholders_equity",
                "operating_income",
                "long_term_debt",
                "cash",
            ],
            "value": [1000.0, 120.0, 165.0, 20.0, 400.0, 150.0, 80.0, 120.0],
            "as_of": ["2026-09-30"] * 8,
            "observed_at": ["2026-10-01"] * 8,
        }
    )
    pq.write_table(aapl_tbl, pit_dir / "AAPL" / "facts.parquet")

    trap_tbl = pa.table(
        {
            "field": [
                "assets",
                "net_income",
                "operating_cash_flow",
                "capex",
                "stockholders_equity",
                "operating_income",
                "long_term_debt",
                "cash",
                # Future filing observed AFTER 2026-10-15 (must be ignored by PIT filter)
                "operating_cash_flow",
            ],
            "value": [1000.0, 110.0, 5.0, 40.0, 300.0, 15.0, 450.0, 20.0, 900.0],
            "as_of": ["2026-09-30"] * 8 + ["2026-10-20"],
            "observed_at": ["2026-10-01"] * 8 + ["2026-10-25"],
        }
    )
    pq.write_table(trap_tbl, pit_dir / "TRAP" / "facts.parquet")

    scanner = LivePortfolioPostmortemScanner(exec_db, journal_db)
    sigs = scanner.scan(as_of=date(2026, 11, 6), watermarks={})
    assert len(sigs) == 2
    assert "3 cycles" in sigs[0].title
    assert "1 stop-loss" in sigs[0].title
    assert "180 settled" in sigs[1].title

    panel = load_empirical_ablation_panel(
        journal_db_path=journal_db,
        execution_db_path=exec_db,
        pit_dir=pit_dir,
        as_of=date(2026, 11, 6),
        cohort_size=6,
    )
    assert len(panel.samples) == 180
    assert panel.n_cohorts == 30
    trap_sample = next(s for s in panel.samples if s.ticker == "TRAP")
    # PIT check: operating_cash_flow=900 observed on 2026-10-25 (> 2026-10-15) must NOT leak
    assert abs(trap_sample.accrual_ratio - (110.0 - 5.0) / 1000.0) < 1e-9
    assert trap_sample.had_protective_stop_exit is True

    proposals = synthesize_proposals_from_signals(sigs, seed=42, ablation_panel=panel)
    assert len(proposals) == 4
    d0 = judge_proposal(proposals[0], prior_n_trials=0, rng=np.random.default_rng(42))
    assert d0.status is ProposalDecisionStatus.PROMOTED_TO_PLAN

    # Verify that on a subsequent cycle with excluded_proposal_ids set to the first 4 IDs,
    # synthesize_proposals_from_signals does not re-emit the 4 canonical proposals and instead
    # synthesizes the empirical postmortem proposals.
    first_cycle_ids = frozenset(p.proposal_id for p in proposals)
    next_proposals = synthesize_proposals_from_signals(
        sigs,
        seed=42,
        ablation_panel=panel,
        excluded_proposal_ids=first_cycle_ids,
    )
    assert len(next_proposals) == 2
    assert all(p.proposal_id not in first_cycle_ids for p in next_proposals)


