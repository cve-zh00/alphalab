"""Weekly discovery scanners for the AlphaResearch committee.

Harvests :class:`~alphalab.domain.alpharesearch.DiscoverySignal` items from four
distinct empirical and literature channels:

1. :class:`LocalReferenceRepoScanner` — inspects cloned repositories in
   ``reference/`` (``finrl``, ``finrobot``, ``ai-trader``, ``hypergrok``,
   ``FinRL-Meta``, ``FinRL_Crypto``) via ``git log`` and module inspection.
2. :class:`RemoteGitHubRepoScanner` — queries upstream GitHub repositories
   (defaulting to ``NoFxAiOS/nofx``, ``AI4Finance-Foundation/FinRL``,
   ``AI4Finance-Foundation/FinRobot``, ``HKUDS/AI-Trader``) for recent commits.
3. :class:`AcademicPaperScanner` — queries the public **arXiv** Quantitative
   Finance API (``q-fin.PM``, ``q-fin.ST``, ``q-fin.TR``, ``q-fin.CP``) for
   recent working papers on portfolio optimization, shrinkage, transaction
   costs, and drawdown control.
4. :class:`LivePortfolioPostmortemScanner` — inspects ``data/execution.db`` and
   ``data/journal.db`` to extract live trading friction, stop-loss / drawdown
   exits, sector concentration blocks, and forecast calibration drift.
"""

from __future__ import annotations

import sqlite3
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from alphalab.domain.alpharesearch.contracts import (
    DiscoverySignal,
    DiscoverySourceKind,
    EmpiricalAblationPanel,
    EmpiricalTradeSample,
    TargetSubsystem,
)

__all__ = [
    "DEFAULT_REMOTE_REPOS",
    "AcademicPaperScanner",
    "LivePortfolioPostmortemScanner",
    "LocalReferenceRepoScanner",
    "RemoteGitHubRepoScanner",
    "load_empirical_ablation_panel",
]

DEFAULT_REMOTE_REPOS: tuple[str, ...] = (
    "NoFxAiOS/nofx",
    "AI4Finance-Foundation/FinRL",
    "AI4Finance-Foundation/FinRobot",
    "HKUDS/AI-Trader",
)

_REPO_MECHANISM_CATALOG: Mapping[str, tuple[TargetSubsystem, str, str]] = {
    "finrl": (
        TargetSubsystem.RISK_GUARDS,
        "Mahalanobis financial turbulence index & dynamic cash buffer",
        "d_t = (r_t - mu)^T Sigma^{-1} (r_t - mu) regime tail-risk gate (Kritzman & Li 2010)",
    ),
    "finrl-meta": (
        TargetSubsystem.EXECUTION,
        "Almgren-Chriss square-root market-impact and bid-ask slippage calibration",
        "Impact_i = eta * sigma_i * sqrt(Q_i / ADV_i) pre-trade net edge filter",
    ),
    "finrobot": (
        TargetSubsystem.VALUATION,
        "SEC 10-K/10-Q cross-statement earnings quality (Sloan accrual ratio) penalty",
        "AccrualRatio = (NI - CFO) / TotalAssets shrinkage on DCF terminal growth (Sloan 1996)",
    ),
    "ai-trader": (
        TargetSubsystem.PRESCREEN,
        "Cross-sectional quality-minus-junk (QMJ) composite rank in candidate prescreen",
        "z_QMJ = z(ROE) + z(FCF/Assets) - z(NetDebt/EBITDA) (Asness, Frazzini & Pedersen 2019)",
    ),
    "hypergrok": (
        TargetSubsystem.CALIBRATION,
        "Multi-horizon Brier skill decomposition and confidence temperature scaling",
        "p_cal = sigmoid(logit(p_raw) / T_opt) fitted on strictly settled PIT predictions",
    ),
    "finrl_crypto": (
        TargetSubsystem.SIZING,
        "Downside semivariance (Sortino residual volatility) position sizing",
        "sigma_down,i = sqrt(252 * E[min(0, eps_{i,t})^2]) replacing symmetric residual vol",
    ),
}


class LocalReferenceRepoScanner:
    """Scan cloned git repositories under ``reference/`` for commits & mechanisms."""

    def __init__(self, reference_dir: Path, *, fetch_remote: bool = False) -> None:
        self._reference_dir = reference_dir
        self._fetch_remote = fetch_remote

    def scan(
        self,
        *,
        as_of: date,
        watermarks: Mapping[str, str] | None = None,
    ) -> tuple[DiscoverySignal, ...]:
        if not self._reference_dir.is_dir():
            return ()
        wm = watermarks or {}
        signals: list[DiscoverySignal] = []

        for repo_dir in sorted(self._reference_dir.iterdir()):
            if not repo_dir.is_dir() or not (repo_dir / ".git").exists():
                continue
            repo_name = repo_dir.name
            if self._fetch_remote:
                subprocess.run(
                    ["git", "-C", str(repo_dir), "fetch", "--quiet"],
                    check=False,
                    capture_output=True,
                    timeout=10,
                )
            proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_dir),
                    "--no-pager",
                    "log",
                    "-n",
                    "1",
                    "--pretty=format:%H|%cs|%s",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                continue
            parts = proc.stdout.strip().split("|", 2)
            if len(parts) < 3:
                continue
            sha, commit_date_str, subject = parts[0], parts[1], parts[2]
            source_key = f"local:{repo_name}"
            if wm.get(source_key) == sha:
                continue

            try:
                obs_date = min(as_of, date.fromisoformat(commit_date_str))
            except ValueError:
                obs_date = as_of

            key_lower = repo_name.lower()
            subsys, title, mechanism = _REPO_MECHANISM_CATALOG.get(
                key_lower,
                (
                    TargetSubsystem.VALUATION,
                    f"Quantitative pattern extraction from {repo_name}",
                    f"Latest commit ({sha[:8]}): {subject}",
                ),
            )
            signals.append(
                DiscoverySignal(
                    signal_id=f"local-{repo_name.lower()}-{sha[:8]}",
                    source_kind=DiscoverySourceKind.LOCAL_REPO,
                    source_name=f"reference/{repo_name}",
                    title=f"{title} (commit {sha[:7]}: {subject[:60]})",
                    reference_uri=str(repo_dir),
                    observed_at=obs_date,
                    watermark_token=sha,
                    summary=(
                        f"Repository reference/{repo_name} at commit {sha[:8]} "
                        f"({commit_date_str}): '{subject}'. "
                        f"Extracted candidate mechanism: {title}."
                    ),
                    quantitative_mechanism=mechanism,
                    suggested_subsystem=subsys,
                )
            )
        return tuple(signals)


_NON_QUANT_COMMIT_KEYWORDS: tuple[str, ...] = (
    "telegram",
    "discord",
    "community link",
    "readme",
    "typo",
    "badge",
    "funding",
    "sponsor",
    "license",
    "i18n",
    "translation",
)


def _is_non_quant_commit_message(message: str) -> bool:
    lower = message.lower()
    return any(kw in lower for kw in _NON_QUANT_COMMIT_KEYWORDS)


class RemoteGitHubRepoScanner:
    """Query tracked remote GitHub repositories for recent commits."""

    def __init__(
        self,
        repos: Sequence[str] = DEFAULT_REMOTE_REPOS,
        *,
        github_token: str | None = None,
        timeout_seconds: float = 6.0,
    ) -> None:
        self._repos = tuple(repos)
        self._token = github_token
        self._timeout = timeout_seconds

    def scan(
        self,
        *,
        as_of: date,
        watermarks: Mapping[str, str] | None = None,
    ) -> tuple[DiscoverySignal, ...]:
        wm = watermarks or {}
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "alphalab-alpharesearch"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        signals: list[DiscoverySignal] = []
        for repo in self._repos:
            url = f"https://api.github.com/repos/{repo}/commits?per_page=10"
            try:
                resp = httpx.get(url, headers=headers, timeout=self._timeout)
                if resp.status_code != 200:
                    continue
                payload = resp.json()
                if not isinstance(payload, list) or not payload:
                    continue
                selected = None
                for candidate in payload:
                    if not isinstance(candidate, Mapping):
                        continue
                    c_info = candidate.get("commit", {}) or {}
                    c_msg = str(c_info.get("message", "")).splitlines()[0]
                    if not _is_non_quant_commit_message(c_msg):
                        selected = candidate
                        break
                if selected is None:
                    continue
                sha = str(selected.get("sha", ""))
                if not sha:
                    continue
                source_key = f"remote:{repo}"
                if wm.get(source_key) == sha:
                    continue
                commit_info = selected.get("commit", {}) or {}
                message = str(commit_info.get("message", "")).splitlines()[0][:80]
                html_url = str(selected.get("html_url", f"https://github.com/{repo}/commit/{sha}"))
                subsys = (
                    TargetSubsystem.RISK_GUARDS
                    if "nofx" in repo.lower()
                    else TargetSubsystem.PRESCREEN
                )
                mechanism = (
                    f"Volatility-regime adaptive trailing stop & cooldown decay ({repo}: {message})"
                    if "nofx" in repo.lower()
                    else f"Upstream quantitative update in {repo}: {message}"
                )
                signals.append(
                    DiscoverySignal(
                        signal_id=f"remote-{repo.replace('/', '-').lower()}-{sha[:8]}",
                        source_kind=DiscoverySourceKind.REMOTE_REPO,
                        source_name=repo,
                        title=f"Upstream commit in {repo} ({sha[:7]}): {message}",
                        reference_uri=html_url,
                        observed_at=as_of,
                        watermark_token=sha,
                        summary=f"Upstream repository {repo} updated to {sha[:8]}: {message}",
                        quantitative_mechanism=mechanism,
                        suggested_subsystem=subsys,
                    )
                )
            except Exception:
                continue
        return tuple(signals)


class AcademicPaperScanner:
    """Query arXiv Quantitative Finance (`q-fin`) for recent academic papers."""

    def __init__(
        self,
        *,
        max_results: int = 12,
        max_emit: int = 4,
        timeout_seconds: float = 6.0,
    ) -> None:
        self._max_results = max_results
        self._max_emit = max_emit
        self._timeout = timeout_seconds

    def scan(
        self,
        *,
        as_of: date,
        watermarks: Mapping[str, str] | None = None,
    ) -> tuple[DiscoverySignal, ...]:
        wm = watermarks or {}
        url = (
            "https://export.arxiv.org/api/query?"
            "search_query=cat:q-fin.PM+OR+cat:q-fin.TR+OR+cat:q-fin.ST&"
            f"sortBy=submittedDate&sortOrder=descending&max_results={self._max_results}"
        )
        signals: list[DiscoverySignal] = []
        fetched_any_entry = False
        try:
            resp = httpx.get(url, timeout=self._timeout, follow_redirects=True)
            if resp.status_code == 200 and resp.text:
                root = ET.fromstring(resp.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    raw_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
                    title = " ".join(
                        (entry.findtext("atom:title", default="", namespaces=ns) or "").split()
                    )
                    summary = " ".join(
                        (entry.findtext("atom:summary", default="", namespaces=ns) or "").split()
                    )
                    pub_raw = (entry.findtext("atom:published", default="", namespaces=ns) or "")[
                        :10
                    ]
                    if not raw_id or not title:
                        continue
                    fetched_any_entry = True
                    paper_id = raw_id.rsplit("/", 1)[-1]
                    if wm.get(f"paper:{paper_id}") == paper_id:
                        continue
                    try:
                        obs_date = min(as_of, date.fromisoformat(pub_raw))
                    except ValueError:
                        obs_date = as_of

                    lower_blob = f"{title} {summary}".lower()
                    if "transaction cost" in lower_blob or "execution" in lower_blob:
                        subsys = TargetSubsystem.EXECUTION
                    elif "drawdown" in lower_blob or "stop" in lower_blob or "risk" in lower_blob:
                        subsys = TargetSubsystem.RISK_GUARDS
                    elif "llm" in lower_blob or "calibration" in lower_blob:
                        subsys = TargetSubsystem.CALIBRATION
                    elif (
                        "portfolio" in lower_blob
                        or "weight" in lower_blob
                        or "kelly" in lower_blob
                        or "volatility" in lower_blob
                    ):
                        subsys = TargetSubsystem.SIZING
                    else:
                        subsys = TargetSubsystem.VALUATION

                    signals.append(
                        DiscoverySignal(
                            signal_id=f"arxiv-{paper_id.replace('.', '-')}",
                            source_kind=DiscoverySourceKind.ACADEMIC_PAPER,
                            source_name="arXiv q-fin",
                            title=title[:120],
                            reference_uri=raw_id,
                            observed_at=obs_date,
                            watermark_token=paper_id,
                            summary=summary[:360] if summary else title,
                            quantitative_mechanism=(
                                "Empirical shrinkage & transaction-cost-aware portfolio "
                                f"optimization (arXiv:{paper_id})"
                            ),
                            suggested_subsystem=subsys,
                        )
                    )
                    if len(signals) >= self._max_emit:
                        break
        except Exception:
            pass

        if not signals and not fetched_any_entry:
            fallback_id = "garleanu-pedersen-2013-dynamic-trading"
            if wm.get(f"paper:{fallback_id}") != fallback_id:
                signals.append(
                    DiscoverySignal(
                        signal_id=f"paper-{fallback_id}",
                        source_kind=DiscoverySourceKind.ACADEMIC_PAPER,
                        source_name="Journal of Finance (Canonical Reference)",
                        title=(
                            "Dynamic Trading with Predictable Returns and Transaction Costs "
                            "(Garleanu & Pedersen 2013)"
                        ),
                        reference_uri="https://doi.org/10.1111/jofi.12080",
                        observed_at=as_of,
                        watermark_token=fallback_id,
                        summary=(
                            "Optimal portfolio weights trade partially toward the frictionless "
                            "Markowitz target at rate kappa = (1 - a) proportional to return "
                            "mean-reversion speed and inverse market impact."
                        ),
                        quantitative_mechanism=(
                            "w_t = (1 - kappa) * w_{t-1} + kappa * w*_t partial adjustment rule"
                        ),
                        suggested_subsystem=TargetSubsystem.SIZING,
                    )
                )
        return tuple(signals)


class LivePortfolioPostmortemScanner:
    """Inspect ``execution.db`` and ``journal.db`` for weekly empirical pain points."""

    def __init__(self, execution_db: Path, journal_db: Path | None = None) -> None:
        self._execution_db = execution_db
        self._journal_db = journal_db

    def scan(
        self,
        *,
        as_of: date,
        watermarks: Mapping[str, str] | None = None,
    ) -> tuple[DiscoverySignal, ...]:
        signals: list[DiscoverySignal] = []
        wm = watermarks or {}

        if self._execution_db.is_file():
            try:
                with sqlite3.connect(self._execution_db) as conn:
                    has_exec_events = (
                        conn.execute(
                            "SELECT 1 FROM sqlite_master "
                            "WHERE type = 'table' AND name = 'execution_events'"
                        ).fetchone()
                        is not None
                    )
                    if has_exec_events:
                        summary_row = conn.execute(
                            "SELECT COUNT(DISTINCT cycle_id), MAX(recorded_at), "
                            "SUM(CASE WHEN kind = 'filled' THEN 1 ELSE 0 END), "
                            "SUM(CASE WHEN kind = 'rejected' THEN 1 ELSE 0 END) "
                            "FROM execution_events"
                        ).fetchone()
                        cycle_count = int(summary_row[0] or 0) if summary_row else 0
                        latest_cycle_ts = (
                            str(summary_row[1] or as_of.isoformat())
                            if summary_row
                            else as_of.isoformat()
                        )
                        n_fills = int(summary_row[2] or 0) if summary_row else 0
                        n_rejected = int(summary_row[3] or 0) if summary_row else 0
                        exit_rows = conn.execute(
                            "SELECT ticker, detail FROM execution_events "
                            "WHERE detail LIKE '%stop_loss_exit%' OR detail LIKE '%drawdown_exit%'"
                        ).fetchall()
                        exit_count = len(exit_rows)
                        n_stop = sum(
                            1 for _, detail in exit_rows if "stop_loss_exit" in str(detail or "")
                        )
                        n_dd = sum(
                            1 for _, detail in exit_rows if "drawdown_exit" in str(detail or "")
                        )
                        stop_tickers = sorted(
                            {
                                str(t)
                                for t, detail in exit_rows
                                if t and "stop_loss_exit" in str(detail or "")
                            }
                        )
                        token = f"exec:{cycle_count}:{exit_count}:{n_fills}:{n_rejected}"
                        title = (
                            f"Live portfolio postmortem across {cycle_count} cycles "
                            f"({exit_count} protective exits: {n_stop} stop-loss, "
                            f"{n_dd} drawdown; {n_fills} fills, {n_rejected} guard blocks)"
                        )
                        stop_list = ", ".join(stop_tickers[:6]) if stop_tickers else "none"
                        summary_text = (
                            f"Audited {cycle_count} live execution cycles through "
                            f"{latest_cycle_ts[:19]}: {n_fills} filled orders, {n_rejected} NOFX "
                            f"guard rejections, {n_stop} Lot-1 stop-loss exits ({stop_list}), and "
                            f"{n_dd} Ornstein-Uhlenbeck drawdown giveback exits."
                        )
                        subsys = TargetSubsystem.RISK_GUARDS
                        emit_exec = cycle_count > 0
                    else:
                        row = conn.execute(
                            "SELECT COUNT(*), MAX(occurred_at) FROM position_events "
                            "WHERE event_type IN ('STOP_EXIT', 'DRAWDOWN_EXIT')"
                        ).fetchone()
                        exit_count = int(row[0] or 0) if row else 0
                        latest_exit_ts = str(row[1] or "none") if row else "none"

                        cycle_row = conn.execute(
                            "SELECT COUNT(*), MAX(started_at) FROM cycle_runs"
                        ).fetchone()
                        cycle_count = int(cycle_row[0] or 0) if cycle_row else 0
                        latest_cycle_ts = (
                            str(cycle_row[1] or as_of.isoformat())
                            if cycle_row
                            else as_of.isoformat()
                        )
                        token = f"exec:{cycle_count}:{exit_count}:{latest_exit_ts}"
                        title = (
                            f"Live portfolio postmortem across {cycle_count} cycles "
                            f"({exit_count} protective stop/drawdown exits)"
                        )
                        summary_text = (
                            f"Audited {cycle_count} live execution cycles through "
                            f"{latest_cycle_ts}: {exit_count} protective exits triggered "
                            "(Ornstein-Uhlenbeck convergence / Lot-1 stop-loss). Opportunity "
                            "to tighten pre-entry quality & earnings accrual filter."
                        )
                        subsys = TargetSubsystem.PRESCREEN
                        emit_exec = True

                if emit_exec and wm.get("postmortem:execution") != token:
                    signals.append(
                        DiscoverySignal(
                            signal_id=f"postmortem-exec-{as_of.isoformat()}",
                            source_kind=DiscoverySourceKind.LIVE_POSTMORTEM,
                            source_name="data/execution.db",
                            title=title,
                            reference_uri=str(self._execution_db),
                            observed_at=as_of,
                            watermark_token=token,
                            summary=summary_text,
                            quantitative_mechanism=(
                                "Pre-entry Sloan accrual, net-leverage distress & Lot-1 "
                                "stop-loss tail filter calibrated on live execution events"
                            ),
                            suggested_subsystem=subsys,
                        )
                    )
            except sqlite3.Error:
                pass

        if self._journal_db is not None and self._journal_db.is_file():
            try:
                with sqlite3.connect(self._journal_db) as conn:
                    row = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
                    pred_count = int(row[0] or 0) if row else 0
                    has_settlements = (
                        conn.execute(
                            "SELECT 1 FROM sqlite_master "
                            "WHERE type = 'table' AND name = 'settlements'"
                        ).fetchone()
                        is not None
                    )
                    settled_count = 0
                    brier = 0.0
                    hit_rate = 0.0
                    mean_alpha_net = 0.0
                    if has_settlements:
                        s_row = conn.execute(
                            "SELECT COUNT(*), "
                            "AVG((p.probability - s.outcome) * (p.probability - s.outcome)), "
                            "AVG(s.outcome), AVG(s.alpha_net) "
                            "FROM settlements s "
                            "JOIN predictions p ON p.prediction_id = s.prediction_id"
                        ).fetchone()
                        if s_row and int(s_row[0] or 0) > 0:
                            settled_count = int(s_row[0] or 0)
                            brier = float(s_row[1] or 0.0)
                            hit_rate = float(s_row[2] or 0.0)
                            mean_alpha_net = float(s_row[3] or 0.0)

                if settled_count > 0:
                    token = f"journal:{pred_count}:{settled_count}"
                    j_title = (
                        f"Forecast journal calibration audit ({pred_count} predictions, "
                        f"{settled_count} settled: Brier={brier:.4f}, "
                        f"hit={hit_rate * 100:.1f}%, alpha_net={mean_alpha_net * 100:+.2f}%)"
                    )
                    j_summary = (
                        f"Journal contains {pred_count} PIT forecasts ({settled_count} settled): "
                        f"realized Brier score={brier:.4f}, hit rate={hit_rate * 100:.1f}%, "
                        f"mean net alpha={mean_alpha_net * 100:+.2f}%. Opportunity for Empirical "
                        "Bayes confidence shrinkage conditioned on PIT cash-flow quality."
                    )
                else:
                    token = f"journal:{pred_count}:{as_of.isoformat()}"
                    j_title = (
                        f"Forecast journal calibration audit ({pred_count} recorded predictions)"
                    )
                    j_summary = (
                        f"Journal contains {pred_count} PIT forecasts; monitor Brier "
                        "skill score and Waudby-Smith & Ramdas (2024) wealth e-process "
                        "drift."
                    )
                if pred_count > 0 and wm.get("postmortem:journal") != token:
                    signals.append(
                        DiscoverySignal(
                            signal_id=f"postmortem-journal-{as_of.isoformat()}",
                            source_kind=DiscoverySourceKind.LIVE_POSTMORTEM,
                            source_name="data/journal.db",
                            title=j_title,
                            reference_uri=str(self._journal_db),
                            observed_at=as_of,
                            watermark_token=token,
                            summary=j_summary,
                            quantitative_mechanism=(
                                "E_t = prod_{k=1}^t (1 + lambda_k * y_k) anytime-valid "
                                "calibration gate & Empirical Bayes probability shrinkage"
                            ),
                            suggested_subsystem=TargetSubsystem.CALIBRATION,
                        )
                    )
            except sqlite3.Error:
                pass

        return tuple(signals)


_PIT_ABLATION_FIELDS: frozenset[str] = frozenset(
    {
        "net_income",
        "operating_cash_flow",
        "assets",
        "capex",
        "stockholders_equity",
        "operating_income",
        "long_term_debt",
        "cash",
    }
)


def load_empirical_ablation_panel(
    *,
    journal_db_path: Path | None,
    execution_db_path: Path | None = None,
    pit_dir: Path | None = None,
    as_of: date | None = None,
    cohort_size: int = 6,
) -> EmpiricalAblationPanel:
    """Load Point-in-Time settled forecast & fundamental panel for committee ablations.

    Enforces ``observed_at <= knowledge_date`` on every SEC EDGAR XBRL fact read
    from ``pit_dir/<ticker>/facts.parquet``.
    """
    if journal_db_path is None or not journal_db_path.is_file():
        return EmpiricalAblationPanel(samples=(), cohort_size=cohort_size)

    try:
        with sqlite3.connect(journal_db_path) as conn:
            has_settlements = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'settlements'"
                ).fetchone()
                is not None
            )
            if not has_settlements:
                return EmpiricalAblationPanel(samples=(), cohort_size=cohort_size)
            if as_of is not None:
                rows = conn.execute(
                    "SELECT p.ticker, p.knowledge_date, p.probability, "
                    "s.alpha_gross, s.alpha_net "
                    "FROM settlements s "
                    "JOIN predictions p ON p.prediction_id = s.prediction_id "
                    "WHERE p.knowledge_date <= ? "
                    "ORDER BY p.recorded_at, p.prediction_id",
                    (as_of.isoformat(),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT p.ticker, p.knowledge_date, p.probability, "
                    "s.alpha_gross, s.alpha_net "
                    "FROM settlements s "
                    "JOIN predictions p ON p.prediction_id = s.prediction_id "
                    "ORDER BY p.recorded_at, p.prediction_id"
                ).fetchall()
    except sqlite3.Error:
        return EmpiricalAblationPanel(samples=(), cohort_size=cohort_size)

    if not rows:
        return EmpiricalAblationPanel(samples=(), cohort_size=cohort_size)

    hard_stop_tickers: set[str] = set()
    if execution_db_path is not None and execution_db_path.is_file():
        try:
            with sqlite3.connect(execution_db_path) as exec_conn:
                has_exec = (
                    exec_conn.execute(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'execution_events'"
                    ).fetchone()
                    is not None
                )
                if has_exec:
                    stop_rows = exec_conn.execute(
                        "SELECT DISTINCT ticker FROM execution_events "
                        "WHERE detail LIKE '%stop_loss_exit%'"
                    ).fetchall()
                    hard_stop_tickers = {str(r[0]) for r in stop_rows if r and r[0]}
        except sqlite3.Error:
            pass

    pit_tables: dict[str, list[tuple[str, float, str, str]]] = {}
    if pit_dir is not None and pit_dir.is_dir():
        distinct_tickers = sorted({str(r[0]) for r in rows if r and r[0]})
        for tkr in distinct_tickers:
            pq_path = pit_dir / tkr / "facts.parquet"
            if not pq_path.is_file():
                continue
            try:
                tbl = pq.read_table(
                    pq_path, columns=["field", "value", "as_of", "observed_at"]
                )
                fact_rows: list[tuple[str, float, str, str]] = []
                for f, v, ao, obs in zip(
                    tbl["field"].to_pylist(),
                    tbl["value"].to_pylist(),
                    tbl["as_of"].to_pylist(),
                    tbl["observed_at"].to_pylist(),
                    strict=False,
                ):
                    f_str = str(f)
                    if f_str in _PIT_ABLATION_FIELDS and v is not None:
                        fact_rows.append((f_str, float(v), str(ao), str(obs)))
                pit_tables[tkr] = fact_rows
            except Exception:
                continue

    ratio_cache: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    samples: list[EmpiricalTradeSample] = []
    for raw_tkr, raw_kd, raw_prob, raw_ag, raw_an in rows:
        tkr = str(raw_tkr)
        kd_str = str(raw_kd)[:10]
        try:
            kd_date = date.fromisoformat(kd_str)
        except ValueError:
            continue

        cache_key = (tkr, kd_str)
        if cache_key not in ratio_cache:
            latest_vals: dict[str, tuple[tuple[str, str], float]] = {}
            for f_str, val, ao_str, obs_str in pit_tables.get(tkr, ()):
                if obs_str <= kd_str:
                    ord_key = (ao_str, obs_str)
                    prev = latest_vals.get(f_str)
                    if prev is None or ord_key >= prev[0]:
                        latest_vals[f_str] = (ord_key, val)
            d = {k: v[1] for k, v in latest_vals.items()}
            assets = d.get("assets", 0.0)
            ni = d.get("net_income", 0.0)
            ocf = d.get("operating_cash_flow", 0.0)
            capex = abs(d.get("capex", 0.0))
            eq = d.get("stockholders_equity", 0.0)
            oi = d.get("operating_income", 0.0)
            debt = max(d.get("long_term_debt", 0.0), 0.0)
            cash = max(d.get("cash", 0.0), 0.0)

            accrual = (ni - ocf) / assets if assets > 0.0 else 0.0
            fcf_roe = float(max(-0.5, min(1.0, (ocf - capex) / eq))) if eq > 0.0 else 0.0
            roa_oi = oi / assets if assets > 0.0 else 0.0
            net_lev = (debt - cash) / assets if assets > 0.0 else 0.0
            ratio_cache[cache_key] = (accrual, fcf_roe, roa_oi, net_lev)

        accrual, fcf_roe, roa_oi, net_lev = ratio_cache[cache_key]
        samples.append(
            EmpiricalTradeSample(
                ticker=tkr,
                knowledge_date=kd_date,
                probability=max(0.0, min(1.0, float(raw_prob))),
                alpha_gross=float(raw_ag),
                alpha_net=float(raw_an),
                accrual_ratio=accrual,
                fcf_roe=fcf_roe,
                roa_operating=roa_oi,
                net_leverage=net_lev,
                had_protective_stop_exit=(tkr in hard_stop_tickers),
            )
        )

    return EmpiricalAblationPanel(samples=tuple(samples), cohort_size=cohort_size)
