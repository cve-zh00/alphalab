"""The ``doctor`` command: an honest, read-only environment check.

This is the first command a new user runs, so it must be useful and honest: it
reports what is and is not configured, never modifies anything, and never
crashes on a missing dependency. Each check returns a status and a short,
actionable line.

Checks:
* Ollama reachable at the configured endpoint, and which models are present.
* Price fixtures present (how many tickers).
* Journal database present (so ``calibration`` has data).
* EDGAR ``User-Agent`` configured (required for live ingest; absent is fine
  for fixture mode but called out).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TextIO

from alphalab.adapters.llm_config import LLMSettings
from alphalab.cli.fixtures import available_tickers, default_fixtures_dir
from alphalab.report.theme import Theme, theme_for

_OLLAMA_TIMEOUT_SECONDS = 2.0


class CheckStatus(StrEnum):
    """Outcome of one doctor check."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    """One named check with a status and a human-readable detail line."""

    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class DoctorOptions:
    """Resolved options for one ``doctor`` invocation."""

    db_path: Path
    as_json: bool
    no_color: bool
    fixtures_dir: Path | None = None


def run_doctor(opts: DoctorOptions, *, out: TextIO, err: TextIO) -> int:
    """Run every read-only check and print the results. Exit 0 always (report)."""
    _ = err
    results = [
        _check_ollama(),
        _check_fixtures(opts.fixtures_dir),
        _check_journal(opts.db_path),
        _check_edgar_user_agent(),
    ]
    if opts.as_json:
        print(_as_json(results), file=out)
        return 0
    theme = theme_for(out, force_disable=opts.no_color)
    print(_format(results, theme), file=out)
    return 0


def _check_ollama() -> CheckResult:
    """Probe the configured Ollama endpoint read-only; list models if up."""
    settings = LLMSettings()
    base = settings.base_url.removesuffix("/v1").rstrip("/")
    url = f"{base}/api/tags"
    try:
        import httpx

        response = httpx.get(url, timeout=_OLLAMA_TIMEOUT_SECONDS)
    except Exception as exc:  # any failure means "not reachable"
        return CheckResult(
            "ollama",
            CheckStatus.WARN,
            f"not reachable at {base} ({type(exc).__name__}). "
            "research works offline with --no-llm.",
        )
    if response.status_code != 200:
        return CheckResult(
            "ollama",
            CheckStatus.WARN,
            f"responded {response.status_code} at {base}; use --no-llm for offline.",
        )
    names = _model_names(response)
    detail = f"up at {base}; models: {', '.join(names) if names else '(none)'}"
    return CheckResult("ollama", CheckStatus.OK, detail)


def _model_names(response: object) -> list[str]:
    """Extract model names from an Ollama ``/api/tags`` response, defensively."""
    try:
        payload = response.json()  # type: ignore[attr-defined]
    except Exception:  # defensive: malformed/absent JSON body
        return []
    models = payload.get("models", []) if isinstance(payload, dict) else []
    names: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = model.get("name") or model.get("model")
            if isinstance(name, str):
                names.append(name)
    return names


def _check_fixtures(fixtures_dir: Path | None) -> CheckResult:
    root = fixtures_dir or default_fixtures_dir()
    tickers = available_tickers(root)
    if not tickers:
        return CheckResult(
            "fixtures",
            CheckStatus.FAIL,
            f"no price fixtures found in {root}.",
        )
    preview = ", ".join(tickers[:8])
    more = "" if len(tickers) <= 8 else f", +{len(tickers) - 8} more"
    return CheckResult(
        "fixtures",
        CheckStatus.OK,
        f"{len(tickers)} tickers in {root} ({preview}{more}).",
    )


def _check_journal(db_path: Path) -> CheckResult:
    if db_path.exists():
        return CheckResult("journal", CheckStatus.OK, f"journal DB present at {db_path}.")
    return CheckResult(
        "journal",
        CheckStatus.WARN,
        f"no journal DB at {db_path}; calibration has nothing to score yet.",
    )


def _check_edgar_user_agent() -> CheckResult:
    agent = os.environ.get("ALPHALAB_EDGAR_USER_AGENT")
    if agent:
        return CheckResult(
            "edgar_user_agent",
            CheckStatus.OK,
            "ALPHALAB_EDGAR_USER_AGENT is set (required for --source live).",
        )
    return CheckResult(
        "edgar_user_agent",
        CheckStatus.WARN,
        "ALPHALAB_EDGAR_USER_AGENT not set; needed only for live EDGAR ingest.",
    )


_STATUS_STYLE: dict[CheckStatus, str] = {
    CheckStatus.OK: "green",
    CheckStatus.WARN: "yellow",
    CheckStatus.FAIL: "red",
}


def _format(results: list[CheckResult], theme: Theme) -> str:
    lines = [theme.bold("ALPHALAB DOCTOR  (read-only environment check)")]
    for result in results:
        badge = theme.paint(result.status.value.upper(), "bold", _STATUS_STYLE[result.status])
        lines.append(f"  [{badge}] {result.name}: {result.detail}")
    lines.append(theme.dim("  This check modified nothing."))
    return "\n".join(lines)


def _as_json(results: list[CheckResult]) -> str:
    payload = [{"name": r.name, "status": r.status.value, "detail": r.detail} for r in results]
    return json.dumps(payload, indent=2, sort_keys=True)
