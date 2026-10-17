#!/usr/bin/env python3
"""Scan a quantitative/agentic trading project across the 7 institutional audit pillars.

Inspects code modules, valuation operators, agentic pipelines, risk guards,
portfolio sizing rules, and live SQLite telemetry (journal.db, abstentions.db,
execution.db, pit store, edgar cache) to produce a factual baseline before
proposing improvements.

Usage:
    python3 scan_project_anatomy.py --root .
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


def _count_py_files(directory: Path) -> dict[str, Any]:
    if not directory.exists():
        return {"exists": False, "files": 0, "lines": 0, "modules": []}
    files = sorted(directory.rglob("*.py"))
    lines = sum(len(f.read_text(errors="ignore").splitlines()) for f in files)
    return {
        "exists": True,
        "files": len(files),
        "lines": lines,
        "modules": [str(f.relative_to(directory)) for f in files if f.name != "__init__.py"],
    }


def _inspect_sqlite(db_path: Path, query: str) -> Any:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            return conn.execute(query).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"


def scan_repository(root: Path) -> dict[str, Any]:
    src = root / "src" / "alphalab"
    data = root / "data"

    pillars = {
        "pillar_1_pit_and_data": {
            "domain_pit": _count_py_files(src / "domain" / "pit"),
            "edgar_cache_files": len(list((data / "edgar_cache").glob("CIK*.json")))
            if (data / "edgar_cache").exists()
            else 0,
            "pit_ticker_partitions": len(
                [p for p in (data / "pit").iterdir() if p.is_dir()]
            )
            if (data / "pit").exists()
            else 0,
            "sp500_constituents_cached": (data / "sp500_constituents.json").exists(),
        },
        "pillar_2_valuation_compute": {
            "domain_compute": _count_py_files(src / "domain" / "compute"),
            "operators": _count_py_files(src / "domain" / "compute" / "operators"),
        },
        "pillar_3_agentic_and_debate": {
            "domain_analyst": _count_py_files(src / "domain" / "analyst"),
            "domain_debate": _count_py_files(src / "domain" / "debate"),
            "domain_contract": _count_py_files(src / "domain" / "contract"),
        },
        "pillar_4_calibration_and_journal": {
            "domain_forecast": _count_py_files(src / "domain" / "forecast"),
            "domain_journal": _count_py_files(src / "domain" / "journal"),
            "journal_summary": {
                "total_predictions": _inspect_sqlite(
                    data / "journal.db", "SELECT COUNT(*) FROM predictions"
                ),
                "by_llm_version": _inspect_sqlite(
                    data / "journal.db",
                    "SELECT COALESCE(llm_version, 'live-llm'), direction, COUNT(*) "
                    "FROM predictions GROUP BY llm_version, direction",
                ),
                "settled_count": _inspect_sqlite(
                    data / "journal.db", "SELECT COUNT(*) FROM settlements"
                ),
                "lessons_count": _inspect_sqlite(
                    data / "journal.db", "SELECT COUNT(*) FROM lessons"
                ),
                "abstentions_by_reason": _inspect_sqlite(
                    data / "abstentions.db",
                    "SELECT reason, COUNT(*) FROM abstentions GROUP BY reason",
                ),
            },
        },
        "pillar_5_statistical_certification": {
            "domain_stats": _count_py_files(src / "domain" / "stats"),
            "domain_backtest": _count_py_files(src / "domain" / "backtest"),
            "domain_skeptic": _count_py_files(src / "domain" / "skeptic"),
        },
        "pillar_6_portfolio_and_costs": {
            "domain_costs": _count_py_files(src / "domain" / "costs"),
            "agentic_portfolio_exists": (src / "app" / "agentic_portfolio.py").exists(),
        },
        "pillar_7_execution_and_nofx_guards": {
            "domain_execution": _count_py_files(src / "domain" / "execution"),
            "autopilot_runner_exists": (src / "app" / "autopilot_runner.py").exists(),
            "recent_execution_cycles": _inspect_sqlite(
                data / "execution.db",
                "SELECT cycle_id, kind, COUNT(*) FROM execution_events "
                "GROUP BY cycle_id, kind ORDER BY MAX(recorded_at) DESC LIMIT 12",
            ),
        },
    }
    return {
        "project_root": str(root.resolve()),
        "research_docs": [
            str(p.relative_to(root)) for p in sorted((root / "docs").glob("*.md"))
        ]
        if (root / "docs").exists()
        else [],
        "pillars": pillars,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root path")
    args = parser.parse_args()
    report = scan_repository(args.root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
