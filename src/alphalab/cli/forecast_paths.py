"""Default on-disk locations for the forward-evaluation clock.

The ``predict``/``settle``/``journal``/``calibration`` commands all read and
write the same two SQLite files and the same price series, so their default
locations live in one place rather than being re-derived (and risking drift)
in four command modules. Every path is overridable from the CLI; these are
only the sensible defaults the commands print on screen so the user always
knows where their record lives.

The journal and abstention databases default under ``data/`` — the repo's
working-data directory, which already holds the real ``data/prices`` series the
settler prices against. Nothing here does I/O; it only computes paths.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "default_abstention_db",
    "default_journal_db",
    "default_prices_dir",
    "project_root",
]


def project_root() -> Path:
    """Return the repository root (four levels up from this file)."""
    return Path(__file__).resolve().parents[3]


def _data_dir() -> Path:
    return project_root() / "data"


def default_journal_db() -> Path:
    """Default path of the append-only prediction journal."""
    return _data_dir() / "journal.db"


def default_abstention_db() -> Path:
    """Default path of the abstention store (declined predictions)."""
    return _data_dir() / "abstentions.db"


def default_prices_dir() -> Path:
    """Default directory of ``<ticker>.csv`` close series used to settle."""
    return _data_dir() / "prices"
