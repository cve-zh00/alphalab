"""The four scheduling cadences (dossier §6/§9).

A cadence names *when* a kind of work runs and, implicitly, *how expensive* it
is allowed to be:

* ``DAILY`` — post-close deterministic refresh (and at most an LLM *quick*
  pass). Minutes, zero cost. Deliberately **not** the deep agentic analysis:
  daily LLM analysis is noise mining and the single largest inflator of the
  trial count, which is exactly what destroys statistical significance (the
  DSR penalises by number of trials).
* ``WEEKLY`` — the full agentic deep-dive on 1-3 names. The debate and the
  valuation live here.
* ``MONTHLY`` — the calibration run: settle due theses, score Brier/ECE,
  update the trial ledger, measure the evaporation curve.
* ``EVENT`` — ad-hoc, triggered by earnings / 8-K. Never calendar-driven.

The enum lives in the pure domain so the calendar logic and the use-case layer
can both reason about cadences without importing each other.
"""

from __future__ import annotations

from enum import StrEnum


class Cadence(StrEnum):
    """A scheduling cadence. See the module docstring for the cost rationale."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    EVENT = "event"
