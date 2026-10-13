"""The two LLM tiers, after the TradingAgents quick-/deep-think split.

* ``QUICK`` — a small, fast model for extraction and classification (the
  analyst role: pull qualitative facts out of text).
* ``DEEP`` — a larger model for synthesis and judgement (the arbiter role:
  weave the thesis narrative).

The enum lives in the pure domain so the pipeline can request a tier without
knowing which concrete model backs it; the adapter maps a tier to a model
name via configuration.
"""

from __future__ import annotations

from enum import StrEnum


class ModelTier(StrEnum):
    """Which capability tier a step requires."""

    QUICK = "quick"
    DEEP = "deep"
