"""The three debate roles, as a closed enumeration.

Kept in its own module (mirroring ``domain/analyst/tiers.py``) so both the pure
contracts and the deterministic router can depend on the role set without any
import cycle. The set is deliberately *closed*: the router's exhaustive route
map (dossier §4, the hardening TradingAgents added after a label-drift crash)
relies on every role being one of these three and nothing else.
"""

from __future__ import annotations

from enum import StrEnum


class DebateRole(StrEnum):
    """Who is speaking (or about to speak) in the debate.

    * ``BULL`` / ``BEAR`` — the two quick-think debaters.
    * ``JUDGE`` — the deep-think arbiter that resolves by reading the history.
    """

    BULL = "bull"
    BEAR = "bear"
    JUDGE = "judge"
