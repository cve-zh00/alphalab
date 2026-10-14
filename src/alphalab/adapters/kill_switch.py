"""A deliberately dumb kill switch for order emission.

Why a file (and an env var), not a config flag
----------------------------------------------
The one control that must work when everything else is on fire has to be
trivial to reason about and trivial to trip by hand. Two mechanisms, both
checked immediately before every order POST:

* a **halt file** on disk (default ``HALT`` at the repository root), and
* the **environment variable** ``ALPHALAB_PAPER_HALT`` set to a truthy value.

Either one engaged blocks *all* order emission. There is no "soft" halt and no
override flag: a kill switch with an override is not a kill switch. Tripping it
is a positive act (``touch HALT``) and clearing it is another (``rm HALT``), so
its state is always visible in the filesystem and never buried in a process's
memory.

The check is pure apart from reading one path and one env var, so the broker
can call it on the hot path without ceremony and tests can point it at a temp
directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DEFAULT_HALT_FILENAME", "HALT_ENV_VAR", "KillSwitch", "KillSwitchState"]

#: Name of the on-disk halt file, resolved relative to the configured root.
DEFAULT_HALT_FILENAME = "HALT"

#: Environment variable that, when truthy, halts order emission regardless of
#: the halt file. ``"0"``, ``"false"``, ``""`` and unset are all "not halted".
HALT_ENV_VAR = "ALPHALAB_PAPER_HALT"

_FALSE_LITERALS = frozenset({"", "0", "false", "no", "off"})


@dataclass(frozen=True)
class KillSwitchState:
    """A snapshot of why (or whether) emission is halted.

    Both reasons are reported rather than short-circuited so an operator sees
    the full picture: a halt file *and* an env var can be engaged at once, and
    hiding one would make clearing the halt a guessing game.
    """

    halt_file_present: bool
    env_var_engaged: bool
    halt_file_path: str

    @property
    def engaged(self) -> bool:
        """Whether order emission must be blocked."""
        return self.halt_file_present or self.env_var_engaged

    def reason(self) -> str:
        """A human-readable, actionable explanation of the current state."""
        if not self.engaged:
            return "kill switch clear"
        causes: list[str] = []
        if self.halt_file_present:
            causes.append(f"halt file present at {self.halt_file_path}")
        if self.env_var_engaged:
            causes.append(f"{HALT_ENV_VAR} is set")
        return "order emission HALTED: " + "; ".join(causes)


class KillSwitch:
    """Checks the halt file and env var at the moment of use.

    Nothing is cached: the whole point is that flipping the switch takes effect
    on the very next check, so the state is read fresh every time.
    """

    __slots__ = ("_env", "_halt_path")

    def __init__(
        self,
        *,
        halt_path: str | Path | None = None,
        env: os._Environ[str] | dict[str, str] | None = None,
    ) -> None:
        self._halt_path = Path(halt_path) if halt_path is not None else Path(DEFAULT_HALT_FILENAME)
        self._env = os.environ if env is None else env

    def state(self) -> KillSwitchState:
        """Read the halt file and env var and return the current state."""
        return KillSwitchState(
            halt_file_present=self._halt_path.exists(),
            env_var_engaged=_is_truthy(self._env.get(HALT_ENV_VAR)),
            halt_file_path=str(self._halt_path),
        )

    def engaged(self) -> bool:
        """Convenience: whether emission is currently blocked."""
        return self.state().engaged


def _is_truthy(value: str | None) -> bool:
    """Interpret an env-var string as a boolean, erring toward "not halted"."""
    if value is None:
        return False
    return value.strip().lower() not in _FALSE_LITERALS
