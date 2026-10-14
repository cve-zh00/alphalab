"""Terminal theming: ANSI colour codes gated on TTY detection.

No third-party formatting dependency (no ``rich``): just the handful of ANSI
escapes we need plus the one rule that matters for piping — **if the sink is
not a TTY, every colour collapses to the empty string**, so ``| tee`` and pipes
receive clean text instead of escape soup. A caller may also force colour off
explicitly (e.g. a ``--no-color`` flag or ``NO_COLOR`` in the environment).

Pure and dependency-free: the only I/O is reading ``stream.isatty()``.
"""

from __future__ import annotations

from typing import IO, Protocol

_RESET = "\x1b[0m"

#: Named SGR codes used by the report. Kept small and legible on purpose.
_CODES: dict[str, str] = {
    "reset": _RESET,
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
}


class SupportsIsatty(Protocol):
    """Minimal stream shape the theme needs to decide on colour."""

    def isatty(self) -> bool: ...


class Theme:
    """A colour palette that is either active (TTY) or inert (pipe/file).

    When ``enabled`` is ``False`` every styling method returns its input
    unchanged, so the same formatting code produces clean text for pipes and
    coloured text for an interactive terminal.
    """

    __slots__ = ("_enabled",)

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Whether colour codes are emitted."""
        return self._enabled

    def paint(self, text: str, *styles: str) -> str:
        """Wrap ``text`` in the named ``styles`` when colour is enabled."""
        if not self._enabled or not styles:
            return text
        prefix = "".join(_CODES[s] for s in styles if s in _CODES)
        if not prefix:
            return text
        return f"{prefix}{text}{_RESET}"

    def bold(self, text: str) -> str:
        """Bold ``text`` (no-op when disabled)."""
        return self.paint(text, "bold")

    def dim(self, text: str) -> str:
        """Dim ``text`` (no-op when disabled)."""
        return self.paint(text, "dim")


def theme_for(
    stream: SupportsIsatty | IO[str],
    *,
    force_disable: bool = False,
) -> Theme:
    """Build a :class:`Theme` for ``stream``, disabling colour off a TTY.

    Args:
        stream: The destination stream; its ``isatty()`` decides colour.
        force_disable: When ``True``, colour is off regardless of the stream
            (used for ``--no-color`` / ``NO_COLOR`` / ``--json``).

    Returns:
        A theme whose colour is on only for an interactive terminal.
    """
    if force_disable:
        return Theme(enabled=False)
    try:
        is_tty = bool(stream.isatty())
    except (AttributeError, ValueError):
        is_tty = False
    return Theme(enabled=is_tty)
