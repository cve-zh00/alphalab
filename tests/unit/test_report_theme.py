"""Unit tests for the report theme: TTY-gated ANSI colour."""

from __future__ import annotations

import io

from alphalab.report.theme import Theme, theme_for


class _FakeTTY(io.StringIO):
    """A StringIO that reports itself as a terminal."""

    def isatty(self) -> bool:
        return True


class TestThemePainting:
    def test_enabled_theme_wraps_text_in_ansi(self) -> None:
        theme = Theme(enabled=True)
        painted = theme.paint("hi", "red")
        assert painted.startswith("\x1b[")
        assert painted.endswith("\x1b[0m")
        assert "hi" in painted

    def test_disabled_theme_returns_text_unchanged(self) -> None:
        theme = Theme(enabled=False)
        assert theme.paint("hi", "red") == "hi"
        assert theme.bold("hi") == "hi"
        assert theme.dim("hi") == "hi"

    def test_unknown_style_is_ignored(self) -> None:
        theme = Theme(enabled=True)
        assert theme.paint("hi", "not-a-style") == "hi"

    def test_no_styles_returns_text(self) -> None:
        assert Theme(enabled=True).paint("hi") == "hi"


class TestTTYDetection:
    def test_colors_disabled_when_not_a_tty(self) -> None:
        """A plain StringIO is not a TTY: colour must be off (pipe-safe)."""
        theme = theme_for(io.StringIO())
        assert theme.enabled is False
        assert theme.paint("x", "green") == "x"

    def test_colors_enabled_on_a_tty(self) -> None:
        theme = theme_for(_FakeTTY())
        assert theme.enabled is True
        assert "\x1b[" in theme.paint("x", "green")

    def test_force_disable_overrides_a_tty(self) -> None:
        theme = theme_for(_FakeTTY(), force_disable=True)
        assert theme.enabled is False

    def test_stream_without_isatty_defaults_to_disabled(self) -> None:
        class _NoIsatty:
            pass

        theme = theme_for(_NoIsatty())  # type: ignore[arg-type]
        assert theme.enabled is False
