"""The alphalab command-line interface.

``alphalab = "alphalab.cli:main"`` is the console-script entrypoint. The CLI is
pure orchestration: it builds requests, runs the existing pipelines, and hands
artifacts to the report formatter. No business logic lives here.
"""

from __future__ import annotations

from alphalab.cli.app import build_parser, main

__all__ = ["build_parser", "main"]
