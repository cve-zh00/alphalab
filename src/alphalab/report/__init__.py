"""Terminal reporting for alphalab research artifacts.

A dependency-free formatter (stdlib + ANSI only) that renders an
:class:`~alphalab.artifact.models.EquityResearchArtifact` for a human reader,
honouring the two visible FinRobot invariants: numbers are labelled
code-calculated, narratives LLM-assisted; and a withheld price is shown
prominently without ever hiding the verdict or the analysis.
"""

from __future__ import annotations

from alphalab.report.artifact_report import format_artifact
from alphalab.report.theme import Theme, theme_for

__all__ = ["Theme", "format_artifact", "theme_for"]
