"""Human-readable terminal report for an :class:`EquityResearchArtifact`.

This module *formats*; it does not decide anything. Every number it prints
comes already-computed on the artifact (or on the ``ProvenancedValue`` lineage
behind the price target). The report encodes the two FinRobot invariants the
dossier insists be *visible on screen*, not merely true in code:

1. **"numbers are code-calculated, narratives are LLM-assisted"** — the price
   target, range, upside and per-method values are explicitly labelled as
   code-calculated; the LLM prose is printed in a clearly separated block.
2. **"withholds the price, never the judgment"** — when the contract gate set
   ``valuation_withheld=True`` the report shows the withholding prominently
   with its ``contract_stamps`` *and still prints the verdict and the full
   analysis*.

A ``--provenance`` view walks ``ProvenancedValue.tree()`` / ``leaves()`` so a
reader can trace the headline number back to its raw inputs.

Pure: the only inputs are the artifact and a :class:`Theme`; output is a
string. No I/O, no business logic.
"""

from __future__ import annotations

from typing import Any

from alphalab.artifact.models import EquityResearchArtifact, Verdict
from alphalab.domain.analyst.contracts import RunMetadata
from alphalab.domain.compute import ProvenancedValue
from alphalab.report.theme import Theme

_DISCLAIMER = (
    "DISCLAIMER: Technical research output, NOT financial advice. "
    "Numbers are code-calculated; narratives are LLM-assisted."
)

_VERDICT_STYLE: dict[Verdict, str] = {
    Verdict.BUY: "green",
    Verdict.HOLD: "yellow",
    Verdict.SELL: "red",
}


def format_artifact(
    artifact: EquityResearchArtifact,
    *,
    theme: Theme,
    run_metadata: RunMetadata | None = None,
    data_source: str | None = None,
    show_provenance: bool = False,
) -> str:
    """Render the full report for ``artifact`` as a single string.

    Args:
        artifact: The gated artifact to present.
        theme: Colour palette (inert when not a TTY).
        run_metadata: Optional run provenance (model, cutoff, dataset hash).
        data_source: Optional human label of where the inputs came from; shown
            prominently so demo/fixture data is never mistaken for real data.
        show_provenance: When ``True``, append the derivation tree of the
            headline number.
    """
    lines: list[str] = []
    lines.append(_header(artifact, theme))
    if data_source:
        lines.append(_data_source_banner(data_source, theme))
    lines.append("")
    lines.extend(_verdict_block(artifact, theme))
    lines.append("")
    lines.extend(_valuation_block(artifact, theme))
    lines.append("")
    lines.extend(_methods_block(artifact, theme))
    lines.append("")
    lines.extend(_narrative_block(artifact, theme))
    if run_metadata is not None:
        lines.append("")
        lines.extend(_metadata_block(run_metadata, theme))
    if show_provenance:
        lines.append("")
        lines.extend(_provenance_block(artifact, theme))
    lines.append("")
    lines.append(theme.dim(_DISCLAIMER))
    return "\n".join(lines)


def _header(artifact: EquityResearchArtifact, theme: Theme) -> str:
    kd = artifact.knowledge_date.isoformat() if artifact.knowledge_date else "n/a"
    title = f"EQUITY RESEARCH — {artifact.ticker}  (knowledge date: {kd})"
    return theme.bold(title)


def _data_source_banner(data_source: str, theme: Theme) -> str:
    return theme.paint(f"DATA SOURCE: {data_source}", "cyan")


def _verdict_block(artifact: EquityResearchArtifact, theme: Theme) -> list[str]:
    style = _VERDICT_STYLE.get(artifact.verdict, "bold")
    badge = theme.paint(f" {artifact.verdict.value} ", "bold", style)
    return [f"VERDICT: {badge}"]


def _valuation_block(artifact: EquityResearchArtifact, theme: Theme) -> list[str]:
    """Price target, range and implied upside — all code-calculated."""
    lines = [theme.bold("VALUATION  [code-calculated]")]
    if artifact.valuation_withheld:
        lines.extend(_withheld_block(artifact, theme))
        return lines
    target = artifact.price_target
    if target is None:
        lines.append("  Price target: (none reported)")
    else:
        lines.append(f"  Price target: {_money(target.value)} {target.unit}")
        lines.append(f"  Current price: {_money(artifact.current_price)}")
        lines.append("  " + _upside_line(target.value, artifact.current_price, theme))
    rng = artifact.price_range
    if rng is not None:
        lines.append(f"  Range: {_money(rng.low)} / {_money(rng.mid)} / {_money(rng.high)}")
    return lines


def _withheld_block(artifact: EquityResearchArtifact, theme: Theme) -> list[str]:
    """The 'withhold the price, never the judgment' panel, shown prominently."""
    header = theme.paint("  PRICE WITHHELD by output contract gate", "bold", "red")
    lines = [header]
    if artifact.contract_stamps:
        lines.append("  Contract stamps:")
        lines.extend(f"    - {theme.paint(stamp, 'red')}" for stamp in artifact.contract_stamps)
    else:
        lines.append("    - (no stamps recorded)")
    rng = artifact.price_range
    if rng is not None:
        lines.append(
            "  Retained range (point withheld): "
            f"{_money(rng.low)} / {_money(rng.mid)} / {_money(rng.high)}"
        )
    lines.append(theme.dim("  The verdict and analysis below are unaffected by the withholding."))
    return lines


def _upside_line(target: float, current: float, theme: Theme) -> str:
    if current <= 0.0:
        return "Implied upside/downside: n/a (no positive current price)"
    pct = (target - current) / current * 100.0
    label = "upside" if pct >= 0.0 else "downside"
    style = "green" if pct >= 0.0 else "red"
    return f"Implied {label}: {theme.paint(f'{pct:+.1f}%', style)}"


def _methods_block(artifact: EquityResearchArtifact, theme: Theme) -> list[str]:
    lines = [theme.bold("METHODS USED  [code-calculated]")]
    if not artifact.methods_used:
        lines.append("  (no methods recorded)")
        return lines
    for name in artifact.methods_used:
        value = artifact.method_values.get(name)
        rendered = _money(value) if value is not None else "n/a"
        lines.append(f"  - {name}: {rendered}")
    if artifact.consensus_relative_span is not None:
        lines.append(f"  Cross-method relative span: {artifact.consensus_relative_span:.3f}")
    return lines


def _narrative_block(artifact: EquityResearchArtifact, theme: Theme) -> list[str]:
    lines = [theme.bold("NARRATIVE  [LLM-assisted]")]
    if not artifact.narrative_slots:
        lines.append("  (no narrative)")
        return lines
    for slot, text in artifact.narrative_slots.items():
        lines.append(f"  {theme.paint(slot, 'blue')}:")
        lines.append(f"    {text}")
    return lines


def _metadata_block(meta: RunMetadata, theme: Theme) -> list[str]:
    return [
        theme.bold("RUN METADATA"),
        f"  Model: {meta.model_name} ({meta.tier.value}, v{meta.llm_version})",
        f"  Model cutoff: {meta.llm_cutoff.isoformat()}",
        f"  Knowledge date: {meta.knowledge_date.isoformat()}",
        f"  Dataset hash: {meta.dataset_hash}",
    ]


def _provenance_block(artifact: EquityResearchArtifact, theme: Theme) -> list[str]:
    lines = [theme.bold("PROVENANCE  [derivation of the headline number]")]
    target = artifact.price_target
    if not isinstance(target, ProvenancedValue):
        lines.append("  (headline number has no provenance tree; it was withheld or untraced)")
        return lines
    lines.extend(_render_tree(target.tree(), depth=0))
    leaves = target.leaves()
    lines.append("  Raw-input leaves:")
    for node in leaves:
        for name, val in node.inputs:
            lines.append(f"    - {name} = {val} ({node.unit})")
    return lines


def _render_tree(tree: dict[str, Any], *, depth: int) -> list[str]:
    indent = "  " + "  " * depth
    op = tree["operator"]
    value = tree["value"]
    unit = tree["unit"]
    lines = [f"{indent}{op} = {value} ({unit})"]
    parents = tree.get("parents", [])
    for parent in parents:
        lines.extend(_render_tree(parent, depth=depth + 1))
    return lines


def _money(value: float) -> str:
    """Format a scalar as a 2-decimal figure (code-calculated numbers only)."""
    return f"{value:,.2f}"
