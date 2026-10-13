"""Pure text utilities for detecting and scrubbing numeric amounts in prose.

A withheld price target is not safe just because the headline slot is nulled:
the same number can hide inside a narrative sentence or a fallback slot and
"resurrect". These helpers recognise a numeric amount in free text — tolerant
of trivial formatting (``150``, ``150.0``, ``150.00``, ``$150``, thousands
separators) — and replace every occurrence with a visible redaction marker.

Pure module: no I/O, no state. Regex and string work only.
"""

from __future__ import annotations

import math
import re

# Redaction marker stamped in place of a scrubbed amount so the withhold is
# visible rather than silent.
REDACTION_MARKER = "[REDACTED]"

# A run of digits with optional thousands separators and an optional decimal
# fraction: matches 150, 1000, 1,250, 150.00, 1,250.5 — the shapes a price
# takes in prose. Either a comma-grouped integer (1,250) or a plain digit run
# (1000); longest-match semantics plus the decimal tail handle 150.00.
_NUMBER_RE = re.compile(r"(?<![\d.])(?:\$\s?)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\d])")


def _canonical(amount: float, decimals: int) -> float:
    """Round an amount to the comparison precision used for matching."""
    return round(amount, decimals)


def _parse_amount(token: str) -> float | None:
    """Parse a matched token (``$1,250.00``) into a float, or None."""
    cleaned = token.replace("$", "").replace(",", "").strip()
    if cleaned == "":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def text_contains_amount(text: str, amount: float, *, decimals: int) -> bool:
    """True when ``text`` mentions ``amount`` at the given decimal precision.

    Non-finite targets (NaN/inf) are never considered "contained": they cannot
    be the legitimate price a narrative quotes, so matching them would be
    meaningless.
    """
    if not math.isfinite(amount):
        return False
    target = _canonical(amount, decimals)
    for match in _NUMBER_RE.finditer(text):
        parsed = _parse_amount(match.group())
        if parsed is None:
            continue
        if _canonical(parsed, decimals) == target:
            return True
    return False


def scrub_amount(text: str, amount: float, *, decimals: int) -> str:
    """Replace every mention of ``amount`` in ``text`` with the marker.

    Trivial formatting variants collapse to the same canonical value, so
    ``150``, ``150.00`` and ``$150`` are all redacted. A non-finite amount is
    returned unchanged (nothing legitimate to scrub).
    """
    if not math.isfinite(amount):
        return text
    target = _canonical(amount, decimals)

    def _replace(match: re.Match[str]) -> str:
        parsed = _parse_amount(match.group())
        if parsed is not None and _canonical(parsed, decimals) == target:
            return REDACTION_MARKER
        return match.group()

    return _NUMBER_RE.sub(_replace, text)


def has_double_decimal(text: str) -> bool:
    """Detect a malformed amount with two decimal separators (``1.234.56``).

    A number carrying two dots (optionally with a currency sign) is the
    signature of a string that was concatenated wrong — e.g. ``$2172.062.06``
    glued from two fragments. Thousands-separated integers like ``1,234,567``
    use commas and never trip this.
    """
    pattern = re.compile(r"\$?\d+\.\d+\.\d+")
    return pattern.search(text) is not None


def has_glued_digits(text: str) -> bool:
    """Detect an implausibly long unbroken digit run that betrays a bad join.

    Two amounts concatenated without a separator (``150250``) or a price glued
    to another figure produce a digit run longer than any real per-share price
    needs. Nine or more consecutive digits (ignoring separators) is the signal;
    legitimately grouped figures carry commas and break the run.
    """
    pattern = re.compile(r"\d{9,}")
    return pattern.search(text) is not None
