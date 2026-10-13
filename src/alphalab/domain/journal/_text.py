"""Pure text utilities for bounding the length of a reflection lesson.

Settle-then-reflect caps a lesson at a few sentences on purpose: short,
bounded prose keeps the context that a downstream model reads from bloating
and degrading. These helpers count sentences and characters so the
:class:`~alphalab.domain.journal.types.Lesson` type can enforce a hard budget.

Pure module: no I/O, no state. Regex and string work only.
"""

from __future__ import annotations

import re

#: A sentence terminator is one or more of ``.``, ``!`` or ``?``. We split on
#: runs of terminators so ``"Done!!!"`` counts as a single sentence and an
#: ellipsis does not inflate the count.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def count_sentences(text: str) -> int:
    """Count the sentences in ``text`` by terminator runs.

    A *sentence* is a non-empty run of characters delimited by terminators
    (``.``, ``!``, ``?``). Trailing whitespace and a final terminator do not
    produce a spurious empty sentence, so ``"A. B."`` counts as two. Text with
    no terminator but some visible content counts as one sentence.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    pieces = [piece for piece in _SENTENCE_SPLIT_RE.split(stripped) if piece.strip()]
    return len(pieces)
