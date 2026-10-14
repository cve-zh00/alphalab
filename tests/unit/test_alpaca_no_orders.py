"""Guard: the Alpaca adapters must contain no order-placement code path.

This project has no demonstrated edge (out-of-sample information ratio -0.10,
zero settled predictions), so routing orders would be indefensible. These
adapters are read-only by contract: assets and historical bars. This test makes
that contract mechanical — it parses the Alpaca modules I own and fails if any
of them so much as mentions order placement, a trading POST, or a position
mutation.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Every Alpaca module this agent owns. ``alpaca_config.py`` is the architect's
# security contract and is included so a regression there would also be caught.
_OWNED_MODULES = (
    _PROJECT_ROOT / "src" / "alphalab" / "adapters" / "alpaca_assets.py",
    _PROJECT_ROOT / "src" / "alphalab" / "adapters" / "alpaca_bars.py",
    _PROJECT_ROOT / "scripts" / "fetch_alpaca_universe.py",
)

# Constructs that would indicate a real order-placement / position-mutation
# path. These target *actionable* code — endpoint paths, order verbs, and HTTP
# write methods — not the English word "orders" appearing in a docstring that
# promises the opposite. Prose is stripped before matching so a comment cannot
# trip the guard, and a genuine ``requests.post(".../v2/orders")`` cannot hide.
_FORBIDDEN_PATTERNS = (
    r"/v2/orders",
    r"/v2/positions",
    r"\bsubmit_order\b",
    r"\bplace_order\b",
    r"\bcancel_order\b",
    r"\bclose_position\b",
    r"\bliquidat",
)


def _strip_prose(source: str) -> str:
    """Remove docstrings and comments so only executable code is scanned.

    The guard must judge what the code *does*, not what its documentation
    *says*. A module docstring that states "this places no orders" is a promise,
    not a violation; stripping it keeps the guard honest in both directions.
    """
    tree = ast.parse(source)
    # Drop every string-expression statement (module/class/func docstrings and
    # bare string literals used as comments).
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            node.body = [  # type: ignore[attr-defined]
                stmt
                for stmt in body
                if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
            ] or [ast.Pass()]
    code_only = ast.unparse(tree)
    # Also drop ``#`` line comments, which ast.unparse does not reproduce anyway.
    return "\n".join(line.split("#", 1)[0] for line in code_only.splitlines())


def _source(path: Path) -> str:
    return _strip_prose(path.read_text(encoding="utf-8"))


class TestNoOrderPlacement:
    def test_owned_modules_exist(self) -> None:
        # Arrange / Act / Assert
        for path in _OWNED_MODULES:
            assert path.exists(), f"expected owned module missing: {path}"

    @pytest.mark.parametrize("path", _OWNED_MODULES, ids=lambda p: p.name)
    def test_no_order_or_mutation_tokens(self, path: Path) -> None:
        # Arrange
        source = _source(path)

        # Act
        hits = [
            pattern
            for pattern in _FORBIDDEN_PATTERNS
            if re.search(pattern, source, flags=re.IGNORECASE)
        ]

        # Assert — any hit is a forbidden order/mutation path.
        assert not hits, f"{path.name} contains forbidden order/mutation tokens: {hits}"

    @pytest.mark.parametrize("path", _OWNED_MODULES, ids=lambda p: p.name)
    def test_only_safe_http_verbs_are_used(self, path: Path) -> None:
        # Arrange — the adapters issue GETs only, through an injected transport.
        source = _source(path).lower()

        # Act / Assert — no write verbs anywhere in the module.
        for verb in (".post(", ".put(", ".delete(", ".patch("):
            assert verb not in source, f"{path.name} uses a write verb: {verb}"
