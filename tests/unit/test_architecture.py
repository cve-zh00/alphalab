"""Architecture guard: the ``alphalab.domain`` layer must stay pure.

The domain layer encodes business rules only. It must never import I/O
libraries nor reach into the ``adapters`` or ``app`` layers. This test
parses every ``.py`` file under ``src/alphalab/domain`` with the standard
library :mod:`ast` module and fails loudly on any violation, reporting the
offending file and the exact import responsible.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

# Resolve the domain package relative to this test file so the check works
# regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DOMAIN_ROOT = _PROJECT_ROOT / "src" / "alphalab" / "domain"

# Layers within alphalab that the domain is forbidden from importing.
_FORBIDDEN_LAYER_PREFIXES = (
    "alphalab.adapters",
    "alphalab.app",
)

# I/O-capable libraries that must never appear in the pure domain layer.
_FORBIDDEN_TOP_LEVEL_MODULES = frozenset(
    {
        "httpx",
        "duckdb",
        "pyarrow",
        "sqlite3",
        "requests",
    }
)


def _iter_domain_files() -> Iterator[Path]:
    """Yield every Python source file under the domain layer."""
    yield from _DOMAIN_ROOT.rglob("*.py")


def _module_is_forbidden(module: str) -> str | None:
    """Return a human-readable reason if ``module`` is forbidden, else None."""
    top_level = module.split(".", 1)[0]
    if top_level in _FORBIDDEN_TOP_LEVEL_MODULES:
        return f"I/O library '{module}'"
    for prefix in _FORBIDDEN_LAYER_PREFIXES:
        if module == prefix or module.startswith(f"{prefix}."):
            return f"forbidden layer '{module}'"
    return None


def _imported_modules(tree: ast.Module) -> Iterator[tuple[str, int]]:
    """Yield (module_name, lineno) for every import in the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            # Relative imports (level > 0) stay inside the domain package and
            # therefore cannot reach adapters/app or external I/O libraries.
            yield node.module, node.lineno


def _uses_builtin_open(tree: ast.Module) -> int | None:
    """Return the line number of a bare ``open(...)`` call, if any."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open"
        ):
            return node.lineno
    return None


def test_domain_root_exists() -> None:
    # Arrange / Act / Assert
    assert _DOMAIN_ROOT.is_dir(), f"domain root not found: {_DOMAIN_ROOT}"


def test_domain_layer_is_pure() -> None:
    # Arrange
    violations: list[str] = []

    # Act
    for path in _iter_domain_files():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        rel = path.relative_to(_PROJECT_ROOT)

        for module, lineno in _imported_modules(tree):
            reason = _module_is_forbidden(module)
            if reason is not None:
                violations.append(f"{rel}:{lineno} imports {reason}")

        open_line = _uses_builtin_open(tree)
        if open_line is not None:
            violations.append(f"{rel}:{open_line} calls builtin open() (I/O not allowed in domain)")

    # Assert
    assert not violations, "domain layer purity violated:\n" + "\n".join(violations)
