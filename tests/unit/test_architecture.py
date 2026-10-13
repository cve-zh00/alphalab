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


# ---------------------------------------------------------------------------
# Raw-filter enforcement guard.
#
# ``resolve_feature`` is the single sanctioned way to turn a stored datum into
# a feature value: it raises ``LeakageError`` on look-ahead and refuses
# unregistered features. The raw filter ``_visible_records`` only *drops*
# unknowable rows silently, so a caller that builds a feature from it bypasses
# both invariants without any signal. This guard makes that bypass mechanical
# to detect: no code outside ``domain/pit`` (and its own filter tests) may
# reference the raw filter by name — everything must go through the gate.
# ---------------------------------------------------------------------------

# The raw, silent filter names that must not leak outside the PIT package.
_RAW_FILTER_NAMES = frozenset({"visible_records", "_visible_records"})

_PIT_ROOT = _DOMAIN_ROOT / "pit"
_TESTS_ROOT = _PROJECT_ROOT / "tests"

# Directories whose files are allowed to reference the raw filter directly:
# the PIT package itself (which owns the filter) and the tests that exercise
# the filter's own behaviour.
_RAW_FILTER_ALLOWED_DIRS = (_PIT_ROOT,)
_RAW_FILTER_ALLOWED_TEST_FILES = frozenset(
    {
        _TESTS_ROOT / "unit" / "test_pit_records.py",
        _TESTS_ROOT / "property" / "test_pit_invariants.py",
        # This architecture test names the raw filter in its own guard literals.
        _TESTS_ROOT / "unit" / "test_architecture.py",
    }
)


def _iter_enforced_files() -> Iterator[Path]:
    """Yield every source/test file subject to the raw-filter guard."""
    yield from (_PROJECT_ROOT / "src").rglob("*.py")
    yield from _TESTS_ROOT.rglob("*.py")


def _is_raw_filter_allowed(path: Path) -> bool:
    """Return True if ``path`` may reference the raw filter directly."""
    resolved = path.resolve()
    for allowed_dir in _RAW_FILTER_ALLOWED_DIRS:
        if resolved.is_relative_to(allowed_dir.resolve()):
            return True
    return resolved in {p.resolve() for p in _RAW_FILTER_ALLOWED_TEST_FILES}


def _raw_filter_references(tree: ast.Module) -> Iterator[tuple[str, int]]:
    """Yield (name, lineno) for every reference to a raw-filter name.

    Catches imports (``from ... import visible_records``), bare name uses, and
    attribute access (``filter.visible_records``), so neither an import nor a
    qualified call can slip through.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _RAW_FILTER_NAMES:
                    yield alias.name, node.lineno
        elif isinstance(node, ast.Name) and node.id in _RAW_FILTER_NAMES:
            yield node.id, node.lineno
        elif isinstance(node, ast.Attribute) and node.attr in _RAW_FILTER_NAMES:
            yield node.attr, node.lineno


def test_domain_root_exists() -> None:
    # Arrange / Act / Assert
    assert _DOMAIN_ROOT.is_dir(), f"domain root not found: {_DOMAIN_ROOT}"


def test_raw_filter_is_only_reachable_through_resolve_feature() -> None:
    """No module outside domain/pit may touch the raw silent filter.

    Every feature value a strategy consumes must flow through
    ``resolve_feature``; the raw filter is reserved for the PIT package and the
    tests that validate the filter itself.
    """
    # Arrange
    violations: list[str] = []

    # Act
    for path in _iter_enforced_files():
        if _is_raw_filter_allowed(path):
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        rel = path.relative_to(_PROJECT_ROOT)
        for name, lineno in _raw_filter_references(tree):
            violations.append(
                f"{rel}:{lineno} references raw filter '{name}'; use resolve_feature instead"
            )

    # Assert
    assert not violations, (
        "raw PIT filter reached outside domain/pit (bypasses the leakage gate):\n"
        + "\n".join(violations)
    )


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
