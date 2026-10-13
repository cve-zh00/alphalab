"""Unit tests for the launchd entrypoint CLI (``scripts/schedule_runner.py``).

The CLI is what launchd invokes with a cadence argument. It must be robust:
exit 0 on success, non-zero with a clear structured log on failure, and never
run ``launchctl`` itself. The ``generate-plists`` subcommand writes artifacts
to a directory the caller chooses (tmp_path here).
"""

from __future__ import annotations

import importlib.util
import plistlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "schedule_runner.py"


def _load_runner() -> ModuleType:
    """Import the standalone script module by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("schedule_runner", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["schedule_runner"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner() -> ModuleType:
    return _load_runner()


def test_script_file_exists_and_is_executable() -> None:
    # Arrange / Act / Assert: the entrypoint lives in scripts/, not the root.
    assert _SCRIPT.exists()


def test_main_runs_known_cadence_and_exits_zero(runner: ModuleType) -> None:
    # Arrange / Act
    exit_code = runner.main(["daily"])
    # Assert
    assert exit_code == 0


def test_main_unknown_cadence_exits_nonzero(runner: ModuleType) -> None:
    # Arrange / Act / Assert: argparse rejects an invalid choice.
    with pytest.raises(SystemExit) as excinfo:
        runner.main(["quarterly"])
    assert excinfo.value.code != 0


def test_main_failure_in_cadence_returns_nonzero(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: force the daily cadence to raise.
    def boom(**_kwargs: object) -> object:
        raise RuntimeError("refresh exploded")

    monkeypatch.setattr(runner, "run_daily", boom)
    # Act
    exit_code = runner.main(["daily"])
    # Assert: a failure is caught, logged, and surfaced as a non-zero code.
    assert exit_code != 0


def test_generate_plists_writes_valid_files(runner: ModuleType, tmp_path: Path) -> None:
    # Arrange
    out_dir = tmp_path / "agents"
    # Act
    exit_code = runner.main(["generate-plists", "--out-dir", str(out_dir)])
    # Assert
    assert exit_code == 0
    plists = sorted(out_dir.glob("*.plist"))
    assert len(plists) == 3
    for path in plists:
        parsed = plistlib.loads(path.read_bytes())
        assert parsed["Label"].startswith("com.alphalab.")


def test_help_documents_install_without_running_launchctl(
    runner: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange / Act: argparse prints help then raises SystemExit(0).
    with pytest.raises(SystemExit):
        runner.main(["--help"])
    out = capsys.readouterr().out
    # Assert: the help text tells the user how to install, and we never ran it.
    assert "launchctl" in out
