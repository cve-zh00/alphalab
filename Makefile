# Resolve a working uv binary. Some environments shadow `uv` with a pyenv
# shim that fails to dispatch in a non-interactive shell, so probe it and fall
# back to the pyenv-managed install when the shim is broken.
UV := $(shell sh -c 'if uv --version >/dev/null 2>&1; then echo uv; else ls $$HOME/.pyenv/versions/*/bin/uv 2>/dev/null | head -n1; fi')

.PHONY: install test lint typecheck check acid-test

install:
	$(UV) sync

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy src

check: lint typecheck test

# Acceptance ("acid") tests: certify a real passive, reject the leakage oracle
# and reject best-of-N noise. The global --cov-fail-under gate is meaningless
# for a partial collection, so coverage is disabled here (--no-cov) WITHOUT
# touching the gate in `make check`. The target fails if any acid test fails.
acid-test:
	$(UV) run pytest -m acceptance -v -p no:cacheprovider --no-cov
