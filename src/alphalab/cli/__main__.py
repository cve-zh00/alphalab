"""Enable ``python -m alphalab.cli``."""

from __future__ import annotations

import sys

from alphalab.cli.app import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
