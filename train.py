#!/usr/bin/env python
"""Entry point kept so the documented ``python train.py`` keeps working.

The pipeline itself lives in the ``lolwin`` package. Install it once
(``pip install -e .``) and every flag of ``python -m lolwin`` works here too.
"""

from __future__ import annotations

import sys

try:
    from lolwin.cli import main
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    sys.exit(
        "The 'lolwin' package is not installed in this interpreter.\n"
        "Install it from the repository root with:  pip install -e '.[dev]'"
    )

if __name__ == "__main__":
    main()
