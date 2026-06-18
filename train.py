#!/usr/bin/env python
"""Thin wrapper so the pipeline can be run as ``python train.py`` from the root.

It just adds ``src/`` to the path and delegates to :func:`lolwin.cli.main`, so
all CLI flags (``--sample``, ``--no-tune``, ``--output-csv`` ...) work here too.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lolwin.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
