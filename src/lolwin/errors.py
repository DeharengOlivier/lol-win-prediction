"""Exception types raised by the package.

Kept in a module of their own so any layer can raise them without importing a
layer it should not depend on.
"""

from __future__ import annotations


class DataFormatError(ValueError):
    """The data on disk does not match the schema the pipeline requires.

    Raised at the loading boundary and nowhere else. Every message names the
    file it came from and the column that caused it, so a malformed export is
    diagnosed where it enters the system rather than three layers later.
    """


__all__ = ["DataFormatError"]
