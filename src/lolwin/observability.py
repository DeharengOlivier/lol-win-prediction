"""Logging for the package.

Two rules, both of them library manners:

- The package never writes to stdout. Progress goes to a logger the calling
  application owns, so a notebook or another program can silence it, redirect
  it, or raise its level.
- Importing the package configures nothing. A ``NullHandler`` is attached so an
  unconfigured application sees no "No handlers could be found" warning, and
  :func:`configure_logging` is opt-in, called by the CLI and by nobody else.

Records use ``key=value`` fields so a run can be grepped rather than read.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "lolwin"

LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_HANDLER_NAME = "lolwin-console"

# Attached at import time so an application that never configures logging still
# sees nothing rather than a warning about missing handlers.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return the package logger for a module.

    Args:
        name: Normally ``__name__``.

    Returns:
        A child of the ``lolwin`` logger.
    """
    return logging.getLogger(name)


def configure_logging(level: str = "WARNING") -> None:
    """Send the package's records to stderr at ``level``.

    Idempotent: calling it again re-levels the existing handler instead of
    stacking a second one, so every record would otherwise be printed twice.

    Args:
        level: One of :data:`LOG_LEVELS`.

    Raises:
        ValueError: ``level`` is not one of :data:`LOG_LEVELS`.
    """
    normalized = level.strip().upper()
    if normalized not in LOG_LEVELS:
        raise ValueError(
            f"Unknown log level {level!r}. Accepted: {', '.join(LOG_LEVELS)}."
        )

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(normalized)

    for handler in logger.handlers:
        if handler.get_name() == _HANDLER_NAME:
            handler.setLevel(normalized)
            return

    handler = logging.StreamHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(normalized)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)


__all__ = ["LOGGER_NAME", "LOG_LEVELS", "configure_logging", "get_logger"]
