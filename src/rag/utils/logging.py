"""Lightweight logging setup.

A real production system would emit structured JSON to a log aggregator. For a
teaching project, a single human-readable handler is enough — students can read
log lines while debugging without needing tooling.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring the root handler on first call.

    Honours the ``LOG_LEVEL`` environment variable (default ``INFO``).
    """
    global _CONFIGURED
    if not _CONFIGURED:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(level)
        _CONFIGURED = True
    return logging.getLogger(name)
