"""Structured logging helper (named logging_utils to avoid shadowing stdlib)."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "maestro", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
        )
        root = logging.getLogger("maestro")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("maestro") else f"maestro.{name}")
